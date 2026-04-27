"""Monkey-patch layer that routes vLLM attention through KVTC."""

from __future__ import annotations

import math
import re
import time
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch

from .common import CalibrationData, PCAEntry
from .gpu_ops import batch_quantize, greedy_bit_allocation
from .pca import apply_rope_inverse, pca_transform
from .vllm_triton import decode_attention_from_kvtc, dense_attention_state, merge_attention_states


def _apply_soft_cap_torch(scores: torch.Tensor, logits_soft_cap: float | None) -> torch.Tensor:
    """Mirror of kvtc.vllm_triton._apply_soft_cap, but accepts the 0-as-disabled sentinel."""
    if logits_soft_cap is None or logits_soft_cap <= 0:
        return scores
    return logits_soft_cap * torch.tanh(scores / logits_soft_cap)


NEG_INF = float("-inf")


@dataclass(frozen=True)
class RequestSpan:
    """Logical request slice inside a vLLM attention step."""

    request_id: str
    start: int
    end: int
    positions: torch.Tensor
    seq_len: int
    query_len: int


@dataclass
class QuantizedTensorSpec:
    """Precomputed PCA/quantization state for one tensor kind."""

    active_components: torch.Tensor
    bit_widths: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor
    projection_basis: torch.Tensor
    basis_t: torch.Tensor
    mean: torch.Tensor
    index_dtype: torch.dtype


@dataclass
class KVTCGroupConfig:
    """Calibration-backed quantization config for one KV head group."""

    group_idx: int
    head_indices: tuple[int, ...]
    key: QuantizedTensorSpec
    value: QuantizedTensorSpec


@dataclass
class QuantizedGroupStorage:
    """Compressed indices for one sequence and one head group."""

    key_indices: torch.Tensor | None = None
    value_indices: torch.Tensor | None = None


@dataclass
class KVTCSequenceState:
    """Per-request KV state for one transformer layer."""

    request_id: str
    raw_keys: List[torch.Tensor] = field(default_factory=list)
    raw_values: List[torch.Tensor] = field(default_factory=list)
    raw_positions: List[torch.Tensor] = field(default_factory=list)
    sinks_keys: torch.Tensor | None = None
    sinks_values: torch.Tensor | None = None
    window_keys: torch.Tensor | None = None
    window_values: torch.Tensor | None = None
    window_positions: torch.Tensor | None = None
    middle_positions: torch.Tensor | None = None
    compressed: Dict[int, QuantizedGroupStorage] = field(default_factory=dict)
    finalized: bool = False

    def has_prefill(self) -> bool:
        return bool(self.raw_keys)


@dataclass
class PatchedLayer:
    """Bookkeeping for a patched attention layer."""

    prefix: str
    layer_idx: int
    layer: Any
    impl: Any
    state: "KVTCLayerState"
    original_forward: Any
    original_do_kv_cache_update: Any | None
    original_kv_cache: Any | None = None


def _to_cpu_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().flatten().tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _num_actual_tokens(attn_metadata: Any, fallback: int) -> int:
    if attn_metadata is None:
        return fallback
    for name in ("num_actual_tokens", "num_input_tokens"):
        value = getattr(attn_metadata, name, None)
        if value is not None:
            return min(int(value), fallback)
    return fallback


def _request_id_from_block_table(attn_metadata: Any, row_idx: int, seq_len: int, start: int, end: int) -> str:
    block_table = getattr(attn_metadata, "block_table", None)
    if block_table is None:
        block_table = getattr(attn_metadata, "block_tables", None)
    if block_table is None:
        return f"row-{row_idx}-seq-{seq_len}-tok-{start}-{end}"
    row = block_table[row_idx]
    if isinstance(row, torch.Tensor):
        block_ids = row.detach().cpu().flatten().tolist()
    else:
        block_ids = list(row)
    filtered = [int(block_id) for block_id in block_ids if int(block_id) >= 0]
    if filtered:
        return "blocks:" + ",".join(str(block_id) for block_id in filtered)
    return f"row-{row_idx}-seq-{seq_len}-tok-{start}-{end}"


def extract_request_spans(
    attn_metadata: Any,
    num_tokens: int,
    *,
    device: torch.device,
) -> List[RequestSpan]:
    """Split one attention step into per-request token spans."""

    actual_tokens = _num_actual_tokens(attn_metadata, num_tokens)
    if attn_metadata is None:
        positions = torch.arange(actual_tokens, device=device, dtype=torch.long)
        return [RequestSpan("request-0", 0, actual_tokens, positions, actual_tokens, actual_tokens)]

    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if query_start_loc is None or seq_lens is None:
        positions = torch.arange(actual_tokens, device=device, dtype=torch.long)
        return [RequestSpan("request-0", 0, actual_tokens, positions, actual_tokens, actual_tokens)]

    qsl = _to_cpu_list(query_start_loc)
    if len(qsl) < 2:
        positions = torch.arange(actual_tokens, device=device, dtype=torch.long)
        return [RequestSpan("request-0", 0, actual_tokens, positions, actual_tokens, actual_tokens)]

    seq_lens_list = _to_cpu_list(seq_lens)
    spans: List[RequestSpan] = []
    for row_idx, start in enumerate(qsl[:-1]):
        end = min(qsl[row_idx + 1], actual_tokens)
        if end <= start:
            continue
        query_len = end - start
        seq_len = seq_lens_list[row_idx] if row_idx < len(seq_lens_list) else end
        pos_start = max(seq_len - query_len, 0)
        positions = torch.arange(pos_start, seq_len, device=device, dtype=torch.long)
        request_id = _request_id_from_block_table(attn_metadata, row_idx, seq_len, start, end)
        spans.append(RequestSpan(request_id, start, end, positions, seq_len, query_len))

    if not spans:
        positions = torch.arange(actual_tokens, device=device, dtype=torch.long)
        spans.append(RequestSpan("request-0", 0, actual_tokens, positions, actual_tokens, actual_tokens))
    return spans


def _is_pure_decode(spans: Sequence[RequestSpan]) -> bool:
    return bool(spans) and all(span.query_len == 1 for span in spans)


def _parse_layer_idx(prefix: str, fallback: int) -> int:
    matches = re.findall(r"\d+", prefix)
    if not matches:
        return fallback
    return int(matches[-1])


def _smallest_index_dtype(bit_widths: torch.Tensor) -> torch.dtype:
    if bit_widths.numel() == 0:
        return torch.uint8
    max_bits = int(bit_widths.max().item())
    if max_bits <= 8:
        return torch.uint8
    if max_bits <= 15:
        return torch.int16
    return torch.int32


def _static_quant_params(entry: PCAEntry) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if entry.bit_widths is not None and entry.scales is not None and entry.zero_points is not None:
        return (
            entry.bit_widths.detach().clone().to(torch.int64),
            entry.scales.detach().clone().to(torch.float32),
            entry.zero_points.detach().clone().to(torch.float32),
        )
    if entry.pca_mins is None or entry.pca_maxs is None:
        raise ValueError(
            "Calibration entry is missing static quantization ranges. "
            "Recompute calibration with the updated calibrator."
        )
    bit_widths = greedy_bit_allocation(entry.eigenvalues, entry.bit_budget)
    mins = entry.pca_mins.to(torch.float32)
    maxs = entry.pca_maxs.to(torch.float32)
    bw = bit_widths.to(torch.float32)
    nonzero = bw > 0
    safe_bw = torch.where(nonzero, bw, torch.ones_like(bw))
    qmax = (2.0 ** safe_bw) - 1.0
    qmax = torch.where(nonzero, qmax, torch.ones_like(qmax))
    span = (maxs - mins).clamp(min=1e-8)
    scales = torch.where(nonzero, span / qmax, torch.ones_like(span))
    zero_points = torch.where(nonzero, -mins / scales, torch.zeros_like(mins))
    return bit_widths.to(torch.int64), scales, zero_points


def _build_quant_spec(entry: PCAEntry) -> QuantizedTensorSpec:
    bit_widths, scales, zero_points = _static_quant_params(entry)
    active = torch.nonzero(bit_widths > 0, as_tuple=False).flatten()
    eigenvectors = entry.eigenvectors.to(torch.float32)
    projection_basis = eigenvectors[:, active].contiguous()
    basis_t = eigenvectors.transpose(0, 1)[active].contiguous()
    return QuantizedTensorSpec(
        active_components=active,
        bit_widths=bit_widths[active].contiguous(),
        scales=scales[active].contiguous(),
        zero_points=zero_points[active].contiguous(),
        projection_basis=projection_basis,
        basis_t=basis_t,
        mean=entry.mean.to(torch.float32).contiguous(),
        index_dtype=_smallest_index_dtype(bit_widths[active]),
    )


def _append_tensor(existing: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
    new = new.contiguous()
    if existing is None:
        return new
    return torch.cat((existing, new), dim=0)


def _dummy_like_cache(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.empty((1,), device=value.device, dtype=value.dtype)
    if isinstance(value, list):
        return [_dummy_like_cache(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_dummy_like_cache(item) for item in value)
    if isinstance(value, dict):
        return {key: _dummy_like_cache(item) for key, item in value.items()}
    return value


def resolve_attention_layers(model: Any) -> List[tuple[str, Any]]:
    """Locate compiled vLLM attention layers for monkey-patching."""

    queue: List[Any] = [model]
    seen: set[int] = set()
    attr_names = (
        "model_runner",
        "worker",
        "driver_worker",
        "model_executor",
        "llm_engine",
        "engine",
        "executor",
        "runner",
        "model",
    )
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        compilation_config = getattr(current, "compilation_config", None)
        static_forward_context = getattr(compilation_config, "static_forward_context", None)
        if isinstance(static_forward_context, dict):
            layers = [
                (prefix, layer)
                for prefix, layer in static_forward_context.items()
                if hasattr(layer, "impl") and hasattr(layer, "head_size")
            ]
            if layers:
                return sorted(layers, key=lambda item: item[0])
        for attr_name in attr_names:
            child = getattr(current, attr_name, None)
            if child is not None:
                queue.append(child)

    module = getattr(model, "model", model)
    if hasattr(module, "named_modules"):
        layers = [
            (name, candidate)
            for name, candidate in module.named_modules()
            if hasattr(candidate, "impl") and hasattr(candidate, "head_size")
        ]
        if layers:
            return layers
    raise ValueError("Could not locate vLLM attention layers to patch.")


class KVTCLayerState:
    """Per-layer KVTC serving state."""

    def __init__(
        self,
        layer_idx: int,
        layer: Any,
        calibration_data: CalibrationData,
        *,
        use_triton: bool = True,
    ) -> None:
        self.layer_idx = layer_idx
        self.layer = layer
        self.impl = getattr(layer, "impl")
        self.calibration_data = calibration_data
        self.use_triton = use_triton
        self.active = False
        self.head_dim = int(getattr(layer, "head_size"))
        self.num_heads = int(getattr(layer, "num_heads", getattr(self.impl, "num_heads", 0)))
        self.num_kv_heads = int(getattr(layer, "num_kv_heads", getattr(self.impl, "num_kv_heads", 0)))
        if self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError(f"Layer {layer_idx} is missing num_heads/num_kv_heads metadata.")
        self.queries_per_kv = max(self.num_heads // self.num_kv_heads, 1)
        self.head_group_size = calibration_data.head_group_size
        self.rope_theta = calibration_data.rope_theta
        self.sink_tokens = calibration_data.sink_tokens
        self.window_tokens = calibration_data.window_tokens
        self.softmax_scale = float(getattr(self.impl, "scale", 1.0 / math.sqrt(self.head_dim)))
        self.logits_soft_cap = getattr(self.impl, "logits_soft_cap", None)
        self.groups = self._build_groups()
        self.sequences: Dict[str, KVTCSequenceState] = {}

    def _build_groups(self) -> Dict[int, KVTCGroupConfig]:
        groups: Dict[int, KVTCGroupConfig] = {}
        for group_idx, start in enumerate(range(0, self.num_kv_heads, self.head_group_size)):
            key_entry = self.calibration_data.entries[(self.layer_idx, group_idx, "keys")]
            value_entry = self.calibration_data.entries[(self.layer_idx, group_idx, "values")]
            head_stop = min(start + self.head_group_size, self.num_kv_heads)
            groups[group_idx] = KVTCGroupConfig(
                group_idx=group_idx,
                head_indices=tuple(range(start, head_stop)),
                key=_build_quant_spec(key_entry),
                value=_build_quant_spec(value_entry),
            )
        return groups

    def has_prefill_data(self) -> bool:
        return any(sequence.has_prefill() for sequence in self.sequences.values())

    def _sequence(self, request_id: str) -> KVTCSequenceState:
        sequence = self.sequences.get(request_id)
        if sequence is None:
            sequence = KVTCSequenceState(request_id=request_id)
            self.sequences[request_id] = sequence
        return sequence

    def capture(self, key: torch.Tensor, value: torch.Tensor, spans: Sequence[RequestSpan]) -> None:
        for span in spans:
            key_tokens = key[span.start : span.end].detach()
            value_tokens = value[span.start : span.end].detach()
            positions = span.positions.detach()
            if key_tokens.numel() == 0:
                continue
            sequence = self._sequence(span.request_id)
            if not self.active:
                sequence.raw_keys.append(key_tokens.contiguous())
                sequence.raw_values.append(value_tokens.contiguous())
                sequence.raw_positions.append(positions.contiguous())
            else:
                self._append_active_tokens(sequence, key_tokens, value_tokens, positions)

    def finalize_prefill(self) -> None:
        for sequence in self.sequences.values():
            self._finalize_sequence(sequence)
        self.active = True

    def _finalize_sequence(self, sequence: KVTCSequenceState) -> None:
        if sequence.finalized:
            return
        if sequence.raw_keys:
            keys = torch.cat(sequence.raw_keys, dim=0)
            values = torch.cat(sequence.raw_values, dim=0)
            positions = torch.cat(sequence.raw_positions, dim=0)
        else:
            sequence.sinks_keys = None
            sequence.sinks_values = None
            sequence.window_keys = None
            sequence.window_values = None
            sequence.window_positions = None
            sequence.middle_positions = None
            sequence.finalized = True
            return

        total_tokens = int(keys.shape[0])
        sink_len = min(self.sink_tokens, total_tokens)
        residual = max(total_tokens - sink_len, 0)
        window_len = min(self.window_tokens, residual)
        middle_start = sink_len
        middle_end = total_tokens - window_len

        sequence.sinks_keys = keys[:sink_len].contiguous()
        sequence.sinks_values = values[:sink_len].contiguous()
        sequence.window_keys = keys[middle_end:].contiguous()
        sequence.window_values = values[middle_end:].contiguous()
        sequence.window_positions = positions[middle_end:].contiguous()
        sequence.middle_positions = None

        middle_keys = keys[middle_start:middle_end]
        middle_values = values[middle_start:middle_end]
        if middle_keys.numel():
            self._compress_middle_chunk(sequence, middle_keys, middle_values, positions[middle_start:middle_end].contiguous())

        sequence.raw_keys.clear()
        sequence.raw_values.clear()
        sequence.raw_positions.clear()
        sequence.finalized = True

    def _append_active_tokens(
        self,
        sequence: KVTCSequenceState,
        key_tokens: torch.Tensor,
        value_tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if not sequence.finalized:
            self._finalize_sequence(sequence)

        for token_idx in range(int(key_tokens.shape[0])):
            key_token = key_tokens[token_idx : token_idx + 1].contiguous()
            value_token = value_tokens[token_idx : token_idx + 1].contiguous()
            pos_token = positions[token_idx : token_idx + 1].contiguous()
            sink_len = 0 if sequence.sinks_keys is None else int(sequence.sinks_keys.shape[0])
            if sink_len < self.sink_tokens:
                sequence.sinks_keys = _append_tensor(sequence.sinks_keys, key_token)
                sequence.sinks_values = _append_tensor(sequence.sinks_values, value_token)
                continue

            sequence.window_keys = _append_tensor(sequence.window_keys, key_token)
            sequence.window_values = _append_tensor(sequence.window_values, value_token)
            sequence.window_positions = _append_tensor(sequence.window_positions, pos_token)

            if self.window_tokens <= 0:
                overflow = sequence.window_keys
                overflow_values = sequence.window_values
                overflow_positions = sequence.window_positions
                sequence.window_keys = sequence.window_keys[0:0]
                sequence.window_values = sequence.window_values[0:0]
                sequence.window_positions = sequence.window_positions[0:0]
                self._compress_middle_chunk(sequence, overflow, overflow_values, overflow_positions)
                continue

            excess = int(sequence.window_keys.shape[0] - self.window_tokens)
            if excess > 0:
                overflow_keys = sequence.window_keys[:excess].contiguous()
                overflow_values = sequence.window_values[:excess].contiguous()
                overflow_positions = sequence.window_positions[:excess].contiguous()
                sequence.window_keys = sequence.window_keys[excess:].contiguous()
                sequence.window_values = sequence.window_values[excess:].contiguous()
                sequence.window_positions = sequence.window_positions[excess:].contiguous()
                self._compress_middle_chunk(sequence, overflow_keys, overflow_values, overflow_positions)

    def _compress_middle_chunk(
        self,
        sequence: KVTCSequenceState,
        middle_keys: torch.Tensor,
        middle_values: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if middle_keys.numel() == 0:
            return
        sequence.middle_positions = _append_tensor(sequence.middle_positions, positions)
        for group_idx, group in self.groups.items():
            start = group.head_indices[0]
            stop = start + len(group.head_indices)
            group_keys = middle_keys[:, start:stop, :]
            group_values = middle_values[:, start:stop, :]
            key_indices = self._quantize(group_keys, positions, group.key, undo_rope=True)
            value_indices = self._quantize(group_values, positions, group.value, undo_rope=False)
            storage = sequence.compressed.setdefault(group_idx, QuantizedGroupStorage())
            storage.key_indices = _append_tensor(storage.key_indices, key_indices)
            storage.value_indices = _append_tensor(storage.value_indices, value_indices)

    def _quantize(
        self,
        tensor: torch.Tensor,
        positions: torch.Tensor,
        spec: QuantizedTensorSpec,
        *,
        undo_rope: bool,
    ) -> torch.Tensor:
        tokens, group_heads, _ = tensor.shape
        if spec.active_components.numel() == 0:
            return torch.empty((tokens, group_heads, 0), device=tensor.device, dtype=spec.index_dtype)
        work = tensor
        if undo_rope:
            work = apply_rope_inverse(
                work,
                positions.to(device=tensor.device, dtype=torch.long),
                rope_theta=self.rope_theta,
                head_dim=self.head_dim,
            )
        rows = work.reshape(tokens * group_heads, self.head_dim).to(torch.float32)
        centered = rows - spec.mean.to(device=rows.device, dtype=rows.dtype)
        # PATCH(lna-lab): pass basis_t (shape [k, dim]) so that pca_transform's
        # `.T` produces [dim, k] for the matmul. spec.projection_basis is stored
        # with shape [dim, k] (column-selected from full eigenvectors) which
        # collapses correctly when k == dim (no zero bit_widths) but breaks for
        # rank-reduced calibrations like Qwen3.6-27B head_dim=256 with 4
        # zero-bit components → 252 active.
        pca_values = pca_transform(centered, spec.basis_t.to(device=rows.device, dtype=rows.dtype))
        indices = batch_quantize(
            pca_values,
            spec.bit_widths.to(device=rows.device),
            spec.scales.to(device=rows.device),
            spec.zero_points.to(device=rows.device),
        )
        return indices.to(spec.index_dtype).reshape(tokens, group_heads, -1).contiguous()

    def decode_request(self, request_id: str, query: torch.Tensor) -> torch.Tensor:
        sequence = self.sequences.get(request_id)
        if sequence is None:
            raise KeyError(f"Layer {self.layer_idx} has no KVTC state for request {request_id}.")

        output = torch.empty_like(query)
        for head_idx in range(self.num_heads):
            kv_head_idx = min(head_idx // self.queries_per_kv, self.num_kv_heads - 1)
            group_idx = kv_head_idx // self.head_group_size
            local_head_idx = kv_head_idx - self.groups[group_idx].head_indices[0]
            output[head_idx] = self._decode_one_head(sequence, self.groups[group_idx], kv_head_idx, local_head_idx, query[head_idx])
        return output

    def decode_request_at(self, request_id: str, query: torch.Tensor, max_position_exclusive: int) -> torch.Tensor:
        """PATCH(lna-lab Phase α): causal-truncated multi-token decode.

        Like ``decode_request`` but only attends over state K/V at positions strictly
        less than ``max_position_exclusive``. Used for spec-verify multi-token batches.
        Phase β step 2a: per-head Python loop replaced with batched torch attention
        (sinks + window dense in single matmul across all heads; middle range still
        per-head until Triton kernel rewrite).
        """
        sequence = self.sequences.get(request_id)
        if sequence is None:
            raise KeyError(f"Layer {self.layer_idx} has no KVTC state for request {request_id}.")
        return self._decode_all_heads(sequence, query, max_position_exclusive=max_position_exclusive)

    def _decode_all_heads(
        self,
        sequence: KVTCSequenceState,
        query: torch.Tensor,
        max_position_exclusive: int | None = None,
    ) -> torch.Tensor:
        """PATCH(lna-lab Phase β step 2a): batched all-heads attention for one query token.

        ``query`` shape: ``[num_heads, head_dim]``. Returns ``[num_heads, head_dim]``.

        - Sinks + window: a single dense torch attention across all heads
          (avoids the 24× Python loop per layer per call).
        - Middle (PCA-compressed): still iterates per group (4 groups for Qwen3.6-27B)
          since the existing decode_attention_from_kvtc Triton kernel is per-head;
          but inside each group the queries-per-kv (6 for Qwen3.6) are batched.
        """
        device = query.device
        dtype = query.dtype
        # Per-head expanded indices: which kv_head each query head reads from.
        # E.g. Qwen3.6-27B num_heads=24 num_kv_heads=4 queries_per_kv=6
        # → kv_head_lookup = [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]
        kv_head_lookup = torch.arange(self.num_heads, device=device) // self.queries_per_kv
        kv_head_lookup = kv_head_lookup.clamp_max(self.num_kv_heads - 1)

        merged_output = torch.zeros(self.num_heads, self.head_dim, device=device, dtype=dtype)
        merged_lse = torch.full((self.num_heads,), NEG_INF, device=device, dtype=torch.float32)

        # Sinks (positions 0..sink_len-1) and Window (positions [middle_end..total-1]).
        for region_keys, region_vals, region_pos in self._iter_dense_regions(
            sequence, max_position_exclusive
        ):
            # region_keys / region_vals shape: [N, num_kv_heads, head_dim]
            if region_keys.numel() == 0:
                continue
            keys_per_head = region_keys.index_select(1, kv_head_lookup).to(dtype)   # [N, num_heads, head_dim]
            vals_per_head = region_vals.index_select(1, kv_head_lookup).to(dtype)   # [N, num_heads, head_dim]
            # scores[n, h] = (keys[n,h,:] · query[h,:]) * scale
            scores = torch.einsum("nhd,hd->nh", keys_per_head.float(), query.float()) * self.softmax_scale
            scores = _apply_soft_cap_torch(scores, self.logits_soft_cap)
            lse = torch.logsumexp(scores, dim=0)            # [num_heads]
            weights = torch.softmax(scores, dim=0)          # [N, num_heads]
            region_out = torch.einsum("nh,nhd->hd", weights, vals_per_head.float()).to(dtype)
            merged_output, merged_lse = self._merge_states(merged_output, merged_lse, region_out, lse)

        # Middle (compressed). Still per-group; each group amortizes dequant across its
        # queries-per-kv heads (6 for Qwen3.6-27B). Could be further batched in Phase β
        # step 2b/c with a dedicated multi-head Triton kernel.
        middle_positions = sequence.middle_positions
        if middle_positions is not None and middle_positions.numel() > 0:
            mid_pos = middle_positions
            mid_mask = None
            if max_position_exclusive is not None:
                mid_mask = mid_pos < max_position_exclusive
                if mid_mask.all():
                    mid_mask = None
                else:
                    mid_pos = mid_pos[mid_mask]
            if mid_pos.numel() > 0:
                for group_idx, group in self.groups.items():
                    storage = sequence.compressed.get(group_idx)
                    if storage is None or storage.key_indices is None or storage.key_indices.shape[0] == 0:
                        continue
                    key_idx = storage.key_indices
                    val_idx = storage.value_indices
                    if mid_mask is not None:
                        key_idx = key_idx[mid_mask]
                        val_idx = val_idx[mid_mask]
                    # Within a group, iterate per kv_head (head_group_size=1 means one
                    # kv_head per group; the queries_per_kv batching is handled by
                    # _attend_compressed_for_kv_head which serves all matching query heads).
                    for local_head_idx, kv_head_idx in enumerate(group.head_indices):
                        # Find which query heads use this kv_head
                        q_mask = (kv_head_lookup == kv_head_idx)
                        if not q_mask.any():
                            continue
                        q_head_indices = q_mask.nonzero(as_tuple=False).flatten()
                        for qh in q_head_indices.tolist():
                            mid_out, mid_lse = decode_attention_from_kvtc(
                                query[qh],
                                key_idx[:, local_head_idx, :].to(device=device),
                                val_idx[:, local_head_idx, :].to(device=device),
                                group.key.scales.to(device=device),
                                group.key.zero_points.to(device=device),
                                group.value.scales.to(device=device),
                                group.value.zero_points.to(device=device),
                                group.key.basis_t.to(device=device),
                                group.value.basis_t.to(device=device),
                                group.key.mean.to(device=device),
                                group.value.mean.to(device=device),
                                mid_pos.to(device=device),
                                rope_theta=self.rope_theta,
                                softmax_scale=self.softmax_scale,
                                logits_soft_cap=self.logits_soft_cap,
                                use_triton=self.use_triton,
                            )
                            cur_out = merged_output[qh]
                            cur_lse = merged_lse[qh]
                            new_out, new_lse = merge_attention_states(
                                cur_out, cur_lse, mid_out.to(dtype), mid_lse,
                            )
                            merged_output[qh] = new_out
                            merged_lse[qh] = new_lse

        return merged_output

    def _iter_dense_regions(self, sequence: KVTCSequenceState, max_position_exclusive: int | None):
        """Yield (keys, values, positions) for sinks and window, applying causal mask."""
        # Sinks: positions 0..sink_len-1 (implicit ordering).
        if sequence.sinks_keys is not None and sequence.sinks_keys.numel():
            sk = sequence.sinks_keys
            sv = sequence.sinks_values
            if max_position_exclusive is not None and max_position_exclusive < sk.shape[0]:
                sk = sk[:max_position_exclusive]
                sv = sv[:max_position_exclusive]
            if sk.numel():
                yield sk, sv, None
        # Window
        if sequence.window_keys is not None and sequence.window_keys.numel():
            wk = sequence.window_keys
            wv = sequence.window_values
            if max_position_exclusive is not None and sequence.window_positions is not None:
                mask = sequence.window_positions < max_position_exclusive
                if not mask.all():
                    wk = wk[mask]
                    wv = wv[mask]
            if wk.numel():
                yield wk, wv, None

    @staticmethod
    def _merge_states(left_out, left_lse, right_out, right_lse):
        """Vectorized log-sum-exp merge across heads (input shapes [num_heads, head_dim] / [num_heads])."""
        is_left_neg = torch.isneginf(left_lse)
        is_right_neg = torch.isneginf(right_lse)
        # Where left is -inf: take right; where right is -inf: take left; else merge.
        max_lse = torch.maximum(left_lse, right_lse)
        left_w = torch.exp(left_lse - max_lse)
        right_w = torch.exp(right_lse - max_lse)
        left_w = torch.where(is_left_neg, torch.zeros_like(left_w), left_w)
        right_w = torch.where(is_right_neg, torch.zeros_like(right_w), right_w)
        denom = left_w + right_w
        # Avoid div-by-zero where both are -inf
        denom_safe = torch.where(denom == 0, torch.ones_like(denom), denom)
        merged = (left_out * left_w.unsqueeze(-1) + right_out * right_w.unsqueeze(-1)) / denom_safe.unsqueeze(-1)
        merged_lse = max_lse + torch.log(denom_safe)
        # Where both -inf, keep -inf
        both_neg = is_left_neg & is_right_neg
        merged_lse = torch.where(both_neg, left_lse, merged_lse)
        return merged, merged_lse

    def _decode_one_head(
        self,
        sequence: KVTCSequenceState,
        group: KVTCGroupConfig,
        kv_head_idx: int,
        local_head_idx: int,
        query: torch.Tensor,
        max_position_exclusive: int | None = None,
    ) -> torch.Tensor:
        device = query.device
        merged_output = torch.zeros(self.head_dim, device=device, dtype=query.dtype)
        merged_lse = torch.tensor(NEG_INF, device=device, dtype=torch.float32)

        # Sinks: positions 0..sink_len-1 (implicit). For causal mask with P, take
        # only positions < P (i.e. up to min(sink_len, P) tokens).
        if sequence.sinks_keys is not None and sequence.sinks_keys.numel():
            sk_keys = sequence.sinks_keys
            sk_vals = sequence.sinks_values
            if max_position_exclusive is not None and max_position_exclusive < sk_keys.shape[0]:
                sk_keys = sk_keys[:max_position_exclusive]
                sk_vals = sk_vals[:max_position_exclusive]
            if sk_keys.numel():
                sink_output, sink_lse = dense_attention_state(
                    query,
                    sk_keys[:, kv_head_idx, :].to(device=device, dtype=query.dtype),
                    sk_vals[:, kv_head_idx, :].to(device=device, dtype=query.dtype),
                    softmax_scale=self.softmax_scale,
                    logits_soft_cap=self.logits_soft_cap,
                )
                merged_output, merged_lse = merge_attention_states(merged_output, merged_lse, sink_output, sink_lse)

        # Middle (compressed): filter via middle_positions < P.
        storage = sequence.compressed.get(group.group_idx)
        middle_positions = sequence.middle_positions
        if (
            storage is not None
            and storage.key_indices is not None
            and storage.value_indices is not None
            and middle_positions is not None
            and storage.key_indices.shape[0] > 0
        ):
            mid_key_idx = storage.key_indices
            mid_val_idx = storage.value_indices
            mid_pos = middle_positions
            if max_position_exclusive is not None:
                mask = mid_pos < max_position_exclusive
                if mask.all():
                    pass
                else:
                    mid_key_idx = mid_key_idx[mask]
                    mid_val_idx = mid_val_idx[mask]
                    mid_pos = mid_pos[mask]
            if mid_key_idx.shape[0] > 0:
                middle_output, middle_lse = decode_attention_from_kvtc(
                    query,
                    mid_key_idx[:, local_head_idx, :].to(device=device),
                    mid_val_idx[:, local_head_idx, :].to(device=device),
                    group.key.scales.to(device=device),
                    group.key.zero_points.to(device=device),
                    group.value.scales.to(device=device),
                    group.value.zero_points.to(device=device),
                    group.key.basis_t.to(device=device),
                    group.value.basis_t.to(device=device),
                    group.key.mean.to(device=device),
                    group.value.mean.to(device=device),
                    mid_pos.to(device=device),
                    rope_theta=self.rope_theta,
                    softmax_scale=self.softmax_scale,
                    logits_soft_cap=self.logits_soft_cap,
                    use_triton=self.use_triton,
                )
                merged_output, merged_lse = merge_attention_states(merged_output, merged_lse, middle_output, middle_lse)

        # Window: filter via window_positions < P.
        if sequence.window_keys is not None and sequence.window_keys.numel():
            w_keys = sequence.window_keys
            w_vals = sequence.window_values
            if max_position_exclusive is not None and sequence.window_positions is not None:
                mask = sequence.window_positions < max_position_exclusive
                if not mask.all():
                    w_keys = w_keys[mask]
                    w_vals = w_vals[mask]
            if w_keys.numel():
                window_output, window_lse = dense_attention_state(
                    query,
                    w_keys[:, kv_head_idx, :].to(device=device, dtype=query.dtype),
                    w_vals[:, kv_head_idx, :].to(device=device, dtype=query.dtype),
                    softmax_scale=self.softmax_scale,
                    logits_soft_cap=self.logits_soft_cap,
                )
                merged_output, merged_lse = merge_attention_states(merged_output, merged_lse, window_output, window_lse)

        return merged_output


class PatchedCacheUpdate:
    """Intercept vLLM cache writes once KVTC is active."""

    def __init__(self, handle: "KVTCHandle", original: Any | None) -> None:
        self.handle = handle
        self.original = original

    def __call__(self, impl: Any, layer: Any, key: torch.Tensor, value: torch.Tensor, kv_cache: Any, slot_mapping: Any) -> Any:
        if self.handle.active:
            return None
        if self.original is None:
            return None
        return self.original(layer, key, value, kv_cache, slot_mapping)


class PatchedForward:
    """Capture KV updates and route decode through KVTC."""

    def __init__(self, handle: "KVTCHandle", state: KVTCLayerState, original: Any) -> None:
        self.handle = handle
        self.state = state
        self.original = original

    def __call__(
        self,
        impl: Any,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Any,
        attn_metadata: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            return self.original(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)

        spans = extract_request_spans(attn_metadata, int(query.shape[0]), device=key.device)
        pure_decode = _is_pure_decode(spans)
        if self.handle.auto_activate and not self.handle.active and pure_decode and self.handle.has_prefill_data():
            self.handle.free_kv_cache()

        # PATCH(lna-lab): when Attention.forward wrapper handles capture (vLLM 0.19+ V1
        # workaround), skip capture here to avoid double-counting. Toggled via state flag
        # set by _wrap_attention_forward_for_capture.
        if not getattr(self.state, "_skip_patched_capture", False):
            self.state.capture(key, value, spans)

        if not self.handle.active:
            return self.original(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)

        if not pure_decode:
            raise RuntimeError("KVTC decode path only supports pure decode batches after activation.")

        extra_kwargs = dict(kwargs)
        output = extra_kwargs.pop("output", args[0] if args else None)
        # PATCH(lna-lab): vLLM 0.19+ unified_attention_with_output passes extra
        # kwargs (output_scale, output_block_scale, ...) that we can safely
        # ignore on the KVTC decode path because we synthesize the output tensor
        # ourselves from the dequantized projection.
        extra_kwargs.pop("output_scale", None)
        extra_kwargs.pop("output_block_scale", None)
        if extra_kwargs or len(args) > 1:
            raise RuntimeError(
                f"Unsupported vLLM attention signature on KVTC decode path: "
                f"unexpected kwargs={list(extra_kwargs)}, extra_args={len(args) - 1}"
            )

        output_tensor = output if output is not None else torch.empty_like(query)
        output_view = output_tensor.view_as(query) if output_tensor.dim() == 2 else output_tensor
        for span in spans:
            query_slice = query[span.start : span.end]
            decoded = self.state.decode_request(span.request_id, query_slice[0])
            output_view[span.start] = decoded
        return output_tensor


class KVTCHandle:
    """Handle returned by hook_model for activation and cleanup."""

    def __init__(self, model: Any, patched_layers: Sequence[PatchedLayer], *, auto_activate: bool = False) -> None:
        self.model = model
        self.patched_layers = list(patched_layers)
        self.auto_activate = auto_activate
        self.active = False
        self.free_timestamp: float | None = None

    def has_prefill_data(self) -> bool:
        return any(layer.state.has_prefill_data() for layer in self.patched_layers)

    def free_kv_cache(self) -> None:
        if self.active:
            return
        for layer in self.patched_layers:
            layer.state.finalize_prefill()
        for layer in self.patched_layers:
            if hasattr(layer.layer, "kv_cache"):
                layer.original_kv_cache = getattr(layer.layer, "kv_cache")
                setattr(layer.layer, "kv_cache", _dummy_like_cache(layer.original_kv_cache))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.active = True
        self.free_timestamp = time.perf_counter()

    def unhook(self) -> None:
        for layer in self.patched_layers:
            layer.impl.forward = layer.original_forward
            if layer.original_do_kv_cache_update is not None:
                layer.impl.do_kv_cache_update = layer.original_do_kv_cache_update
            if layer.original_kv_cache is not None and hasattr(layer.layer, "kv_cache"):
                setattr(layer.layer, "kv_cache", layer.original_kv_cache)
        self.active = False


def hook_model(
    model: Any,
    calibration_data: CalibrationData,
    *,
    auto_activate: bool = False,
    use_triton: bool = True,
) -> KVTCHandle:
    """Patch vLLM attention layers so decode reads from KVTC state."""

    patched_layers: List[PatchedLayer] = []
    resolved = resolve_attention_layers(model)
    # PATCH(lna-lab): for hybrid attention models (e.g. Qwen3.6 / Qwen3-Next /
    # Mamba-mixed) only the *full_attention* layers carry KV state and end up in
    # ``static_forward_context``. Their prefixes parse to sparse indices like
    # 3, 7, 11, ..., 63 — but our calibration is built with a *contiguous*
    # remap (kvtc/calibrate.py iterates past_key_values.layers and skips KV-less
    # layers, indexing 0..N-1). So when the resolved set is sparse we use the
    # iteration position as ``layer_idx`` to keep the two sides symmetric.
    parsed_indices = [_parse_layer_idx(prefix, i) for i, (prefix, _) in enumerate(resolved)]
    is_sparse = parsed_indices != list(range(len(resolved)))
    for fallback_idx, (prefix, layer) in enumerate(resolved):
        layer_idx = fallback_idx if is_sparse else _parse_layer_idx(prefix, fallback_idx)
        impl = layer.impl
        patched_layers.append(
            PatchedLayer(
                prefix=prefix,
                layer_idx=layer_idx,
                layer=layer,
                impl=impl,
                state=KVTCLayerState(layer_idx, layer, calibration_data, use_triton=use_triton),
                original_forward=impl.forward,
                original_do_kv_cache_update=getattr(impl, "do_kv_cache_update", None),
            )
        )

    handle = KVTCHandle(model, patched_layers, auto_activate=auto_activate)
    for layer in patched_layers:
        layer.impl.forward = types.MethodType(PatchedForward(handle, layer.state, layer.original_forward), layer.impl)
        if layer.original_do_kv_cache_update is not None:
            layer.impl.do_kv_cache_update = types.MethodType(
                PatchedCacheUpdate(handle, layer.original_do_kv_cache_update),
                layer.impl,
            )
    setattr(model, "_kvtc_handle", handle)
    return handle


def _wrap_attention_forward_for_capture(layer: Any, state: "KVTCLayerState", handle: "KVTCHandle") -> None:
    """PATCH(lna-lab Phase α): Install Attention.forward wrapper for capture + multi-token decode.

    vLLM 0.19 V1 routes prefill (and spec-verify multi-token batches) through
    ``torch.ops.vllm.unified_attention_with_output`` whose dispatcher does NOT honour
    runtime monkey-patches of ``self.impl.forward`` for those batches (only the
    single-token decode path reaches the patched method). So:

    - **prefill** never reaches ``PatchedForward``                  (Phase B/C finding)
    - **MTP/DFlash spec verify** (M=2..5 query tokens) never reaches ``PatchedForward``,
      AND ``pure_decode and has_prefill_data`` is never true → ``auto_activate`` never
      triggers → KVTC effectively stays inactive                    (Phase D finding)

    This wrapper handles both:

    1. Capture: every call's K/V is appended to ``state`` (handles prefill, verify,
       and post-activation single decode).
    2. Activation: the first multi-token "verify-shaped" call (i.e., query_len > 1
       AND seq_len > query_len, meaning prior cached context exists) explicitly
       triggers ``handle.free_kv_cache()`` so subsequent decodes go through KVTC.
    3. Decode replacement: when ``handle.active``, the wrapper computes attention
       output via ``state.decode_request`` for each query token (with causal mask
       implicit in the per-token state slice), and returns that instead of calling
       through to ``original_forward`` (which would attend over the dummy cache).
    """
    from vllm.forward_context import get_forward_context

    original_forward = layer.forward
    state._skip_patched_capture = True  # PatchedForward will not double-capture

    def patched_attention_forward(query, key, value, output_shape=None):
        try:
            fwd_ctx = get_forward_context()
            attn_metadata = fwd_ctx.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata.get(layer.layer_name, None)
        except Exception:
            attn_metadata = None

        num_tokens = query.shape[0]
        # K/V come in 2D as [num_tokens, num_kv_heads * head_dim]; reshape to 3D.
        k3d = key.view(-1, state.num_kv_heads, state.head_dim) if (key is not None and key.dim() == 2) else key
        v3d = value.view(-1, state.num_kv_heads, state.head_dim) if (value is not None and value.dim() == 2) else value

        spans = []
        if k3d is not None and v3d is not None:
            spans = extract_request_spans(attn_metadata, num_tokens, device=k3d.device)
            state.capture(k3d, v3d, spans)

        # "decode-or-verify shaped" = prior cached context exists (single-token decode
        # OR multi-token spec verify); not a fresh prefill batch.
        max_query_len = max((s.query_len for s in spans), default=0)
        max_seq_len = max((s.seq_len for s in spans), default=0)
        is_decode_or_verify = (max_query_len >= 1) and (max_seq_len > max_query_len)

        # Activation trigger: first decode-or-verify call after capture has prefill data.
        # For single-token decode this matches the original PatchedForward trigger; for
        # spec verify (M=2..5) it is the only way activation fires under vLLM 0.19 V1
        # because the verify path bypasses ``impl.forward`` and never reports pure_decode.
        if (handle.auto_activate and not handle.active
                and is_decode_or_verify and handle.has_prefill_data()):
            handle.free_kv_cache()

        # Active KVTC path: compute output ourselves, replace original_forward.
        if handle.active and is_decode_or_verify and spans:
            return _kvtc_compute_attention(
                state, query, k3d, v3d, spans, num_tokens, output_shape, layer
            )

        return original_forward(query, key, value, output_shape=output_shape)

    layer.forward = patched_attention_forward


def _kvtc_compute_attention(
    state: "KVTCLayerState",
    query: torch.Tensor,
    k3d: torch.Tensor,
    v3d: torch.Tensor,
    spans: List["RequestSpan"],
    num_tokens: int,
    output_shape: Any,
    layer: Any,
) -> torch.Tensor:
    """Compute attention via KVTC for the M query tokens of one Attention.forward call.

    Each query token at sequence position p attends over the per-request KVTC state
    truncated to positions < p (causal). Result is reshaped back to the 2D
    ``[num_tokens, num_heads * head_dim]`` layout vLLM's ``Attention.forward`` returns.
    """
    # Reshape query to [num_tokens, num_heads, head_dim].
    if query.dim() == 2:
        q3d = query.view(num_tokens, state.num_heads, state.head_dim)
    else:
        q3d = query

    out_3d = torch.empty(
        (num_tokens, state.num_heads, state.head_dim),
        device=q3d.device,
        dtype=q3d.dtype,
    )

    for span in spans:
        # Per-token: causal slice of state. We rely on the fact that this call's
        # tokens were just appended to state at positions [seq_len-query_len .. seq_len-1].
        # For each query token i (relative to span), attend over state truncated at
        # absolute position seq_len - query_len + i (exclusive upper bound).
        for i in range(span.query_len):
            global_pos = span.seq_len - span.query_len + i
            tok_idx = span.start + i
            # decode_one_token_at_position computes attention over the per-request
            # state truncated to positions < global_pos.
            decoded = state.decode_request_at(span.request_id, q3d[tok_idx], global_pos)
            out_3d[tok_idx] = decoded

    # Reshape back to vLLM's 2D output layout.
    hidden = state.num_heads * state.head_dim
    result = out_3d.reshape(num_tokens, hidden)
    return result


def free_kv_cache(model_or_handle: Any) -> KVTCHandle:
    """Finalize KVTC state and release vLLM's paged KV cache."""

    if isinstance(model_or_handle, KVTCHandle):
        handle = model_or_handle
    else:
        handle = getattr(model_or_handle, "_kvtc_handle", None)
        if handle is None:
            raise ValueError("Model has not been patched with hook_model().")
    handle.free_kv_cache()
    return handle


# ---------------------------------------------------------------------------
# vLLM 0.19+ child-process wrapper (Lna-Lab patch)
# ---------------------------------------------------------------------------
#
# Since vLLM 0.18, LLMEngine spawns a separate EngineCore process; the model
# nn.Module lives in that child. Direct hook_model(model) from the user-facing
# process can no longer find attention layers. Instead, we route through
# llm.llm_engine.apply_model(func) which executes ``func(model)`` inside the
# EngineCore worker (requires VLLM_ALLOW_INSECURE_SERIALIZATION=1 because the
# function is pickled across the IPC boundary).


def _install_in_worker(calibration_bytes: bytes, auto_activate: bool, use_triton: bool):
    """Closure executed inside the EngineCore worker process."""

    def _install(model):
        import io
        import pickle
        from kvtc.vllm_backend import hook_model, _wrap_attention_forward_for_capture
        cal = pickle.load(io.BytesIO(calibration_bytes))
        handle = hook_model(model, cal, auto_activate=auto_activate, use_triton=use_triton)
        # Save handle on the model for later free_kv_cache_engine call.
        model._kvtc_handle = handle
        # PATCH(lna-lab): vLLM 0.19+ V1 routes prefill batches through a fast-path
        # that bypasses runtime monkey-patches of ``impl.forward`` (only DECODE
        # batches reach the patched method). To reliably capture prefill K/V we
        # additionally wrap each Attention nn.Module ``forward``.
        for pl in handle.patched_layers:
            _wrap_attention_forward_for_capture(pl.layer, pl.state, handle)
        return {
            "num_layers": len(handle.patched_layers),
            "auto_activate": auto_activate,
            "use_triton": use_triton,
        }

    return _install


def hook_engine(
    llm: Any,
    calibration_data: CalibrationData,
    *,
    auto_activate: bool = False,
    use_triton: bool = True,
) -> Dict[str, Any]:
    """Install KVTC hooks on a vLLM 0.19+ ``LLM`` (apply_model RPC variant).

    Requires ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` in the env so vLLM's
    ZMQ encoder will pickle our installer function across the IPC boundary
    to the EngineCore child process.

    Returns a dict summary (workers' return values aggregated). The actual
    :class:`KVTCHandle` lives inside the child process and is reachable from
    user code only via further :func:`apply_model` calls (see
    :func:`free_kv_cache_engine`).
    """

    import pickle
    cal_bytes = pickle.dumps(calibration_data)
    install = _install_in_worker(cal_bytes, auto_activate, use_triton)
    results = llm.llm_engine.apply_model(install)
    if results:
        return results[0]  # uniproc / driver worker
    return {"num_layers": 0}


def free_kv_cache_engine(llm: Any) -> Dict[str, Any]:
    """vLLM 0.19+ wrapper around :func:`free_kv_cache` via ``apply_model``."""

    def _free(model):
        from kvtc.vllm_backend import free_kv_cache
        handle = free_kv_cache(model)
        return {"freed_layers": len(handle.patched_layers)}

    results = llm.llm_engine.apply_model(_free)
    if results:
        return results[0]
    return {"freed_layers": 0}


__all__ = [
    "KVTCLayerState",
    "KVTCHandle",
    "PatchedCacheUpdate",
    "PatchedForward",
    "extract_request_spans",
    "free_kv_cache",
    "free_kv_cache_engine",
    "hook_model",
    "hook_engine",
    "resolve_attention_layers",
]
