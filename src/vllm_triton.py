"""KVTC decode attention helpers for vLLM integration.

The serving path stores PCA-quantized indices directly on GPU and reconstructs
keys/values on demand. A Triton block kernel is provided for the common
decode-only case (`query_len == 1`), with a torch fallback for unsupported
shapes or environments.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised in environments without Triton.
    triton = None
    tl = None
    HAS_TRITON = False

from .pca import apply_rope


NEG_INF = float("-inf")


def _empty_state(
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.zeros(head_dim, device=device, dtype=dtype)
    lse = torch.tensor(NEG_INF, device=device, dtype=torch.float32)
    return output, lse


def _dequantize_active(
    indices: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
) -> torch.Tensor:
    return (indices.to(torch.float32) - zero_points.unsqueeze(0)) * scales.unsqueeze(0)


def reconstruct_vectors(
    indices: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    basis_t: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct dense vectors from active PCA component indices.

    Args:
        indices: `[tokens, active_components]` integer tensor.
        scales: `[active_components]` float tensor.
        zero_points: `[active_components]` float tensor.
        basis_t: `[active_components, head_dim]` PCA basis rows.
        mean: `[head_dim]` vector mean.
    """

    if indices.numel() == 0:
        return mean.unsqueeze(0).expand(indices.shape[0], -1).contiguous()
    coords = _dequantize_active(indices, scales, zero_points)
    return coords @ basis_t.to(coords.device, coords.dtype) + mean.unsqueeze(0)


def _apply_soft_cap(scores: torch.Tensor, logits_soft_cap: float | None) -> torch.Tensor:
    if logits_soft_cap is None or logits_soft_cap <= 0:
        return scores
    return logits_soft_cap * torch.tanh(scores / logits_soft_cap)


def dense_attention_state(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    softmax_scale: float,
    logits_soft_cap: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute attention over an exact dense KV segment."""

    if keys.numel() == 0:
        return _empty_state(query.shape[-1], device=query.device, dtype=query.dtype)

    scores = torch.matmul(keys.to(torch.float32), query.to(torch.float32))
    scores = scores * softmax_scale
    scores = _apply_soft_cap(scores, logits_soft_cap)

    lse = torch.logsumexp(scores, dim=0)
    weights = torch.softmax(scores, dim=0).to(values.dtype)
    output = torch.sum(weights.unsqueeze(-1) * values, dim=0)
    return output.to(query.dtype), lse.to(torch.float32)


def merge_attention_states(
    left_output: torch.Tensor,
    left_lse: torch.Tensor,
    right_output: torch.Tensor,
    right_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge two normalized attention states using their log-sum-exp values."""

    if torch.isneginf(left_lse):
        return right_output, right_lse
    if torch.isneginf(right_lse):
        return left_output, left_lse

    max_lse = torch.maximum(left_lse, right_lse)
    left_weight = torch.exp(left_lse - max_lse)
    right_weight = torch.exp(right_lse - max_lse)
    denom = left_weight + right_weight
    merged = (left_output * left_weight + right_output * right_weight) / denom
    return merged, max_lse + torch.log(denom)


def decode_attention_torch(
    query: torch.Tensor,
    key_indices: torch.Tensor,
    value_indices: torch.Tensor,
    key_scales: torch.Tensor,
    key_zero_points: torch.Tensor,
    value_scales: torch.Tensor,
    value_zero_points: torch.Tensor,
    key_basis_t: torch.Tensor,
    value_basis_t: torch.Tensor,
    key_mean: torch.Tensor,
    value_mean: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_theta: float,
    softmax_scale: float,
    logits_soft_cap: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch reference path for attention over one compressed KV chunk."""

    head_dim = query.shape[-1]
    if positions.numel() == 0:
        return _empty_state(head_dim, device=query.device, dtype=query.dtype)

    keys = reconstruct_vectors(
        key_indices,
        key_scales,
        key_zero_points,
        key_basis_t,
        key_mean,
    )
    keys = apply_rope(
        keys.unsqueeze(1),
        positions.to(device=keys.device, dtype=torch.long),
        rope_theta=rope_theta,
        head_dim=head_dim,
    ).squeeze(1)
    values = reconstruct_vectors(
        value_indices,
        value_scales,
        value_zero_points,
        value_basis_t,
        value_mean,
    )
    return dense_attention_state(
        query,
        keys.to(query.dtype),
        values.to(query.dtype),
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
    )


def _rope_cos_sin(
    positions: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (base / head_dim))
    angles = positions.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    return angles.cos().contiguous(), angles.sin().contiguous()


if HAS_TRITON:

    @triton.jit
    def _decode_block_kernel(
        query_ptr,
        key_indices_ptr,
        value_indices_ptr,
        key_scales_ptr,
        key_zero_points_ptr,
        value_scales_ptr,
        value_zero_points_ptr,
        key_basis_ptr,
        value_basis_ptr,
        key_mean_ptr,
        value_mean_ptr,
        cos_ptr,
        sin_ptr,
        output_ptr,
        lse_ptr,
        seq_len,
        key_components,
        value_components,
        head_dim,
        softmax_scale,
        key_stride_t,
        key_stride_c,
        value_stride_t,
        value_stride_c,
        key_basis_stride_c,
        key_basis_stride_d,
        value_basis_stride_c,
        value_basis_stride_d,
        cos_stride_t,
        cos_stride_d,
        output_stride_b,
        output_stride_d,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_CK: tl.constexpr,
        BLOCK_CV: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        token_offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        dim_offsets = tl.arange(0, BLOCK_D)

        token_mask = token_offsets < seq_len
        dim_mask = dim_offsets < head_dim

        query = tl.load(query_ptr + dim_offsets, mask=dim_mask, other=0.0).to(tl.float32)

        key_vectors = tl.load(
            key_mean_ptr + dim_offsets[None, :],
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        value_vectors = tl.load(
            value_mean_ptr + dim_offsets[None, :],
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        key_comp_offsets = tl.arange(0, BLOCK_CK)
        key_comp_mask = key_comp_offsets < key_components
        key_scales = tl.load(
            key_scales_ptr + key_comp_offsets,
            mask=key_comp_mask,
            other=0.0,
        ).to(tl.float32)
        key_zero_points = tl.load(
            key_zero_points_ptr + key_comp_offsets,
            mask=key_comp_mask,
            other=0.0,
        ).to(tl.float32)
        key_coords = tl.load(
            key_indices_ptr
            + token_offsets[:, None] * key_stride_t
            + key_comp_offsets[None, :] * key_stride_c,
            mask=token_mask[:, None] & key_comp_mask[None, :],
            other=0,
        ).to(tl.float32)
        key_coords = (key_coords - key_zero_points[None, :]) * key_scales[None, :]
        key_basis = tl.load(
            key_basis_ptr
            + key_comp_offsets[:, None] * key_basis_stride_c
            + dim_offsets[None, :] * key_basis_stride_d,
            mask=key_comp_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        key_vectors += tl.dot(key_coords, key_basis)

        value_comp_offsets = tl.arange(0, BLOCK_CV)
        value_comp_mask = value_comp_offsets < value_components
        value_scales = tl.load(
            value_scales_ptr + value_comp_offsets,
            mask=value_comp_mask,
            other=0.0,
        ).to(tl.float32)
        value_zero_points = tl.load(
            value_zero_points_ptr + value_comp_offsets,
            mask=value_comp_mask,
            other=0.0,
        ).to(tl.float32)
        value_coords = tl.load(
            value_indices_ptr
            + token_offsets[:, None] * value_stride_t
            + value_comp_offsets[None, :] * value_stride_c,
            mask=token_mask[:, None] & value_comp_mask[None, :],
            other=0,
        ).to(tl.float32)
        value_coords = (value_coords - value_zero_points[None, :]) * value_scales[None, :]
        value_basis = tl.load(
            value_basis_ptr
            + value_comp_offsets[:, None] * value_basis_stride_c
            + dim_offsets[None, :] * value_basis_stride_d,
            mask=value_comp_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        value_vectors += tl.dot(value_coords, value_basis)

        half_offsets = tl.arange(0, BLOCK_D // 2)
        half_mask = half_offsets < (head_dim // 2)
        cos = tl.load(
            cos_ptr
            + token_offsets[:, None] * cos_stride_t
            + half_offsets[None, :] * cos_stride_d,
            mask=token_mask[:, None] & half_mask[None, :],
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            sin_ptr
            + token_offsets[:, None] * cos_stride_t
            + half_offsets[None, :] * cos_stride_d,
            mask=token_mask[:, None] & half_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        even_offsets = half_offsets * 2
        odd_offsets = even_offsets + 1
        even = tl.load(
            key_mean_ptr + dim_offsets[None, :],
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        odd = even
        even = tl.where(dim_offsets[None, :] == even_offsets[:, None], key_vectors, even)
        odd = tl.where(dim_offsets[None, :] == odd_offsets[:, None], key_vectors, odd)

        key_even = key_vectors[:, 0::2]
        key_odd = key_vectors[:, 1::2]
        rotated_even = key_even * cos - key_odd * sin
        rotated_odd = key_even * sin + key_odd * cos
        key_vectors = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
        key_vectors += tl.load(
            key_mean_ptr + dim_offsets[None, :],
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        key_vectors = tl.where(dim_offsets[None, :] % 2 == 0, rotated_even, key_vectors)
        key_vectors = tl.where(dim_offsets[None, :] % 2 == 1, rotated_odd, key_vectors)

        scores = tl.sum(key_vectors * query[None, :], axis=1)
        scores *= softmax_scale
        scores = tl.where(token_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        probs = tl.exp(scores - block_max)
        probs = tl.where(token_mask, probs, 0.0)
        block_sum = tl.sum(probs, axis=0)
        output = tl.sum(value_vectors * probs[:, None], axis=0) / block_sum

        tl.store(
            output_ptr + pid * output_stride_b + dim_offsets * output_stride_d,
            output,
            mask=dim_mask,
        )
        tl.store(lse_ptr + pid, block_max + tl.log(block_sum))


def _decode_attention_triton(
    query: torch.Tensor,
    key_indices: torch.Tensor,
    value_indices: torch.Tensor,
    key_scales: torch.Tensor,
    key_zero_points: torch.Tensor,
    value_scales: torch.Tensor,
    value_zero_points: torch.Tensor,
    key_basis_t: torch.Tensor,
    value_basis_t: torch.Tensor,
    key_mean: torch.Tensor,
    value_mean: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_theta: float,
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    head_dim = int(query.shape[-1])
    seq_len = int(positions.numel())
    if seq_len == 0:
        return _empty_state(head_dim, device=query.device, dtype=query.dtype)

    if head_dim > 128:
        # PATCH(lna-lab Phase β): the original kernel hardcoded BLOCK_D=128 and uses
        # PyTorch-style slicing (key_vectors[:, 0::2]) for RoPE, which doesn't expand
        # cleanly to BLOCK_D=256 in Triton 3.6 ("Cannot make_shape_compatible"). For
        # head_dim>128 we fall back to torch until Phase β step 2 rewrites the kernel.
        raise ValueError("The reference Triton kernel supports head dimensions up to 128.")

    block_t = 64
    block_d = 128
    block_ck = 128
    block_cv = 128
    num_blocks = math.ceil(seq_len / block_t)

    cos, sin = _rope_cos_sin(
        positions,
        head_dim,
        rope_theta,
        device=query.device,
    )
    block_outputs = torch.empty(
        (num_blocks, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    block_lse = torch.empty(num_blocks, device=query.device, dtype=torch.float32)

    _decode_block_kernel[(num_blocks,)](
        query.contiguous().to(torch.float32),
        key_indices.contiguous(),
        value_indices.contiguous(),
        key_scales.contiguous().to(torch.float32),
        key_zero_points.contiguous().to(torch.float32),
        value_scales.contiguous().to(torch.float32),
        value_zero_points.contiguous().to(torch.float32),
        key_basis_t.contiguous().to(torch.float32),
        value_basis_t.contiguous().to(torch.float32),
        key_mean.contiguous().to(torch.float32),
        value_mean.contiguous().to(torch.float32),
        cos,
        sin,
        block_outputs,
        block_lse,
        seq_len,
        int(key_indices.shape[-1]),
        int(value_indices.shape[-1]),
        head_dim,
        float(softmax_scale),
        key_indices.stride(0),
        key_indices.stride(1),
        value_indices.stride(0),
        value_indices.stride(1),
        key_basis_t.stride(0),
        key_basis_t.stride(1),
        value_basis_t.stride(0),
        value_basis_t.stride(1),
        cos.stride(0),
        cos.stride(1),
        block_outputs.stride(0),
        block_outputs.stride(1),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        BLOCK_CK=block_ck,
        BLOCK_CV=block_cv,
    )

    merged_output, merged_lse = _empty_state(
        head_dim,
        device=query.device,
        dtype=query.dtype,
    )
    for block_idx in range(num_blocks):
        merged_output, merged_lse = merge_attention_states(
            merged_output,
            merged_lse,
            block_outputs[block_idx].to(query.dtype),
            block_lse[block_idx],
        )
    return merged_output, merged_lse


def decode_attention_from_kvtc(
    query: torch.Tensor,
    key_indices: torch.Tensor,
    value_indices: torch.Tensor,
    key_scales: torch.Tensor,
    key_zero_points: torch.Tensor,
    value_scales: torch.Tensor,
    value_zero_points: torch.Tensor,
    key_basis_t: torch.Tensor,
    value_basis_t: torch.Tensor,
    key_mean: torch.Tensor,
    value_mean: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_theta: float,
    softmax_scale: float,
    logits_soft_cap: float | None = None,
    use_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode attention over one compressed chunk for one query head."""

    # PATCH(lna-lab Phase β): treat soft_cap=0 same as None (Qwen3.6/LNARIZE pass 0
    # as a "disabled" sentinel; the original strict-None check silently routed those
    # models through the slow torch fallback). head_dim and component caps stay at 128
    # until Phase β step 2 rewrites the kernel to handle BLOCK_D=256 / Qwen3.6-27B.
    soft_cap_disabled = (logits_soft_cap is None) or (logits_soft_cap == 0)
    if (
        use_triton
        and HAS_TRITON
        and query.is_cuda
        and key_indices.is_cuda
        and value_indices.is_cuda
        and soft_cap_disabled
        and int(query.shape[-1]) <= 128
        and int(key_indices.shape[-1]) <= 128
        and int(value_indices.shape[-1]) <= 128
    ):
        return _decode_attention_triton(
            query,
            key_indices,
            value_indices,
            key_scales,
            key_zero_points,
            value_scales,
            value_zero_points,
            key_basis_t,
            value_basis_t,
            key_mean,
            value_mean,
            positions,
            rope_theta=rope_theta,
            softmax_scale=softmax_scale,
        )

    return decode_attention_torch(
        query,
        key_indices,
        value_indices,
        key_scales,
        key_zero_points,
        value_scales,
        value_zero_points,
        key_basis_t,
        value_basis_t,
        key_mean,
        value_mean,
        positions,
        rope_theta=rope_theta,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
    )
