"""Calibration utilities for KVTC."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch

from .common import CalibrationData
from .pca import PCACalibrator


@dataclass
class KVTCCalibrator:
    """Collect model KV samples and compute calibration artifacts."""

    head_group_size: int = 1
    rope_theta: float = 10000.0
    samples_collected: int = 0
    _pca: PCACalibrator = field(init=False)

    def __post_init__(self) -> None:
        self._pca = PCACalibrator(head_group_size=self.head_group_size, rope_theta=self.rope_theta)

    def collect_samples(
        self,
        model,
        tokenizer,
        texts: Sequence[str],
        max_samples: int = 100,
    ) -> None:
        """Collect KV cache samples from a causal LM."""

        device = next(model.parameters()).device
        for text in texts[:max_samples]:
            encoded = tokenizer(text, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded, use_cache=True)
            positions = torch.arange(encoded["input_ids"].shape[1], device=device)
            past = outputs.past_key_values
            # Handle DynamicCache (.layers), legacy DynamicCache (.key_cache), and tuple formats
            if hasattr(past, 'layers'):
                # Newest transformers DynamicCache with .layers[i].keys/.values
                # PATCH(lna-lab): hybrid attention support — contiguous index for full-attn layers
                contig_idx = 0
                for layer in past.layers:
                    if not hasattr(layer, 'keys') or layer.keys is None:
                        continue  # skip linear/mamba layers (no KV cache)
                    self._pca.collect(contig_idx, "keys", layer.keys[0].transpose(0, 1).detach().cpu(), positions.cpu())
                    self._pca.collect(contig_idx, "values", layer.values[0].transpose(0, 1).detach().cpu())
                    contig_idx += 1
            elif hasattr(past, 'key_cache'):
                # Older DynamicCache with .key_cache/.value_cache lists
                for layer_idx in range(len(past.key_cache)):
                    self._pca.collect(layer_idx, "keys", past.key_cache[layer_idx][0].transpose(0, 1).detach().cpu(), positions.cpu())
                    self._pca.collect(layer_idx, "values", past.value_cache[layer_idx][0].transpose(0, 1).detach().cpu())
            else:
                # Legacy tuple-of-tuples format
                for layer_idx, (keys, values) in enumerate(past):
                    self._pca.collect(layer_idx, "keys", keys[0].transpose(0, 1).detach().cpu(), positions.cpu())
                    self._pca.collect(layer_idx, "values", values[0].transpose(0, 1).detach().cpu())
            self.samples_collected += 1

    def compute_calibration(self, bit_budget_ratio: float = 0.25) -> CalibrationData:
        """Compute the final calibration data."""

        return self._pca.compute(bit_budget_ratio=bit_budget_ratio)

    def save(self, path: str | Path, calibration_data: CalibrationData) -> None:
        """Persist calibration data to disk."""

        torch.save(calibration_data, Path(path))

    @staticmethod
    def load(path: str | Path) -> CalibrationData:
        """Load calibration data from disk."""

        return torch.load(Path(path), weights_only=False)
