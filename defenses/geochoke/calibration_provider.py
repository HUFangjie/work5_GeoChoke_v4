from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CalibrationTensor:
    vector: np.ndarray
    source: str
    scale: float
    l2_norm: float
    mean: float
    std: float
    max_abs: float


class CalibrationTensorProvider:
    """Build protocol calibration tensors with model-update dimension and realistic layer scales."""

    def __init__(self, codec: Any, cfg: Any, device: str = "cpu") -> None:
        self.codec = codec
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(int(getattr(cfg, "calibration_seed", 20240617)))

    def build(self, model: torch.nn.Module, proxy_loader: Any | None = None) -> list[CalibrationTensor]:
        tensors: list[CalibrationTensor] = []
        empirical = self._empirical_update(model, proxy_loader)
        if empirical is not None:
            tensors.append(self._record(empirical, "proxy_one_step_empirical_update", 1.0))

        state = model.state_dict()
        flat_state = self.codec.flatten_state_dict(state).astype(np.float64, copy=False)
        base_scale = max(float(np.std(flat_state)), float(np.mean(np.abs(flat_state))), 1e-3)
        empirical_norm = float(np.linalg.norm(empirical)) if empirical is not None else base_scale * np.sqrt(self.codec.total_dimension)
        layer_scales = self._layer_scales(state, base_scale)
        target_count = int(self.cfg.calibration_vectors)
        scale_grid = np.geomspace(0.25, 4.0, max(1, target_count))
        while len(tensors) < target_count:
            scale = float(scale_grid[len(tensors) % len(scale_grid)])
            vector = np.zeros(self.codec.total_dimension, dtype=np.float64)
            cursor = 0
            for name, tensor in state.items():
                width = int(tensor.numel())
                sigma = layer_scales[name] * scale
                vector[cursor : cursor + width] = self.rng.normal(0.0, sigma, size=width)
                cursor += width
            norm = np.linalg.norm(vector)
            target_norm = max(1e-12, empirical_norm * scale)
            vector = vector * (target_norm / max(norm, 1e-12))
            tensors.append(self._record(vector, f"layer_scaled_reference_distribution_{len(tensors)}", scale))
        return tensors[:target_count]

    def _empirical_update(self, model: torch.nn.Module, proxy_loader: Any | None) -> np.ndarray | None:
        if proxy_loader is None:
            return None
        try:
            images, labels = next(iter(proxy_loader))
        except StopIteration:
            return None
        local = copy.deepcopy(model).to(self.device)
        local.train()
        opt = torch.optim.SGD(local.parameters(), lr=float(getattr(self.cfg, "calibration_reference_lr", 0.01)))
        loss_fn = torch.nn.CrossEntropyLoss()
        images, labels = images.to(self.device), labels.to(self.device)
        opt.zero_grad()
        logits = local(images)
        num_classes = int(logits.shape[1])
        labels = labels.long()
        valid_labels = (labels >= 0) & (labels < num_classes)
        if bool(valid_labels.all()):
            training_labels = labels
        else:
            # Proxy data is intentionally unlabeled (MNISTProvider returns -1).
            # Use deterministic model pseudo-labels so calibration can estimate a
            # realistic one-step update without requiring private/proxy labels.
            training_labels = logits.detach().argmax(dim=1)
        loss = loss_fn(logits, training_labels)
        loss.backward()
        opt.step()
        return self.codec.flatten_state_dict(local.state_dict()) - self.codec.flatten_state_dict(model.state_dict())

    def _layer_scales(self, state: dict[str, torch.Tensor], fallback: float) -> dict[str, float]:
        scales = {}
        for name, tensor in state.items():
            array = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
            scales[name] = max(float(np.std(array)), float(np.mean(np.abs(array))) * 0.1, fallback * 0.05, 1e-6)
        return scales

    @staticmethod
    def _record(vector: np.ndarray, source: str, scale: float) -> CalibrationTensor:
        vector = vector.astype(np.float64, copy=True)
        return CalibrationTensor(
            vector=vector,
            source=source,
            scale=float(scale),
            l2_norm=float(np.linalg.norm(vector)),
            mean=float(np.mean(vector)),
            std=float(np.std(vector)),
            max_abs=float(np.max(np.abs(vector))),
        )
