from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    numel: int


class ModelUpdateCodec:
    """Deterministic trainable-parameter vector codec.

    The codec records `named_parameters()` order at construction time. It only
    flattens trainable floating-point parameters and never relies on incidental
    `state_dict` insertion order while reconstructing model updates.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.specs: list[ParamSpec] = [
            ParamSpec(name=name, shape=tuple(parameter.shape), dtype=parameter.dtype, numel=parameter.numel())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.is_floating_point()
        ]
        self.trainable_names = tuple(spec.name for spec in self.specs)
        self.total_dimension = sum(spec.numel for spec in self.specs)

    def flatten_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        arrays: list[np.ndarray] = []
        for spec in self.specs:
            if spec.name not in state_dict:
                raise KeyError(f"state_dict is missing trainable parameter {spec.name}")
            tensor = state_dict[spec.name]
            if tuple(tensor.shape) != spec.shape:
                raise ValueError(f"shape mismatch for {spec.name}: expected {spec.shape}, got {tuple(tensor.shape)}")
            arrays.append(tensor.detach().cpu().to(torch.float64).reshape(-1).numpy())
        return np.concatenate(arrays).astype(np.float64, copy=False) if arrays else np.empty(0, dtype=np.float64)

    def unflatten_to_update_state_dict(self, flat_update: Iterable[float]) -> Dict[str, torch.Tensor]:
        vector = np.asarray(flat_update, dtype=np.float64)
        if vector.size != self.total_dimension:
            raise ValueError(f"flat update length {vector.size} != expected {self.total_dimension}")
        update: Dict[str, torch.Tensor] = {}
        offset = 0
        for spec in self.specs:
            next_offset = offset + spec.numel
            update[spec.name] = torch.as_tensor(vector[offset:next_offset], dtype=spec.dtype).reshape(spec.shape).clone()
            offset = next_offset
        return update

    def unflatten_to_state_dict(
        self,
        flat_update: Iterable[float],
        reference_state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return a full state_dict-shaped object containing update tensors for trainable params.

        Non-trainable entries are cloned from the reference for structural
        compatibility, but callers should use :meth:`apply_update_to_state_dict`
        when they need to add an update to a model.
        """

        update = self.unflatten_to_update_state_dict(flat_update)
        result = {name: tensor.detach().cpu().clone() for name, tensor in reference_state_dict.items()}
        for name, tensor in update.items():
            result[name] = tensor
        return result

    def apply_update_to_state_dict(
        self,
        base_state_dict: Dict[str, torch.Tensor],
        flat_update: Iterable[float],
        step_size: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        update = self.unflatten_to_update_state_dict(flat_update)
        result = {name: tensor.detach().cpu().clone() for name, tensor in base_state_dict.items()}
        for spec in self.specs:
            result[spec.name] = base_state_dict[spec.name].detach().cpu() + float(step_size) * update[spec.name]
        return result

    def update_to_state_dict(self, base_state: Dict[str, torch.Tensor], flat_update: Iterable[float]) -> Dict[str, torch.Tensor]:
        """Backward-compatible alias for applying an update with step size 1."""

        return self.apply_update_to_state_dict(base_state, flat_update, step_size=1.0)
