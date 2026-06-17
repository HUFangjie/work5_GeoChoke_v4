from __future__ import annotations
from typing import Any, Callable

DATASET_REGISTRY: dict[str, type] = {}

def register_dataset(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset already registered: {name}")
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def create_dataset_provider(cfg: Any) -> Any:
    try:
        provider_cls = DATASET_REGISTRY[cfg.dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {cfg.dataset_name}. Available: {sorted(DATASET_REGISTRY)}") from exc
    return provider_cls(cfg)
