from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

MODEL_REGISTRY: dict[str, Callable[[], Any]] = {}

def register_model(name: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    def decorator(factory: Callable[[], Any]) -> Callable[[], Any]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model already registered: {name}")
        MODEL_REGISTRY[name] = factory
        return factory
    return decorator

@dataclass(frozen=True)
class RegisteredModelFactory:
    name: str
    def create(self) -> Any:
        return create_model(self.name)

def create_model(name: str) -> Any:
    try:
        return MODEL_REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown model: {name}. Available: {sorted(MODEL_REGISTRY)}") from exc

def create_model_factory(cfg: Any) -> RegisteredModelFactory:
    if cfg.model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {cfg.model_name}. Available: {sorted(MODEL_REGISTRY)}")
    return RegisteredModelFactory(cfg.model_name)
