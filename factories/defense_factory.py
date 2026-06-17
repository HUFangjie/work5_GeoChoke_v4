from __future__ import annotations
from typing import Any, Callable

DEFENSE_REGISTRY: dict[str, Callable[[Any], Any]] = {}

def register_defense(name: str) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    def decorator(factory: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if name in DEFENSE_REGISTRY:
            raise ValueError(f"Defense already registered: {name}")
        DEFENSE_REGISTRY[name] = factory
        return factory
    return decorator

def create_defense(cfg: Any, crypto_bundle: Any | None = None) -> Any:
    try:
        return DEFENSE_REGISTRY[cfg.defense_name](cfg)
    except KeyError as exc:
        raise ValueError(f"Unknown defense: {cfg.defense_name}. Available: {sorted(DEFENSE_REGISTRY)}") from exc
