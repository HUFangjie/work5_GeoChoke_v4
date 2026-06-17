from __future__ import annotations
from typing import Any, Callable

ATTACK_REGISTRY: dict[str, Callable[[Any], Any]] = {}

def register_attack(name: str) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    def decorator(factory: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if name in ATTACK_REGISTRY:
            raise ValueError(f"Attack already registered: {name}")
        ATTACK_REGISTRY[name] = factory
        return factory
    return decorator

def create_attack(cfg: Any) -> Any:
    name = getattr(cfg, "attack_name", getattr(cfg, "attack_type", "none"))
    try:
        return ATTACK_REGISTRY[name](cfg)
    except KeyError as exc:
        raise ValueError(f"Unknown attack: {name}. Available: {sorted(ATTACK_REGISTRY)}") from exc
