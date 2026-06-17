from __future__ import annotations
from typing import Any, Callable



def _ensure_default_registrations() -> None:
    # Import for side effects: registers built-in components. Safe if already imported.
    import factories.defaults  # noqa: F401

ATTACK_REGISTRY: dict[str, Callable[[Any], Any]] = {}

def register_attack(name: str) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    def decorator(factory: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if name in ATTACK_REGISTRY:
            raise ValueError(f"Attack already registered: {name}")
        ATTACK_REGISTRY[name] = factory
        return factory
    return decorator

def create_attack(cfg: Any) -> Any:
    _ensure_default_registrations()
    name = getattr(cfg, "attack_name", getattr(cfg, "attack_type", "none"))
    try:
        return ATTACK_REGISTRY[name](cfg)
    except KeyError as exc:
        raise ValueError(f"Unknown attack: {name}. Available: {sorted(ATTACK_REGISTRY)}") from exc
