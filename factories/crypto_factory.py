from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

CRYPTO_REGISTRY: dict[str, Callable[[Any], Any]] = {}

def register_crypto(name: str) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    def decorator(factory: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if name in CRYPTO_REGISTRY:
            raise ValueError(f"Crypto backend already registered: {name}")
        CRYPTO_REGISTRY[name] = factory
        return factory
    return decorator

@dataclass(frozen=True)
class CryptoBundle:
    public_backend: Any
    context_manager: Any
    decryption_service: Any

def create_crypto_backend(cfg: Any) -> CryptoBundle:
    try:
        return CRYPTO_REGISTRY[cfg.crypto_backend_name](cfg)
    except KeyError as exc:
        raise ValueError(f"Unknown crypto backend: {cfg.crypto_backend_name}. Available: {sorted(CRYPTO_REGISTRY)}") from exc
