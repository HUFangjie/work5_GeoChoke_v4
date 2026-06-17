from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import tenseal as ts


@dataclass(frozen=True)
class PublicCKKSContextBundle:
    """Public-only CKKS material safe for clients and aggregation server."""

    profile_id: str
    public_context: Any
    slot_count: int
    parameters: Dict[str, Any]


@dataclass(frozen=True)
class SecretCKKSContextBundle:
    """Secret CKKS material owned only by AuthorizedDecryptionService."""

    profile_id: str
    secret_context: Any
    public_context: Any
    slot_count: int
    parameters: Dict[str, Any]


class CKKSContextManager:
    """Owns per-profile CKKS secret contexts and exports public-only bundles.

    This object must be constructed by the authorized decryption side. Clients and
    the aggregation server receive only the result of :meth:`public_bundles` and
    therefore have no method or object reference that can reveal a secret key.
    """

    def __init__(self, profiles: Mapping[str, Dict[str, Any]]) -> None:
        self._profiles = dict(profiles)
        self._secret_bundles: Dict[str, SecretCKKSContextBundle] = {}

    def initialize(self) -> None:
        for profile_id, cfg in self._profiles.items():
            self._validate_profile(profile_id, cfg)
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=cfg["poly_modulus_degree"],
                coeff_mod_bit_sizes=cfg["coeff_mod_bit_sizes"],
            )
            context.global_scale = 2 ** int(cfg["global_scale_bits"])
            context.generate_galois_keys()
            public_context = ts.context_from(context.serialize(save_secret_key=False))
            self._secret_bundles[profile_id] = SecretCKKSContextBundle(
                profile_id=profile_id,
                secret_context=context,
                public_context=public_context,
                slot_count=int(cfg["poly_modulus_degree"]) // 2,
                parameters=dict(cfg),
            )

    def public_bundles(self) -> Dict[str, PublicCKKSContextBundle]:
        return {
            pid: PublicCKKSContextBundle(
                profile_id=bundle.profile_id,
                public_context=bundle.public_context,
                slot_count=bundle.slot_count,
                parameters=dict(bundle.parameters),
            )
            for pid, bundle in self._secret_bundles.items()
        }

    def secret_context(self, profile_id: str) -> Any:
        return self._secret_bundles[profile_id].secret_context

    def public_context(self, profile_id: str) -> Any:
        return self._secret_bundles[profile_id].public_context

    def slot_count(self, profile_id: str) -> int:
        return self._secret_bundles[profile_id].slot_count

    @staticmethod
    def _validate_profile(profile_id: str, cfg: Mapping[str, Any]) -> None:
        required = {"poly_modulus_degree", "coeff_mod_bit_sizes", "global_scale_bits"}
        missing = required.difference(cfg)
        if missing:
            raise ValueError(f"CKKS profile {profile_id} is missing {sorted(missing)}")
        degree = int(cfg["poly_modulus_degree"])
        if degree < 8192 or degree & (degree - 1):
            raise ValueError(
                f"CKKS profile {profile_id} must use a power-of-two poly_modulus_degree >= 8192"
            )
        coeff_bits = [int(x) for x in cfg["coeff_mod_bit_sizes"]]
        if len(coeff_bits) < 3 or coeff_bits[0] > 60 or coeff_bits[-1] > 60:
            raise ValueError(f"CKKS profile {profile_id} has invalid coefficient modulus chain")
        max_total_coeff_bits_128 = {8192: 218, 16384: 438, 32768: 881}
        if degree not in max_total_coeff_bits_128:
            raise ValueError(f"CKKS profile {profile_id} degree is not in the built-in 128-bit safety table")
        if sum(coeff_bits) > max_total_coeff_bits_128[degree]:
            raise ValueError(f"CKKS profile {profile_id} exceeds conservative 128-bit coefficient modulus budget")
        scale_bits = int(cfg["global_scale_bits"])
        if scale_bits <= 0 or scale_bits >= min(coeff_bits[1:-1]):
            raise ValueError(f"CKKS profile {profile_id} scale must fit inside middle primes")
