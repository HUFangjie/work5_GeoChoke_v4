from __future__ import annotations

from typing import Any

from defenses.base import DefenseStrategy


class NoDefense(DefenseStrategy):
    """DefenseStrategy that disables GeoChoke adaptation and keeps one CKKS profile."""

    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id
        self.initialized = False

    def initialize(self, model: Any, crypto_backend: Any, proxy_loader: Any, decrypt_aggregate_fn: Any = None) -> None:
        self.initialized = True

    def get_profile_for_round(self, round_id: int) -> str:
        return self.profile_id

    def after_aggregate(
        self,
        previous_model: Any,
        candidate_model: Any,
        current_profile_id: str,
        round_id: int,
    ) -> dict[str, Any]:
        return {
            "previous_cfi": None,
            "candidate_cfi": None,
            "fragility_injection_score": 0.0,
            "current_calibrated_error_energy": None,
            "unconstrained_target_error_energy": None,
            "target_next_error_energy": None,
            "selected_next_profile": current_profile_id,
            "profile_switching_indicator": False,
            "defense_enabled": False,
        }
