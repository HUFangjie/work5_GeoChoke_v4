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
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "previous_cfi": None,
            "candidate_cfi": None,
            "fragility_injection_score": 0.0,
            "raw_functional_drift": 0.0,
            "raw_candidate_cfi": 0.0,
            "raw_cfi_injection": 0.0,
            "normalized_drift_risk": 0.0,
            "normalized_cfi_risk": 0.0,
            "normalized_injection_risk": 0.0,
            "raw_candidate_risk": 0.0,
            "accepted_candidate_risk": 0.0,
            "cfi_injection": 0.0,
            "candidate_functional_drift": 0.0,
            "accepted_update_scale": 1.0,
            "candidate_rejected": False,
            "rollback_triggered": False,
            "cusum_score": 0.0,
            "current_calibrated_error_energy": None,
            "unconstrained_target_error_energy": None,
            "target_next_error_energy": None,
            "selected_next_profile": current_profile_id,
            "next_profile_id": current_profile_id,
            "current_profile_id": current_profile_id,
            "profile_switching_indicator": False,
            "profile_transition_reason": "no_defense",
            "defense_enabled": False,
        }
