from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class GeoChokeControlDecision:
    previous_cfi: float
    candidate_cfi: float
    fragility_injection_score: float
    current_calibrated_error_energy: float
    unconstrained_target_error_energy: float
    target_next_error_energy: float
    selected_next_profile: str
    profile_switching_indicator: bool

    def as_metrics(self) -> dict[str, Any]:
        return {
            "previous_cfi": self.previous_cfi,
            "candidate_cfi": self.candidate_cfi,
            "fragility_injection_score": self.fragility_injection_score,
            "current_calibrated_error_energy": self.current_calibrated_error_energy,
            "unconstrained_target_error_energy": self.unconstrained_target_error_energy,
            "target_next_error_energy": self.target_next_error_energy,
            "selected_next_profile": self.selected_next_profile,
            "profile_switching_indicator": self.profile_switching_indicator,
        }


class GeoChokeController:
    """Implements the GeoChoke next-round CKKS profile controller."""

    def __init__(self, cfg: Any, calibration: Mapping[str, Mapping[str, Any]]) -> None:
        if not calibration:
            raise ValueError("GeoChokeController requires at least one calibrated profile")
        self.cfg = cfg
        self.calibration = dict(calibration)

    def select(self, prev_cfi: float, cand_cfi: float, current_profile_id: str) -> tuple[str, dict[str, Any]]:
        if current_profile_id not in self.calibration:
            raise KeyError(f"current profile {current_profile_id} is not calibrated")
        if not np.isfinite(prev_cfi) or not np.isfinite(cand_cfi):
            raise ValueError("CFI values must be finite")
        gamma = float(self.cfg.gamma)
        rho = float(self.cfg.rho)
        if gamma + rho <= 0.0:
            raise ValueError("GeoChoke gamma + rho must be positive")
        fragility_delta = max(0.0, float(cand_cfi) - float(prev_cfi))
        current_energy = float(self.calibration[current_profile_id]["mse"])
        unconstrained_target = (
            float(self.cfg.lambda_) * fragility_delta
            - float(prev_cfi)
            + 2.0 * rho * current_energy
        ) / (2.0 * (gamma + rho))
        maximum_energy = max(float(profile["mse"]) for profile in self.calibration.values())
        target_energy = float(np.clip(unconstrained_target, 0.0, maximum_energy))
        selected_profile = min(
            self.calibration,
            key=lambda profile_id: abs(float(self.calibration[profile_id]["mse"]) - target_energy),
        )
        decision = GeoChokeControlDecision(
            previous_cfi=float(prev_cfi),
            candidate_cfi=float(cand_cfi),
            fragility_injection_score=fragility_delta,
            current_calibrated_error_energy=current_energy,
            unconstrained_target_error_energy=float(unconstrained_target),
            target_next_error_energy=target_energy,
            selected_next_profile=selected_profile,
            profile_switching_indicator=selected_profile != current_profile_id,
        )
        return selected_profile, decision.as_metrics()
