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
    normalized_previous_cfi: float
    normalized_fragility_injection: float
    normalized_current_error_energy: float
    normalized_target_error_energy: float

    def as_metrics(self) -> dict[str, Any]:
        return self.__dict__.copy()


class GeoChokeController:
    """GeoChoke next-round CKKS profile controller in normalized log-energy space."""

    def __init__(self, cfg: Any, calibration: Mapping[str, Mapping[str, Any]]) -> None:
        if not calibration:
            raise ValueError("GeoChokeController requires at least one calibrated profile")
        self.cfg = cfg
        self.calibration = dict(calibration)
        energies = np.asarray([max(float(profile["mse"]), 1e-30) for profile in self.calibration.values()], dtype=np.float64)
        self.log_u_min = float(np.min(np.log10(energies)))
        self.log_u_max = float(np.max(np.log10(energies)))
        self.log_denominator = max(self.log_u_max - self.log_u_min, 1e-12)

    def _normalize_energy(self, energy: float) -> float:
        log_energy = np.log10(max(float(energy), 1e-30))
        return float(np.clip((log_energy - self.log_u_min) / self.log_denominator, 0.0, 1.0))

    def _denormalize_energy(self, normalized_energy: float) -> float:
        clipped = float(np.clip(normalized_energy, 0.0, 1.0))
        return float(10.0 ** (self.log_u_min + clipped * self.log_denominator))

    def select(self, prev_cfi: float, cand_cfi: float, current_profile_id: str, cfi_scale: float = 1.0) -> tuple[str, dict[str, Any]]:
        if current_profile_id not in self.calibration:
            raise KeyError(f"current profile {current_profile_id} is not calibrated")
        if not np.isfinite(prev_cfi) or not np.isfinite(cand_cfi):
            raise ValueError("CFI values must be finite")
        gamma = float(self.cfg.gamma)
        rho = float(self.cfg.rho)
        if gamma + rho <= 0.0:
            raise ValueError("GeoChoke gamma + rho must be positive")
        raw_delta = max(0.0, float(cand_cfi) - float(prev_cfi))
        scale = max(float(cfi_scale), 1e-12)
        normalized_prev_cfi = float(np.clip(float(prev_cfi) / scale, 0.0, 10.0))
        normalized_delta = float(np.clip(raw_delta / scale, 0.0, 10.0))
        current_energy = max(float(self.calibration[current_profile_id]["mse"]), 1e-30)
        normalized_current_energy = self._normalize_energy(current_energy)
        unconstrained_normalized = (
            float(self.cfg.lambda_) * normalized_delta
            - normalized_prev_cfi
            + 2.0 * rho * normalized_current_energy
        ) / (2.0 * (gamma + rho))
        target_normalized = float(np.clip(unconstrained_normalized, 0.0, 1.0))
        target_energy = self._denormalize_energy(target_normalized)
        target_log_energy = np.log10(max(target_energy, 1e-30))
        selected_profile = min(
            self.calibration,
            key=lambda profile_id: abs(np.log10(max(float(self.calibration[profile_id]["mse"]), 1e-30)) - target_log_energy),
        )
        decision = GeoChokeControlDecision(
            previous_cfi=float(prev_cfi),
            candidate_cfi=float(cand_cfi),
            fragility_injection_score=raw_delta,
            current_calibrated_error_energy=current_energy,
            unconstrained_target_error_energy=self._denormalize_energy(unconstrained_normalized),
            target_next_error_energy=target_energy,
            selected_next_profile=selected_profile,
            profile_switching_indicator=selected_profile != current_profile_id,
            normalized_previous_cfi=normalized_prev_cfi,
            normalized_fragility_injection=normalized_delta,
            normalized_current_error_energy=normalized_current_energy,
            normalized_target_error_energy=target_normalized,
        )
        return selected_profile, decision.as_metrics()
