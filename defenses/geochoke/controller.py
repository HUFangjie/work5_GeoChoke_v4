from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class GeoChokeController:
    """GeoChoke next-round CKKS profile controller over calibrated valid profiles."""

    def __init__(self, cfg: Any, calibration: Mapping[str, Mapping[str, Any]]) -> None:
        self.cfg = cfg
        self.calibration = {pid: dict(metrics) for pid, metrics in calibration.items() if metrics.get("valid_profile", True)}
        if not self.calibration:
            raise ValueError("GeoChokeController requires at least one valid calibrated profile")
        self.profile_order = sorted(self.calibration, key=lambda pid: float(self.calibration[pid]["mse"]))

    def select(self, prev_cfi: float, cand_cfi: float, current_profile_id: str) -> tuple[str, dict[str, Any]]:
        if current_profile_id not in self.calibration:
            current_profile_id = self.profile_order[0]
        if not np.isfinite(prev_cfi) or not np.isfinite(cand_cfi):
            raise ValueError("CFI values must be finite")
        if prev_cfi < -1e-12 or cand_cfi < -1e-12:
            raise ValueError(f"CFI values must be nonnegative: previous={prev_cfi}, candidate={cand_cfi}")
        prev_cfi = max(0.0, float(prev_cfi))
        cand_cfi = max(0.0, float(cand_cfi))
        gamma = float(self.cfg.gamma)
        rho = float(self.cfg.rho)
        lambda_value = float(self.cfg.lambda_)
        if lambda_value <= 0.0 or gamma <= 0.0 or rho <= 0.0:
            raise ValueError("GeoChoke lambda_, gamma, and rho must each be positive")
        fragility_delta = max(0.0, cand_cfi - prev_cfi)
        current_energy = float(self.calibration[current_profile_id]["mse"])
        unconstrained_target = (lambda_value * fragility_delta - prev_cfi + 2.0 * rho * current_energy) / (2.0 * (gamma + rho))
        min_energy = float(self.calibration[self.profile_order[0]]["mse"])
        max_energy = float(self.calibration[self.profile_order[-1]]["mse"])
        below = unconstrained_target < min_energy
        exceeds = unconstrained_target > max_energy
        clipped_target = float(np.clip(unconstrained_target, min_energy, max_energy))
        selected_profile = min(self.profile_order, key=lambda pid: abs(float(self.calibration[pid]["mse"]) - clipped_target))
        selected_energy = float(self.calibration[selected_profile]["mse"])
        return selected_profile, {
            "previous_cfi": prev_cfi,
            "candidate_cfi": cand_cfi,
            "fragility_injection_score": fragility_delta,
            "cfi_nonnegative_check": bool(prev_cfi >= 0.0 and cand_cfi >= 0.0 and fragility_delta >= 0.0),
            "current_profile_id": current_profile_id,
            "selected_next_profile": selected_profile,
            "selected_next_profile_id": selected_profile,
            "current_error_energy": current_energy,
            "current_calibrated_error_energy": current_energy,
            "unconstrained_target_error_energy": float(unconstrained_target),
            "clipped_target_error_energy": clipped_target,
            "target_next_error_energy": clipped_target,
            "selected_profile_error_energy": selected_energy,
            "profile_switch": selected_profile != current_profile_id,
            "profile_switching_indicator": selected_profile != current_profile_id,
            "profile_rank": self.profile_order.index(selected_profile),
            "max_supported_error_energy": max_energy,
            "target_energy_exceeds_profile_range": bool(exceeds),
            "target_energy_below_profile_range": bool(below),
        }
