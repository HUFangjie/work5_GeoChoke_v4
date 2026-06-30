from __future__ import annotations

import logging

import numpy as np
from typing import Any, Mapping

from crypto.update_codec import ModelUpdateCodec
from defenses.base import DefenseStrategy
from defenses.geochoke.calibration import ProfileCalibrator
from defenses.geochoke.calibration_provider import CalibrationTensorProvider
from defenses.geochoke.cfi_estimator import CFIEstimator
from defenses.geochoke.controller import GeoChokeController
from defenses.geochoke.tangent_commitment import TangentCommitter


class GeoChokeDefense(DefenseStrategy):
    def __init__(self, cfg: Any, profiles: Mapping[str, dict[str, Any]], device: str = "cpu", output_dir: str | None = None) -> None:
        self.cfg = cfg
        self.profiles = dict(profiles)
        self.device = device
        self.output_dir = output_dir
        self._validate_config()
        self.next_profile = cfg.initial_profile_id
        self.history: list[dict[str, Any]] = []
        self._previous_cfi_cache: float | None = None

    def initialize(self, model: Any, crypto_backend: Any, proxy_loader: Any, decrypt_aggregate_fn: Any = None) -> None:
        if decrypt_aggregate_fn is None:
            raise ValueError("GeoChoke calibration requires aggregate-only decryption callable")
        self.codec = ModelUpdateCodec(model)
        self.proxy_loader = proxy_loader
        self.tangent_committer = TangentCommitter(self.codec, proxy_loader, self.cfg, self.device)
        provider = CalibrationTensorProvider(self.codec, self.cfg, self.device)
        representative_tensors = provider.build(model, proxy_loader)
        calibrator = ProfileCalibrator(
            crypto_backend,
            decrypt_aggregate_fn,
            self.profiles,
            self.cfg,
            output_dir=self.output_dir,
        )
        self.calibration, self.perturbation_bank = calibrator.calibrate(representative_tensors)
        self.profile_range_warning = calibrator.profile_range_warning
        if self.profile_range_warning:
            logging.getLogger("geochoke").warning("GeoChoke profile range may be too weak to suppress DBA or adjacent calibrated MSE values are nearly identical.")
        self.estimator = CFIEstimator(self.codec, proxy_loader, self.perturbation_bank, self.device)
        self.controller = GeoChokeController(self.cfg, self.calibration)

    def get_profile_for_round(self, round_id: int) -> str:
        return self.next_profile

    def after_aggregate(self, previous_model: Any, candidate_model: Any, current_profile_id: str, round_id: int) -> dict[str, Any]:
        cfi_fis_enabled = bool(getattr(self.cfg, "cfi_fis_enabled", True))
        precision_control_enabled = bool(getattr(self.cfg, "precision_control_enabled", True))
        ablation_variant = str(getattr(self.cfg, "ablation_variant", "full"))
        tangent_commitment_enabled = bool(getattr(self.cfg, "tangent_commitment_enabled", True))

        if current_profile_id not in self.calibration:
            current_profile_id = self.controller.profile_order[0]
        current_energy = float(self.calibration[current_profile_id]["mse"])
        current_rank = self.controller.profile_order.index(current_profile_id)
        max_energy = float(self.calibration[self.controller.profile_order[-1]]["mse"])

        if cfi_fis_enabled:
            previous_cfi = self._previous_cfi_cache
            previous_cfi_source = "cached_previous_round_candidate_cfi"
            if previous_cfi is None:
                previous_cfi = self.estimator.estimate(previous_model)
                previous_cfi_source = "estimated_initial_previous_model_cfi"
            candidate_cfi = self.estimator.estimate(candidate_model)
            if previous_cfi < -1e-12 or candidate_cfi < -1e-12:
                raise ValueError(f"GeoChoke CFI became negative: previous={previous_cfi}, candidate={candidate_cfi}")
            next_profile, metrics = self.controller.select(previous_cfi, candidate_cfi, current_profile_id)
            fragility_injection_score = float(metrics.get("fragility_injection_score", max(0.0, candidate_cfi - previous_cfi)))
            self._previous_cfi_cache = candidate_cfi
            cached_previous_cfi_for_next_round = float(candidate_cfi)
        else:
            previous_cfi = None
            candidate_cfi = None
            previous_cfi_source = "cfi_fis_disabled"
            fragility_injection_score = 0.0
            next_profile = current_profile_id
            self._previous_cfi_cache = None
            cached_previous_cfi_for_next_round = None
            metrics = {
                "previous_cfi": None,
                "candidate_cfi": None,
                "fragility_injection_score": fragility_injection_score,
                "cfi_nonnegative_check": True,
                "current_profile_id": current_profile_id,
                "selected_next_profile": current_profile_id,
                "selected_next_profile_id": current_profile_id,
                "current_error_energy": current_energy,
                "current_calibrated_error_energy": current_energy,
                "unconstrained_target_error_energy": current_energy,
                "clipped_target_error_energy": current_energy,
                "target_next_error_energy": current_energy,
                "selected_profile_error_energy": current_energy,
                "profile_switch": False,
                "profile_switching_indicator": False,
                "profile_rank": current_rank,
                "max_supported_error_energy": max_energy,
                "target_energy_exceeds_profile_range": False,
                "target_energy_below_profile_range": False,
            }

        if not precision_control_enabled:
            next_profile = current_profile_id
            metrics.update(
                {
                    "selected_next_profile": current_profile_id,
                    "selected_next_profile_id": current_profile_id,
                    "target_next_error_energy": current_energy,
                    "selected_profile_error_energy": current_energy,
                    "profile_switch": False,
                    "profile_switching_indicator": False,
                    "profile_rank": current_rank,
                    "target_energy_exceeds_profile_range": False,
                    "target_energy_below_profile_range": False,
                }
            )

        if tangent_commitment_enabled:
            tau_raw = self.cfg.tangent_tau_max / (1.0 + self.cfg.tangent_lambda * fragility_injection_score)
            tangent_tau = max(self.cfg.tangent_tau_min, min(self.cfg.tangent_tau_max, tau_raw))
        else:
            tangent_tau = self.cfg.tangent_tau_max
        self.next_profile = next_profile
        metrics.update(
            {
                "ablation_variant": ablation_variant,
                "cfi_fis_enabled": cfi_fis_enabled,
                "precision_control_enabled": precision_control_enabled,
                "previous_cfi_source": previous_cfi_source,
                "cached_previous_cfi_for_next_round": cached_previous_cfi_for_next_round,
                "reference_profile_id": self.cfg.reference_profile_id,
                "cfi_reference_profile_id": self.cfg.reference_profile_id,
                "perturbation_count": int(self.cfg.perturbation_count),
                "perturbation_scale": float(self.cfg.perturbation_scale),
                "profile_calibration_residual_mse": current_energy,
                "reference_residual_mse": float(self.calibration[self.cfg.reference_profile_id]["mse"]),
                "profile_range_warning": bool(getattr(self, "profile_range_warning", False)),
                "tangent_commitment_enabled": tangent_commitment_enabled,
                "tangent_tau": float(tangent_tau),
                "tangent_tau_min": float(self.cfg.tangent_tau_min),
                "tangent_tau_max": float(self.cfg.tangent_tau_max),
                "tangent_lambda": float(self.cfg.tangent_lambda),
            }
        )
        self.history.append({"round": round_id, "current_profile": current_profile_id, **metrics})
        return metrics

    def commit_update(self, update_vector: Any, previous_model: Any, round_id: int, tangent_tau: float):
        original_norm = float(np.linalg.norm(update_vector))
        eps = float(getattr(self.cfg, "tangent_eps", 1e-12))
        if not self.cfg.tangent_commitment_enabled:
            return update_vector, {
                "tangent_commitment_enabled": False,
                "tangent_basis_rank_actual": 0,
                "tangent_tau": float(tangent_tau),
                "tangent_rho": 1.0,
                "tangent_parallel_norm": original_norm,
                "tangent_perp_norm": 0.0,
                "tangent_null_ratio": 0.0,
                "tangent_committed_update_norm": original_norm,
                "tangent_original_update_norm": original_norm,
                "tangent_update_shrink_ratio": float(original_norm / (original_norm + eps)),
            }
        return self.tangent_committer.commit(update_vector, previous_model, round_id, tangent_tau)

    def _validate_config(self) -> None:
        if self.cfg.initial_profile_id not in self.profiles:
            raise ValueError(f"initial_profile_id {self.cfg.initial_profile_id!r} is not defined in ckks_profiles")
        if self.cfg.reference_profile_id not in self.profiles:
            raise ValueError(f"reference_profile_id {self.cfg.reference_profile_id!r} is not defined in ckks_profiles")
        positive_fields = ["lambda_", "gamma", "rho", "calibration_vectors", "perturbation_count", "perturbation_scale", "tangent_basis_rank", "tangent_max_proxy_batches", "tangent_tau_max", "tangent_tau_min", "tangent_lambda", "tangent_eps", "tangent_refresh_interval"]
        for field in positive_fields:
            value = getattr(self.cfg, field)
            if float(value) <= 0.0:
                raise ValueError(f"GeoChokeConfig.{field} must be > 0")
