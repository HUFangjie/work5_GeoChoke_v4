from __future__ import annotations

import logging
from typing import Any, Mapping

from crypto.update_codec import ModelUpdateCodec
from defenses.base import DefenseStrategy
from defenses.geochoke.calibration import ProfileCalibrator
from defenses.geochoke.calibration_provider import CalibrationTensorProvider
from defenses.geochoke.cfi_estimator import CFIEstimator
from defenses.geochoke.controller import GeoChokeController


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
        previous_cfi = self._previous_cfi_cache
        previous_cfi_source = "cached_previous_round_candidate_cfi"
        if previous_cfi is None:
            previous_cfi = self.estimator.estimate(previous_model)
            previous_cfi_source = "estimated_initial_previous_model_cfi"
        candidate_cfi = self.estimator.estimate(candidate_model)
        if previous_cfi < -1e-12 or candidate_cfi < -1e-12:
            raise ValueError(f"GeoChoke CFI became negative: previous={previous_cfi}, candidate={candidate_cfi}")
        next_profile, metrics = self.controller.select(previous_cfi, candidate_cfi, current_profile_id)
        self._previous_cfi_cache = candidate_cfi
        self.next_profile = next_profile
        metrics.update(
            {
                "previous_cfi_source": previous_cfi_source,
                "cached_previous_cfi_for_next_round": float(candidate_cfi),
                "reference_profile_id": self.cfg.reference_profile_id,
                "cfi_reference_profile_id": self.cfg.reference_profile_id,
                "perturbation_count": int(self.cfg.perturbation_count),
                "perturbation_scale": float(self.cfg.perturbation_scale),
                "profile_calibration_residual_mse": float(self.calibration[current_profile_id]["mse"]),
                "reference_residual_mse": float(self.calibration[self.cfg.reference_profile_id]["mse"]),
                "profile_range_warning": bool(getattr(self, "profile_range_warning", False)),
            }
        )
        self.history.append({"round": round_id, "current_profile": current_profile_id, **metrics})
        return metrics

    def _validate_config(self) -> None:
        if self.cfg.initial_profile_id not in self.profiles:
            raise ValueError(f"initial_profile_id {self.cfg.initial_profile_id!r} is not defined in ckks_profiles")
        if self.cfg.reference_profile_id not in self.profiles:
            raise ValueError(f"reference_profile_id {self.cfg.reference_profile_id!r} is not defined in ckks_profiles")
        positive_fields = ["lambda_", "gamma", "rho", "calibration_vectors", "perturbation_count", "perturbation_scale"]
        for field in positive_fields:
            value = getattr(self.cfg, field)
            if float(value) <= 0.0:
                raise ValueError(f"GeoChokeConfig.{field} must be > 0")
