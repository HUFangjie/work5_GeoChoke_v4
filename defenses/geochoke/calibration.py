from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np


class ProfileCalibrator:
    """Offline profile calibration using real CKKS encrypt/decrypt residuals."""

    def __init__(self, crypto_backend: Any, decrypt_aggregate_fn: Any, profiles: Mapping[str, Dict[str, Any]], cfg: Any) -> None:
        self.crypto_backend = crypto_backend
        self.decrypt_aggregate = decrypt_aggregate_fn
        self.profiles = dict(profiles)
        self.cfg = cfg

    def calibrate(self, representative_vectors: list[np.ndarray]) -> tuple[dict[str, dict[str, Any]], list[np.ndarray]]:
        if not representative_vectors:
            raise ValueError("calibration requires at least one representative vector")
        dimension = int(representative_vectors[0].size)
        for vector in representative_vectors:
            if vector.size != dimension:
                raise ValueError("all representative vectors must have the model-update dimension")
        calibration: dict[str, dict[str, Any]] = {}
        for profile_id in self.profiles:
            mses: list[float] = []
            max_errors: list[float] = []
            residual_samples: list[np.ndarray] = []
            for vector in representative_vectors[: self.cfg.calibration_vectors]:
                encrypted = self.crypto_backend.encrypt_update(vector, profile_id)
                decrypted = self.decrypt_aggregate(encrypted, profile_id)
                residual = decrypted - vector
                mses.append(float(np.mean(residual * residual)))
                max_errors.append(float(np.max(np.abs(residual))))
                residual_samples.append(residual.astype(np.float64, copy=True))
            calibration[profile_id] = {
                "mse": float(np.mean(mses)),
                "max_abs_error": float(np.max(max_errors)),
                "residual_samples": residual_samples,
            }
        reference_profile = next(iter(self.profiles))
        residual_source = calibration[reference_profile]["residual_samples"]
        perturbation_bank: list[np.ndarray] = []
        index = 0
        while len(perturbation_bank) < self.cfg.perturbation_count:
            perturbation_bank.append(residual_source[index % len(residual_source)].copy() * self.cfg.perturbation_scale)
            index += 1
        return calibration, perturbation_bank
