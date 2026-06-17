from __future__ import annotations

import csv
import os
from typing import Any, Dict, Mapping

import numpy as np


class ProfileCalibrator:
    """Offline profile calibration using real CKKS encrypt/decrypt residuals."""

    def __init__(
        self,
        crypto_backend: Any,
        decrypt_aggregate_fn: Any,
        profiles: Mapping[str, Dict[str, Any]],
        cfg: Any,
        output_dir: str | None = None,
    ) -> None:
        self.crypto_backend = crypto_backend
        self.decrypt_aggregate = decrypt_aggregate_fn
        self.profiles = dict(profiles)
        self.cfg = cfg
        self.output_dir = output_dir

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
            relative_errors: list[float] = []
            for vector in representative_vectors[: self.cfg.calibration_vectors]:
                encrypted = self.crypto_backend.encrypt_update(vector, profile_id)
                decrypted = self.decrypt_aggregate(encrypted, profile_id)
                residual = decrypted - vector
                mses.append(float(np.mean(residual * residual)))
                max_errors.append(float(np.max(np.abs(residual))))
                residual_samples.append(residual.astype(np.float64, copy=True))
                relative_errors.append(float(np.linalg.norm(residual) / max(np.linalg.norm(vector), 1e-12)))
            stacked_residuals = np.vstack(residual_samples)
            calibration[profile_id] = {
                "mse": float(np.mean(mses)),
                "max_abs_error": float(np.max(max_errors)),
                "relative_l2_error": float(np.mean(relative_errors)),
                "residual_mean": float(np.mean(stacked_residuals)),
                "residual_std": float(np.std(stacked_residuals)),
                "residual_samples": residual_samples,
            }
        reference_profile = next(iter(self.profiles))
        residual_source = calibration[reference_profile]["residual_samples"]
        perturbation_bank: list[np.ndarray] = []
        index = 0
        while len(perturbation_bank) < self.cfg.perturbation_count:
            perturbation_bank.append(residual_source[index % len(residual_source)].copy() * self.cfg.perturbation_scale)
            index += 1
        self._save_calibration(calibration)
        return calibration, perturbation_bank

    def _save_calibration(self, calibration: dict[str, dict[str, Any]]) -> None:
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        rows = []
        residual_payload: dict[str, np.ndarray] = {}
        for profile_id, metrics in calibration.items():
            rows.append(
                {
                    "profile_id": profile_id,
                    "sample_count": len(metrics["residual_samples"]),
                    "mse": metrics["mse"],
                    "max_abs_error": metrics["max_abs_error"],
                    "relative_l2_error": metrics["relative_l2_error"],
                    "residual_mean": metrics["residual_mean"],
                    "residual_std": metrics["residual_std"],
                }
            )
            residual_payload[profile_id] = np.vstack(metrics["residual_samples"])
        with open(os.path.join(self.output_dir, "direct_profile_calibration.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["profile_id", "sample_count", "mse", "max_abs_error", "relative_l2_error", "residual_mean", "residual_std"])
            writer.writeheader()
            writer.writerows(rows)
        np.savez(os.path.join(self.output_dir, "reference_residual_bank.npz"), **residual_payload)
