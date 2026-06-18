from __future__ import annotations

import csv
import hashlib
import os
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from defenses.geochoke.calibration_provider import CalibrationTensor


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

    def calibrate(self, representative_tensors: Sequence[CalibrationTensor]) -> tuple[dict[str, dict[str, Any]], list[np.ndarray]]:
        if not representative_tensors:
            raise ValueError("calibration requires at least one representative tensor")
        dimension = int(representative_tensors[0].vector.size)
        for tensor in representative_tensors:
            if tensor.vector.size != dimension:
                raise ValueError("all representative tensors must have the model-update dimension")
        calibration: dict[str, dict[str, Any]] = {}
        vectors = list(representative_tensors[: int(self.cfg.calibration_vectors)])
        for profile_id in self.profiles:
            mses: list[float] = []
            max_errors: list[float] = []
            residual_samples: list[np.ndarray] = []
            relative_errors: list[float] = []
            for tensor in vectors:
                vector = tensor.vector
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
        perturbation_bank, perturbation_rows = self._build_perturbation_bank(calibration)
        self._save_calibration(calibration, vectors, perturbation_rows)
        return calibration, perturbation_bank

    def _build_perturbation_bank(self, calibration: dict[str, dict[str, Any]]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        reference_profile = self.cfg.reference_profile_id
        if reference_profile not in calibration:
            raise ValueError(f"reference_profile_id {reference_profile!r} was not calibrated")
        residuals = calibration[reference_profile]["residual_samples"]
        stacked = np.vstack(residuals)
        rng = np.random.default_rng(int(getattr(self.cfg, "perturbation_seed", 20240618)))
        bank: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        for index in range(int(self.cfg.perturbation_count)):
            weights = rng.normal(0.0, 1.0, size=len(residuals))
            weights /= max(np.linalg.norm(weights), 1e-12)
            perturbation = np.tensordot(weights, stacked, axes=(0, 0)).astype(np.float64, copy=False)
            perturbation *= float(self.cfg.perturbation_scale)
            digest = hashlib.sha256(perturbation.tobytes()).hexdigest()
            bank.append(perturbation.copy())
            rows.append(
                {
                    "perturbation_id": index,
                    "source_profile_id": reference_profile,
                    "source_distribution": "protocol_aligned_reference_residual_distribution",
                    "l2_norm": float(np.linalg.norm(perturbation)),
                    "mean": float(np.mean(perturbation)),
                    "std": float(np.std(perturbation)),
                    "max_abs": float(np.max(np.abs(perturbation))),
                    "sha256": digest,
                }
            )
        return bank, rows

    def _save_calibration(self, calibration: dict[str, dict[str, Any]], tensors: Sequence[CalibrationTensor], perturbation_rows: list[dict[str, Any]]) -> None:
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        profile_rows = []
        residual_payload: dict[str, np.ndarray] = {}
        for profile_id, metrics in calibration.items():
            profile_rows.append(
                {
                    "profile_id": profile_id,
                    "sample_count": len(metrics["residual_samples"]),
                    "mse": metrics["mse"],
                    "max_abs_error": metrics["max_abs_error"],
                    "relative_l2_error": metrics["relative_l2_error"],
                    "residual_mean": metrics["residual_mean"],
                    "residual_std": metrics["residual_std"],
                    "residual_role": "reference_residual" if profile_id == self.cfg.reference_profile_id else "profile_calibration_residual",
                    "reference_profile_id": self.cfg.reference_profile_id,
                }
            )
            residual_payload[profile_id] = np.vstack(metrics["residual_samples"])
        with open(os.path.join(self.output_dir, "direct_profile_calibration.csv"), "w", newline="") as handle:
            fieldnames = ["profile_id", "sample_count", "mse", "max_abs_error", "relative_l2_error", "residual_mean", "residual_std", "residual_role", "reference_profile_id"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(profile_rows)
        with open(os.path.join(self.output_dir, "calibration_tensor_sources.csv"), "w", newline="") as handle:
            fieldnames = ["source", "scale", "l2_norm", "mean", "std", "max_abs"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{k: getattr(tensor, k) for k in fieldnames} for tensor in tensors])
        with open(os.path.join(self.output_dir, "perturbation_bank.csv"), "w", newline="") as handle:
            fieldnames = ["perturbation_id", "source_profile_id", "source_distribution", "l2_norm", "mean", "std", "max_abs", "sha256"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(perturbation_rows)
        np.savez(os.path.join(self.output_dir, "reference_residual_bank.npz"), **residual_payload)
