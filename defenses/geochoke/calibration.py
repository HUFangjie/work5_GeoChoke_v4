from __future__ import annotations

import csv
import os
from typing import Any, Dict, Mapping

import numpy as np


class ProfileCalibrator:
    """Offline profile calibration from real CKKS aggregation-pipeline residuals."""

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
        if self.cfg.reference_profile_id not in self.profiles:
            raise ValueError(f"unknown reference_profile_id: {self.cfg.reference_profile_id}")
        dimension = int(representative_vectors[0].size)
        for vector in representative_vectors:
            if vector.size != dimension:
                raise ValueError("all representative vectors must have the model-update dimension")
        calibration: dict[str, dict[str, Any]] = {}
        rng = np.random.default_rng(20240618)
        sample_count = max(1, int(self.cfg.calibration_vectors))
        repetitions = max(1, int(self.cfg.calibration_repetitions))
        for profile_id in self.profiles:
            mses: list[float] = []
            max_errors: list[float] = []
            residual_samples: list[np.ndarray] = []
            relative_errors: list[float] = []
            norm_ratios: list[float] = []
            for sample_idx in range(sample_count):
                base = representative_vectors[sample_idx % len(representative_vectors)].astype(np.float64, copy=False)
                for _rep in range(repetitions):
                    client_vectors = self._client_vectors(base, rng)
                    weights = rng.dirichlet(np.ones(len(client_vectors))).astype(np.float64)
                    encrypted = [self.crypto_backend.encrypt_update(vector, profile_id) for vector in client_vectors]
                    weighted = [self.crypto_backend.multiply_plain(ciphertext, float(weight)) for ciphertext, weight in zip(encrypted, weights)]
                    aggregate_ciphertext = self.crypto_backend.add_ciphertexts(weighted)
                    decrypted = self.decrypt_aggregate(aggregate_ciphertext, profile_id)
                    plaintext_reference = np.zeros_like(base, dtype=np.float64)
                    for vector, weight in zip(client_vectors, weights):
                        plaintext_reference += float(weight) * vector
                    residual = decrypted - plaintext_reference
                    if not np.all(np.isfinite(residual)):
                        raise ValueError(f"non-finite CKKS calibration residual for profile {profile_id}")
                    reference_norm = max(float(np.linalg.norm(plaintext_reference)), 1e-12)
                    decrypted_norm = float(np.linalg.norm(decrypted))
                    mses.append(float(np.mean(residual * residual)))
                    max_errors.append(float(np.max(np.abs(residual))))
                    residual_samples.append(residual.astype(np.float64, copy=True))
                    relative_errors.append(float(np.linalg.norm(residual) / reference_norm))
                    norm_ratios.append(decrypted_norm / reference_norm)
            stacked_residuals = np.vstack(residual_samples)
            calibration[profile_id] = {
                "mse": float(np.mean(mses)),
                "max_abs_error": float(np.max(max_errors)),
                "relative_l2_error": float(np.mean(relative_errors)),
                "norm_ratio": float(np.mean(norm_ratios)),
                "residual_mean": float(np.mean(stacked_residuals)),
                "residual_std": float(np.std(stacked_residuals)),
                "residual_samples": residual_samples,
            }
        residual_source = calibration[self.cfg.reference_profile_id]["residual_samples"]
        perturbation_bank: list[np.ndarray] = []
        index = 0
        while len(perturbation_bank) < self.cfg.perturbation_count:
            perturbation_bank.append(residual_source[index % len(residual_source)].copy() * self.cfg.perturbation_scale)
            index += 1
        self._save_calibration(calibration)
        return calibration, perturbation_bank

    @staticmethod
    def _client_vectors(base: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
        return [
            base,
            -0.5 * base + rng.normal(0.0, 0.002, size=base.shape),
            0.25 * base + rng.normal(0.0, 0.002, size=base.shape),
        ]

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
                    "norm_ratio": metrics["norm_ratio"],
                    "residual_mean": metrics["residual_mean"],
                    "residual_std": metrics["residual_std"],
                }
            )
            residual_payload[profile_id] = np.vstack(metrics["residual_samples"])
        with open(os.path.join(self.output_dir, "direct_profile_calibration.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["profile_id", "sample_count", "mse", "max_abs_error", "relative_l2_error", "norm_ratio", "residual_mean", "residual_std"])
            writer.writeheader()
            writer.writerows(rows)
        np.savez(os.path.join(self.output_dir, "reference_residual_bank.npz"), **residual_payload)
