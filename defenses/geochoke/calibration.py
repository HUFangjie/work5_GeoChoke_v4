from __future__ import annotations

import csv
import hashlib
import os
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from defenses.geochoke.calibration_provider import CalibrationTensor


class ProfileCalibrator:
    """Offline profile calibration using real CKKS encrypt/decrypt residuals."""

    def __init__(self, crypto_backend: Any, decrypt_aggregate_fn: Any, profiles: Mapping[str, Dict[str, Any]], cfg: Any, output_dir: str | None = None) -> None:
        self.crypto_backend = crypto_backend
        self.decrypt_aggregate = decrypt_aggregate_fn
        self.profiles = dict(profiles)
        self.cfg = cfg
        self.output_dir = output_dir
        self.profile_rows: list[dict[str, Any]] = []
        self.profile_range_warning = False

    def calibrate(self, representative_tensors: Sequence[CalibrationTensor]) -> tuple[dict[str, dict[str, Any]], list[np.ndarray]]:
        if not representative_tensors:
            raise ValueError("calibration requires at least one representative tensor")
        dimension = int(representative_tensors[0].vector.size)
        if any(tensor.vector.size != dimension for tensor in representative_tensors):
            raise ValueError("all representative tensors must have the model-update dimension")
        calibration: dict[str, dict[str, Any]] = {}
        vectors = list(representative_tensors[: int(self.cfg.calibration_vectors)])
        residual_payload: dict[str, np.ndarray] = {}
        for profile_id, profile_cfg in self.profiles.items():
            row_base = {
                "profile_id": profile_id,
                "global_scale_bits": profile_cfg.get("global_scale_bits"),
                "coeff_mod_bit_sizes": profile_cfg.get("coeff_mod_bit_sizes"),
            }
            try:
                mses: list[float] = []
                max_abs: list[float] = []
                l2_norms: list[float] = []
                l2_ratios: list[float] = []
                residual_samples: list[np.ndarray] = []
                for tensor in vectors:
                    vector = tensor.vector
                    encrypted = self.crypto_backend.encrypt_update(vector, profile_id)
                    decrypted = self.decrypt_aggregate(encrypted, profile_id)
                    residual = decrypted - vector
                    residual_norm = float(np.linalg.norm(residual))
                    vector_norm = float(np.linalg.norm(vector))
                    mses.append(float(np.mean(residual * residual)))
                    max_abs.append(float(np.max(np.abs(residual))))
                    l2_norms.append(residual_norm)
                    l2_ratios.append(residual_norm / max(vector_norm, 1e-12))
                    residual_samples.append(residual.astype(np.float64, copy=True))
                stacked = np.vstack(residual_samples)
                metrics = {
                    "mse": float(np.mean(mses)),
                    "calibrated_mse": float(np.mean(mses)),
                    "calibrated_rmse": float(np.sqrt(np.mean(mses))),
                    "max_abs_error": float(np.max(max_abs)),
                    "residual_max_abs_mean": float(np.mean(max_abs)),
                    "relative_l2_error": float(np.mean(l2_ratios)),
                    "residual_l2_ratio_mean": float(np.mean(l2_ratios)),
                    "residual_l2_norm_mean": float(np.mean(l2_norms)),
                    "residual_mean": float(np.mean(stacked)),
                    "residual_std": float(np.std(stacked)),
                    "residual_samples": residual_samples,
                    "valid_profile": True,
                    "skip_reason": "",
                    "global_scale_bits": profile_cfg.get("global_scale_bits"),
                    "coeff_mod_bit_sizes": profile_cfg.get("coeff_mod_bit_sizes"),
                }
                calibration[profile_id] = metrics
                residual_payload[profile_id] = stacked
                self.profile_rows.append({**row_base, **{k: metrics[k] for k in ["calibrated_mse", "calibrated_rmse", "residual_l2_norm_mean", "residual_l2_ratio_mean", "residual_max_abs_mean", "valid_profile", "skip_reason"]}})
            except Exception as exc:
                self.profile_rows.append({**row_base, "calibrated_mse": None, "calibrated_rmse": None, "residual_l2_norm_mean": None, "residual_l2_ratio_mean": None, "residual_max_abs_mean": None, "valid_profile": False, "skip_reason": str(exc)})
        calibration = dict(sorted(calibration.items(), key=lambda item: float(item[1]["calibrated_mse"])))
        if self.cfg.reference_profile_id not in calibration:
            raise ValueError(f"reference_profile_id {self.cfg.reference_profile_id!r} is not a valid calibrated profile")
        perturbation_bank, perturbation_rows = self._build_perturbation_bank(calibration)
        self._check_profile_range(calibration)
        self._save_calibration(calibration, vectors, perturbation_rows, residual_payload)
        return calibration, perturbation_bank

    def _check_profile_range(self, calibration: dict[str, dict[str, Any]]) -> None:
        mses = [float(metrics["calibrated_mse"]) for metrics in calibration.values()]
        for left, right in zip(mses, mses[1:]):
            if abs(right - left) <= max(1e-18, abs(left) * 0.05):
                self.profile_range_warning = True
        ratios = [float(metrics["residual_l2_ratio_mean"]) for metrics in calibration.values()]
        if ratios and max(ratios) < 1e-3:
            self.profile_range_warning = True

    def _build_perturbation_bank(self, calibration: dict[str, dict[str, Any]]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        reference_profile = self.cfg.reference_profile_id
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
            rows.append({"perturbation_id": index, "source_profile_id": reference_profile, "source_distribution": "protocol_aligned_reference_residual_distribution", "l2_norm": float(np.linalg.norm(perturbation)), "mean": float(np.mean(perturbation)), "std": float(np.std(perturbation)), "max_abs": float(np.max(np.abs(perturbation))), "sha256": digest})
        return bank, rows

    def _save_calibration(self, calibration: dict[str, dict[str, Any]], tensors: Sequence[CalibrationTensor], perturbation_rows: list[dict[str, Any]], residual_payload: dict[str, np.ndarray]) -> None:
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        root_output_dir = os.path.dirname(self.output_dir)
        sorted_rows = sorted(self.profile_rows, key=lambda row: (row["calibrated_mse"] is None, float(row["calibrated_mse"] or 0.0)))
        fieldnames = ["profile_id", "global_scale_bits", "coeff_mod_bit_sizes", "calibrated_mse", "calibrated_rmse", "residual_l2_norm_mean", "residual_l2_ratio_mean", "residual_max_abs_mean", "valid_profile", "skip_reason"]
        for path in [os.path.join(self.output_dir, "direct_profile_calibration.csv"), os.path.join(root_output_dir, "profile_calibration.csv")]:
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sorted_rows)
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
