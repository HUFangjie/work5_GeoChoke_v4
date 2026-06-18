from __future__ import annotations

import csv
import os
from typing import Any, Iterable, Sequence

import numpy as np

from crypto.validation_result import PipelineValidationResult


class AggregationPipelineValidator:
    """Validates the real Enc -> weight -> add -> Dec aggregation pipeline."""

    def __init__(
        self,
        crypto_backend: Any,
        decrypt_for_validation: Any,
        rtol: float,
        atol: float,
        norm_ratio_tolerance: float,
    ) -> None:
        self.crypto_backend = crypto_backend
        self.decrypt_for_validation = decrypt_for_validation
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.norm_ratio_tolerance = float(norm_ratio_tolerance)

    def validate_profile(
        self,
        profile_id: str,
        vectors: Sequence[np.ndarray],
        weights: Sequence[float],
    ) -> PipelineValidationResult:
        if not vectors:
            raise ValueError("pipeline validation requires at least one vector")
        if len(vectors) != len(weights):
            raise ValueError("pipeline validation vectors and weights length mismatch")
        arrays = [np.asarray(vector, dtype=np.float64) for vector in vectors]
        dimension = arrays[0].size
        if any(array.size != dimension for array in arrays):
            raise ValueError("all validation vectors must have the same dimension")
        weight_array = np.asarray(weights, dtype=np.float64)
        if not np.all(np.isfinite(weight_array)):
            raise ValueError("pipeline validation weights must be finite")
        encrypted_updates = [self.crypto_backend.encrypt_update(vector, profile_id) for vector in arrays]
        weighted_updates = [
            self.crypto_backend.multiply_plain(ciphertext, float(weight))
            for ciphertext, weight in zip(encrypted_updates, weight_array)
        ]
        aggregate_ciphertext = self.crypto_backend.add_ciphertexts(weighted_updates)
        decrypted = self.decrypt_for_validation(aggregate_ciphertext, profile_id)
        plaintext_reference = np.zeros(dimension, dtype=np.float64)
        for vector, weight in zip(arrays, weight_array):
            plaintext_reference += float(weight) * vector
        if decrypted.shape != plaintext_reference.shape:
            raise ValueError("pipeline validation decrypted dimension mismatch")
        if not np.all(np.isfinite(decrypted)):
            raise ValueError("pipeline validation decrypted values contain NaN or Inf")
        difference = decrypted - plaintext_reference
        mse = float(np.mean(difference * difference))
        max_abs_error = float(np.max(np.abs(difference)))
        reference_norm = float(np.linalg.norm(plaintext_reference))
        error_norm = float(np.linalg.norm(difference))
        decrypted_norm = float(np.linalg.norm(decrypted))
        relative_l2_error = error_norm / max(reference_norm, 1e-12)
        norm_ratio = decrypted_norm / max(reference_norm, 1e-12)
        allclose = bool(np.allclose(decrypted, plaintext_reference, rtol=self.rtol, atol=self.atol))
        ratio_ok = abs(norm_ratio - 1.0) < self.norm_ratio_tolerance
        return PipelineValidationResult(
            profile_id=profile_id,
            number_of_vectors=len(arrays),
            weights=[float(weight) for weight in weight_array],
            mse=mse,
            max_abs_error=max_abs_error,
            relative_l2_error=float(relative_l2_error),
            norm_ratio=float(norm_ratio),
            passed=allclose and ratio_ok,
        )

    def validate_profiles(
        self,
        profile_ids: Iterable[str],
        vectors: Sequence[np.ndarray],
        weights: Sequence[float],
        output_dir: str | None = None,
    ) -> list[PipelineValidationResult]:
        results = [self.validate_profile(profile_id, vectors, weights) for profile_id in profile_ids]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "aggregation_pipeline_validation.csv"), "w", newline="") as handle:
                fieldnames = [
                    "profile_id",
                    "number_of_vectors",
                    "weights",
                    "aggregate_mse",
                    "aggregate_max_abs_error",
                    "relative_l2_error",
                    "norm_ratio",
                    "passed",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([result.as_row() for result in results])
        failed = [result for result in results if not result.passed]
        if failed:
            details = ", ".join(f"{r.profile_id}: ratio={r.norm_ratio:.6g}, rel={r.relative_l2_error:.6g}" for r in failed)
            raise ValueError(f"CKKS aggregation pipeline validation failed: {details}")
        return results
