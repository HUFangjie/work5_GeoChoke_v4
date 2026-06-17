from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from crypto.update_codec import ModelUpdateCodec
from defenses.base import DefenseStrategy
from defenses.geochoke.calibration import ProfileCalibrator
from defenses.geochoke.cfi_estimator import CFIEstimator
from defenses.geochoke.controller import GeoChokeController


class GeoChokeDefense(DefenseStrategy):
    def __init__(self, cfg: Any, profiles: Mapping[str, dict[str, Any]], device: str = "cpu") -> None:
        self.cfg = cfg
        self.profiles = dict(profiles)
        self.device = device
        self.next_profile = cfg.initial_profile_id
        self.history: list[dict[str, Any]] = []

    def initialize(self, model: Any, crypto_backend: Any, proxy_loader: Any, decrypt_aggregate_fn: Any = None) -> None:
        if decrypt_aggregate_fn is None:
            raise ValueError("GeoChoke calibration requires aggregate-only decryption callable")
        self.codec = ModelUpdateCodec(model)
        representative_vectors = self._representative_vectors(model)
        self.calibration, self.perturbation_bank = ProfileCalibrator(
            crypto_backend,
            decrypt_aggregate_fn,
            self.profiles,
            self.cfg,
        ).calibrate(representative_vectors)
        self.estimator = CFIEstimator(self.codec, proxy_loader, self.perturbation_bank, self.device)
        self.controller = GeoChokeController(self.cfg, self.calibration)

    def get_profile_for_round(self, round_id: int) -> str:
        if round_id < self.cfg.warmup_rounds:
            return self.cfg.initial_profile_id
        return self.next_profile

    def after_aggregate(self, previous_model: Any, candidate_model: Any, current_profile_id: str, round_id: int) -> dict[str, Any]:
        previous_cfi = self.estimator.estimate(previous_model)
        candidate_cfi = self.estimator.estimate(candidate_model)
        next_profile, metrics = self.controller.select(previous_cfi, candidate_cfi, current_profile_id)
        if round_id >= self.cfg.warmup_rounds:
            self.next_profile = next_profile
        self.history.append({"round": round_id, "current_profile": current_profile_id, **metrics})
        return metrics

    def _representative_vectors(self, model: Any) -> list[np.ndarray]:
        initial = self.codec.flatten_state_dict(model.state_dict())
        rng = np.random.default_rng(20240617)
        vectors = [initial * 0.0]
        for _ in range(max(1, self.cfg.calibration_vectors - 1)):
            vectors.append(rng.normal(loc=0.0, scale=0.01, size=self.codec.total_dimension).astype(np.float64))
        return vectors
