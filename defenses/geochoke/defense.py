from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from crypto.update_codec import ModelUpdateCodec
from defenses.base import DefenseStrategy
from defenses.geochoke.calibration import ProfileCalibrator
from defenses.geochoke.cfi_estimator import CFIEstimator
from defenses.geochoke.controller import GeoChokeController


class GeoChokeDefense(DefenseStrategy):
    def __init__(self, cfg: Any, profiles: Mapping[str, dict[str, Any]], device: str = "cpu", output_dir: str | None = None) -> None:
        self.cfg = cfg
        self.profiles = dict(profiles)
        self.device = device
        self.output_dir = output_dir
        self.next_profile = cfg.initial_profile_id
        self.history: list[dict[str, Any]] = []
        self.benign_cfi_history: list[float] = []

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
            output_dir=self.output_dir,
        ).calibrate(representative_vectors)
        self.estimator = CFIEstimator(self.codec, proxy_loader, self.perturbation_bank, self.device)
        self.controller = GeoChokeController(self.cfg, self.calibration)

    def get_profile_for_round(self, round_id: int) -> str:
        if round_id < self.cfg.warmup_rounds:
            return self.cfg.initial_profile_id
        return self.next_profile

    def after_aggregate(
        self,
        previous_model: Any,
        candidate_model: Any,
        current_profile_id: str,
        round_id: int,
        aggregate_update: np.ndarray | None = None,
        server_lr: float = 1.0,
        previous_state: dict[str, Any] | None = None,
        model_factory: Any | None = None,
        candidate_scales: list[float] | None = None,
    ) -> dict[str, Any]:
        previous_cfi = self.estimator.estimate(previous_model)
        if round_id < self.cfg.warmup_rounds:
            self.benign_cfi_history.append(previous_cfi)
        scales = candidate_scales or [1.0]
        gate_candidates = self._score_candidates(
            previous_model,
            candidate_model,
            aggregate_update,
            server_lr,
            previous_state,
            model_factory,
            scales,
            previous_cfi,
        )
        selected = next((item for item in gate_candidates if item["risk"] <= self.cfg.gate_risk_threshold), gate_candidates[-1])
        if selected["scale"] > 0.0 and selected["risk"] > self.cfg.gate_risk_threshold:
            selected = gate_candidates[-1]
        candidate_cfi = float(selected["candidate_cfi"])
        next_profile, metrics = self.controller.select(previous_cfi, candidate_cfi, current_profile_id, cfi_scale=self._cfi_scale())
        if round_id >= self.cfg.warmup_rounds:
            self.next_profile = next_profile
        decision = {
            **metrics,
            "accepted_update_scale": float(selected["scale"]),
            "candidate_functional_drift": float(selected["functional_drift"]),
            "candidate_cfi": candidate_cfi,
            "cfi_injection": float(selected["cfi_injection"]),
            "candidate_rejected": bool(float(selected["scale"]) == 0.0),
            "gate_risk": float(selected["risk"]),
            "gate_risk_threshold": float(self.cfg.gate_risk_threshold),
            "current_profile_id": current_profile_id,
            "next_profile_id": next_profile,
        }
        self.history.append({"round": round_id, "current_profile": current_profile_id, **decision})
        return decision

    def _score_candidates(
        self,
        previous_model: Any,
        full_candidate_model: Any,
        aggregate_update: np.ndarray | None,
        server_lr: float,
        previous_state: dict[str, Any] | None,
        model_factory: Any | None,
        candidate_scales: list[float],
        previous_cfi: float,
    ) -> list[dict[str, float]]:
        ordered_scales = sorted({float(scale) for scale in candidate_scales}, reverse=True)
        if 0.0 not in ordered_scales:
            ordered_scales.append(0.0)
        scored: list[dict[str, float]] = []
        for scale in ordered_scales:
            if scale == 1.0 or aggregate_update is None or previous_state is None or model_factory is None:
                candidate = full_candidate_model if scale == 1.0 else previous_model
            elif scale == 0.0:
                candidate = previous_model
            else:
                state = self.codec.apply_update_to_state_dict(previous_state, aggregate_update, step_size=float(server_lr) * scale)
                candidate = model_factory()
                candidate.load_state_dict(state)
            functional_drift = self.estimator.functional_drift(previous_model, candidate)
            candidate_cfi = self.estimator.estimate(candidate)
            cfi_injection = max(0.0, candidate_cfi - previous_cfi)
            risk = functional_drift + float(self.cfg.gate_kappa) * cfi_injection
            scored.append(
                {
                    "scale": float(scale),
                    "functional_drift": float(functional_drift),
                    "candidate_cfi": float(candidate_cfi),
                    "cfi_injection": float(cfi_injection),
                    "risk": float(risk),
                }
            )
        return scored

    def _cfi_scale(self) -> float:
        values = np.asarray(self.benign_cfi_history, dtype=np.float64)
        if values.size == 0:
            return 1.0
        return max(float(np.percentile(values, 95)), 1e-12)

    def _representative_vectors(self, model: Any) -> list[np.ndarray]:
        initial = self.codec.flatten_state_dict(model.state_dict())
        rng = np.random.default_rng(20240617)
        vectors = [initial * 0.0]
        for _ in range(max(1, self.cfg.calibration_vectors - 1)):
            vectors.append(rng.normal(loc=0.0, scale=0.01, size=self.codec.total_dimension).astype(np.float64))
        return vectors
