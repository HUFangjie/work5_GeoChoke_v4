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
        self.benign_metric_history: list[dict[str, float]] = []
        self.cusum_score = 0.0
        self.consecutive_safe_rounds = 0
        self.recovery_rounds_remaining = 0
        self.last_safe_state: dict[str, Any] | None = None
        self._pending_rollback_state: dict[str, Any] | None = None
        self._current_max_candidate_scale = 1.0
        self._warmup_update_rounds: list[list[np.ndarray]] = []
        self._warmup_calibration_finalized = False

    def initialize(self, model: Any, crypto_backend: Any, proxy_loader: Any, decrypt_aggregate_fn: Any = None) -> None:
        if decrypt_aggregate_fn is None:
            raise ValueError("GeoChoke calibration requires aggregate-only decryption callable")
        self.crypto_backend = crypto_backend
        self.decrypt_aggregate_fn = decrypt_aggregate_fn
        self.proxy_loader = proxy_loader
        self.codec = ModelUpdateCodec(model)
        self.last_safe_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
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


    def record_warmup_client_updates(self, clean_updates: list[np.ndarray], round_id: int) -> None:
        if round_id < self.cfg.warmup_rounds and clean_updates:
            self._warmup_update_rounds.append([np.asarray(update, dtype=np.float64).copy() for update in clean_updates])

    def finalize_warmup_calibration(self) -> None:
        if self._warmup_calibration_finalized or not self._warmup_update_rounds:
            return
        calibration: dict[str, dict[str, Any]] = {}
        for profile_id in self.profiles:
            residual_samples: list[np.ndarray] = []
            mses: list[float] = []
            max_errors: list[float] = []
            relative_errors: list[float] = []
            norm_ratios: list[float] = []
            for round_updates in self._warmup_update_rounds:
                weights = np.full(len(round_updates), 1.0 / max(1, len(round_updates)), dtype=np.float64)
                encrypted = [self.crypto_backend.encrypt_update(update, profile_id) for update in round_updates]
                weighted = [self.crypto_backend.multiply_plain(ciphertext, float(weight)) for ciphertext, weight in zip(encrypted, weights)]
                aggregate_ciphertext = self.crypto_backend.add_ciphertexts(weighted)
                decrypted = self.decrypt_aggregate_fn(aggregate_ciphertext, profile_id)
                plaintext = np.zeros_like(round_updates[0], dtype=np.float64)
                for update, weight in zip(round_updates, weights):
                    plaintext += float(weight) * update
                residual = decrypted - plaintext
                if not np.all(np.isfinite(residual)):
                    raise ValueError(f"non-finite warmup CKKS residual for profile {profile_id}")
                residual_samples.append(residual.astype(np.float64, copy=True))
                mses.append(float(np.mean(residual * residual)))
                max_errors.append(float(np.max(np.abs(residual))))
                reference_norm = max(float(np.linalg.norm(plaintext)), 1e-12)
                relative_errors.append(float(np.linalg.norm(residual) / reference_norm))
                norm_ratios.append(float(np.linalg.norm(decrypted)) / reference_norm)
            stacked = np.vstack(residual_samples)
            calibration[profile_id] = {
                "mse": float(np.mean(mses)),
                "max_abs_error": float(np.max(max_errors)),
                "relative_l2_error": float(np.mean(relative_errors)),
                "norm_ratio": float(np.mean(norm_ratios)),
                "residual_mean": float(np.mean(stacked)),
                "residual_std": float(np.std(stacked)),
                "residual_samples": residual_samples,
            }
        reference_samples = calibration[self.cfg.reference_profile_id]["residual_samples"]
        self.calibration = calibration
        self.perturbation_bank = [
            reference_samples[index % len(reference_samples)].copy() * self.cfg.perturbation_scale
            for index in range(self.cfg.perturbation_count)
        ]
        self.estimator = CFIEstimator(self.codec, self.proxy_loader, self.perturbation_bank, self.device)
        self.controller = GeoChokeController(self.cfg, self.calibration)
        self._warmup_calibration_finalized = True

    def get_profile_for_round(self, round_id: int) -> str:
        if round_id < self.cfg.warmup_rounds:
            return self.cfg.initial_profile_id
        return self.next_profile

    def get_pending_rollback_state(self) -> dict[str, Any] | None:
        state = self._pending_rollback_state
        self._pending_rollback_state = None
        if state is None:
            return None
        return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}

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
        scales = self._effective_candidate_scales(candidate_scales or [1.0])
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
        raw_metrics = self._raw_metrics(gate_candidates)
        thresholds = self._thresholds(raw_metrics)
        gate_candidates = [self._with_normalized_risk(item, thresholds) for item in gate_candidates]
        raw_metrics = self._raw_metrics(gate_candidates)

        if round_id < self.cfg.warmup_rounds:
            self.benign_metric_history.append(
                {
                    "functional_drift": raw_metrics["functional_drift"],
                    "candidate_cfi": raw_metrics["candidate_cfi"],
                    "cfi_injection": raw_metrics["cfi_injection"],
                }
            )

        raw_candidate_risk = float(raw_metrics["normalized_risk"])
        self.cusum_score = self._update_cusum(raw_candidate_risk)
        cumulative_anomaly = bool(self.cfg.enable_cumulative_gate and self.cusum_score > float(self.cfg.cusum_threshold))
        selected = self._select_candidate(gate_candidates)
        candidate_rejected = bool(float(selected["scale"]) == 0.0)
        rollback_triggered = False
        if cumulative_anomaly and self.last_safe_state is not None:
            selected = self._zero_scale_candidate(gate_candidates)
            candidate_rejected = True
            rollback_triggered = True
            self._pending_rollback_state = {name: tensor.detach().cpu().clone() for name, tensor in self.last_safe_state.items()}
            self.recovery_rounds_remaining = int(self.cfg.rollback_recovery_rounds)

        accepted_metrics = selected
        accepted_safe = bool(accepted_metrics["normalized_risk"] <= 1.0 and not cumulative_anomaly and not rollback_triggered)
        if accepted_safe:
            self.consecutive_safe_rounds += 1
            if selected.get("state") is not None:
                self.last_safe_state = {name: tensor.detach().cpu().clone() for name, tensor in selected["state"].items()}
            elif previous_state is not None:
                self.last_safe_state = {name: tensor.detach().cpu().clone() for name, tensor in previous_state.items()}
        else:
            self.consecutive_safe_rounds = 0
        if self.recovery_rounds_remaining > 0 and not rollback_triggered:
            self.recovery_rounds_remaining -= 1

        next_profile, controller_metrics = self.controller.select(
            previous_cfi,
            float(raw_metrics["candidate_cfi"]),
            current_profile_id,
            cfi_scale=self._cfi_scale(),
            raw_candidate_risk=raw_candidate_risk,
            cusum_score=self.cusum_score,
            candidate_rejected=candidate_rejected,
            rollback_triggered=rollback_triggered,
            consecutive_safe_rounds=self.consecutive_safe_rounds,
        )
        if round_id >= self.cfg.warmup_rounds:
            self.next_profile = next_profile
        decision = {
            **controller_metrics,
            "raw_functional_drift": float(raw_metrics["functional_drift"]),
            "raw_candidate_cfi": float(raw_metrics["candidate_cfi"]),
            "raw_cfi_injection": float(raw_metrics["cfi_injection"]),
            "normalized_drift_risk": float(raw_metrics["drift_risk"]),
            "normalized_cfi_risk": float(raw_metrics["cfi_risk"]),
            "normalized_injection_risk": float(raw_metrics["injection_risk"]),
            "raw_candidate_risk": raw_candidate_risk,
            "accepted_candidate_risk": float(accepted_metrics["normalized_risk"]),
            "accepted_update_scale": float(accepted_metrics["scale"]),
            "candidate_functional_drift": float(accepted_metrics["functional_drift"]),
            "candidate_cfi": float(accepted_metrics["candidate_cfi"]),
            "cfi_injection": float(accepted_metrics["cfi_injection"]),
            "candidate_rejected": candidate_rejected,
            "rollback_triggered": rollback_triggered,
            "cusum_score": float(self.cusum_score),
            "cumulative_anomaly": cumulative_anomaly,
            "consecutive_safe_rounds": int(self.consecutive_safe_rounds),
            "recovery_rounds_remaining": int(self.recovery_rounds_remaining),
            "drift_threshold": float(thresholds["drift"]),
            "cfi_threshold": float(thresholds["cfi"]),
            "injection_threshold": float(thresholds["injection"]),
            "current_profile_id": current_profile_id,
            "next_profile_id": next_profile,
        }
        self.history.append({"round": round_id, "current_profile": current_profile_id, **decision})
        return decision

    def _effective_candidate_scales(self, candidate_scales: list[float]) -> list[float]:
        self._current_max_candidate_scale = 1.0
        if self.recovery_rounds_remaining > 0:
            self._current_max_candidate_scale = min(self._current_max_candidate_scale, float(self.cfg.recovery_max_candidate_scale))
        scales = sorted({float(scale) for scale in candidate_scales}, reverse=True)
        if 1.0 not in scales:
            scales.insert(0, 1.0)
        if 0.0 not in scales:
            scales.append(0.0)
        return scales

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
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for scale in candidate_scales:
            state = None
            if aggregate_update is not None and previous_state is not None:
                state = self.codec.apply_update_to_state_dict(previous_state, aggregate_update, step_size=float(server_lr) * scale)
                candidate = model_factory() if model_factory is not None else full_candidate_model
                candidate.load_state_dict(state)
            else:
                candidate = full_candidate_model if scale == 1.0 else previous_model
            functional_drift = self.estimator.functional_drift(previous_model, candidate)
            candidate_cfi = self.estimator.estimate(candidate)
            cfi_injection = max(0.0, candidate_cfi - previous_cfi)
            scored.append(
                {
                    "scale": float(scale),
                    "functional_drift": float(functional_drift),
                    "candidate_cfi": float(candidate_cfi),
                    "cfi_injection": float(cfi_injection),
                    "state": state,
                }
            )
        return scored

    def _thresholds(self, raw_metrics: dict[str, float]) -> dict[str, float]:
        if not self.benign_metric_history:
            return {
                "drift": max(float(raw_metrics["functional_drift"]), float(self.cfg.min_threshold)),
                "cfi": max(float(raw_metrics["candidate_cfi"]), float(self.cfg.min_threshold)),
                "injection": max(float(raw_metrics["cfi_injection"]), float(self.cfg.min_threshold)),
            }
        window = self.benign_metric_history[-max(1, int(self.cfg.warmup_stats_window)) :]
        return {
            "drift": self._robust_threshold([item["functional_drift"] for item in window]),
            "cfi": self._robust_threshold([item["candidate_cfi"] for item in window]),
            "injection": self._robust_threshold([item["cfi_injection"] for item in window]),
        }

    def _robust_threshold(self, values: list[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return float(self.cfg.min_threshold)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        median_mad = median + 3.0 * 1.4826 * mad
        quantile = float(np.quantile(array, float(self.cfg.threshold_quantile)))
        return max(median_mad, quantile, float(self.cfg.min_threshold))

    def _with_normalized_risk(self, item: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
        drift_risk = float(item["functional_drift"]) / max(float(thresholds["drift"]), float(self.cfg.min_threshold))
        cfi_risk = float(item["candidate_cfi"]) / max(float(thresholds["cfi"]), float(self.cfg.min_threshold))
        injection_risk = float(item["cfi_injection"]) / max(float(thresholds["injection"]), float(self.cfg.min_threshold))
        normalized_risk = max(
            drift_risk,
            float(self.cfg.gate_kappa_cfi) * cfi_risk,
            float(self.cfg.gate_kappa_injection) * injection_risk,
        )
        return {
            **item,
            "drift_risk": drift_risk,
            "cfi_risk": cfi_risk,
            "injection_risk": injection_risk,
            "normalized_risk": float(normalized_risk),
            "selectable": float(item["scale"]) <= self._current_max_candidate_scale,
        }

    @staticmethod
    def _raw_metrics(candidates: list[dict[str, Any]]) -> dict[str, float]:
        raw = next((item for item in candidates if float(item["scale"]) == 1.0), candidates[0])
        return raw

    @staticmethod
    def _select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        safe = [item for item in candidates if item.get("selectable", True) and float(item["normalized_risk"]) <= 1.0]
        if safe:
            return max(safe, key=lambda item: float(item["scale"]))
        selectable = [item for item in candidates if item.get("selectable", True)]
        return min(selectable or candidates, key=lambda item: abs(float(item["scale"])))

    @staticmethod
    def _zero_scale_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return min(candidates, key=lambda item: abs(float(item["scale"])))

    def _update_cusum(self, raw_candidate_risk: float) -> float:
        previous = self.cusum_score
        if raw_candidate_risk <= 1.0:
            previous = max(0.0, previous - float(self.cfg.cusum_decay))
        return max(0.0, previous + raw_candidate_risk - 1.0)

    def _cfi_scale(self) -> float:
        values = np.asarray([item["candidate_cfi"] for item in self.benign_metric_history], dtype=np.float64)
        if values.size == 0:
            return 1.0
        return max(float(np.percentile(values, 95)), float(self.cfg.min_threshold))

    def _representative_vectors(self, model: Any) -> list[np.ndarray]:
        initial = self.codec.flatten_state_dict(model.state_dict())
        rng = np.random.default_rng(20240617)
        vectors = [initial * 0.0]
        for _ in range(max(1, self.cfg.calibration_vectors - 1)):
            vectors.append(rng.normal(loc=0.0, scale=0.01, size=self.codec.total_dimension).astype(np.float64))
        return vectors
