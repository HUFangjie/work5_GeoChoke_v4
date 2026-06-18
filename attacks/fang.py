from __future__ import annotations

from typing import Any

import numpy as np

from attacks.base import AttackStrategy


class FangMeanAttack(AttackStrategy):
    """Mean-aggregation attack wrapper.

    With ``oracle_mean_replacement=True`` this runs the explicit oracle pressure
    test that uses clean aggregate information. Without oracle access it is an
    explicitly named sign-flip-scaled fallback and does not claim to be the
    original Fang robust-aggregator attack.
    """

    def __init__(
        self,
        aggregation: str = "weighted_mean",
        max_norm: float = 10.0,
        search_steps: int = 10,
        whitebox: bool = False,
        target_scale: float = 3.0,
        oracle_mean_replacement: bool = False,
    ) -> None:
        if aggregation != "weighted_mean":
            raise ValueError("FangMeanAttack is only compatible with weighted_mean aggregation")
        self.max_norm = float(max_norm)
        self.search_steps = int(search_steps)
        self.whitebox = bool(whitebox)
        self.target_scale = float(target_scale)
        self.oracle_mean_replacement = bool(oracle_mean_replacement)
        self.effective_attack_name = "oracle_mean_replacement" if self.oracle_mean_replacement else "sign_flip_scaled"

    def craft_update(self, client_id: int, clean_update: np.ndarray, global_model: Any, attacker_context: dict[str, Any]) -> np.ndarray:
        clean_update = np.asarray(clean_update, dtype=np.float64)
        observations = self._observations(clean_update, attacker_context)
        clean_norms = np.linalg.norm(observations, axis=1)
        base_norm = float(np.median(clean_norms)) if clean_norms.size else float(np.linalg.norm(clean_update))
        max_update_norm = max(base_norm, np.linalg.norm(clean_update), 1e-12) * self.max_norm
        if self.oracle_mean_replacement or attacker_context.get("oracle_mean_replacement", False):
            candidate = self._whitebox_replacement(clean_update, attacker_context, max_update_norm)
        else:
            candidate = self._blackbox_mean_deviation(clean_update, observations, attacker_context, max_update_norm)
        return self._project(candidate, max_update_norm).astype(np.float64, copy=False)

    def _whitebox_replacement(self, clean_update: np.ndarray, attacker_context: dict[str, Any], max_update_norm: float) -> np.ndarray:
        clean_aggregate = np.asarray(attacker_context.get("clean_aggregate", clean_update), dtype=np.float64)
        benign_weighted_sum = np.asarray(attacker_context.get("benign_weighted_sum", clean_aggregate), dtype=np.float64)
        malicious_total_weight = float(attacker_context.get("malicious_total_weight", attacker_context.get("malicious_weight", 1.0)))
        malicious_total_weight = max(malicious_total_weight, 1e-12)
        target_aggregate = -self.target_scale * clean_aggregate
        replacement = (target_aggregate - benign_weighted_sum) / malicious_total_weight
        if not np.all(np.isfinite(replacement)):
            replacement = -self.target_scale * clean_aggregate
        if np.linalg.norm(replacement) <= max_update_norm:
            return replacement
        direction = replacement / (np.linalg.norm(replacement) + 1e-12)
        best_candidate = direction * max_update_norm
        best_objective = self._objective(best_candidate, benign_weighted_sum, malicious_total_weight, target_aggregate)
        for scale in np.linspace(0.25, 1.0, max(2, self.search_steps)):
            candidate = direction * max_update_norm * scale
            objective = self._objective(candidate, benign_weighted_sum, malicious_total_weight, target_aggregate)
            if objective > best_objective:
                best_candidate = candidate
                best_objective = objective
        return best_candidate

    @staticmethod
    def _objective(candidate: np.ndarray, benign_weighted_sum: np.ndarray, malicious_total_weight: float, target_aggregate: np.ndarray) -> float:
        simulated = benign_weighted_sum + malicious_total_weight * candidate
        return -float(np.linalg.norm(simulated - target_aggregate))

    def _blackbox_mean_deviation(self, clean_update: np.ndarray, observations: np.ndarray, attacker_context: dict[str, Any], max_update_norm: float) -> np.ndarray:
        self.effective_attack_name = "sign_flip_scaled"
        benign_center = observations.mean(axis=0)
        center_norm = np.linalg.norm(benign_center)
        if center_norm <= 1e-12:
            direction = -np.sign(clean_update)
            direction[direction == 0.0] = -1.0
            direction = direction / (np.linalg.norm(direction) + 1e-12)
        else:
            direction = -benign_center / center_norm
        malicious_fraction = attacker_context.get("malicious_weight", 1.0 / max(1, attacker_context.get("num_selected", 1)))
        best_candidate = clean_update.copy()
        best_objective = -np.inf
        for magnitude in np.linspace(np.linalg.norm(clean_update) + 1e-12, max_update_norm, max(2, self.search_steps)):
            candidate = direction * magnitude
            simulated_shift = malicious_fraction * candidate + (1.0 - malicious_fraction) * benign_center
            objective = np.linalg.norm(simulated_shift - benign_center)
            if objective > best_objective:
                best_objective = objective
                best_candidate = candidate
        return best_candidate

    def _observations(self, clean_update: np.ndarray, attacker_context: dict[str, Any]) -> np.ndarray:
        if self.oracle_mean_replacement or attacker_context.get("oracle_mean_replacement", False):
            updates = attacker_context.get("all_clean_updates") or attacker_context.get("oracle_all_updates")
        else:
            updates = attacker_context.get("observable_updates")
        updates = updates or [clean_update]
        return np.vstack([np.asarray(update, dtype=np.float64) for update in updates])

    @staticmethod
    def _project(update: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(update))
        if max_norm > 0.0 and norm > max_norm:
            return update * (max_norm / (norm + 1e-12))
        return update



class SignFlipScaledAttack(FangMeanAttack):
    """Explicit non-oracle sign-flip-scaled poisoning baseline."""

    def __init__(self, aggregation: str = "weighted_mean", max_norm: float = 10.0, search_steps: int = 10, target_scale: float = 3.0) -> None:
        super().__init__(aggregation=aggregation, max_norm=max_norm, search_steps=search_steps, target_scale=target_scale, oracle_mean_replacement=False)
        self.effective_attack_name = "sign_flip_scaled"


FangAttack = FangMeanAttack
