from __future__ import annotations

from typing import Any

import numpy as np

from attacks.base import AttackStrategy


class FangMeanAttack(AttackStrategy):
    """Mean-aggregation adaptation of Fang-style local model poisoning.

    For weighted mean aggregation, the attack searches a bounded malicious update
    magnitude along the direction that maximally moves the aggregate away from
    the observed benign center. Krum/TrimmedMean-specific Fang variants are not
    silently reused for this aggregation rule.
    """

    def __init__(self, aggregation: str = "weighted_mean", max_norm: float = 5.0, search_steps: int = 8) -> None:
        if aggregation != "weighted_mean":
            raise ValueError("FangMeanAttack is only compatible with weighted_mean aggregation")
        self.max_norm = float(max_norm)
        self.search_steps = int(search_steps)

    def craft_update(self, client_id: int, clean_update: np.ndarray, global_model: Any, attacker_context: dict[str, Any]) -> np.ndarray:
        observable_updates = attacker_context.get("observable_updates") or [clean_update]
        observations = np.vstack([np.asarray(update, dtype=np.float64) for update in observable_updates])
        benign_center = observations.mean(axis=0)
        center_norm = np.linalg.norm(benign_center)
        if center_norm <= 1e-12:
            direction = -np.sign(clean_update)
            direction[direction == 0.0] = -1.0
            direction = direction / (np.linalg.norm(direction) + 1e-12)
        else:
            direction = -benign_center / center_norm
        clean_norm = np.linalg.norm(clean_update) + 1e-12
        maximum_norm = self.max_norm * clean_norm
        malicious_fraction = attacker_context.get("malicious_weight", 1.0 / max(1, attacker_context.get("num_selected", 1)))
        best_candidate = clean_update.copy()
        best_objective = -np.inf
        for magnitude in np.linspace(clean_norm, maximum_norm, max(2, self.search_steps)):
            candidate = direction * magnitude
            simulated_shift = malicious_fraction * candidate + (1.0 - malicious_fraction) * benign_center
            objective = np.linalg.norm(simulated_shift - benign_center)
            if objective > best_objective:
                best_objective = objective
                best_candidate = candidate
        candidate_norm = np.linalg.norm(best_candidate)
        if candidate_norm > maximum_norm:
            best_candidate = best_candidate * (maximum_norm / candidate_norm)
        return best_candidate.astype(np.float64, copy=False)


FangAttack = FangMeanAttack
