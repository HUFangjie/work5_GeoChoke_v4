from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

from attacks.base import AttackStrategy


class ALIEAttack(AttackStrategy):
    """Coordinate-wise ALIE attack applied before CKKS encryption."""

    def __init__(self, z: float | None = None, std_floor: float = 1e-6, oracle_all_updates: bool = False) -> None:
        self.z = z
        self.std_floor = std_floor
        self.oracle_all_updates = oracle_all_updates

    def craft_update(self, client_id: int, clean_update: np.ndarray, global_model: Any, attacker_context: dict[str, Any]) -> np.ndarray:
        observable_updates = attacker_context.get("observable_updates") or [clean_update]
        if self.oracle_all_updates:
            observable_updates = attacker_context.get("oracle_all_updates") or observable_updates
        observations = np.vstack([np.asarray(update, dtype=np.float64) for update in observable_updates])
        mean = observations.mean(axis=0)
        std = np.maximum(observations.std(axis=0), self.std_floor)
        malicious_count = max(1, int(attacker_context.get("num_malicious", 1)))
        selected_count = max(malicious_count + 1, int(attacker_context.get("num_selected", malicious_count + 1)))
        z_value = self.z if self.z is not None else self._default_z(malicious_count, selected_count)
        benign_direction = np.sign(mean)
        benign_direction[benign_direction == 0.0] = 1.0
        return mean - float(z_value) * std * benign_direction

    @staticmethod
    def _default_z(num_malicious: int, num_selected: int) -> float:
        quantile = (num_selected - num_malicious) / max(1, num_selected)
        quantile = min(0.99, max(0.51, quantile))
        return float(NormalDist().inv_cdf(quantile))
