from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

from attacks.base import AttackStrategy


class ALIEAttack(AttackStrategy):
    """White-box capable coordinate-wise ALIE attack applied before CKKS encryption."""

    def __init__(
        self,
        z: float | None = None,
        std_floor: float = 1e-6,
        oracle_all_updates: bool = False,
        whitebox: bool = True,
        whitebox_z: float = 2.5,
        strength: float = 1.5,
    ) -> None:
        self.z = z
        self.std_floor = std_floor
        self.oracle_all_updates = oracle_all_updates
        self.whitebox = whitebox
        self.whitebox_z = float(whitebox_z)
        self.strength = float(strength)

    def craft_update(self, client_id: int, clean_update: np.ndarray, global_model: Any, attacker_context: dict[str, Any]) -> np.ndarray:
        observations = self._observations(clean_update, attacker_context)
        mean = observations.mean(axis=0)
        std = np.maximum(observations.std(axis=0), self.std_floor)
        malicious_count = max(1, int(attacker_context.get("num_malicious", 1)))
        selected_count = max(malicious_count + 1, int(attacker_context.get("num_selected", malicious_count + 1)))
        theoretical_z = self._default_z(malicious_count, selected_count)
        z_value = self.z if self.z is not None else theoretical_z
        if self.whitebox or attacker_context.get("whitebox", False):
            z_value = max(float(z_value), self.whitebox_z)
        direction = self._attack_direction(mean, clean_update, attacker_context)
        crafted = mean - self.strength * float(z_value) * std * direction
        max_norm = attacker_context.get("attack_max_norm")
        if max_norm is not None:
            crafted = self._project(crafted, float(max_norm))
        return crafted.astype(np.float64, copy=False)

    def _observations(self, clean_update: np.ndarray, attacker_context: dict[str, Any]) -> np.ndarray:
        if self.whitebox or attacker_context.get("whitebox", False) or self.oracle_all_updates:
            observable_updates = attacker_context.get("all_clean_updates") or attacker_context.get("oracle_all_updates")
        else:
            observable_updates = attacker_context.get("observable_updates")
        observable_updates = observable_updates or [clean_update]
        return np.vstack([np.asarray(update, dtype=np.float64) for update in observable_updates])

    @staticmethod
    def _attack_direction(mean: np.ndarray, clean_update: np.ndarray, attacker_context: dict[str, Any]) -> np.ndarray:
        clean_aggregate = attacker_context.get("clean_aggregate")
        if clean_aggregate is not None:
            direction = np.sign(np.asarray(clean_aggregate, dtype=np.float64))
        else:
            direction = np.sign(mean)
        direction[direction == 0.0] = np.sign(clean_update[direction == 0.0])
        direction[direction == 0.0] = 1.0
        return direction

    @staticmethod
    def _project(update: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(update))
        if max_norm > 0.0 and norm > max_norm:
            return update * (max_norm / (norm + 1e-12))
        return update

    @staticmethod
    def _default_z(num_malicious: int, num_selected: int) -> float:
        quantile = (num_selected - num_malicious) / max(1, num_selected)
        quantile = min(0.99, max(0.51, quantile))
        return float(NormalDist().inv_cdf(quantile))
