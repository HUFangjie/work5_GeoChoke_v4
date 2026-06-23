from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def _coefficient(round_id: int, coordinate: int, key_index: int) -> float:
    seed = f"aion-hprf|{round_id}|{coordinate}|{key_index}".encode()
    raw = hashlib.shake_256(seed).digest(8)
    integer = int.from_bytes(raw, "big")
    return (integer / float(2**64 - 1)) * 2.0 - 1.0


class KeyHomomorphicPRF:
    """Deterministic linear key-homomorphic PRF used by Aion masks."""

    def __init__(self, key_dim: int = 16, hmax: float = 1.0) -> None:
        self.key_dim = int(key_dim)
        self.hmax = float(hmax)
        if self.key_dim <= 0:
            raise ValueError("hprf key_dim must be positive")

    def mask(self, key: Iterable[int | float], round_id: int, dimension: int) -> np.ndarray:
        key_array = np.asarray(list(key), dtype=np.float64)
        if key_array.size != self.key_dim:
            raise ValueError(f"HPRF key dimension {key_array.size} != expected {self.key_dim}")
        coeffs = np.empty((int(dimension), self.key_dim), dtype=np.float64)
        for coord in range(int(dimension)):
            for idx in range(self.key_dim):
                coeffs[coord, idx] = _coefficient(round_id, coord, idx)
        raw = coeffs @ key_array
        scale = max(float(self.hmax), 1e-12) * max(self.key_dim, 1) * 1024.0
        return raw / scale

    def assert_homomorphic_property(self, keys: list[Iterable[int | float]], round_id: int, dimension: int, atol: float = 1e-9) -> float:
        masks = [self.mask(key, round_id, dimension) for key in keys]
        aggregate_key = np.sum([np.asarray(list(key), dtype=np.float64) for key in keys], axis=0)
        err = float(np.max(np.abs(np.sum(masks, axis=0) - self.mask(aggregate_key, round_id, dimension))))
        if err > atol:
            raise AssertionError(f"HPRF additivity error {err} > {atol}")
        return err
