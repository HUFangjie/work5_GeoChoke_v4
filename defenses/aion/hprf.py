from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def _coefficient(round_id: int, coordinate: int, key_index: int) -> int:
    seed = f"aion-hprf|{round_id}|{coordinate}|{key_index}".encode()
    raw = hashlib.shake_256(seed).digest(8)
    return int.from_bytes(raw, "big") % 2049 - 1024


class KeyHomomorphicPRF:
    """Deterministic linear key-homomorphic PRF with raw integer masks.

    `hmax` is the target max absolute value after normalization. DMC then scales
    this normalized raw mask by 10 ** (decimal_places + extra_digits).
    """

    def __init__(self, key_dim: int = 16, hmax: float = 1.0) -> None:
        self.key_dim = int(key_dim)
        self.hmax = float(hmax)
        if self.key_dim <= 0:
            raise ValueError("hprf key_dim must be positive")
        if self.hmax <= 0.0:
            raise ValueError("hprf hmax must be positive")

    def raw_mask(self, key: Iterable[int | float], round_id: int, dimension: int) -> np.ndarray:
        key_array = np.asarray(list(key), dtype=np.float64)
        if key_array.size != self.key_dim:
            raise ValueError(f"HPRF key dimension {key_array.size} != expected {self.key_dim}")
        coeffs = np.empty((int(dimension), self.key_dim), dtype=np.float64)
        for coord in range(int(dimension)):
            for idx in range(self.key_dim):
                coeffs[coord, idx] = _coefficient(round_id, coord, idx)
        raw = coeffs @ key_array
        scale = max(1.0, float(self.key_dim) * 1024.0 * 4096.0)
        return raw / scale * self.hmax

    def mask(self, key: Iterable[int | float], round_id: int, dimension: int) -> np.ndarray:
        return self.raw_mask(key, round_id, dimension)

    def dmc_scale(self, raw_or_normalized_mask: np.ndarray, decimal_places: int, extra_digits: int) -> np.ndarray:
        return np.asarray(raw_or_normalized_mask, dtype=np.float64) / float(10 ** (int(decimal_places) + int(extra_digits)))

    def assert_homomorphic_property(self, keys: list[Iterable[int | float]], round_id: int, dimension: int, atol: float = 1e-9) -> float:
        raw_masks = [self.raw_mask(key, round_id, dimension) for key in keys]
        aggregate_key = np.sum([np.asarray(list(key), dtype=np.float64) for key in keys], axis=0)
        err = float(np.max(np.abs(np.sum(raw_masks, axis=0) - self.raw_mask(aggregate_key, round_id, dimension))))
        if err > atol:
            raise AssertionError(f"HPRF additivity error {err} > {atol}")
        return err
