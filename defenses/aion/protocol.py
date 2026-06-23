from __future__ import annotations

from typing import Any
from fractions import Fraction

import numpy as np

from defenses.aion.hprf import KeyHomomorphicPRF
from defenses.aion.vss import FeldmanVSS, Share, VSSSharing


class AionProtocol:
    def __init__(self, cfg: Any, seed: int = 7) -> None:
        self.cfg = cfg
        threshold = cfg.vss_threshold if cfg.vss_threshold is not None else cfg.malicious_aggregator_count + 1
        self.vss = FeldmanVSS(cfg.aggregator_count, cfg.malicious_aggregator_count, threshold, seed=seed)
        self.hprf = KeyHomomorphicPRF(cfg.hprf_key_dim, cfg.hprf_hmax)
        self.client_keys: dict[int, np.ndarray] = {}
        self.client_sharings: dict[int, list[VSSSharing]] = {}
        self.invalid_share_count = 0

    def ensure_client(self, client_id: int) -> np.ndarray:
        if client_id in self.client_keys:
            return self.client_keys[client_id]
        key = np.asarray([self.vss.random_scalar(f"client-{client_id}-{j}") % 4096 for j in range(self.cfg.hprf_key_dim)], dtype=np.int64)
        sharings = [self.vss.share_secret(int(value)) for value in key]
        for sharing in sharings:
            for share in sharing.shares:
                if not self.vss.verify_share(share, sharing.commitments):
                    self.invalid_share_count += 1
                    if self.cfg.strict_protocol_checks:
                        raise ValueError("VSS share verification failed during Aion initialization")
        self.client_keys[client_id] = key
        self.client_sharings[client_id] = sharings
        return key

    def client_raw_mask(self, client_id: int, round_id: int, dimension: int) -> np.ndarray:
        return self.hprf.raw_mask(self.ensure_client(client_id), round_id, dimension)

    def client_mask(self, client_id: int, round_id: int, dimension: int) -> np.ndarray:
        return self.client_raw_mask(client_id, round_id, dimension)

    def reconstruct_aggregated_mask(self, client_ids: list[int], round_id: int, dimension: int, mean: bool = True, mode: str = "amr") -> tuple[np.ndarray, dict[str, Any]]:
        if mode not in {"amr", "asr"}:
            raise ValueError(f"unsupported Aion reconstruction mode: {mode}")
        if not client_ids:
            return np.zeros(dimension, dtype=np.float64), {"aion_reconstruction_success": False, "aion_secret_recovered": False}
        for client_id in client_ids:
            self.ensure_client(client_id)
        if mode == "asr":
            return self._reconstruct_mask_asr(client_ids, round_id, dimension, mean)
        return self._reconstruct_mask_amr(client_ids, round_id, dimension, mean)

    def _reconstruct_mask_amr(self, client_ids: list[int], round_id: int, dimension: int, mean: bool) -> tuple[np.ndarray, dict[str, Any]]:
        quorum = max(2 * self.cfg.malicious_aggregator_count + 1, self.vss.threshold)
        quorum = min(quorum, self.cfg.aggregator_count)
        aggregator_key_shares: list[np.ndarray] = []
        aggregator_xs: list[int] = []
        for agg_idx in range(quorum):
            x = agg_idx + 1
            key_share = []
            for dim_idx in range(self.cfg.hprf_key_dim):
                share_sets = [self.client_sharings[cid][dim_idx].shares for cid in client_ids]
                commitments = [self.client_sharings[cid][dim_idx].commitments for cid in client_ids]
                agg_shares = self.vss.aggregate_shares(share_sets)
                agg_commitments = self.vss.aggregate_commitments(commitments)
                share = agg_shares[agg_idx]
                if not self.vss.verify_share(share, agg_commitments):
                    self.invalid_share_count += 1
                    if self.cfg.strict_protocol_checks:
                        raise ValueError("aggregated VSS share verification failed")
                key_share.append(share.y)
            aggregator_xs.append(x)
            aggregator_key_shares.append(np.asarray(key_share, dtype=np.float64))
        lambdas = self._lagrange_at_zero(aggregator_xs)
        mask = np.zeros(dimension, dtype=np.float64)
        for coeff, key_share in zip(lambdas, aggregator_key_shares):
            mask += coeff * self.hprf.raw_mask(key_share, round_id, dimension)
        if mean:
            mask = mask / float(len(client_ids))
        direct = np.mean(np.stack([self.client_raw_mask(cid, round_id, dimension) for cid in client_ids], axis=0), axis=0) if mean else np.sum([self.client_raw_mask(cid, round_id, dimension) for cid in client_ids], axis=0)
        add_err = float(np.max(np.abs(mask - direct))) if mask.size else 0.0
        return mask, {
            "aion_reconstruction_mode": "amr",
            "aion_reconstruction_success": True,
            "aion_secret_recovered": False,
            "aion_hprf_additivity_max_error": add_err,
        }

    def _reconstruct_mask_asr(self, client_ids: list[int], round_id: int, dimension: int, mean: bool) -> tuple[np.ndarray, dict[str, Any]]:
        agg_key = []
        for dim_idx in range(self.cfg.hprf_key_dim):
            share_sets = [self.client_sharings[cid][dim_idx].shares for cid in client_ids]
            commitments = [self.client_sharings[cid][dim_idx].commitments for cid in client_ids]
            agg_shares = self.vss.aggregate_shares(share_sets)
            agg_commitments = self.vss.aggregate_commitments(commitments)
            for share in agg_shares[: self.vss.threshold]:
                if not self.vss.verify_share(share, agg_commitments):
                    self.invalid_share_count += 1
                    if self.cfg.strict_protocol_checks:
                        raise ValueError("aggregated VSS share verification failed")
            agg_key.append(self.vss.reconstruct(agg_shares[: self.vss.threshold]))
        key_array = np.asarray(agg_key, dtype=np.float64)
        if mean:
            key_array = key_array / float(len(client_ids))
        mask = self.hprf.raw_mask(key_array, round_id, dimension)
        direct = np.mean(np.stack([self.client_raw_mask(cid, round_id, dimension) for cid in client_ids], axis=0), axis=0) if mean else np.sum([self.client_raw_mask(cid, round_id, dimension) for cid in client_ids], axis=0)
        add_err = float(np.max(np.abs(mask - direct))) if mask.size else 0.0
        return mask, {
            "aion_reconstruction_mode": "asr",
            "aion_reconstruction_success": True,
            "aion_secret_recovered": True,
            "aion_hprf_additivity_max_error": add_err,
        }

    def _lagrange_at_zero(self, xs: list[int]) -> list[float]:
        coeffs: list[float] = []
        for j, xj in enumerate(xs):
            coeff = Fraction(1, 1)
            for m, xm in enumerate(xs):
                if m == j:
                    continue
                coeff *= Fraction(-xm, xj - xm)
            coeffs.append(float(coeff))
        return coeffs
