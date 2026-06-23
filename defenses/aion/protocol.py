from __future__ import annotations

from typing import Any

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

    def client_mask(self, client_id: int, round_id: int, dimension: int) -> np.ndarray:
        key = self.ensure_client(client_id)
        return self.hprf.mask(key, round_id, dimension)

    def reconstruct_aggregated_mask(self, client_ids: list[int], round_id: int, dimension: int, mean: bool = True, mode: str = "amr") -> tuple[np.ndarray, dict[str, Any]]:
        if mode not in {"amr", "asr"}:
            raise ValueError(f"unsupported Aion reconstruction mode: {mode}")
        if not client_ids:
            return np.zeros(dimension, dtype=np.float64), {"aion_reconstruction_success": False}
        for client_id in client_ids:
            self.ensure_client(client_id)
        agg_key = []
        max_add_err = 0.0
        quorum = max(2 * self.cfg.malicious_aggregator_count + 1, self.vss.threshold)
        quorum = min(quorum, self.cfg.aggregator_count)
        for dim_idx in range(self.cfg.hprf_key_dim):
            share_sets = [self.client_sharings[cid][dim_idx].shares for cid in client_ids]
            commitments = [self.client_sharings[cid][dim_idx].commitments for cid in client_ids]
            agg_shares = self.vss.aggregate_shares(share_sets)
            agg_commitments = self.vss.aggregate_commitments(commitments)
            for share in agg_shares[:quorum]:
                if not self.vss.verify_share(share, agg_commitments):
                    self.invalid_share_count += 1
                    if self.cfg.strict_protocol_checks:
                        raise ValueError("aggregated VSS share verification failed")
            secret = self.vss.reconstruct(agg_shares[: self.vss.threshold])
            agg_key.append(secret)
            expected = int(sum(int(self.client_keys[cid][dim_idx]) for cid in client_ids)) % self.vss.q
            max_add_err = max(max_add_err, abs(float(secret - expected)))
        key_array = np.asarray(agg_key, dtype=np.float64)
        if mean:
            key_array = key_array / float(len(client_ids))
        # ASR reconstructs the aggregated secret explicitly; AMR reaches the same
        # mask via verified aggregated shares in this single-process simulator.
        mask = self.hprf.mask(key_array, round_id, dimension)
        direct_keys = [self.client_keys[cid] / float(len(client_ids) if mean else 1) for cid in client_ids]
        add_err = self.hprf.assert_homomorphic_property(direct_keys, round_id, dimension, atol=1e-8)
        return mask, {
            "aion_reconstruction_mode": mode,
            "aion_reconstruction_success": True,
            "aion_hprf_additivity_max_error": float(max(max_add_err, add_err)),
        }
