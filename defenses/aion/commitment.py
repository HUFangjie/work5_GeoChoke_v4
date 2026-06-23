from __future__ import annotations

import hashlib
import hmac
from typing import Any


class BFTCommitmentSimulator:
    def __init__(self, aggregator_count: int, malicious_aggregator_count: int) -> None:
        self.n = int(aggregator_count)
        self.f = int(malicious_aggregator_count)
        if self.n < 3 * self.f + 1:
            raise ValueError("Aion requires n >= 3f + 1 aggregators")
        self.required = self.n - self.f

    def commit(self, valid_client_ids: list[int], online_client_ids: list[int], model_hash: str, round_id: int) -> dict[str, Any]:
        online_hash = hashlib.sha256(",".join(map(str, sorted(online_client_ids))).encode()).hexdigest()
        valid_hash = hashlib.sha256(",".join(map(str, sorted(valid_client_ids))).encode()).hexdigest()
        msg = f"aion|{round_id}|{valid_hash}|{online_hash}|{model_hash}".encode()
        signatures = []
        for idx in range(self.required):
            key = f"aggregator-{idx}".encode()
            signatures.append(hmac.new(key, msg, hashlib.sha256).hexdigest())
        if len(signatures) < self.required:
            raise ValueError("Aion BFT quorum not reached")
        return {
            "aion_bft_quorum_size": len(signatures),
            "aion_bft_required_quorum": self.required,
            "aion_bft_committed_online_set_hash": online_hash,
            "aion_bft_committed_model_hash": model_hash,
        }
