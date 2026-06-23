from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable

RFC3526_2048_PRIME = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E08"
    "8A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD"
    "3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E"
    "7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899F"
    "A5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF05"
    "98DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C"
    "62F356208552BB9ED529077096966D670C354E4ABC9804F1746C"
    "08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2E"
    "C07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA9"
    "56AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
GENERATOR = 4


@dataclass(frozen=True)
class Share:
    x: int
    y: int


@dataclass(frozen=True)
class VSSSharing:
    secret: int
    coefficients: tuple[int, ...]
    shares: tuple[Share, ...]
    commitments: tuple[int, ...]


class FeldmanVSS:
    def __init__(self, aggregator_count: int, malicious_aggregator_count: int = 0, threshold: int | None = None, seed: int = 7) -> None:
        self.n = int(aggregator_count)
        self.f = int(malicious_aggregator_count)
        self.threshold = int(threshold) if threshold is not None else self.f + 1
        if self.n <= 0 or self.threshold <= 0 or self.threshold > self.n:
            raise ValueError("invalid VSS n/threshold")
        self.p = RFC3526_2048_PRIME
        self.q = (self.p - 1) // 2
        self.g = GENERATOR
        self.rng = random.Random(seed)

    def random_scalar(self, label: str = "") -> int:
        digest = hashlib.sha256(f"aion-vss|{label}|{self.rng.getrandbits(256)}".encode()).digest()
        return int.from_bytes(digest, "big") % self.q

    def share_secret(self, secret: int) -> VSSSharing:
        secret %= self.q
        coeffs = [secret] + [self.random_scalar(f"coeff-{i}") for i in range(1, self.threshold)]
        shares = tuple(Share(x, self._poly_eval(coeffs, x)) for x in range(1, self.n + 1))
        commitments = tuple(pow(self.g, coeff, self.p) for coeff in coeffs)
        return VSSSharing(secret, tuple(coeffs), shares, commitments)

    def verify_share(self, share: Share, commitments: Iterable[int]) -> bool:
        lhs = pow(self.g, share.y % self.q, self.p)
        rhs = 1
        for power, commitment in enumerate(tuple(commitments)):
            rhs = (rhs * pow(int(commitment), pow(share.x, power, self.q), self.p)) % self.p
        return lhs == rhs

    def aggregate_shares(self, share_sets: Iterable[Iterable[Share]]) -> tuple[Share, ...]:
        totals: dict[int, int] = {}
        for shares in share_sets:
            for share in shares:
                totals[share.x] = (totals.get(share.x, 0) + share.y) % self.q
        return tuple(Share(x, totals[x]) for x in sorted(totals))

    def aggregate_commitments(self, commitments_list: Iterable[Iterable[int]]) -> tuple[int, ...]:
        result = [1] * self.threshold
        for commitments in commitments_list:
            for idx, commitment in enumerate(tuple(commitments)[: self.threshold]):
                result[idx] = (result[idx] * int(commitment)) % self.p
        return tuple(result)

    def reconstruct(self, shares: Iterable[Share]) -> int:
        selected = tuple(shares)
        if len(selected) < self.threshold:
            raise ValueError(f"need at least {self.threshold} shares to reconstruct")
        selected = selected[: self.threshold]
        secret = 0
        for j, share_j in enumerate(selected):
            num = 1
            den = 1
            xj = share_j.x % self.q
            for m, share_m in enumerate(selected):
                if m == j:
                    continue
                xm = share_m.x % self.q
                num = (num * (-xm)) % self.q
                den = (den * (xj - xm)) % self.q
            secret = (secret + share_j.y * num * pow(den, -1, self.q)) % self.q
        return secret

    def _poly_eval(self, coeffs: list[int], x: int) -> int:
        total = 0
        power = 1
        for coeff in coeffs:
            total = (total + coeff * power) % self.q
            power = (power * x) % self.q
        return total
