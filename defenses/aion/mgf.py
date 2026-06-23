from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MGFResult:
    valid_uploads: list[Any]
    metrics: dict[str, Any]


class MaskedGradientFilter:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.previous_bound: float | None = None
        self.previous_global_norms: list[float] = []

    def alpha(self, previous_inf_norm: float) -> float:
        return float(self.cfg.mask_ratio_beta) * max(float(previous_inf_norm), float(self.cfg.bound_min)) / max(float(self.cfg.hprf_hmax), 1e-12)

    def extra_digits(self, client_count: int) -> int:
        if not self.cfg.enable_dmc_dmr:
            return 0
        return max(0, int(np.ceil(np.log10(max(1, client_count)))))

    def filter(self, uploads: list[Any], round_id: int) -> MGFResult:
        if not self.cfg.enable_mgf or not uploads:
            return MGFResult(uploads, self._metrics(uploads, [], 0.0, 0.0))
        norms = np.asarray([float(upload.metadata.get("aion_masked_update_norm", 0.0)) for upload in uploads], dtype=np.float64)
        if self.previous_bound is None:
            quantile = float(np.quantile(norms, float(self.cfg.initial_bound_quantile))) if norms.size else 0.0
            bound = max(float(self.cfg.bound_min), float(self.cfg.initial_bound_multiplier) * quantile)
        else:
            drift = float(np.mean(self.previous_global_norms[-2:])) if self.previous_global_norms else 0.0
            mu = 1.0 + min(1.0, drift / max(self.previous_bound, 1e-12))
            bound = max(float(self.cfg.bound_min), self.previous_bound * mu)
        valid = [upload for upload, norm in zip(uploads, norms) if norm <= bound]
        filtered = [upload.client_id for upload, norm in zip(uploads, norms) if norm > bound]
        if not valid and uploads:
            best_idx = int(np.argmin(norms))
            valid = [uploads[best_idx]]
            filtered = [upload.client_id for idx, upload in enumerate(uploads) if idx != best_idx]
        self.previous_bound = bound
        mu = bound / max(float(np.mean(norms)) if norms.size else bound, 1e-12)
        return MGFResult(valid, self._metrics(uploads, filtered, bound, mu))

    def note_global_update(self, norm: float) -> None:
        self.previous_global_norms.append(float(norm))
        self.previous_global_norms = self.previous_global_norms[-3:]

    def _metrics(self, uploads: list[Any], filtered_ids: list[int], bound: float, mu: float) -> dict[str, Any]:
        norms = [float(upload.metadata.get("aion_masked_update_norm", 0.0)) for upload in uploads]
        total = len(uploads)
        return {
            "aion_bound": float(bound),
            "aion_mu": float(mu),
            "aion_valid_client_count": int(total - len(filtered_ids)),
            "aion_filtered_client_count": int(len(filtered_ids)),
            "aion_filtered_client_ids": list(filtered_ids),
            "aion_masked_norm_mean": float(np.mean(norms)) if norms else 0.0,
            "aion_masked_norm_max": float(np.max(norms)) if norms else 0.0,
            "aion_filter_rate": float(len(filtered_ids) / max(total, 1)),
        }
