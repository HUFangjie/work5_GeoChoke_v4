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
        self.history: list[dict[str, float]] = []

    def alpha(self, previous_inf_norm: float) -> float:
        return float(self.cfg.mask_ratio_beta) * max(float(previous_inf_norm), float(self.cfg.bound_min)) / max(float(self.cfg.hprf_hmax), 1e-12)

    def extra_digits(self, client_count: int) -> int:
        if not self.cfg.enable_dmc_dmr:
            return 0
        return max(0, int(np.ceil(np.log10(max(1, 2 * int(client_count))))))

    def masked_vector(self, upload: Any) -> np.ndarray:
        payload = getattr(upload, "protocol_payload", {}) or {}
        if "aion_masked_update_vector" not in payload:
            raise ValueError("Aion upload is missing aion_masked_update_vector protocol payload")
        return np.asarray(payload["aion_masked_update_vector"], dtype=np.float64)

    def filter(self, uploads: list[Any], round_id: int, min_clients_per_round: int = 1) -> MGFResult:
        if not self.cfg.enable_mgf or not uploads:
            return MGFResult(uploads, self._metrics(uploads, [], np.array([], dtype=np.float64), 0.0, 0.0, 1.0, 1.0, 0.0, 0.0))
        vectors = [self.masked_vector(upload) for upload in uploads]
        norms = np.asarray([float(np.linalg.norm(vector)) for vector in vectors], dtype=np.float64)
        finite_mask = np.isfinite(norms)
        finite_norms = norms[finite_mask]
        if finite_norms.size == 0:
            if self.cfg.fail_open_when_too_few_valid:
                return MGFResult(list(uploads), self._metrics(uploads, [], norms, float("inf"), self.previous_bound, 1.0, 1.0, 0.0, 0.0))
            raise ValueError("Aion MGF received no finite masked update norms")
        previous_bound = float(self.previous_bound) if self.previous_bound is not None else None
        bound, mu_raw, mu_clipped, numerator, denominator = self._compute_bound(finite_norms, round_id)
        if self.previous_bound is None or round_id < int(self.cfg.warmup_rounds_for_bound):
            kth = min(max(int(min_clients_per_round), 1), finite_norms.size) - 1
            min_required_bound = float(np.partition(finite_norms, kth)[kth])
            bound = max(bound, min_required_bound)
        valid = [upload for upload, norm in zip(uploads, norms) if np.isfinite(norm) and norm <= bound]
        filtered = [upload for upload, norm in zip(uploads, norms) if (not np.isfinite(norm)) or norm > bound]
        if not valid and uploads and self.cfg.fail_safe_keep_one:
            best_idx = int(np.argmin(norms))
            valid = [uploads[best_idx]]
            filtered = [upload for idx, upload in enumerate(uploads) if idx != best_idx]
        if len(valid) < int(min_clients_per_round):
            if self.cfg.fail_open_when_too_few_valid:
                valid = list(uploads)
                filtered = []
            else:
                raise ValueError(f"Aion MGF valid client count {len(valid)} < min_clients_per_round {min_clients_per_round}")
        filtered_ids = [upload.client_id for upload in filtered]
        self.previous_bound = bound
        return MGFResult(valid, self._metrics(uploads, filtered_ids, norms, bound, previous_bound, mu_raw, mu_clipped, numerator, denominator))

    def note_round_result(self, global_l2_norm: float, alpha: float, aggregated_mask_linf: float) -> None:
        self.history.append({
            "global_l2_norm": float(global_l2_norm),
            "alpha": float(alpha),
            "aggregated_mask_linf": float(aggregated_mask_linf),
        })
        self.history = self.history[-3:]

    def _compute_bound(self, norms: np.ndarray, round_id: int) -> tuple[float, float, float, float, float]:
        if self.previous_bound is None or round_id < int(self.cfg.warmup_rounds_for_bound) or len(self.history) < 2 or not self.cfg.use_paper_evolving_bound:
            quantile = float(np.quantile(norms, float(self.cfg.initial_bound_quantile))) if norms.size else 0.0
            bound = float(self.cfg.initial_bound_multiplier) * quantile
            bound = self._clip_bound(max(float(self.cfg.bound_min), bound))
            return bound, 1.0, 1.0, 0.0, 0.0
        prev = self.history[-1]
        pre_prev = self.history[-2]
        numerator = prev["global_l2_norm"] + prev["alpha"] * prev["aggregated_mask_linf"]
        denominator = pre_prev["global_l2_norm"] + pre_prev["alpha"] * pre_prev["aggregated_mask_linf"]
        mu_raw = numerator / max(denominator, 1e-12)
        mu_clipped = min(float(self.cfg.mu_max), max(float(self.cfg.mu_min), float(mu_raw)))
        bound = self._clip_bound(float(self.previous_bound) * mu_clipped)
        return bound, float(mu_raw), float(mu_clipped), float(numerator), float(denominator)

    def _clip_bound(self, bound: float) -> float:
        bound = max(float(self.cfg.bound_min), float(bound))
        if self.cfg.bound_max is not None:
            bound = min(float(self.cfg.bound_max), bound)
        return bound

    def _metrics(self, uploads: list[Any], filtered_ids: list[int], norms: np.ndarray, bound: float, previous_bound: float | None, mu_raw: float, mu_clipped: float, numerator: float, denominator: float) -> dict[str, Any]:
        total = len(uploads)
        filtered_set = set(filtered_ids)
        valid_ids = [upload.client_id for upload in uploads if upload.client_id not in filtered_set]
        malicious_total = sum(1 for upload in uploads if upload.metadata.get("is_malicious", False))
        filtered_malicious = sum(1 for upload in uploads if upload.client_id in filtered_set and upload.metadata.get("is_malicious", False))
        filtered_benign = len(filtered_ids) - filtered_malicious
        passed_malicious = malicious_total - filtered_malicious
        passed_benign = (total - malicious_total) - filtered_benign
        quantile = float(np.quantile(norms, float(self.cfg.initial_bound_quantile))) if norms.size else 0.0
        return {
            "aion_mgf_enabled": bool(self.cfg.enable_mgf),
            "aion_bound_init_mode": self.cfg.bound_init_mode,
            "aion_bound": float(bound),
            "aion_previous_bound": previous_bound,
            "aion_mu": float(mu_clipped),
            "aion_mu_raw": float(mu_raw),
            "aion_mu_clipped": float(mu_clipped),
            "aion_bound_numerator": float(numerator),
            "aion_bound_denominator": float(denominator),
            "aion_valid_client_ids": valid_ids,
            "aion_filtered_client_ids": list(filtered_ids),
            "aion_valid_client_count": int(len(valid_ids)),
            "aion_filtered_client_count": int(len(filtered_ids)),
            "aion_filter_rate": float(len(filtered_ids) / max(total, 1)),
            "aion_filtered_malicious_count": int(filtered_malicious),
            "aion_filtered_benign_count": int(filtered_benign),
            "aion_passed_malicious_count": int(passed_malicious),
            "aion_passed_benign_count": int(passed_benign),
            "aion_malicious_pass_rate": float(passed_malicious / max(malicious_total, 1)),
            "aion_masked_norm_mean": float(np.mean(norms)) if norms.size else 0.0,
            "aion_masked_norm_median": float(np.median(norms)) if norms.size else 0.0,
            "aion_masked_norm_std": float(np.std(norms)) if norms.size else 0.0,
            "aion_masked_norm_max": float(np.max(norms)) if norms.size else 0.0,
            "aion_masked_norm_quantile": quantile,
        }
