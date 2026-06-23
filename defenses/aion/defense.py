from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

from crypto.update_codec import ModelUpdateCodec
from defenses.base import DefenseStrategy
from defenses.aion.commitment import BFTCommitmentSimulator
from defenses.aion.metrics import append_csv, write_json
from defenses.aion.mgf import MaskedGradientFilter
from defenses.aion.protocol import AionProtocol


class AionDefense(DefenseStrategy):
    """Single-process simulation of Aion single-mask secure aggregation."""

    def __init__(self, cfg: Any, profiles: dict[str, dict[str, Any]], device: str = "cpu", output_dir: str | None = None) -> None:
        self.cfg = cfg
        self.profiles = dict(profiles)
        self.device = device
        self.output_dir = output_dir
        self.protocol = AionProtocol(cfg)
        self.mgf = MaskedGradientFilter(cfg)
        self.bft = BFTCommitmentSimulator(cfg.aggregator_count, cfg.malicious_aggregator_count)
        self.round_metrics: dict[int, dict[str, Any]] = {}
        self.last_global_update_inf_norm = float(cfg.hprf_hmax)
        self.min_clients_per_round = 1
        self._validate_config()

    def initialize(self, model: Any, crypto_backend: Any, proxy_loader: Any, decrypt_aggregate_fn: Any = None) -> None:
        self.codec = ModelUpdateCodec(model)
        self.dimension = self.codec.total_dimension
        os.makedirs(self.output_dir or "./outputs/aion", exist_ok=True)
        write_json(os.path.join(self.output_dir or "./outputs/aion", "aion_vss_init_summary.json"), self._vss_summary())
        validation = {"hprf_key_dim": self.cfg.hprf_key_dim, "validated": True, "note": "validated per round when clients are available"}
        write_json(os.path.join(self.output_dir or "./outputs/aion", "aion_hprf_validation.json"), validation)

    def get_profile_for_round(self, round_id: int) -> str:
        return self.cfg.profile_id

    def use_uniform_aggregation_weights(self) -> bool:
        return True

    def prepare_client_upload(self, client_id: int, update_vector: Any, round_id: int, profile_id: str, num_samples: int, metadata: dict[str, Any] | None = None):
        update = np.asarray(update_vector, dtype=np.float64)
        key = self.protocol.ensure_client(int(client_id))
        alpha = self.mgf.alpha(self.last_global_update_inf_norm)
        selected_count = int((metadata or {}).get("num_selected") or 1)
        extra_digits = self.mgf.extra_digits(selected_count)
        raw_mask = self.protocol.client_raw_mask(int(client_id), round_id, update.size)
        scaled_mask = self.protocol.hprf.dmc_scale(raw_mask, self.cfg.decimal_places, extra_digits) if self.cfg.enable_dmc_dmr else raw_mask
        masked = update + alpha * scaled_mask
        actual_mask_linf = float(np.linalg.norm(scaled_mask, ord=np.inf)) if scaled_mask.size else 0.0
        actual_mask_ratio = float(np.linalg.norm(alpha * scaled_mask, ord=np.inf) / max(self.last_global_update_inf_norm, float(self.cfg.bound_min))) if scaled_mask.size else 0.0
        defense_metadata = {
            "aion_client_id": int(client_id),
            "aion_round_id": int(round_id),
            "aion_masked_update_norm": float(np.linalg.norm(masked)),
            "aion_mask_norm": float(np.linalg.norm(alpha * scaled_mask)),
            "aion_alpha": float(alpha),
            "aion_key_commitment_hash": hashlib.sha256(np.asarray(key, dtype=np.int64).tobytes()).hexdigest(),
            "aion_extra_digits": int(extra_digits),
            "aion_dmc_enabled": bool(self.cfg.enable_dmc_dmr),
            "aion_dmr_enabled": bool(self.cfg.enable_dmc_dmr),
            "aion_rounding_decimal_places": int(self.cfg.decimal_places),
            "aion_hprf_hmax": float(self.cfg.hprf_hmax),
            "aion_actual_mask_linf": actual_mask_linf,
            "aion_actual_mask_ratio": actual_mask_ratio,
            "aion_protocol_payload": {
                "aion_masked_update_vector": masked.copy(),
                "aion_client_id": int(client_id),
                "aion_round_id": int(round_id),
                "aion_alpha": float(alpha),
                "aion_extra_digits": int(extra_digits),
            },
        }
        return masked, defense_metadata

    def filter_uploads_before_aggregation(self, uploads: list[Any], round_id: int):
        self._check_uniform_aggregation(uploads)
        result = self.mgf.filter(list(uploads), round_id, min_clients_per_round=self.min_clients_per_round)
        valid_ids = [upload.client_id for upload in result.valid_uploads]
        online_ids = [upload.client_id for upload in uploads]
        bft_metrics = self.bft.commit(valid_ids, online_ids, self._model_hash_placeholder(round_id), round_id)
        metrics = self._base_metrics(round_id)
        metrics.update(result.metrics)
        metrics.update(bft_metrics)
        metrics.update({
            "aion_alpha": float(result.valid_uploads[0].metadata.get("aion_alpha", self.mgf.alpha(self.last_global_update_inf_norm))) if result.valid_uploads else self.mgf.alpha(self.last_global_update_inf_norm),
            "aion_mask_ratio_beta": float(self.cfg.mask_ratio_beta),
            "aion_vss_valid_client_count": int(len(self.protocol.client_keys)),
            "aion_vss_invalid_share_count": int(self.protocol.invalid_share_count),
            "aion_extra_digits": max([int(u.metadata.get("aion_extra_digits", 0)) for u in uploads], default=0),
            "aion_dmc_enabled": bool(self.cfg.enable_dmc_dmr),
            "aion_dmr_enabled": bool(self.cfg.enable_dmc_dmr),
            "aion_hprf_hmax": float(self.cfg.hprf_hmax),
            "aion_actual_mask_linf": max([float(u.metadata.get("aion_actual_mask_linf", 0.0)) for u in uploads], default=0.0),
            "aion_actual_mask_ratio": max([float(u.metadata.get("aion_actual_mask_ratio", 0.0)) for u in uploads], default=0.0),
        })
        self.round_metrics.setdefault(round_id, {}).update(metrics)
        return result.valid_uploads, metrics

    def unmask_aggregate_update(self, aggregate_update: Any, uploads: list[Any], round_id: int, weights: list[float]):
        masked_aggregate = np.asarray(aggregate_update, dtype=np.float64)
        valid_ids = [int(upload.client_id) for upload in uploads]
        alpha = float(uploads[0].metadata.get("aion_alpha", self.mgf.alpha(self.last_global_update_inf_norm))) if uploads else self.mgf.alpha(self.last_global_update_inf_norm)
        raw_aggregate_mask, metrics = self.protocol.reconstruct_aggregated_mask(valid_ids, round_id, masked_aggregate.size, mean=True, mode=self.cfg.reconstruction_mode)
        extra_digits = int(uploads[0].metadata.get("aion_extra_digits", 0)) if uploads else 0
        aggregate_mask = self.protocol.hprf.dmc_scale(raw_aggregate_mask, self.cfg.decimal_places, extra_digits) if self.cfg.enable_dmc_dmr else raw_aggregate_mask
        unmasked = masked_aggregate - alpha * aggregate_mask
        if self.cfg.enable_dmc_dmr:
            unmasked = np.round(unmasked, int(self.cfg.decimal_places))
        unmasked_norm = float(np.linalg.norm(unmasked))
        self.last_global_update_inf_norm = float(np.linalg.norm(unmasked, ord=np.inf)) if unmasked.size else 0.0
        aggregated_mask_linf = float(np.linalg.norm(aggregate_mask, ord=np.inf)) if aggregate_mask.size else 0.0
        self.mgf.note_round_result(unmasked_norm, alpha, aggregated_mask_linf)
        metrics.update({
            "aion_aggregated_mask_linf": aggregated_mask_linf,
            "aion_aggregated_mask_norm": float(np.linalg.norm(alpha * aggregate_mask)),
            "aion_unmasked_aggregate_norm": unmasked_norm,
            "aion_rounding_decimal_places": int(self.cfg.decimal_places),
            "aion_dmc_enabled": bool(self.cfg.enable_dmc_dmr),
            "aion_dmr_enabled": bool(self.cfg.enable_dmc_dmr),
        })
        self.round_metrics.setdefault(round_id, {}).update(metrics)
        append_csv(os.path.join(self.output_dir or "./outputs/aion", "aion_protocol_metrics.csv"), {"round": round_id, **self.round_metrics.get(round_id, {})})
        return unmasked, metrics

    def after_aggregate(self, previous_model: Any, candidate_model: Any, current_profile_id: str, round_id: int) -> dict[str, Any]:
        metrics = self._base_metrics(round_id)
        metrics.update(self.round_metrics.get(round_id, {}))
        metrics.update({
            "previous_cfi": None,
            "candidate_cfi": None,
            "fragility_injection_score": None,
            "current_calibrated_error_energy": None,
            "unconstrained_target_error_energy": None,
            "target_next_error_energy": None,
            "selected_next_profile": self.cfg.profile_id,
            "profile_switching_indicator": False,
        })
        return metrics

    def get_round_metrics(self, round_id: int) -> dict[str, Any]:
        return dict(self.round_metrics.get(round_id, {}))

    def _base_metrics(self, round_id: int) -> dict[str, Any]:
        return {
            "defense_name": "aion",
            "defense_enabled": True,
            "secure_aggregation_scheme": "aion_single_mask",
            "aion_reconstruction_mode": self.cfg.reconstruction_mode,
            "selected_next_profile": self.cfg.profile_id,
        }

    def _check_uniform_aggregation(self, uploads: list[Any]) -> None:
        if self.cfg.allow_weighted_mean:
            return
        counts = {upload.num_samples for upload in uploads}
        if len(counts) > 1:
            raise ValueError("Aion HPRF masks support only uniform mean; unequal sample-count weighted_mean is unsafe")

    def _model_hash_placeholder(self, round_id: int) -> str:
        return hashlib.sha256(f"aion-model-round-{round_id}".encode()).hexdigest()

    def _vss_summary(self) -> dict[str, Any]:
        return {
            "aggregator_count": self.cfg.aggregator_count,
            "malicious_aggregator_count": self.cfg.malicious_aggregator_count,
            "threshold": self.protocol.vss.threshold,
            "field_modulus_bits": self.cfg.field_modulus_bits,
            "subgroup_order_bits": self.protocol.vss.q.bit_length(),
            "valid_client_count": len(self.protocol.client_keys),
            "invalid_share_count": self.protocol.invalid_share_count,
        }

    def _validate_config(self) -> None:
        if self.cfg.profile_id not in self.profiles:
            raise ValueError(f"Aion profile_id {self.cfg.profile_id!r} is not defined in ckks_profiles")
        if self.cfg.reconstruction_mode not in {"amr", "asr"}:
            raise ValueError("Aion reconstruction_mode must be 'amr' or 'asr'")
        if self.cfg.reconstruction_mode == "asr" and self.cfg.strict_protocol_checks:
            raise ValueError("Aion ASR requires CCS/disjoint client subsets; use reconstruction_mode='amr' in this simulator")
