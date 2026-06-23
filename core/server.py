from __future__ import annotations

import copy
import random
import time
from typing import Any, Callable, Iterable

import numpy as np
import torch

from core.types import ClientUpload
from crypto.ciphertext_payload import EncryptedUpdate
from crypto.update_codec import ModelUpdateCodec


class AggregationServer:
    """Aggregation server without CKKS secret keys.

    The server owns only the global model, public-key crypto backend, and a
    narrow aggregate-decryption callable. It never decrypts individual client
    ciphertexts and computes aggregation weights from server-side sample counts.
    """

    def __init__(
        self,
        cfg: Any,
        model: torch.nn.Module,
        codec: ModelUpdateCodec,
        crypto_backend: Any,
        defense: Any,
        model_factory: Callable[[], torch.nn.Module],
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.codec = codec
        self.crypto_backend = crypto_backend
        self.defense = defense
        self.model_factory = model_factory

    def sample_clients(self, round_id: int) -> list[int]:
        rng = random.Random(self.cfg.seed + round_id)
        client_ids = list(range(self.cfg.num_clients))
        rng.shuffle(client_ids)
        return client_ids[: self.cfg.clients_per_round]

    def aggregate_encrypted(self, uploads: Iterable[ClientUpload]) -> tuple[EncryptedUpdate, float, list[float]]:
        upload_list = list(uploads)
        if len(upload_list) < self.cfg.min_clients_per_round:
            raise ValueError("too few aggregation participants")
        profile_id = upload_list[0].profile_id
        if any(upload.profile_id != profile_id for upload in upload_list):
            raise ValueError("profile mismatch among client uploads")
        if getattr(self.defense, "use_uniform_aggregation_weights", lambda: False)():
            if self.cfg.aggregation == "weighted_mean" and len({upload.num_samples for upload in upload_list}) > 1:
                raise ValueError("Aion currently supports only uniform mean; unequal weighted_mean would weight masks incorrectly")
            weights = [1.0 / len(upload_list) for _ in upload_list]
        else:
            total_samples = sum(upload.num_samples for upload in upload_list)
            if total_samples <= 0:
                raise ValueError("selected clients have no samples")
            weights = [upload.num_samples / total_samples for upload in upload_list]
        start_time = time.perf_counter()
        weighted_ciphertexts = [
            self.crypto_backend.multiply_plain(upload.encrypted_update, weight)
            for upload, weight in zip(upload_list, weights)
        ]
        aggregate = self.crypto_backend.add_ciphertexts(weighted_ciphertexts)
        aggregation_time = time.perf_counter() - start_time
        return aggregate, aggregation_time, weights

    def apply_round(
        self,
        uploads: Iterable[ClientUpload],
        round_id: int,
        decrypt_aggregate_fn: Callable[[EncryptedUpdate, str], np.ndarray],
        plaintext_reference_update: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        original_upload_list = list(uploads)
        upload_list = original_upload_list
        defense_filter_metrics: dict[str, Any] = {}
        if hasattr(self.defense, "filter_uploads_before_aggregation"):
            if hasattr(self.defense, "min_clients_per_round"):
                self.defense.min_clients_per_round = int(self.cfg.min_clients_per_round)
            upload_list, defense_filter_metrics = self.defense.filter_uploads_before_aggregation(upload_list, round_id)
        previous_state = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
        aggregate_ciphertext, aggregation_time, weights = self.aggregate_encrypted(upload_list)
        profile_id = upload_list[0].profile_id
        start_time = time.perf_counter()
        decrypted_update = decrypt_aggregate_fn(aggregate_ciphertext, profile_id)
        decryption_time = time.perf_counter() - start_time
        unmask_metrics: dict[str, Any] = {}
        if hasattr(self.defense, "unmask_aggregate_update"):
            decrypted_update, unmask_metrics = self.defense.unmask_aggregate_update(
                aggregate_update=decrypted_update,
                uploads=upload_list,
                round_id=round_id,
                weights=weights,
            )
        candidate_state = self.codec.apply_update_to_state_dict(
            previous_state,
            decrypted_update,
            step_size=self.cfg.server_lr,
        )
        candidate_model = self.model_factory()
        candidate_model.load_state_dict(candidate_state)
        geo_metrics = self.defense.after_aggregate(self.model, candidate_model, profile_id, round_id)
        default_tau = float(getattr(self.cfg.geochoke, "tangent_tau_max", 1.0))
        if hasattr(self.defense, "commit_update"):
            committed_update, tangent_metrics = self.defense.commit_update(
                decrypted_update,
                self.model,
                round_id,
                geo_metrics.get("tangent_tau", default_tau),
            )
        else:
            committed_update = decrypted_update
            original_norm = float(np.linalg.norm(decrypted_update))
            tangent_metrics = {
                "tangent_commitment_enabled": False,
                "tangent_basis_rank_actual": 0,
                "tangent_tau": geo_metrics.get("tangent_tau", default_tau),
                "tangent_rho": 1.0,
                "tangent_parallel_norm": original_norm,
                "tangent_perp_norm": 0.0,
                "tangent_null_ratio": 0.0,
                "tangent_original_update_norm": original_norm,
                "tangent_committed_update_norm": original_norm,
                "tangent_update_shrink_ratio": 1.0,
            }
        final_state = self.codec.apply_update_to_state_dict(
            previous_state,
            committed_update,
            step_size=self.cfg.server_lr,
        )
        self.model.load_state_dict(final_state)
        reference_mse = None
        reference_max_error = None
        plaintext_norm = None
        residual_l2_norm = None
        residual_l2_ratio = None
        if plaintext_reference_update is not None:
            if isinstance(plaintext_reference_update, dict):
                refs = [plaintext_reference_update[upload.client_id] for upload in upload_list if upload.client_id in plaintext_reference_update]
                if refs:
                    plaintext_reference_update = np.mean(np.stack(refs, axis=0), axis=0)
                else:
                    plaintext_reference_update = None
            if plaintext_reference_update is not None and plaintext_reference_update.shape != decrypted_update.shape:
                raise ValueError("plaintext reference update dimension mismatch")
            if plaintext_reference_update is not None:
                difference = decrypted_update - plaintext_reference_update
                plaintext_norm = float(np.linalg.norm(plaintext_reference_update))
                residual_l2_norm = float(np.linalg.norm(difference))
                residual_l2_ratio = residual_l2_norm / max(plaintext_norm, 1e-12)
                reference_mse = float(np.mean(difference * difference))
                reference_max_error = float(np.max(np.abs(difference)))
        serialized = self.crypto_backend.serialize(aggregate_ciphertext)
        return decrypted_update, {
            "encrypted_aggregation_time": aggregation_time,
            "original_selected_client_count": len(original_upload_list),
            "decryption_time": decryption_time,
            "ciphertext_block_count": aggregate_ciphertext.block_count,
            "serialized_ciphertext_bytes": sum(len(chunk["payload"]) for chunk in serialized["chunks"]),
            "aggregate_update_norm": float(np.linalg.norm(decrypted_update)),
            "committed_aggregate_update_norm": float(np.linalg.norm(committed_update)),
            "plaintext_aggregate_update_norm": plaintext_norm,
            "decrypted_aggregate_update_norm": float(np.linalg.norm(decrypted_update)),
            "ckks_residual_l2_norm": residual_l2_norm,
            "ckks_residual_l2_ratio": residual_l2_ratio,
            "ckks_residual_max_abs": reference_max_error,
            "aggregate_ckks_mse": reference_mse,
            "aggregate_maximum_absolute_error": reference_max_error,
            **defense_filter_metrics,
            **unmask_metrics,
            **geo_metrics,
            **tangent_metrics,
        }
