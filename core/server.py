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
        upload_list = list(uploads)
        previous_state = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
        aggregate_ciphertext, aggregation_time, weights = self.aggregate_encrypted(upload_list)
        profile_id = upload_list[0].profile_id
        start_time = time.perf_counter()
        decrypted_update = decrypt_aggregate_fn(aggregate_ciphertext, profile_id)
        decryption_time = time.perf_counter() - start_time
        candidate_state = self.codec.apply_update_to_state_dict(
            previous_state,
            decrypted_update,
            step_size=self.cfg.server_lr,
        )
        candidate_model = self.model_factory()
        candidate_model.load_state_dict(candidate_state)
        geo_metrics = self.defense.after_aggregate(self.model, candidate_model, profile_id, round_id)
        self.model.load_state_dict(candidate_state)
        reference_mse = None
        reference_max_error = None
        if plaintext_reference_update is not None:
            if plaintext_reference_update.shape != decrypted_update.shape:
                raise ValueError("plaintext reference update dimension mismatch")
            difference = decrypted_update - plaintext_reference_update
            reference_mse = float(np.mean(difference * difference))
            reference_max_error = float(np.max(np.abs(difference)))
        serialized = self.crypto_backend.serialize(aggregate_ciphertext)
        return decrypted_update, {
            "encrypted_aggregation_time": aggregation_time,
            "decryption_time": decryption_time,
            "ciphertext_block_count": aggregate_ciphertext.block_count,
            "serialized_ciphertext_bytes": sum(len(chunk["payload"]) for chunk in serialized["chunks"]),
            "aggregate_update_norm": float(np.linalg.norm(decrypted_update)),
            "aggregate_ckks_mse": reference_mse,
            "aggregate_maximum_absolute_error": reference_max_error,
            **geo_metrics,
        }
