from __future__ import annotations

import copy
import time
from typing import Any, Dict

import numpy as np

from core.local_trainer import LocalTrainer
from core.types import ClientUpload, LocalUpdateRecord


class Client:
    """Federated client with local data, public CKKS backend, and optional attack module."""

    def __init__(
        self,
        client_id: int,
        loader: Any,
        model_factory: Any,
        codec_factory: Any,
        crypto_backend: Any,
        attack_strategy: Any,
        malicious: bool,
        cfg: Any,
    ) -> None:
        self.client_id = client_id
        self.loader = loader
        self.model_factory = model_factory
        self.codec_factory = codec_factory
        self.crypto_backend = crypto_backend
        self.attack_strategy = attack_strategy
        self.malicious = malicious
        self.cfg = cfg

    def compute_clean_update(self, global_state: Dict[str, Any]) -> LocalUpdateRecord:
        local_model = self.model_factory()
        local_model.load_state_dict(copy.deepcopy(global_state))
        trainer = LocalTrainer(self.cfg.local_epochs, self.cfg.local_lr, self.cfg.device)
        train_loss = trainer.train(local_model, self.loader)
        codec = self.codec_factory(local_model)
        local_flat = codec.flatten_state_dict(local_model.state_dict())
        global_flat = codec.flatten_state_dict(global_state)
        clean_update = local_flat - global_flat
        return LocalUpdateRecord(
            client_id=self.client_id,
            num_samples=len(self.loader.dataset),
            clean_update=clean_update,
            train_loss=float(train_loss),
        )

    def encrypt_update(
        self,
        clean_record: LocalUpdateRecord,
        profile_id: str,
        attacker_context: Dict[str, Any],
    ) -> ClientUpload:
        before = clean_record.clean_update.copy()
        start_time = time.perf_counter()
        attack_applied = bool(self.malicious and attacker_context.get("attack_enabled", True))
        if attack_applied:
            final_update = self.attack_strategy.craft_update(
                self.client_id,
                before,
                None,
                attacker_context,
            )
        else:
            final_update = before
        attack_time = time.perf_counter() - start_time
        if not np.all(np.isfinite(final_update)):
            raise ValueError(f"client {self.client_id} produced a non-finite update")
        encrypted = self.crypto_backend.encrypt_update(final_update, profile_id)
        encryption_time = self.crypto_backend.last_encryption_time
        before_norm = float(np.linalg.norm(before))
        after_norm = float(np.linalg.norm(final_update))
        cosine = float(np.dot(before, final_update) / (before_norm * after_norm + 1e-12))
        metadata = {
            "train_loss": clean_record.train_loss,
            "attack_time": attack_time,
            "encryption_time": encryption_time,
            "malicious_update_norm_before": before_norm,
            "malicious_update_norm_after": after_norm,
            "cosine_before_after": cosine,
            "attack_applied_before_encryption": attack_applied,
            "is_malicious": bool(self.malicious),
        }
        return ClientUpload(
            client_id=self.client_id,
            num_samples=clean_record.num_samples,
            profile_id=profile_id,
            encrypted_update=encrypted,
            metadata=metadata,
        )
