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
        defense_strategy: Any | None = None,
    ) -> None:
        self.client_id = client_id
        self.loader = loader
        self.model_factory = model_factory
        self.codec_factory = codec_factory
        self.crypto_backend = crypto_backend
        self.attack_strategy = attack_strategy
        self.malicious = malicious
        self.cfg = cfg
        self.defense_strategy = defense_strategy

    def compute_local_update(self, global_state: Dict[str, Any], round_id: int | None = None) -> LocalUpdateRecord:
        if (
            self.malicious
            and round_id is not None
            and hasattr(self.attack_strategy, "should_poison")
            and self.attack_strategy.should_poison(self.client_id, round_id)
        ):
            return self.attack_strategy.train_local_update(
                self.client_id,
                self.loader,
                self.model_factory,
                self.codec_factory,
                global_state,
                round_id,
            )
        local_model = self.model_factory()
        local_model.load_state_dict(copy.deepcopy(global_state))
        trainer = LocalTrainer(self.cfg.local_epochs, self.cfg.local_lr, self.cfg.device)
        train_loss = trainer.train(local_model, self.loader)
        codec = self.codec_factory(local_model)
        local_flat = codec.flatten_state_dict(local_model.state_dict())
        global_flat = codec.flatten_state_dict(global_state)
        local_update = local_flat - global_flat
        return LocalUpdateRecord(
            client_id=self.client_id,
            num_samples=len(self.loader.dataset),
            local_update=local_update,
            train_loss=float(train_loss),
            metadata={"update_type": "benign", "dba_attack_active": False},
        )


    def compute_clean_update(self, global_state: Dict[str, Any], round_id: int | None = None) -> LocalUpdateRecord:
        """Backward-compatible wrapper; records now carry local_update."""
        return self.compute_local_update(global_state, round_id)

    def encrypt_update(
        self,
        update_record: LocalUpdateRecord,
        profile_id: str,
        attacker_context: Dict[str, Any],
        round_id: int | None = None,
    ) -> ClientUpload:
        before = update_record.local_update.copy()
        start_time = time.perf_counter()
        if self.malicious:
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
        defense_metadata = {}
        prepared_update = final_update
        if self.defense_strategy is not None and hasattr(self.defense_strategy, "prepare_client_upload"):
            prepared_update, defense_metadata = self.defense_strategy.prepare_client_upload(
                client_id=self.client_id,
                update_vector=final_update,
                round_id=int(round_id or 0),
                profile_id=profile_id,
                num_samples=update_record.num_samples,
                metadata={**update_record.metadata, "num_selected": attacker_context.get("num_selected")},
            )
        encrypted = self.crypto_backend.encrypt_update(prepared_update, profile_id)
        encryption_time = self.crypto_backend.last_encryption_time
        before_norm = float(np.linalg.norm(before))
        after_norm = float(np.linalg.norm(final_update))
        cosine = float(np.dot(before, final_update) / (before_norm * after_norm + 1e-12))
        benign_norm_mean = float(attacker_context.get("benign_selected_update_norm_mean", 0.0))
        dba_extra = {}
        if update_record.metadata.get("dba_attack_active", False):
            poisoned_norm = float(update_record.metadata.get("poisoned_update_norm", after_norm))
            dba_extra = {
                "benign_selected_update_norm_mean": benign_norm_mean,
                "poisoned_to_benign_norm_ratio": poisoned_norm / max(benign_norm_mean, 1e-12),
            }
        metadata = {
            "train_loss": update_record.train_loss,
            "attack_time": attack_time,
            "encryption_time": encryption_time,
            "malicious_update_norm_before": before_norm,
            "malicious_update_norm_after": after_norm,
            "cosine_before_after": cosine,
            "attack_applied_before_encryption": bool(update_record.metadata.get("dba_attack_active", False) or (self.malicious and self.cfg.attack_name != "dba")),
            "is_malicious": bool(self.malicious),
            **update_record.metadata,
            **dba_extra,
            **defense_metadata,
        }
        return ClientUpload(
            client_id=self.client_id,
            num_samples=update_record.num_samples,
            profile_id=profile_id,
            encrypted_update=encrypted,
            metadata=metadata,
        )
