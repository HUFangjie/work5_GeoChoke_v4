from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np

from core.client import Client
from core.server import AggregationServer
from crypto.aggregation_pipeline_validator import AggregationPipelineValidator
from crypto.update_codec import ModelUpdateCodec
from evaluation.evaluator import Evaluator
from utils.validation import model_l2_norm


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class FederatedCoordinator:
    """Federated-learning orchestration over injected abstract components."""

    def __init__(
        self,
        cfg: Any,
        dataset_provider: Any,
        model_factory: Any,
        crypto_backend: Any,
        decryption_service: Any,
        defense: Any,
        attack: Any,
        logger: Any,
    ) -> None:
        self.cfg = cfg
        self.dataset_provider = dataset_provider
        self.model_factory = model_factory
        self.crypto_backend = crypto_backend
        self.decryption_service = decryption_service
        self.defense = defense
        self.attack = attack
        self.logger = logger

    def run(self) -> list[dict[str, Any]]:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        splits = self.dataset_provider.build()
        _write_csv(os.path.join(self.cfg.output_dir, "client_partitions.csv"), splits.client_metadata)

        global_model = self.model_factory.create()
        codec = ModelUpdateCodec(global_model)
        calibration_dir = os.path.join(self.cfg.output_dir, "crypto_calibration")
        self.defense.initialize(
            global_model,
            self.crypto_backend,
            splits.proxy_loader,
            self.decryption_service.decrypt_for_offline_calibration,
        )
        self._validate_crypto_pipeline(codec.total_dimension, calibration_dir)

        clients = [
            Client(
                client_id=client_id,
                loader=loader,
                model_factory=self.model_factory.create,
                codec_factory=lambda model: ModelUpdateCodec(model),
                crypto_backend=self.crypto_backend,
                attack_strategy=self.attack,
                malicious=client_id in self.cfg.malicious_client_ids,
                cfg=self.cfg,
            )
            for client_id, loader in enumerate(splits.client_loaders)
        ]
        server = AggregationServer(
            cfg=self.cfg,
            model=global_model,
            codec=codec,
            crypto_backend=self.crypto_backend,
            defense=self.defense,
            model_factory=self.model_factory.create,
        )
        evaluator = Evaluator(splits.test_loader, self.cfg.device)
        round_rows: list[dict[str, Any]] = []
        attack_rows: list[dict[str, Any]] = []
        for round_id in range(self.cfg.num_rounds):
            profile_id = self.defense.get_profile_for_round(round_id)
            selected_client_ids = server.sample_clients(round_id)
            malicious_selected = [client_id for client_id in selected_client_ids if client_id in self.cfg.malicious_client_ids]
            self.logger.info("round=%s profile=%s selected=%s", round_id, profile_id, selected_client_ids)
            global_state = {name: tensor.detach().cpu().clone() for name, tensor in server.model.state_dict().items()}
            local_records = [clients[client_id].compute_local_update(global_state, round_id) for client_id in selected_client_ids]
            record_by_client = {record.client_id: record for record in local_records}
            observable_updates = [record_by_client[client_id].local_update for client_id in selected_client_ids if client_id not in self.cfg.malicious_client_ids]
            if not observable_updates:
                observable_updates = [record_by_client[client_id].local_update for client_id in selected_client_ids]
            benign_selected_update_norm_mean = float(np.mean([np.linalg.norm(update) for update in observable_updates])) if observable_updates else 0.0
            if self.cfg.alie_oracle_all_updates:
                self.logger.warning("ALIE oracle_all_updates reproduction mode is enabled")
            total_samples = sum(record.num_samples for record in local_records)
            uploads = []
            plaintext_updates_for_metrics = []
            plaintext_weights_for_metrics = []
            for client_id in selected_client_ids:
                record = record_by_client[client_id]
                aggregation_weight = record.num_samples / total_samples if total_samples else 0.0
                attacker_context = {
                    "num_selected": len(selected_client_ids),
                    "num_malicious": len(malicious_selected),
                    "observable_updates": observable_updates,
                    "oracle_all_updates": [record.local_update for record in local_records],
                    "malicious_weight": aggregation_weight,
                    "benign_selected_update_norm_mean": benign_selected_update_norm_mean,
                }
                upload = clients[client_id].encrypt_update(record, profile_id, attacker_context)
                uploads.append(upload)
                if self.cfg.enable_plaintext_reference_metrics:
                    if clients[client_id].malicious:
                        reference_update = clients[client_id].attack_strategy.craft_update(
                            client_id,
                            record.local_update.copy(),
                            None,
                            attacker_context,
                        )
                    else:
                        reference_update = record.local_update.copy()
                    plaintext_updates_for_metrics.append(reference_update)
                    plaintext_weights_for_metrics.append(aggregation_weight)
            plaintext_reference_update = None
            if self.cfg.enable_plaintext_reference_metrics and plaintext_updates_for_metrics:
                plaintext_reference_update = np.zeros_like(plaintext_updates_for_metrics[0])
                for reference_update, reference_weight in zip(plaintext_updates_for_metrics, plaintext_weights_for_metrics):
                    plaintext_reference_update += reference_weight * reference_update
            decrypted_update, metrics = server.apply_round(
                uploads,
                round_id,
                self.decryption_service.decrypt_aggregate,
                plaintext_reference_update=plaintext_reference_update,
            )
            test_loss, test_accuracy = evaluator.evaluate(server.model)
            dba_metrics = evaluator.evaluate_dba(server.model, self.attack, self.cfg, profile_id, uploads)
            self.logger.info(
                "round=%s clean_test_accuracy=%.6f test_loss=%.6f global_trigger_asr=%s "
                "local_trigger_1_asr=%s local_trigger_2_asr=%s local_trigger_3_asr=%s local_trigger_4_asr=%s "
                "attack_active=%s active_malicious_clients=%s poisoned_sample_count=%s effective_poison_ratio=%s current_ckks_profile=%s",
                round_id,
                test_accuracy,
                test_loss,
                dba_metrics.get("global_trigger_asr"),
                dba_metrics.get("local_trigger_1_asr"),
                dba_metrics.get("local_trigger_2_asr"),
                dba_metrics.get("local_trigger_3_asr"),
                dba_metrics.get("local_trigger_4_asr"),
                dba_metrics.get("attack_active", False),
                dba_metrics.get("active_malicious_clients", []),
                dba_metrics.get("poisoned_sample_count", 0),
                dba_metrics.get("effective_poison_ratio", 0.0),
                profile_id,
            )
            profile_cfg = self.cfg.ckks_profiles[profile_id]
            round_row = {
                "round": round_id,
                "selected_clients": selected_client_ids,
                "malicious_selected_clients": malicious_selected,
                "train_loss": float(np.mean([upload.metadata["train_loss"] for upload in uploads])),
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "clean_test_accuracy": test_accuracy,
                "global_model_norm": model_l2_norm(server.model),
                "profile_id": profile_id,
                "poly_modulus_degree": profile_cfg["poly_modulus_degree"],
                "scale_bits": profile_cfg["global_scale_bits"],
                "coeff_modulus_bits": profile_cfg["coeff_mod_bit_sizes"],
                "encryption_time": float(sum(upload.metadata["encryption_time"] for upload in uploads)),
                "attack_name": self.cfg.attack_name,
                "aggregation_pipeline_residual_mse": metrics.get("aggregate_ckks_mse"),
                "aggregation_pipeline_residual_max_abs": metrics.get("aggregate_maximum_absolute_error"),
                **metrics,
                **dba_metrics,
            }
            round_rows.append(round_row)
            for upload in uploads:
                attack_rows.append(
                    {
                        "round": round_id,
                        "client_id": upload.client_id,
                        "attack_name": self.cfg.attack_name if (upload.metadata["is_malicious"] and (self.cfg.attack_name != "dba" or upload.metadata.get("dba_attack_active", False))) else "none",
                        "is_malicious": upload.metadata["is_malicious"],
                        "malicious_update_norm_before": upload.metadata["malicious_update_norm_before"],
                        "malicious_update_norm_after": upload.metadata["malicious_update_norm_after"],
                        "cosine_similarity_before_after": upload.metadata["cosine_before_after"],
                        "attack_computation_time": upload.metadata["attack_time"],
                        "attack_applied_before_encryption": upload.metadata["attack_applied_before_encryption"],
                        "update_type": upload.metadata.get("update_type"),
                        "dba_attack_active": upload.metadata.get("dba_attack_active", False),
                        "dba_trigger_id": upload.metadata.get("dba_trigger_id"),
                        "poisoned_sample_count": upload.metadata.get("poisoned_sample_count", 0),
                        "dba_seen_sample_count": upload.metadata.get("dba_seen_sample_count", 0),
                        "effective_poison_ratio": upload.metadata.get("effective_poison_ratio", 0.0),
                        "dba_scale_factor": upload.metadata.get("dba_scale_factor", 0.0),
                        "update_norm_before_scale": upload.metadata.get("update_norm_before_scale"),
                        "update_norm_after_scale": upload.metadata.get("update_norm_after_scale"),
                        "poisoned_update_norm": upload.metadata.get("poisoned_update_norm"),
                        "benign_selected_update_norm_mean": upload.metadata.get("benign_selected_update_norm_mean"),
                        "poisoned_to_benign_norm_ratio": upload.metadata.get("poisoned_to_benign_norm_ratio"),
                    }
                )
            _write_csv(os.path.join(self.cfg.output_dir, "fl_ckks_geochoke_metrics.csv"), round_rows)
            _write_csv(os.path.join(self.cfg.output_dir, "attack_metrics.csv"), attack_rows)
        _write_csv(os.path.join(self.cfg.output_dir, "fl_ckks_geochoke_metrics.csv"), round_rows)
        _write_csv(os.path.join(self.cfg.output_dir, "attack_metrics.csv"), attack_rows)
        return round_rows

    def _validate_crypto_pipeline(self, dimension: int, output_dir: str) -> None:
        validator = AggregationPipelineValidator(
            self.crypto_backend,
            self.decryption_service.decrypt_for_offline_calibration,
            rtol=self.cfg.pipeline_validation_rtol,
            atol=self.cfg.pipeline_validation_atol,
            norm_ratio_tolerance=self.cfg.pipeline_validation_norm_ratio_tolerance,
        )
        base = np.linspace(-0.25, 0.25, min(dimension, self.crypto_backend.slot_count(next(iter(self.cfg.ckks_profiles)))), dtype=np.float64)
        if base.size == 0:
            base = np.array([0.0], dtype=np.float64)
        vectors = [base, -0.5 * base + 0.01, 0.25 * base - 0.02]
        weights = [0.2, 0.3, 0.5]
        validator.validate_profiles(self.cfg.ckks_profiles.keys(), vectors, weights, output_dir=output_dir)
