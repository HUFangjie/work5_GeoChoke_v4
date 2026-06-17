from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np

from attacks.alie import ALIEAttack
from attacks.fang import FangMeanAttack
from attacks.no_attack import NoAttack
from config import CONFIG
from core.client import Client
from core.decryption_service import AuthorizedDecryptionService
from core.server import AggregationServer
from crypto.ckks_backend import CKKSBackend
from crypto.ckks_context_manager import CKKSContextManager
from crypto.update_codec import ModelUpdateCodec
from data.mnist import MNISTProvider
from defenses.geochoke.defense import GeoChokeDefense
from evaluation.evaluator import Evaluator
import models.mnist_cnn  # registers model
from models.registry import create_model
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.validation import model_l2_norm


def build_attack(cfg: Any) -> Any:
    if cfg.attack_type == "alie":
        return ALIEAttack(cfg.alie_z, oracle_all_updates=cfg.alie_oracle_all_updates)
    if cfg.attack_type == "fang_mean":
        return FangMeanAttack(cfg.aggregation, cfg.fang_max_norm, cfg.fang_search_steps)
    return NoAttack()


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(cfg: Any = CONFIG) -> list[dict[str, Any]]:
    set_seed(cfg.seed)
    logger = setup_logger(cfg.log_level)
    os.makedirs(cfg.output_dir, exist_ok=True)

    splits = MNISTProvider(cfg).build()
    _write_csv(os.path.join(cfg.output_dir, "client_partitions.csv"), splits.client_metadata)

    global_model = create_model(cfg.model_name)
    codec = ModelUpdateCodec(global_model)

    secret_context_manager = CKKSContextManager(cfg.ckks_profiles)
    secret_context_manager.initialize()
    decryption_service = AuthorizedDecryptionService(secret_context_manager)
    public_crypto_backend = CKKSBackend(secret_context_manager.public_bundles())
    public_crypto_backend.initialize_profiles()

    defense = GeoChokeDefense(cfg.geochoke, cfg.ckks_profiles, device=cfg.device)
    defense.initialize(global_model, public_crypto_backend, splits.proxy_loader, decryption_service.decrypt_aggregate)

    attack = build_attack(cfg)
    clients = [
        Client(
            client_id=client_id,
            loader=loader,
            model_factory=lambda: create_model(cfg.model_name),
            codec_factory=lambda model: ModelUpdateCodec(model),
            crypto_backend=public_crypto_backend,
            attack_strategy=attack,
            malicious=client_id in cfg.malicious_client_ids,
            cfg=cfg,
        )
        for client_id, loader in enumerate(splits.client_loaders)
    ]

    server = AggregationServer(
        cfg=cfg,
        model=global_model,
        codec=codec,
        crypto_backend=public_crypto_backend,
        defense=defense,
        model_factory=lambda: create_model(cfg.model_name),
    )
    evaluator = Evaluator(splits.test_loader, cfg.device)

    round_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    for round_id in range(cfg.num_rounds):
        profile_id = defense.get_profile_for_round(round_id)
        selected_client_ids = server.sample_clients(round_id)
        malicious_selected = [client_id for client_id in selected_client_ids if client_id in cfg.malicious_client_ids]
        logger.info("round=%s profile=%s selected=%s", round_id, profile_id, selected_client_ids)

        global_state = {name: tensor.detach().cpu().clone() for name, tensor in server.model.state_dict().items()}
        clean_records = [clients[client_id].compute_clean_update(global_state) for client_id in selected_client_ids]
        record_by_client = {record.client_id: record for record in clean_records}
        observable_updates = [record_by_client[client_id].clean_update for client_id in malicious_selected]
        if cfg.alie_oracle_all_updates:
            logger.warning("ALIE oracle_all_updates reproduction mode is enabled")
        total_samples = sum(record.num_samples for record in clean_records)
        uploads = []
        for client_id in selected_client_ids:
            record = record_by_client[client_id]
            malicious_weight = record.num_samples / total_samples if total_samples else 0.0
            attacker_context = {
                "num_selected": len(selected_client_ids),
                "num_malicious": len(malicious_selected),
                "observable_updates": observable_updates,
                "oracle_all_updates": [record.clean_update for record in clean_records],
                "malicious_weight": malicious_weight,
            }
            uploads.append(clients[client_id].encrypt_update(record, profile_id, attacker_context))

        decrypted_update, metrics = server.apply_round(uploads, round_id, decryption_service.decrypt_aggregate)
        test_loss, test_accuracy = evaluator.evaluate(server.model)
        profile_cfg = cfg.ckks_profiles[profile_id]
        round_row = {
            "round": round_id,
            "selected_clients": selected_client_ids,
            "malicious_selected_clients": malicious_selected,
            "train_loss": float(np.mean([upload.metadata["train_loss"] for upload in uploads])),
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "global_model_norm": model_l2_norm(server.model),
            "profile_id": profile_id,
            "poly_modulus_degree": profile_cfg["poly_modulus_degree"],
            "scale_bits": profile_cfg["global_scale_bits"],
            "coeff_modulus_bits": profile_cfg["coeff_mod_bit_sizes"],
            "encryption_time": float(sum(upload.metadata["encryption_time"] for upload in uploads)),
            "attack_type": cfg.attack_type,
            **metrics,
        }
        round_rows.append(round_row)
        for upload in uploads:
            attack_rows.append(
                {
                    "round": round_id,
                    "client_id": upload.client_id,
                    "attack_type": cfg.attack_type if upload.metadata["is_malicious"] else "none",
                    "is_malicious": upload.metadata["is_malicious"],
                    "malicious_update_norm_before": upload.metadata["malicious_update_norm_before"],
                    "malicious_update_norm_after": upload.metadata["malicious_update_norm_after"],
                    "cosine_similarity_before_after": upload.metadata["cosine_before_after"],
                    "attack_computation_time": upload.metadata["attack_time"],
                    "attack_applied_before_encryption": upload.metadata["attack_applied_before_encryption"],
                }
            )
        _write_csv(os.path.join(cfg.output_dir, f"round_{round_id}.csv"), [round_row])

    _write_csv(os.path.join(cfg.output_dir, "fl_ckks_geochoke_metrics.csv"), round_rows)
    _write_csv(os.path.join(cfg.output_dir, "attack_metrics.csv"), attack_rows)
    return round_rows
