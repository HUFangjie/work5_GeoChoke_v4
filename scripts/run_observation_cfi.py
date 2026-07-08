#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CKKS_PROFILES, make_config  # noqa: E402
from core.client import Client  # noqa: E402
from core.server import AggregationServer  # noqa: E402
from core.types import ClientUpload  # noqa: E402
from crypto.update_codec import ModelUpdateCodec  # noqa: E402
from defenses.geochoke.calibration import ProfileCalibrator  # noqa: E402
from defenses.geochoke.calibration_provider import CalibrationTensorProvider  # noqa: E402
from defenses.geochoke.cfi_estimator import CFIEstimator  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from factories.attack_factory import create_attack  # noqa: E402
from factories.crypto_factory import create_crypto_backend  # noqa: E402
from factories.dataset_factory import create_dataset_provider  # noqa: E402
from factories.model_factory import create_model_factory  # noqa: E402
from utils.seed import set_seed  # noqa: E402


_ATTACK_PREFIX = {
    "dba_multi": "dba",
    "neurotoxin": "neurotoxin",
    "three_dfed": "three_dfed",
    "a3fl": "a3fl",
}
_ATTACK_ACTIVE_KEYS = ("attack_applied_before_encryption", "dba_attack_active", "backdoor_attack_active")
_BANK_FILE = "observation_perturbation_bank.npz"
_META_FILE = "observation_calibration_meta.json"


class _NoCommitDefense:
    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id

    def get_profile_for_round(self, round_id: int) -> str:
        return self.profile_id

    def after_aggregate(self, previous_model: Any, candidate_model: Any, current_profile_id: str, round_id: int) -> dict[str, Any]:
        return {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _attack_overrides(args: argparse.Namespace, num_clients: int) -> dict[str, Any]:
    if args.attack == "none":
        return {"malicious_client_ids": []}
    prefix = _ATTACK_PREFIX[args.attack]
    requested_malicious = max(1, int(round(num_clients * args.malicious_fraction)))
    malicious_count = min(max(1, num_clients - 1), requested_malicious)
    malicious_ids = list(range(1, min(num_clients, malicious_count + 1)))
    overrides: dict[str, Any] = {
        "malicious_client_ids": malicious_ids,
        f"{prefix}_poison_ratio": args.poison_ratio,
        f"{prefix}_attack_start_round": args.attack_start_round,
        f"{prefix}_attack_end_round": args.attack_end_round,
    }
    if prefix == "dba":
        overrides["dba_num_trigger_parts"] = min(4, len(malicious_ids))
    return overrides


def build_observation_config(args: argparse.Namespace):
    num_clients = int(getattr(args, "num_clients", 10))
    dataset_overrides = {
        "partition_type": "dirichlet",
        "dirichlet_alpha": args.alpha,
        "download": bool(args.download_data),
        "quick_data_limit": args.quick_data_limit,
        "proxy_size": args.proxy_size,
        "test_size": args.test_size,
    }
    config_dataset = args.dataset if args.dataset in {"mnist", "fashion_mnist", "cifar10"} else "mnist"
    cfg = make_config(
        dataset=config_dataset,
        attack=args.attack,
        defense="none",
        scale="standard",
        seed=args.seed,
        output_dir=args.output_dir,
        dataset_overrides=dataset_overrides,
        training_overrides={
            "num_rounds": args.num_rounds,
            "num_clients": num_clients,
            "clients_per_round": int(getattr(args, "clients_per_round", num_clients)),
            "min_clients_per_round": int(getattr(args, "min_clients_per_round", min(2, num_clients))),
        },
        attack_overrides=_attack_overrides(args, num_clients),
        geochoke_overrides={
            "reference_profile_id": args.observation_reference_profile,
            "calibration_vectors": args.observation_calibration_vectors,
            "perturbation_count": args.observation_perturbation_count,
            "perturbation_scale": args.observation_perturbation_scale,
            "tangent_commitment_enabled": False,
        },
    )
    cfg.ckks_profiles = CKKS_PROFILES
    if config_dataset != args.dataset:
        cfg.dataset_name = args.dataset
    return cfg


def _save_perturbation_bank(path: Path, perturbation_bank: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"p{index}": perturbation for index, perturbation in enumerate(perturbation_bank)})


def _load_perturbation_bank(path: Path) -> list[np.ndarray]:
    payload = np.load(path)
    return [payload[key].astype(np.float64, copy=True) for key in sorted(payload.files, key=lambda item: int(item[1:]))]


def build_or_load_estimator(cfg: Any, model: Any, proxy_loader: Any, crypto_backend: Any, decryption_service: Any, calibration_dir: Path) -> CFIEstimator:
    codec = ModelUpdateCodec(model)
    bank_path = calibration_dir / _BANK_FILE
    meta_path = calibration_dir / _META_FILE
    if bank_path.exists():
        perturbation_bank = _load_perturbation_bank(bank_path)
    else:
        provider = CalibrationTensorProvider(codec, cfg.geochoke, cfg.device)
        representative_tensors = provider.build(model, proxy_loader)
        calibrator = ProfileCalibrator(
            crypto_backend,
            decryption_service.decrypt_for_offline_calibration,
            cfg.ckks_profiles,
            cfg.geochoke,
            output_dir=str(calibration_dir),
        )
        _calibration, perturbation_bank = calibrator.calibrate(representative_tensors)
        _save_perturbation_bank(bank_path, perturbation_bank)
        meta_path.write_text(
            json.dumps(
                {
                    "reference_profile_id": cfg.geochoke.reference_profile_id,
                    "calibration_vectors": cfg.geochoke.calibration_vectors,
                    "perturbation_count": cfg.geochoke.perturbation_count,
                    "perturbation_scale": cfg.geochoke.perturbation_scale,
                },
                indent=2,
            )
        )
    return CFIEstimator(codec, proxy_loader, perturbation_bank, cfg.device)


def _make_clients(cfg: Any, splits: Any, model_factory: Any, crypto_backend: Any, attack: Any) -> list[Client]:
    return [
        Client(
            client_id=client_id,
            loader=loader,
            model_factory=model_factory.create,
            codec_factory=lambda model: ModelUpdateCodec(model),
            crypto_backend=crypto_backend,
            attack_strategy=attack,
            malicious=client_id in cfg.malicious_client_ids,
            cfg=cfg,
        )
        for client_id, loader in enumerate(splits.client_loaders)
    ]


def _upload_attack_active(upload: ClientUpload) -> bool:
    return any(bool(upload.metadata.get(key, False)) for key in _ATTACK_ACTIVE_KEYS) or upload.metadata.get("update_type") == "poisoned"


def _activity_metrics(uploads: list[ClientUpload], selected_client_ids: list[int]) -> dict[str, Any]:
    active_malicious_clients = [upload.client_id for upload in uploads if _upload_attack_active(upload)]
    poisoned_sample_count = int(sum(upload.metadata.get("poisoned_sample_count", 0) or 0 for upload in uploads))
    dba_seen_count = int(sum(upload.metadata.get("dba_seen_sample_count", 0) or 0 for upload in uploads))
    effective_values = [
        float(upload.metadata["effective_poison_ratio"])
        for upload in uploads
        if _upload_attack_active(upload) and upload.metadata.get("effective_poison_ratio") is not None
    ]
    if dba_seen_count > 0:
        effective_poison_ratio = poisoned_sample_count / dba_seen_count
    elif effective_values:
        effective_poison_ratio = float(np.mean(effective_values))
    else:
        effective_poison_ratio = None
    return {
        "attack_active": bool(active_malicious_clients),
        "active_malicious_clients": active_malicious_clients,
        "num_malicious_selected": len(active_malicious_clients),
        "malicious_selected_ratio": len(active_malicious_clients) / max(1, len(selected_client_ids)),
        "poisoned_sample_count": poisoned_sample_count,
        "effective_poison_ratio": effective_poison_ratio,
    }


def _build_uploads(clients: list[Client], records_by_client: dict[int, Any], selected_client_ids: list[int], cfg: Any, profile_id: str) -> list[ClientUpload]:
    observable_updates = [records_by_client[client_id].local_update for client_id in selected_client_ids if client_id not in cfg.malicious_client_ids]
    if not observable_updates:
        observable_updates = [records_by_client[client_id].local_update for client_id in selected_client_ids]
    benign_norm_mean = float(np.mean([np.linalg.norm(update) for update in observable_updates])) if observable_updates else 0.0
    total_samples = sum(records_by_client[client_id].num_samples for client_id in selected_client_ids)
    uploads = []
    for client_id in selected_client_ids:
        record = records_by_client[client_id]
        aggregation_weight = record.num_samples / total_samples if total_samples else 0.0
        attacker_context = {
            "num_selected": len(selected_client_ids),
            "num_malicious": sum(1 for selected_id in selected_client_ids if selected_id in cfg.malicious_client_ids),
            "observable_updates": observable_updates,
            "oracle_all_updates": [records_by_client[selected_id].local_update for selected_id in selected_client_ids],
            "malicious_weight": aggregation_weight,
            "benign_selected_update_norm_mean": benign_norm_mean,
        }
        uploads.append(clients[client_id].encrypt_update(record, profile_id, attacker_context))
    return uploads


def run_observation(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.observation_cfi:
        raise ValueError("run_observation_cfi.py is an observation entrypoint; pass --observation_cfi to enable logging")
    if args.attack != "none" and not 0 <= args.attack_start_round <= args.attack_end_round < args.num_rounds:
        raise ValueError("attack schedule must satisfy 0 <= attack_start_round <= attack_end_round < num_rounds")
    if not 0.0 < args.malicious_fraction <= 1.0:
        raise ValueError("--malicious_fraction must satisfy 0.0 < malicious_fraction <= 1.0")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_observation_config(args)
    dataset_provider = create_dataset_provider(cfg)
    splits = dataset_provider.build()
    model_factory = create_model_factory(cfg)
    global_model = model_factory.create()
    codec = ModelUpdateCodec(global_model)
    crypto_bundle = create_crypto_backend(cfg)
    estimator = build_or_load_estimator(
        cfg,
        global_model,
        splits.proxy_loader,
        crypto_bundle.public_backend,
        crypto_bundle.decryption_service,
        Path(args.observation_calibration_dir),
    )
    evaluator = Evaluator(splits.test_loader, cfg.device)
    attack = create_attack(cfg)
    clients = _make_clients(cfg, splits, model_factory, crypto_bundle.public_backend, attack)
    server = AggregationServer(cfg, global_model, codec, crypto_bundle.public_backend, _NoCommitDefense(cfg.geochoke.initial_profile_id), model_factory.create)
    rows: list[dict[str, Any]] = []
    for round_id in range(cfg.num_rounds):
        profile_id = cfg.geochoke.initial_profile_id
        selected_client_ids = server.sample_clients(round_id)
        global_state = {name: tensor.detach().cpu().clone() for name, tensor in server.model.state_dict().items()}
        records = [clients[client_id].compute_local_update(global_state, round_id) for client_id in selected_client_ids]
        records_by_client = {record.client_id: record for record in records}
        uploads = _build_uploads(clients, records_by_client, selected_client_ids, cfg, profile_id)
        aggregate_ciphertext, _aggregation_time, _weights = server.aggregate_encrypted(uploads)
        decrypted_update = crypto_bundle.decryption_service.decrypt_aggregate(aggregate_ciphertext, profile_id)
        candidate_state = codec.apply_update_to_state_dict(global_state, decrypted_update, step_size=cfg.server_lr)
        candidate_model = model_factory.create()
        candidate_model.load_state_dict(candidate_state)

        candidate_cfi = estimator.estimate(candidate_model)
        test_loss, clean_test_accuracy = evaluator.evaluate(candidate_model)
        backdoor_metrics = evaluator.evaluate_backdoor(candidate_model, attack, cfg, profile_id, uploads)
        activity_metrics = _activity_metrics(uploads, selected_client_ids)
        row = {
            "dataset": cfg.dataset_name,
            "attack": cfg.attack_name,
            "seed": cfg.seed,
            "round": round_id,
            "state_type": args.observation_state_type,
            "attack_start_round": args.attack_start_round,
            "attack_end_round": args.attack_end_round,
            "alpha": args.alpha,
            "poison_ratio": args.poison_ratio,
            "malicious_fraction": args.malicious_fraction,
            "num_clients": cfg.num_clients,
            "clients_per_round": cfg.clients_per_round,
            "current_ckks_profile": profile_id,
            "reference_profile_id": cfg.geochoke.reference_profile_id,
            "perturbation_count": cfg.geochoke.perturbation_count,
            "perturbation_scale": cfg.geochoke.perturbation_scale,
            "candidate_cfi": candidate_cfi,
            "cfi": candidate_cfi,
            "clean_test_accuracy": clean_test_accuracy,
            "test_loss": test_loss,
            "global_trigger_asr": backdoor_metrics.get("global_trigger_asr"),
            "global_trigger_raw_asr": backdoor_metrics.get("global_trigger_raw_asr"),
            "global_clean_target_rate": backdoor_metrics.get("global_clean_target_rate"),
            "num_selected": len(selected_client_ids),
            **activity_metrics,
        }
        for key, value in backdoor_metrics.items():
            row.setdefault(key, value)
        rows.append(row)
        print(
            "observation_cfi "
            f"round={round_id} state_type={args.observation_state_type} "
            f"candidate_cfi={candidate_cfi:.8g} clean_test_accuracy={clean_test_accuracy:.6f} "
            f"global_trigger_asr={row['global_trigger_asr'] if row['global_trigger_asr'] is not None else 'NaN'} "
            f"attack_active={row['attack_active']}",
            flush=True,
        )
        server.model.load_state_dict(candidate_state)
        _write_csv(output_dir / args.observation_output_name, rows)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log pre-commit candidate CFI for one encrypted FedAvg/attack run.")
    parser.add_argument("--observation_cfi", action="store_true")
    parser.add_argument("--observation_state_type", choices=["benign", "poisoned"], default="benign")
    parser.add_argument("--observation_output_name", default="observation_cfi_curve.csv")
    parser.add_argument("--observation_calibration_dir", default="./outputs_observation/shared_calibration")
    parser.add_argument("--observation_reference_profile", default="ckks_s28")
    parser.add_argument("--observation_calibration_vectors", type=int, default=16)
    parser.add_argument("--observation_perturbation_count", type=int, default=16)
    parser.add_argument("--observation_perturbation_scale", type=float, default=1.0)
    parser.add_argument("--dataset", default="fashion_mnist", help="Dataset: fashion_mnist / cifar10 / mnist")
    parser.add_argument("--attack", default="none", choices=["none", *_ATTACK_PREFIX], help="Current run attack")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--poison_ratio", type=float, default=0.3)
    parser.add_argument("--malicious_fraction", type=float, default=0.2)
    parser.add_argument("--attack_start_round", type=int, default=20)
    parser.add_argument("--attack_end_round", type=int, default=49)
    parser.add_argument("--output_dir", default="./outputs_observation/manual/fashion_mnist/benign/seed7")
    parser.add_argument("--num_clients", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--clients_per_round", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--min_clients_per_round", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--quick_data_limit", type=int, default=6000, help=argparse.SUPPRESS)
    parser.add_argument("--proxy_size", type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument("--test_size", type=int, default=2000, help=argparse.SUPPRESS)
    parser.add_argument("--no_download_data", action="store_false", dest="download_data", default=True, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    rows = run_observation(parse_args(argv))
    print(f"wrote {len(rows)} observation CFI rows")


if __name__ == "__main__":
    main()
