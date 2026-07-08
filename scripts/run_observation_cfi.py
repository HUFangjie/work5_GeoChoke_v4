#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
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
from crypto.update_codec import ModelUpdateCodec  # noqa: E402
from defenses.geochoke.calibration import ProfileCalibrator  # noqa: E402
from defenses.geochoke.calibration_provider import CalibrationTensorProvider  # noqa: E402
from defenses.geochoke.cfi_estimator import CFIEstimator  # noqa: E402
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


class _PreCommitNoDefense:
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


def _attack_overrides(attack: str, num_rounds: int, poison_ratio: float, num_clients: int) -> dict[str, Any]:
    if attack not in _ATTACK_PREFIX:
        raise ValueError(f"Unsupported observation attack: {attack}")
    prefix = _ATTACK_PREFIX[attack]
    malicious_count = 4 if prefix in {"dba", "three_dfed"} else 2
    malicious_ids = list(range(1, min(num_clients, malicious_count + 1)))
    overrides: dict[str, Any] = {
        "malicious_client_ids": malicious_ids,
        f"{prefix}_poison_ratio": poison_ratio,
        f"{prefix}_attack_start_round": 0,
        f"{prefix}_attack_end_round": max(0, num_rounds - 1),
    }
    if prefix == "dba":
        overrides["dba_num_trigger_parts"] = min(4, len(malicious_ids))
    return overrides


def build_observation_config(args: argparse.Namespace, attack_name: str):
    num_clients = int(getattr(args, "num_clients", 10))
    attack_overrides = _attack_overrides(args.attack, args.num_rounds, args.poison_ratio, num_clients) if attack_name != "none" else {"malicious_client_ids": []}
    dataset_overrides = {
        "partition_type": "dirichlet",
        "dirichlet_alpha": args.alpha,
        "download": bool(args.download_data),
    }
    for key in ["quick_data_limit", "proxy_size", "test_size"]:
        value = getattr(args, key, None)
        if value is not None:
            dataset_overrides[key] = value
    config_dataset = args.dataset if args.dataset in {"mnist", "fashion_mnist", "cifar10"} else "mnist"
    cfg = make_config(
        dataset=config_dataset,
        attack=attack_name,
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
        attack_overrides=attack_overrides,
        geochoke_overrides={
            "reference_profile_id": "ckks_s28",
            "calibration_vectors": int(getattr(args, "calibration_vectors", 16)),
            "perturbation_count": int(getattr(args, "perturbation_count", 16)),
            "perturbation_scale": float(getattr(args, "perturbation_scale", 1.0)),
            "tangent_commitment_enabled": False,
        },
    )
    cfg.ckks_profiles = CKKS_PROFILES
    if config_dataset != args.dataset:
        cfg.dataset_name = args.dataset
    return cfg


def _make_clients(cfg: Any, splits: Any, model_factory: Any, codec_factory: Any, crypto_backend: Any, attack: Any) -> list[Client]:
    return [
        Client(
            client_id=client_id,
            loader=loader,
            model_factory=model_factory.create,
            codec_factory=codec_factory,
            crypto_backend=crypto_backend,
            attack_strategy=attack,
            malicious=client_id in cfg.malicious_client_ids,
            cfg=cfg,
        )
        for client_id, loader in enumerate(splits.client_loaders)
    ]


def _run_state_type(
    cfg: Any,
    splits: Any,
    model_factory: Any,
    crypto_backend: Any,
    decryption_service: Any,
    estimator: CFIEstimator,
    initial_state: dict[str, Any],
    state_type: str,
    requested_attack: str,
    alpha: float,
    poison_ratio: float,
) -> list[dict[str, Any]]:
    attack = create_attack(cfg)
    global_model = model_factory.create()
    global_model.load_state_dict(initial_state)
    codec = ModelUpdateCodec(global_model)
    clients = _make_clients(cfg, splits, model_factory, lambda model: ModelUpdateCodec(model), crypto_backend, attack)
    server = AggregationServer(cfg, global_model, codec, crypto_backend, _PreCommitNoDefense(cfg.geochoke.initial_profile_id), model_factory.create)
    rows: list[dict[str, Any]] = []
    profile_id = cfg.geochoke.initial_profile_id
    for round_id in range(cfg.num_rounds):
        selected_client_ids = server.sample_clients(round_id)
        malicious_selected = [client_id for client_id in selected_client_ids if client_id in cfg.malicious_client_ids]
        global_state = {name: tensor.detach().cpu().clone() for name, tensor in server.model.state_dict().items()}
        local_records = [clients[client_id].compute_local_update(global_state, round_id) for client_id in selected_client_ids]
        record_by_client = {record.client_id: record for record in local_records}
        observable_updates = [record_by_client[client_id].local_update for client_id in selected_client_ids if client_id not in cfg.malicious_client_ids]
        if not observable_updates:
            observable_updates = [record_by_client[client_id].local_update for client_id in selected_client_ids]
        benign_norm_mean = float(np.mean([np.linalg.norm(update) for update in observable_updates])) if observable_updates else 0.0
        total_samples = sum(record.num_samples for record in local_records)
        uploads = []
        for client_id in selected_client_ids:
            record = record_by_client[client_id]
            aggregation_weight = record.num_samples / total_samples if total_samples else 0.0
            attacker_context = {
                "num_selected": len(selected_client_ids),
                "num_malicious": len(malicious_selected),
                "observable_updates": observable_updates,
                "oracle_all_updates": [record.local_update for record in local_records],
                "malicious_weight": aggregation_weight,
                "benign_selected_update_norm_mean": benign_norm_mean,
            }
            uploads.append(clients[client_id].encrypt_update(record, profile_id, attacker_context))
        aggregate_ciphertext, _aggregation_time, _weights = server.aggregate_encrypted(uploads)
        decrypted_update = decryption_service.decrypt_aggregate(aggregate_ciphertext, profile_id)
        candidate_state = codec.apply_update_to_state_dict(global_state, decrypted_update, step_size=cfg.server_lr)
        candidate_model = model_factory.create()
        candidate_model.load_state_dict(candidate_state)
        cfi = estimator.estimate(candidate_model)
        rows.append(
            {
                "dataset": cfg.dataset_name,
                "attack": requested_attack,
                "seed": cfg.seed,
                "round": round_id,
                "state_type": state_type,
                "alpha": alpha,
                "poison_ratio": poison_ratio,
                "cfi": cfi,
                "reference_profile_id": cfg.geochoke.reference_profile_id,
                "perturbation_count": cfg.geochoke.perturbation_count,
                "perturbation_scale": cfg.geochoke.perturbation_scale,
            }
        )
        server.model.load_state_dict(candidate_state)
    return rows


def _auroc(benign: np.ndarray, poisoned: np.ndarray) -> float:
    scores = np.concatenate([benign, poisoned])
    labels = np.concatenate([np.zeros_like(benign), np.ones_like(poisoned)])
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    n_pos = float(np.sum(labels == 1))
    n_neg = float(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pos_rank_sum = float(np.sum(ranks[labels == 1]))
    return (pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def _ks_statistic(benign: np.ndarray, poisoned: np.ndarray) -> float:
    values = np.sort(np.unique(np.concatenate([benign, poisoned])))
    if values.size == 0:
        return float("nan")
    benign_sorted = np.sort(benign)
    poisoned_sorted = np.sort(poisoned)
    b_cdf = np.searchsorted(benign_sorted, values, side="right") / max(1, benign_sorted.size)
    p_cdf = np.searchsorted(poisoned_sorted, values, side="right") / max(1, poisoned_sorted.size)
    return float(np.max(np.abs(b_cdf - p_cdf)))


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    benign = np.array([float(row["cfi"]) for row in samples if row["state_type"] == "benign"], dtype=float)
    poisoned = np.array([float(row["cfi"]) for row in samples if row["state_type"] == "poisoned"], dtype=float)
    benign_mean = float(np.mean(benign)) if benign.size else float("nan")
    poisoned_mean = float(np.mean(poisoned)) if poisoned.size else float("nan")
    benign_std = float(np.std(benign, ddof=1)) if benign.size > 1 else 0.0
    poisoned_std = float(np.std(poisoned, ddof=1)) if poisoned.size > 1 else 0.0
    pooled = math.sqrt(((benign.size - 1) * benign_std**2 + (poisoned.size - 1) * poisoned_std**2) / max(1, benign.size + poisoned.size - 2))
    first = samples[0]
    return {
        "dataset": first["dataset"],
        "attack": first["attack"],
        "seed": first["seed"],
        "benign_mean_cfi": benign_mean,
        "benign_std_cfi": benign_std,
        "poisoned_mean_cfi": poisoned_mean,
        "poisoned_std_cfi": poisoned_std,
        "cfi_gap": poisoned_mean - benign_mean,
        "cfi_ratio": poisoned_mean / max(benign_mean, 1e-12),
        "cohen_d": (poisoned_mean - benign_mean) / max(pooled, 1e-12),
        "auroc": _auroc(benign, poisoned),
        "ks_statistic": _ks_statistic(benign, poisoned),
    }


def plot_distribution(samples: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    benign = np.array([float(row["cfi"]) for row in samples if row["state_type"] == "benign"], dtype=float)
    poisoned = np.array([float(row["cfi"]) for row in samples if row["state_type"] == "poisoned"], dtype=float)
    dataset = samples[0]["dataset"]
    attack = samples[0]["attack"]
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bins = min(20, max(3, int(math.sqrt(max(len(benign), len(poisoned))))))
    ax.hist(benign, bins=bins, density=True, alpha=0.45, color="0.25", label="benign", edgecolor="white")
    ax.hist(poisoned, bins=bins, density=True, alpha=0.45, color="0.65", label="poisoned", edgecolor="white")
    if benign.size:
        ax.axvline(float(np.mean(benign)), color="0.10", linestyle="--", linewidth=1.2, label="benign mean")
    if poisoned.size:
        ax.axvline(float(np.mean(poisoned)), color="0.45", linestyle="-", linewidth=1.2, label="poisoned mean")
    ax.set_xlabel("CFI")
    ax.set_ylabel("Density")
    ax.set_title(f"Observation CFI distribution: {dataset} / {attack}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_observation_cfi_distribution.pdf")
    fig.savefig(output_dir / "fig_observation_cfi_distribution.png", dpi=300)
    plt.close(fig)


def run_observation(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    benign_cfg = build_observation_config(args, "none")
    poisoned_cfg = build_observation_config(args, args.attack)
    dataset_provider = create_dataset_provider(benign_cfg)
    splits = dataset_provider.build()
    model_factory = create_model_factory(benign_cfg)
    initial_model = model_factory.create()
    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in initial_model.state_dict().items()}
    codec = ModelUpdateCodec(initial_model)
    crypto_bundle = create_crypto_backend(benign_cfg)

    calibration_dir = output_dir / "crypto_calibration"
    provider = CalibrationTensorProvider(codec, benign_cfg.geochoke, benign_cfg.device)
    representative_tensors = provider.build(initial_model, splits.proxy_loader)
    calibrator = ProfileCalibrator(
        crypto_bundle.public_backend,
        crypto_bundle.decryption_service.decrypt_for_offline_calibration,
        benign_cfg.ckks_profiles,
        benign_cfg.geochoke,
        output_dir=str(calibration_dir),
    )
    _calibration, perturbation_bank = calibrator.calibrate(representative_tensors)
    estimator = CFIEstimator(codec, splits.proxy_loader, perturbation_bank, benign_cfg.device)

    benign_rows = _run_state_type(benign_cfg, splits, model_factory, crypto_bundle.public_backend, crypto_bundle.decryption_service, estimator, initial_state, "benign", args.attack, args.alpha, args.poison_ratio)
    set_seed(args.seed)
    poisoned_rows = _run_state_type(poisoned_cfg, splits, model_factory, crypto_bundle.public_backend, crypto_bundle.decryption_service, estimator, initial_state, "poisoned", args.attack, args.alpha, args.poison_ratio)
    samples = benign_rows + poisoned_rows
    summary = summarize(samples)
    _write_csv(output_dir / "observation_cfi_samples.csv", samples)
    _write_csv(output_dir / "observation_cfi_summary.csv", [summary])
    plot_distribution(samples, output_dir)
    return samples, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-commit CFI observation experiment for benign and poisoned candidates.")
    parser.add_argument("--dataset", default="fashion_mnist", help="Dataset: fashion_mnist / cifar10 / mnist")
    parser.add_argument("--attack", default="dba_multi", choices=sorted(_ATTACK_PREFIX), help="Backdoor attack for poisoned candidates")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_rounds", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--poison_ratio", type=float, default=0.2)
    parser.add_argument("--output_dir", default="./outputs_observation")
    parser.add_argument("--num_clients", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--clients_per_round", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--min_clients_per_round", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--quick_data_limit", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--proxy_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--test_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--calibration_vectors", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--perturbation_count", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--perturbation_scale", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--no_download_data", action="store_false", dest="download_data", default=True, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    samples, summary = run_observation(parse_args(argv))
    print(f"wrote {len(samples)} CFI samples")
    print(summary)


if __name__ == "__main__":
    main()
