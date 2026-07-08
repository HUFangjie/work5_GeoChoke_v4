#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
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
from core.types import ClientUpload, LocalUpdateRecord  # noqa: E402
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


def _attack_overrides(attack: str, num_rounds: int, poison_ratio: float, num_clients: int, attack_start_round: int, malicious_fraction: float) -> dict[str, Any]:
    if attack not in _ATTACK_PREFIX:
        raise ValueError(f"Unsupported observation attack: {attack}")
    prefix = _ATTACK_PREFIX[attack]
    requested_malicious = max(1, int(round(num_clients * malicious_fraction)))
    malicious_count = min(max(1, num_clients - 1), requested_malicious)
    malicious_ids = list(range(1, min(num_clients, malicious_count + 1)))
    overrides: dict[str, Any] = {
        "malicious_client_ids": malicious_ids,
        f"{prefix}_poison_ratio": poison_ratio,
        f"{prefix}_attack_start_round": min(max(0, attack_start_round), max(0, num_rounds - 1)),
        f"{prefix}_attack_end_round": max(0, num_rounds - 1),
    }
    if prefix == "dba":
        overrides["dba_num_trigger_parts"] = min(4, len(malicious_ids))
    return overrides


def build_observation_config(args: argparse.Namespace, attack_name: str):
    num_clients = int(getattr(args, "num_clients", 10))
    attack_overrides = (
        _attack_overrides(args.attack, args.num_rounds, args.poison_ratio, num_clients, args.warmup_rounds, args.malicious_fraction)
        if attack_name != "none"
        else {"malicious_client_ids": []}
    )
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


def _make_clients(cfg: Any, splits: Any, model_factory: Any, codec_factory: Any, crypto_backend: Any, attack: Any, force_clean: bool = False) -> list[Client]:
    return [
        Client(
            client_id=client_id,
            loader=loader,
            model_factory=model_factory.create,
            codec_factory=codec_factory,
            crypto_backend=crypto_backend,
            attack_strategy=attack,
            malicious=False if force_clean else client_id in cfg.malicious_client_ids,
            cfg=cfg,
        )
        for client_id, loader in enumerate(splits.client_loaders)
    ]


def _upload_attack_active(upload: ClientUpload) -> bool:
    return any(bool(upload.metadata.get(key, False)) for key in _ATTACK_ACTIVE_KEYS) or upload.metadata.get("update_type") == "poisoned"


def _attack_activity_metrics(uploads: list[ClientUpload], selected_client_ids: list[int]) -> dict[str, Any]:
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


def _compute_uploads(
    clients: list[Client],
    selected_client_ids: list[int],
    records_by_client: dict[int, LocalUpdateRecord],
    malicious_client_ids: set[int],
    profile_id: str,
) -> list[ClientUpload]:
    observable_updates = [records_by_client[client_id].local_update for client_id in selected_client_ids if client_id not in malicious_client_ids]
    if not observable_updates:
        observable_updates = [records_by_client[client_id].local_update for client_id in selected_client_ids]
    benign_norm_mean = float(np.mean([np.linalg.norm(update) for update in observable_updates])) if observable_updates else 0.0
    total_samples = sum(records_by_client[client_id].num_samples for client_id in selected_client_ids)
    uploads: list[ClientUpload] = []
    for client_id in selected_client_ids:
        record = records_by_client[client_id]
        aggregation_weight = record.num_samples / total_samples if total_samples else 0.0
        attacker_context = {
            "num_selected": len(selected_client_ids),
            "num_malicious": sum(1 for selected_id in selected_client_ids if selected_id in malicious_client_ids),
            "observable_updates": observable_updates,
            "oracle_all_updates": [records_by_client[selected_id].local_update for selected_id in selected_client_ids],
            "malicious_weight": aggregation_weight,
            "benign_selected_update_norm_mean": benign_norm_mean,
        }
        uploads.append(clients[client_id].encrypt_update(record, profile_id, attacker_context))
    return uploads


def _candidate_from_uploads(
    server: AggregationServer,
    codec: ModelUpdateCodec,
    model_factory: Any,
    decryption_service: Any,
    uploads: list[ClientUpload],
    profile_id: str,
    global_state: dict[str, Any],
):
    aggregate_ciphertext, _aggregation_time, _weights = server.aggregate_encrypted(uploads)
    decrypted_update = decryption_service.decrypt_aggregate(aggregate_ciphertext, profile_id)
    candidate_state = codec.apply_update_to_state_dict(global_state, decrypted_update, step_size=server.cfg.server_lr)
    candidate_model = model_factory.create()
    candidate_model.load_state_dict(candidate_state)
    return candidate_state, candidate_model


def _candidate_row(
    cfg: Any,
    requested_attack: str,
    driver_update: str,
    round_id: int,
    state_type: str,
    alpha: float,
    poison_ratio: float,
    profile_id: str,
    previous_cfi: float,
    candidate_cfi: float,
    benign_candidate_cfi: float,
    uploads: list[ClientUpload],
    selected_client_ids: list[int],
    candidate_model: Any,
    evaluator: Evaluator,
    eval_attack: Any,
    eval_cfg: Any,
) -> dict[str, Any]:
    fis = max(0.0, candidate_cfi - previous_cfi)
    test_loss, clean_test_accuracy = evaluator.evaluate(candidate_model)
    backdoor_metrics = evaluator.evaluate_backdoor(candidate_model, eval_attack, eval_cfg, profile_id, uploads)
    activity_metrics = _attack_activity_metrics(uploads, selected_client_ids)
    row = {
        "dataset": cfg.dataset_name,
        "attack": requested_attack,
        "seed": cfg.seed,
        "round": round_id,
        "state_type": state_type,
        "paired_attack": requested_attack,
        "driver_update": driver_update,
        "alpha": alpha,
        "poison_ratio": poison_ratio,
        "reference_profile_id": cfg.geochoke.reference_profile_id,
        "perturbation_count": cfg.geochoke.perturbation_count,
        "perturbation_scale": cfg.geochoke.perturbation_scale,
        "current_ckks_profile": profile_id,
        "previous_cfi": previous_cfi,
        "candidate_cfi": candidate_cfi,
        "cfi": candidate_cfi,
        "fragility_injection_score": fis,
        "cfi_gap_to_benign_candidate": candidate_cfi - benign_candidate_cfi if state_type == "poisoned" else 0.0,
        "test_loss": test_loss,
        "clean_test_accuracy": clean_test_accuracy,
        "test_accuracy": clean_test_accuracy,
        "global_trigger_asr": backdoor_metrics.get("global_trigger_asr"),
        "global_trigger_raw_asr": backdoor_metrics.get("global_trigger_raw_asr"),
        "global_clean_target_rate": backdoor_metrics.get("global_clean_target_rate"),
        "num_selected": len(selected_client_ids),
        **activity_metrics,
    }
    for key, value in backdoor_metrics.items():
        row.setdefault(key, value)
    return row


def _run_paired_observation(
    benign_cfg: Any,
    poisoned_cfg: Any,
    splits: Any,
    model_factory: Any,
    crypto_backend: Any,
    decryption_service: Any,
    estimator: CFIEstimator,
    evaluator: Evaluator,
    initial_state: dict[str, Any],
    requested_attack: str,
    alpha: float,
    poison_ratio: float,
    driver_update: str = "benign",
) -> list[dict[str, Any]]:
    clean_attack = create_attack(benign_cfg)
    poisoned_attack = create_attack(poisoned_cfg)
    clean_clients = _make_clients(benign_cfg, splits, model_factory, lambda model: ModelUpdateCodec(model), crypto_backend, clean_attack, force_clean=True)
    poisoned_clients = _make_clients(poisoned_cfg, splits, model_factory, lambda model: ModelUpdateCodec(model), crypto_backend, poisoned_attack)

    driver_model = model_factory.create()
    driver_model.load_state_dict(initial_state)
    codec = ModelUpdateCodec(driver_model)
    server = AggregationServer(benign_cfg, driver_model, codec, crypto_backend, _PreCommitNoDefense(benign_cfg.geochoke.initial_profile_id), model_factory.create)
    profile_id = benign_cfg.geochoke.initial_profile_id
    rows: list[dict[str, Any]] = []
    malicious_client_ids = set(poisoned_cfg.malicious_client_ids)

    warmup_rounds = int(getattr(benign_cfg, "observation_warmup_rounds", 0))
    for round_id in range(benign_cfg.num_rounds):
        selected_client_ids = server.sample_clients(round_id)
        global_state = {name: tensor.detach().cpu().clone() for name, tensor in server.model.state_dict().items()}
        previous_model = model_factory.create()
        previous_model.load_state_dict(global_state)
        previous_cfi = estimator.estimate(previous_model)

        clean_records = [clean_clients[client_id].compute_local_update(global_state, round_id) for client_id in selected_client_ids]
        clean_records_by_client = {record.client_id: record for record in clean_records}
        clean_uploads = _compute_uploads(clean_clients, selected_client_ids, clean_records_by_client, set(), profile_id)
        benign_candidate_state, benign_candidate_model = _candidate_from_uploads(server, codec, model_factory, decryption_service, clean_uploads, profile_id, global_state)
        if round_id < warmup_rounds:
            server.model.load_state_dict(benign_candidate_state)
            print(
                "observation_warmup "
                f"round={round_id} driver_update=benign "
                f"previous_cfi={previous_cfi:.8g}",
                flush=True,
            )
            continue

        poisoned_records_by_client: dict[int, LocalUpdateRecord] = {}
        for client_id in selected_client_ids:
            if client_id in malicious_client_ids:
                poisoned_records_by_client[client_id] = poisoned_clients[client_id].compute_local_update(global_state, round_id)
            else:
                poisoned_records_by_client[client_id] = clean_records_by_client[client_id]

        poisoned_uploads = _compute_uploads(poisoned_clients, selected_client_ids, poisoned_records_by_client, malicious_client_ids, profile_id)
        poisoned_candidate_state, poisoned_candidate_model = _candidate_from_uploads(server, codec, model_factory, decryption_service, poisoned_uploads, profile_id, global_state)

        benign_candidate_cfi = estimator.estimate(benign_candidate_model)
        poisoned_candidate_cfi = estimator.estimate(poisoned_candidate_model)
        benign_row = _candidate_row(
            benign_cfg,
            requested_attack,
            driver_update,
            round_id,
            "benign",
            alpha,
            poison_ratio,
            profile_id,
            previous_cfi,
            benign_candidate_cfi,
            benign_candidate_cfi,
            clean_uploads,
            selected_client_ids,
            benign_candidate_model,
            evaluator,
            poisoned_attack,
            poisoned_cfg,
        )
        poisoned_row = _candidate_row(
            poisoned_cfg,
            requested_attack,
            driver_update,
            round_id,
            "poisoned",
            alpha,
            poison_ratio,
            profile_id,
            previous_cfi,
            poisoned_candidate_cfi,
            benign_candidate_cfi,
            poisoned_uploads,
            selected_client_ids,
            poisoned_candidate_model,
            evaluator,
            poisoned_attack,
            poisoned_cfg,
        )
        rows.extend([benign_row, poisoned_row])
        print(
            "observation_pair "
            f"round={round_id} driver_update={driver_update} "
            f"previous_cfi={previous_cfi:.8g} "
            f"benign_candidate_cfi={benign_candidate_cfi:.8g} "
            f"poisoned_candidate_cfi={poisoned_candidate_cfi:.8g} "
            f"benign_fis={benign_row['fragility_injection_score']:.8g} "
            f"poisoned_fis={poisoned_row['fragility_injection_score']:.8g} "
            f"benign_acc={benign_row['clean_test_accuracy']:.6f} "
            f"poisoned_acc={poisoned_row['clean_test_accuracy']:.6f} "
            f"poisoned_asr={poisoned_row['global_trigger_asr'] if poisoned_row['global_trigger_asr'] is not None else 'NA'}",
            flush=True,
        )
        if driver_update == "poisoned":
            server.model.load_state_dict(poisoned_candidate_state)
        else:
            server.model.load_state_dict(benign_candidate_state)
    return rows


def _auroc(benign: np.ndarray, poisoned: np.ndarray) -> float:
    if benign.size == 0 or poisoned.size == 0:
        return float("nan")
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
    if benign.size == 0 or poisoned.size == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([benign, poisoned])))
    if values.size == 0:
        return float("nan")
    benign_sorted = np.sort(benign)
    poisoned_sorted = np.sort(poisoned)
    b_cdf = np.searchsorted(benign_sorted, values, side="right") / max(1, benign_sorted.size)
    p_cdf = np.searchsorted(poisoned_sorted, values, side="right") / max(1, poisoned_sorted.size)
    return float(np.max(np.abs(b_cdf - p_cdf)))


def _metric_stats(benign: np.ndarray, poisoned: np.ndarray, prefix: str) -> dict[str, float]:
    benign_mean = float(np.mean(benign)) if benign.size else float("nan")
    poisoned_mean = float(np.mean(poisoned)) if poisoned.size else float("nan")
    benign_std = float(np.std(benign, ddof=1)) if benign.size > 1 else 0.0
    poisoned_std = float(np.std(poisoned, ddof=1)) if poisoned.size > 1 else 0.0
    pooled = math.sqrt(((benign.size - 1) * benign_std**2 + (poisoned.size - 1) * poisoned_std**2) / max(1, benign.size + poisoned.size - 2))
    return {
        f"benign_mean_{prefix}": benign_mean,
        f"benign_std_{prefix}": benign_std,
        f"poisoned_mean_{prefix}": poisoned_mean,
        f"poisoned_std_{prefix}": poisoned_std,
        f"{prefix}_gap": poisoned_mean - benign_mean,
        f"{prefix}_ratio": poisoned_mean / max(benign_mean, 1e-12),
        f"{prefix}_cohen_d": (poisoned_mean - benign_mean) / max(pooled, 1e-12),
        f"{prefix}_auroc": _auroc(benign, poisoned),
        f"{prefix}_ks_statistic": _ks_statistic(benign, poisoned),
    }


def summarize(samples: list[dict[str, Any]], fallback_samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    context = samples or fallback_samples or []
    if not context:
        return {}
    benign_rows = [row for row in samples if row["state_type"] == "benign"]
    poisoned_rows = [row for row in samples if row["state_type"] == "poisoned"]
    benign_cfi = np.array([float(row["candidate_cfi"]) for row in benign_rows], dtype=float)
    poisoned_cfi = np.array([float(row["candidate_cfi"]) for row in poisoned_rows], dtype=float)
    benign_fis = np.array([float(row["fragility_injection_score"]) for row in benign_rows], dtype=float)
    poisoned_fis = np.array([float(row["fragility_injection_score"]) for row in poisoned_rows], dtype=float)
    by_round: dict[int, dict[str, dict[str, Any]]] = {}
    for row in samples:
        by_round.setdefault(int(row["round"]), {})[str(row["state_type"])] = row
    paired_rounds = [pair for pair in by_round.values() if "benign" in pair and "poisoned" in pair]
    candidate_positive = [float(pair["poisoned"]["candidate_cfi"]) > float(pair["benign"]["candidate_cfi"]) for pair in paired_rounds]
    fis_positive = [float(pair["poisoned"]["fragility_injection_score"]) > float(pair["benign"]["fragility_injection_score"]) for pair in paired_rounds]
    first = context[0]
    summary = {
        "dataset": first["dataset"],
        "attack": first["attack"],
        "seed": first["seed"],
        "driver_update": first.get("driver_update"),
        "round_count": len(paired_rounds),
        "paired_positive_candidate_gap_fraction": float(np.mean(candidate_positive)) if candidate_positive else float("nan"),
        "paired_positive_fis_gap_fraction": float(np.mean(fis_positive)) if fis_positive else float("nan"),
        "mean_global_trigger_asr_poisoned": float(np.nanmean([row["global_trigger_asr"] for row in poisoned_rows if row.get("global_trigger_asr") is not None])) if any(row.get("global_trigger_asr") is not None for row in poisoned_rows) else float("nan"),
        "mean_clean_test_accuracy_benign": float(np.mean([float(row["clean_test_accuracy"]) for row in benign_rows])) if benign_rows else float("nan"),
        "mean_clean_test_accuracy_poisoned": float(np.mean([float(row["clean_test_accuracy"]) for row in poisoned_rows])) if poisoned_rows else float("nan"),
    }
    summary.update(_metric_stats(benign_cfi, poisoned_cfi, "candidate_cfi"))
    summary.update(_metric_stats(benign_fis, poisoned_fis, "fis"))
    return summary


def _stealth_successful_samples(samples: list[dict[str, Any]], min_success_asr: float, max_stealth_acc_drop: float) -> list[dict[str, Any]]:
    by_round: dict[int, dict[str, dict[str, Any]]] = {}
    for row in samples:
        by_round.setdefault(int(row["round"]), {})[str(row["state_type"])] = row
    selected: list[dict[str, Any]] = []
    for pair in by_round.values():
        poisoned = pair.get("poisoned")
        if not poisoned:
            continue
        benign = pair.get("benign")
        asr = poisoned.get("global_trigger_asr")
        if not benign or asr is None:
            continue
        clean_drop = float(benign["clean_test_accuracy"]) - float(poisoned["clean_test_accuracy"])
        if bool(poisoned.get("attack_active", False)) and float(asr) >= min_success_asr and clean_drop <= max_stealth_acc_drop:
            selected.append(benign)
            selected.append(poisoned)
    return selected


def _plot_metric_distribution(samples: list[dict[str, Any]], output_dir: Path, metric: str, stem: str, xlabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    benign = np.array([float(row[metric]) for row in samples if row["state_type"] == "benign"], dtype=float) * 1e8
    poisoned = np.array([float(row[metric]) for row in samples if row["state_type"] == "poisoned"], dtype=float) * 1e8
    if benign.size == 0 and poisoned.size == 0:
        return
    dataset = samples[0]["dataset"]
    attack = samples[0]["attack"]
    driver_update = samples[0].get("driver_update", "benign")
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bins = min(20, max(3, int(math.sqrt(max(len(benign), len(poisoned), 1)))))
    ax.hist(benign, bins=bins, density=True, alpha=0.45, color="0.25", label="benign", edgecolor="white")
    ax.hist(poisoned, bins=bins, density=True, alpha=0.45, color="0.65", label="poisoned", edgecolor="white")
    if benign.size:
        ax.axvline(float(np.mean(benign)), color="0.10", linestyle="--", linewidth=1.2, label="benign mean")
    if poisoned.size:
        ax.axvline(float(np.mean(poisoned)), color="0.45", linestyle="-", linewidth=1.2, label="poisoned mean")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset} / {attack} / driver={driver_update}")
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def plot_distributions(samples: list[dict[str, Any]], output_dir: Path) -> None:
    _plot_metric_distribution(samples, output_dir, "candidate_cfi", "fig_observation_candidate_cfi_distribution", "Candidate CFI (×10^{-8})")
    _plot_metric_distribution(samples, output_dir, "fragility_injection_score", "fig_observation_fis_distribution", "Fragility Injection Score (×10^{-8})")


def run_observation(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.warmup_rounds < 0 or args.warmup_rounds >= args.num_rounds:
        raise ValueError("--warmup_rounds must satisfy 0 <= warmup_rounds < num_rounds")
    if not 0.0 < args.malicious_fraction <= 1.0:
        raise ValueError("--malicious_fraction must satisfy 0.0 < malicious_fraction <= 1.0")
    set_seed(args.seed)
    benign_cfg = build_observation_config(args, "none")
    poisoned_cfg = build_observation_config(args, args.attack)
    benign_cfg.observation_warmup_rounds = args.warmup_rounds
    poisoned_cfg.observation_warmup_rounds = args.warmup_rounds
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
    evaluator = Evaluator(splits.test_loader, benign_cfg.device)

    samples = _run_paired_observation(
        benign_cfg,
        poisoned_cfg,
        splits,
        model_factory,
        crypto_bundle.public_backend,
        crypto_bundle.decryption_service,
        estimator,
        evaluator,
        initial_state,
        args.attack,
        args.alpha,
        args.poison_ratio,
        driver_update=args.driver_update,
    )
    all_round_summary = summarize(samples)
    stealth_successful_samples = _stealth_successful_samples(samples, args.min_success_asr, args.max_stealth_acc_drop)
    stealth_successful_summary = summarize(stealth_successful_samples, fallback_samples=samples)
    stealth_successful_summary["min_success_asr"] = args.min_success_asr
    stealth_successful_summary["max_stealth_acc_drop"] = args.max_stealth_acc_drop
    stealth_successful_summary["warmup_rounds"] = args.warmup_rounds
    stealth_successful_summary["malicious_fraction"] = args.malicious_fraction
    stealth_successful_summary["stealth_successful_round_count"] = int(len({row["round"] for row in stealth_successful_samples if row["state_type"] == "poisoned"}))
    all_round_summary["warmup_rounds"] = args.warmup_rounds
    all_round_summary["malicious_fraction"] = args.malicious_fraction
    _write_csv(output_dir / "observation_cfi_samples.csv", samples)
    _write_csv(output_dir / "observation_cfi_summary_all.csv", [all_round_summary])
    _write_csv(output_dir / "observation_cfi_summary.csv", [stealth_successful_summary])
    _write_csv(output_dir / "observation_cfi_summary_successful.csv", [stealth_successful_summary])
    plot_distributions(samples, output_dir)
    return samples, stealth_successful_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired pre-commit CFI observation experiment for benign and poisoned candidates.")
    parser.add_argument("--dataset", default="fashion_mnist", help="Dataset: fashion_mnist / cifar10 / mnist")
    parser.add_argument("--attack", default="dba_multi", choices=sorted(_ATTACK_PREFIX), help="Backdoor attack for poisoned candidates")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--poison_ratio", type=float, default=0.3)
    parser.add_argument("--output_dir", default="./outputs_observation")
    parser.add_argument("--driver_update", choices=["benign", "poisoned"], default="benign")
    parser.add_argument("--warmup_rounds", type=int, default=20)
    parser.add_argument("--malicious_fraction", type=float, default=0.2)
    parser.add_argument("--min_success_asr", type=float, default=0.2)
    parser.add_argument("--max_stealth_acc_drop", type=float, default=0.05)
    parser.add_argument("--num_clients", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--clients_per_round", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--min_clients_per_round", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--quick_data_limit", type=int, default=6000, help=argparse.SUPPRESS)
    parser.add_argument("--proxy_size", type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument("--test_size", type=int, default=2000, help=argparse.SUPPRESS)
    parser.add_argument("--calibration_vectors", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--perturbation_count", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--perturbation_scale", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--no_download_data", action="store_false", dest="download_data", default=True, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    samples, summary = run_observation(parse_args(argv))
    print(f"wrote {len(samples)} paired CFI sample rows")
    print(summary)


if __name__ == "__main__":
    main()
