from __future__ import annotations

import copy
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import CKKS_PROFILES, strong_geochoke_config
from factories import defaults  # noqa: F401
from factories.attack_factory import create_attack
from factories.crypto_factory import create_crypto_backend
from factories.dataset_factory import create_dataset_provider
from factories.defense_factory import create_defense
from factories.model_factory import create_model_factory
from core.coordinator import FederatedCoordinator
from utils.logger import setup_logger
from utils.seed import set_seed


def run_profile(profile_id: str, output_root: str) -> dict[str, float | str | None]:
    cfg = strong_geochoke_config()
    cfg.defense_name = "none"
    cfg.attack_name = "dba"
    cfg.geochoke.initial_profile_id = profile_id
    cfg.geochoke.reference_profile_id = profile_id
    cfg.ckks_profiles = {profile_id: copy.deepcopy(CKKS_PROFILES[profile_id])}
    cfg.output_dir = os.path.join(output_root, f"fixed_{profile_id}")
    set_seed(cfg.seed)
    logger = setup_logger(cfg.log_level)
    dataset_provider = create_dataset_provider(cfg)
    model_factory = create_model_factory(cfg)
    crypto_bundle = create_crypto_backend(cfg)
    defense = create_defense(cfg, crypto_bundle)
    attack = create_attack(cfg)
    rows = FederatedCoordinator(cfg, dataset_provider, model_factory, crypto_bundle.public_backend, crypto_bundle.decryption_service, defense, attack, logger).run()
    asrs = [float(row.get("global_trigger_asr", 0.0) or 0.0) for row in rows]
    clean = [float(row.get("clean_test_accuracy", 0.0) or 0.0) for row in rows]
    post = [float(row.get("global_trigger_asr", 0.0) or 0.0) for row in rows if int(row.get("round", 0)) > cfg.dba_attack_end_round]
    ratios = [float(row["ckks_residual_l2_ratio"]) for row in rows if row.get("ckks_residual_l2_ratio") is not None]
    return {
        "profile_id": profile_id,
        "final_clean_acc": clean[-1] if clean else None,
        "peak_global_trigger_asr": max(asrs) if asrs else None,
        "final_global_trigger_asr": asrs[-1] if asrs else None,
        "post_attack_asr_auc": sum(post),
        "clean_acc_drop_vs_ckks_s40": None,
        "residual_l2_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "residual_l2_ratio_max": max(ratios) if ratios else None,
    }


def main() -> None:
    output_root = os.path.join("outputs", "profile_sweep")
    os.makedirs(output_root, exist_ok=True)
    rows = [run_profile(profile_id, output_root) for profile_id in CKKS_PROFILES]
    baseline = next((row["final_clean_acc"] for row in rows if row["profile_id"] == "ckks_s40"), None)
    for row in rows:
        if baseline is not None and row["final_clean_acc"] is not None:
            row["clean_acc_drop_vs_ckks_s40"] = float(baseline) - float(row["final_clean_acc"])
    path = os.path.join(output_root, "profile_sweep_summary.csv")
    with open(path, "w", newline="") as handle:
        fieldnames = ["profile_id", "final_clean_acc", "peak_global_trigger_asr", "final_global_trigger_asr", "post_attack_asr_auc", "clean_acc_drop_vs_ckks_s40", "residual_l2_ratio_mean", "residual_l2_ratio_max"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
