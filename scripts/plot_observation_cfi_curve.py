#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float_series(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        values.append(float("nan") if value in {None, "", "None"} else float(value))
    return values


def plot_curves(benign_csv: Path, poisoned_csv: Path, attack_start_round: int, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    benign = _read_rows(benign_csv)
    poisoned = _read_rows(poisoned_csv)
    benign_rounds = [int(row["round"]) for row in benign]
    poisoned_rounds = [int(row["round"]) for row in poisoned]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(benign_rounds, [value * 1e8 for value in _float_series(benign, "candidate_cfi")], color="0.20", linewidth=1.6, label="benign")
    ax.plot(poisoned_rounds, [value * 1e8 for value in _float_series(poisoned, "candidate_cfi")], color="0.60", linewidth=1.6, label="poisoned")
    ax.axvline(attack_start_round, color="0.10", linestyle="--", linewidth=1.0, label="attack start")
    ax.set_xlabel("Round")
    ax.set_ylabel("Candidate CFI (×10^{-8})")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_observation_candidate_cfi_curve.pdf")
    fig.savefig(output_dir / "fig_observation_candidate_cfi_curve.png", dpi=300)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6.0, 3.6))
    ax1.plot(benign_rounds, _float_series(benign, "clean_test_accuracy"), color="0.20", linewidth=1.6, label="benign acc")
    ax1.plot(poisoned_rounds, _float_series(poisoned, "clean_test_accuracy"), color="0.60", linewidth=1.6, label="poisoned acc")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Clean accuracy")
    ax2 = ax1.twinx()
    ax2.plot(poisoned_rounds, _float_series(poisoned, "global_trigger_asr"), color="0.40", linestyle="--", linewidth=1.4, label="poisoned ASR")
    ax2.set_ylabel("ASR")
    ax1.axvline(attack_start_round, color="0.10", linestyle=":", linewidth=1.0)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_observation_accuracy_asr_curve.pdf")
    fig.savefig(output_dir / "fig_observation_accuracy_asr_curve.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot manual benign/poisoned observation CFI curves.")
    parser.add_argument("--benign_csv", required=True)
    parser.add_argument("--poisoned_csv", required=True)
    parser.add_argument("--attack_start_round", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_curves(Path(args.benign_csv), Path(args.poisoned_csv), args.attack_start_round, Path(args.output_dir))


if __name__ == "__main__":
    main()
