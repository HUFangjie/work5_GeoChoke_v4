import csv

import tests.test_factories  # registers dummy dataset
from scripts.run_observation_cfi import parse_args, run_observation


def test_observation_cfi_script_outputs_current_run_curve(tmp_path):
    calibration_dir = tmp_path / "shared_calibration"
    args = parse_args([
        "--observation_cfi",
        "--observation_state_type", "poisoned",
        "--dataset", "dummy",
        "--attack", "dba_multi",
        "--seed", "7",
        "--num_rounds", "1",
        "--alpha", "0.5",
        "--poison_ratio", "0.3",
        "--malicious_fraction", "0.5",
        "--attack_start_round", "0",
        "--attack_end_round", "0",
        "--output_dir", str(tmp_path / "run"),
        "--observation_calibration_dir", str(calibration_dir),
        "--num_clients", "2",
        "--clients_per_round", "2",
        "--min_clients_per_round", "2",
        "--observation_calibration_vectors", "1",
        "--observation_perturbation_count", "1",
    ])
    rows = run_observation(args)
    path = tmp_path / "run" / "observation_cfi_curve.csv"
    assert path.exists()
    assert (calibration_dir / "observation_perturbation_bank.npz").exists()
    with path.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert len(csv_rows) == 1
    row = csv_rows[0]
    assert row["state_type"] == "poisoned"
    assert float(row["candidate_cfi"]) >= 0.0
    assert row["candidate_cfi"] == row["cfi"]
    assert "clean_test_accuracy" in row
    assert "global_trigger_asr" in row
    assert int(row["num_selected"]) == 2
