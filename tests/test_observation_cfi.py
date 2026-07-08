import csv
from collections import defaultdict

import tests.test_factories  # registers dummy dataset
from scripts.run_observation_cfi import parse_args, run_observation


def test_observation_cfi_script_outputs_paired_samples(tmp_path):
    args = parse_args([
        "--dataset", "dummy",
        "--attack", "dba_multi",
        "--seed", "7",
        "--num_rounds", "1",
        "--alpha", "0.5",
        "--poison_ratio", "0.3",
        "--output_dir", str(tmp_path),
        "--num_clients", "2",
        "--clients_per_round", "2",
        "--min_clients_per_round", "2",
        "--calibration_vectors", "1",
        "--perturbation_count", "1",
        "--driver_update", "benign",
        "--min_success_asr", "0.2",
    ])
    run_observation(args)
    samples_path = tmp_path / "observation_cfi_samples.csv"
    successful_summary_path = tmp_path / "observation_cfi_summary_successful.csv"
    assert samples_path.exists()
    assert successful_summary_path.exists()
    with samples_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["state_type"] for row in rows} == {"benign", "poisoned"}
    by_round = defaultdict(dict)
    for row in rows:
        by_round[int(row["round"])][row["state_type"]] = row
        assert float(row["candidate_cfi"]) >= 0.0
        assert float(row["fragility_injection_score"]) >= 0.0
    for pair in by_round.values():
        assert set(pair) == {"benign", "poisoned"}
        benign = pair["benign"]
        poisoned = pair["poisoned"]
        for key in ["previous_cfi", "reference_profile_id", "perturbation_count", "perturbation_scale"]:
            assert benign[key] == poisoned[key]
        assert int(poisoned["num_malicious_selected"]) > 0
