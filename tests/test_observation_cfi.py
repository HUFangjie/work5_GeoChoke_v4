import csv

import tests.test_factories  # registers dummy dataset
from scripts.run_observation_cfi import parse_args, run_observation


def test_observation_cfi_script_outputs_samples(tmp_path):
    args = parse_args([
        "--dataset", "dummy",
        "--attack", "dba_multi",
        "--seed", "7",
        "--num_rounds", "1",
        "--alpha", "0.5",
        "--poison_ratio", "0.2",
        "--output_dir", str(tmp_path),
        "--num_clients", "2",
        "--clients_per_round", "2",
        "--min_clients_per_round", "2",
        "--calibration_vectors", "1",
        "--perturbation_count", "1",
    ])
    samples, _summary = run_observation(args)
    path = tmp_path / "observation_cfi_samples.csv"
    assert path.exists()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["state_type"] for row in rows} == {"benign", "poisoned"}
    assert all(float(row["cfi"]) >= 0.0 for row in rows)
    assert all("test_accuracy" in row for row in rows)
    assert all("global_trigger_asr" in row for row in rows)
    assert {row["reference_profile_id"] for row in rows} == {samples[0]["reference_profile_id"]}
    assert {row["perturbation_count"] for row in rows} == {str(samples[0]["perturbation_count"])}
    assert {row["perturbation_scale"] for row in rows} == {str(samples[0]["perturbation_scale"])}
