from config import CKKS_PROFILES, ExperimentConfig, GeoChokeConfig
from core.coordinator import run


def test_end_to_end_smoke(tmp_path):
    cfg = ExperimentConfig(
        num_rounds=2,
        quick_data_limit=240,
        proxy_size=16,
        test_size=32,
        output_dir=str(tmp_path),
        download_data=True,
        ckks_profiles=CKKS_PROFILES,
        geochoke=GeoChokeConfig(calibration_vectors=1, perturbation_count=1),
    )
    rows = run(cfg)
    assert len(rows) == 2
    assert rows[0]["ciphertext_block_count"] >= 1
    assert rows[0]["selected_next_profile"] in CKKS_PROFILES
    assert rows[1]["profile_id"] in CKKS_PROFILES
