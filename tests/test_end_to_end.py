from config import ExperimentConfig, GeoChokeConfig, CKKS_PROFILES
from core.coordinator import run

def test_end_to_end_smoke(tmp_path):
    cfg=ExperimentConfig(num_rounds=1,quick_data_limit=180,proxy_size=16,test_size=32,output_dir=str(tmp_path),download_data=True,ckks_profiles=CKKS_PROFILES,geochoke=GeoChokeConfig(calibration_vectors=1,perturbation_count=1))
    rows=run(cfg); assert len(rows)==1; assert rows[0]['ciphertext_block_count']>=1; assert rows[0]['selected_next_profile'] in CKKS_PROFILES
