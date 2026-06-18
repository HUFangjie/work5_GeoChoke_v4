from dataclasses import dataclass, field
from typing import Dict, List, Any

CKKS_PROFILES: Dict[str, Dict[str, Any]] = {
    "high_precision": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 40, 40, 60], "global_scale_bits": 40},
    "medium_precision": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 35, 35, 60], "global_scale_bits": 35},
    "low_precision": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 30, 30, 60], "global_scale_bits": 30},
}

@dataclass
class GeoChokeConfig:
    initial_profile_id: str = "high_precision"
    reference_profile_id: str = "high_precision"
    calibration_vectors: int = 8
    perturbation_count: int = 8
    perturbation_scale: float = 1.0
    lambda_: float = 1.0
    gamma: float = 1.0
    rho: float = 1.0

@dataclass
class ExperimentConfig:
    seed: int = 7
    device: str = "cpu"
    num_clients: int = 5
    clients_per_round: int = 5
    min_clients_per_round: int = 2
    malicious_client_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    num_rounds: int = 30
    local_epochs: int = 1
    batch_size: int = 32
    local_lr: float = 0.01
    server_lr: float = 1.0
    partition_type: str = "iid"
    dirichlet_alpha: float = 0.5
    crypto_backend_name: str = "ckks"
    defense_name: str = "geochoke"
    attack_name: str = "dba"  # none, alie, fang_mean, dba
    aggregation: str = "weighted_mean"
    model_name: str = "mnist_cnn"
    dataset_name: str = "mnist"
    data_dir: str = "./data_cache"
    download_data: bool = True
    quick_data_limit: int = 600
    proxy_size: int = 64
    test_size: int = 256
    ckks_profiles: Dict[str, Dict[str, Any]] = field(default_factory=lambda: CKKS_PROFILES)
    geochoke: GeoChokeConfig = field(default_factory=GeoChokeConfig)
    output_dir: str = "./outputs"
    log_level: str = "INFO"
    enable_plaintext_reference_metrics: bool = False
    pipeline_validation_rtol: float = 5e-2
    pipeline_validation_atol: float = 5e-2
    pipeline_validation_norm_ratio_tolerance: float = 5e-2
    alie_z: float | None = None
    alie_oracle_all_updates: bool = False
    fang_max_norm: float = 5.0
    fang_search_steps: int = 6

    dba_target_label: int = 2
    dba_poison_ratio: float = 0.3125
    dba_local_epochs: int = 10
    dba_local_lr: float = 0.05
    dba_scale_factor: float = 1.0
    dba_attack_mode: str = "multi_shot"  # multi_shot, single_shot
    dba_attack_start_round: int = 10
    dba_attack_end_round: int = 29
    dba_poison_interval: int = 1
    dba_num_trigger_parts: int = 4
    dba_trigger_size: int = 4
    dba_trigger_gap: int = 2
    dba_trigger_location: str = "top_left"
    dba_trigger_value: float = 1.0

CONFIG = ExperimentConfig()
