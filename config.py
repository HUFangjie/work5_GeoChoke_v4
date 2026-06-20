from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any

CKKS_PROFILES: Dict[str, Dict[str, Any]] = {
    "ckks_s40": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 40, 40, 60], "global_scale_bits": 40},
    "ckks_s38": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 38, 38, 60], "global_scale_bits": 38},
    "ckks_s36": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 36, 36, 60], "global_scale_bits": 36},
    "ckks_s34": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 34, 34, 60], "global_scale_bits": 34},
    "ckks_s32": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 32, 32, 60], "global_scale_bits": 32},
    "ckks_s30": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 30, 30, 60], "global_scale_bits": 30},
    "ckks_s28": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 28, 28, 60], "global_scale_bits": 28},
    "ckks_s26": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 26, 26, 60], "global_scale_bits": 26},
    "ckks_s24": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 24, 24, 60], "global_scale_bits": 24},
    "ckks_s22": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 22, 22, 60], "global_scale_bits": 22},
    "ckks_s20": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 20, 20, 60], "global_scale_bits": 20},
}

@dataclass
class GeoChokeConfig:
    initial_profile_id: str = "ckks_s40"
    reference_profile_id: str = "ckks_s28"
    calibration_vectors: int = 16
    perturbation_count: int = 16
    perturbation_scale: float = 1.0
    lambda_: float = 50.0
    gamma: float = 0.01
    rho: float = 0.01
    tangent_commitment_enabled: bool = True
    tangent_basis_rank: int = 16
    tangent_max_proxy_batches: int = 16
    tangent_tau_max: float = 0.20
    tangent_tau_min: float = 0.02
    tangent_lambda: float = 10.0
    tangent_eps: float = 1e-12
    tangent_refresh_interval: int = 1

@dataclass
class ExperimentConfig:
    seed: int = 7
    device: str = "cpu"
    num_clients: int = 5
    clients_per_round: int = 5
    min_clients_per_round: int = 2
    malicious_client_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    num_rounds: int = 50
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
    num_classes: int = 10
    data_dir: str = "./data_cache"
    download_data: bool = True
    quick_data_limit: int = 600
    proxy_size: int = 64
    test_size: int = 256
    ckks_profiles: Dict[str, Dict[str, Any]] = field(default_factory=lambda: CKKS_PROFILES)
    geochoke: GeoChokeConfig = field(default_factory=GeoChokeConfig)
    output_dir: str = "./outputs"
    log_level: str = "INFO"
    enable_plaintext_reference_metrics: bool = True
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
    dba_multi_shot_scale_factor: float = 1.0
    dba_single_shot_scale_factor: float = 20.0
    dba_attack_mode: str = "multi_shot"  # multi_shot, single_shot
    dba_attack_start_round: int = 10
    dba_attack_end_round: int = 19
    dba_poison_interval: int = 1
    dba_num_trigger_parts: int = 4
    dba_trigger_size: int = 4
    dba_trigger_gap: int = 2
    dba_trigger_location: str = "top_left"
    dba_trigger_value: float = 1.0


def strong_geochoke_config() -> ExperimentConfig:
    return ExperimentConfig()


def mild_geochoke_config() -> ExperimentConfig:
    return ExperimentConfig(
        geochoke=GeoChokeConfig(
            initial_profile_id="ckks_s40",
            reference_profile_id="ckks_s30",
            calibration_vectors=8,
            perturbation_count=8,
            perturbation_scale=1.0,
            lambda_=1.0,
            gamma=1.0,
            rho=1.0,
        )
    )

CONFIG = strong_geochoke_config()
