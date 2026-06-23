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
    tangent_basis_rank: int = 128
    tangent_max_proxy_batches: int = 64
    tangent_tau_max: float = 0.30
    tangent_tau_min: float = 0.02
    tangent_lambda: float = 10.0
    tangent_eps: float = 1e-12
    tangent_refresh_interval: int = 1


@dataclass
class AionConfig:
    profile_id: str = "ckks_s40"

    # Multi-aggregator secure aggregation parameters
    aggregator_count: int = 8
    malicious_aggregator_count: int = 0
    reconstruction_mode: str = "amr"  # amr, asr
    field_modulus_bits: int = 2048
    vss_threshold: int | None = None

    # HPRF / mask parameters
    hprf_key_dim: int = 16
    mask_ratio_beta: float = 0.2
    hprf_hmax: float = 1.0
    decimal_places: int = 8

    # MGF bound initialization
    bound_init_mode: str = "warmup_quantile"  # warmup_quantile, fixed_from_clean_rounds
    warmup_rounds_for_bound: int = 2
    initial_bound_multiplier: float = 1.00
    initial_bound_quantile: float = 0.80

    # Paper evolving bound
    use_paper_evolving_bound: bool = True
    mu_min: float = 0.50
    mu_max: float = 1.20
    bound_min: float = 1e-12
    bound_max: float | None = None

    # Protocol switches
    enable_mgf: bool = True
    enable_dmc_dmr: bool = True
    fail_safe_keep_one: bool = False
    fail_open_when_too_few_valid: bool = False
    log_client_filter_details: bool = True
    enable_vss_verification: bool = True
    allow_weighted_mean: bool = False
    strict_protocol_checks: bool = True


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "cuda"

    # Federated learning setting
    num_clients: int = 20
    clients_per_round: int = 20
    min_clients_per_round: int = 2
    malicious_client_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    num_rounds: int = 30
    local_epochs: int = 1
    batch_size: int = 32
    local_lr: float = 0.01
    server_lr: float = 1.0
    partition_type: str = "iid"
    dirichlet_alpha: float = 0.5

    # Crypto / defense / attack setting
    crypto_backend_name: str = "ckks"
    defense_name: str = "geochoke"  # none, geochoke, aion
    attack_name: str = "dba"  # none, alie, fang_mean, dba
    aggregation: str = "weighted_mean"  # weighted_mean, mean

    # Model / dataset setting
    model_name: str = "mnist_cnn"
    dataset_name: str = "mnist"
    num_classes: int = 10
    data_dir: str = "./data_cache"
    download_data: bool = True
    quick_data_limit: int = 59000
    proxy_size: int = 1000
    test_size: int = 10000

    # Defense configs
    ckks_profiles: Dict[str, Dict[str, Any]] = field(default_factory=lambda: CKKS_PROFILES)
    geochoke: GeoChokeConfig = field(default_factory=GeoChokeConfig)
    aion: AionConfig = field(default_factory=AionConfig)

    # Output / logging
    output_dir: str = "./outputs"
    log_level: str = "INFO"

    # Pipeline validation
    enable_plaintext_reference_metrics: bool = True
    pipeline_validation_rtol: float = 5e-2
    pipeline_validation_atol: float = 5e-2
    pipeline_validation_norm_ratio_tolerance: float = 5e-2

    # ALIE attack parameters
    alie_z: float | None = None
    alie_oracle_all_updates: bool = False

    # Fang attack parameters
    fang_max_norm: float = 5.0
    fang_search_steps: int = 6

    # DBA attack parameters
    dba_target_label: int = 2
    dba_poison_ratio: float = 0.3125
    dba_local_epochs: int = 10
    dba_local_lr: float = 0.05
    dba_scale_factor: float = 1.0
    dba_multi_shot_scale_factor: float = 1.0
    dba_single_shot_scale_factor: float = 20.0
    dba_attack_mode: str = "multi_shot"  # multi_shot, single_shot
    dba_attack_start_round: int = 5
    dba_attack_end_round: int = 15
    dba_poison_interval: int = 1
    dba_num_trigger_parts: int = 4
    dba_trigger_size: int = 4
    dba_trigger_gap: int = 2
    dba_trigger_location: str = "top_left"
    dba_trigger_value: float = 1.0


def strong_geochoke_config() -> ExperimentConfig:
    return ExperimentConfig()


def aion_config() -> ExperimentConfig:
    return ExperimentConfig(
        defense_name="aion",
        attack_name="dba",
        aggregation="mean",
        aion=AionConfig(
            profile_id="ckks_s40",
            aggregator_count=8,
            malicious_aggregator_count=0,
            reconstruction_mode="amr",
            field_modulus_bits=2048,
            vss_threshold=None,
            hprf_key_dim=16,
            mask_ratio_beta=0.2,
            hprf_hmax=1.0,
            decimal_places=8,
            bound_init_mode="warmup_quantile",
            warmup_rounds_for_bound=2,
            initial_bound_multiplier=1.00,
            initial_bound_quantile=0.80,
            use_paper_evolving_bound=True,
            mu_min=0.50,
            mu_max=1.20,
            bound_min=1e-12,
            bound_max=None,
            enable_mgf=True,
            enable_dmc_dmr=True,
            fail_safe_keep_one=False,
            fail_open_when_too_few_valid=False,
            log_client_filter_details=True,
            enable_vss_verification=True,
            allow_weighted_mean=False,
            strict_protocol_checks=True,
        ),
    )


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
            tangent_commitment_enabled=True,
            tangent_basis_rank=64,
            tangent_max_proxy_batches=64,
            tangent_tau_max=0.5,
            tangent_tau_min=0.02,
            tangent_lambda=10.0,
            tangent_eps=1e-12,
            tangent_refresh_interval=1,
        )
    )


# CONFIG = strong_geochoke_config()
CONFIG = aion_config()
