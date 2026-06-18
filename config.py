from dataclasses import dataclass, field
from typing import Dict, List, Any


CKKS_PROFILES: Dict[str, Dict[str, Any]] = {
    "high_precision": {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "global_scale_bits": 40,
    },
    "medium_precision": {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 30, 30, 60],
        "global_scale_bits": 30,
    },
    "low_precision": {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 24, 24, 60],
        "global_scale_bits": 24,
    },
}


@dataclass
class GeoChokeConfig:
    initial_profile_id: str = "high_precision"
    warmup_rounds: int = 10
    reference_profile_id: str = "medium_precision"
    calibration_vectors: int = 32
    calibration_repetitions: int = 8
    perturbation_count: int = 32
    perturbation_scale: float = 1.0

    # Gate thresholds are learned from warmup statistics; this value remains as a safe bootstrap floor only.
    gate_risk_threshold: float = 1.0
    gate_kappa: float = 1.0
    gate_kappa_cfi: float = 1.0
    gate_kappa_injection: float = 1.0
    warmup_stats_window: int = 5
    threshold_quantile: float = 0.95
    min_threshold: float = 1e-12

    enable_cumulative_gate: bool = True
    cusum_threshold: float = 2.0
    cusum_decay: float = 0.25
    rollback_recovery_rounds: int = 3
    recovery_max_candidate_scale: float = 0.25

    lambda_: float = 20.0
    gamma: float = 0.25
    rho: float = 0.5
    safe_rounds_for_profile_recovery: int = 3


@dataclass
class ExperimentConfig:
    seed: int = 7
    device: str = "cpu"
    num_clients: int = 10
    clients_per_round: int = 10
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
    attack_name: str = "dba"  # none, alie, fang_mean, sign_flip_scaled, dba
    attack_type: str = "dba"  # backward-compatible alias
    aggregation: str = "weighted_mean"
    model_name: str = "mnist_cnn"
    dataset_name: str = "mnist"
    data_dir: str = "./data_cache"
    download_data: bool = True
    quick_data_limit: int = 59000
    proxy_size: int = 1000
    test_size: int = 10000
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
    alie_whitebox_z: float = 2.5
    alie_strength: float = 1.5

    attack_start_round: int = 10
    attack_end_round: int = 29
    attack_whitebox: bool = False
    oracle_mean_replacement: bool = False

    fang_max_norm: float = 1.0
    fang_search_steps: int = 10
    fang_target_scale: float = 1.0

    dba_target_label: int = 2
    dba_poison_ratio: float = 0.3125
    dba_local_epochs: int = 10
    dba_local_lr: float = 0.05
    dba_scale_factor: float = 1.0
    dba_attack_mode: str = "multi_shot"
    dba_attack_start_round: int = 10
    dba_attack_end_round: int = 29
    dba_poison_interval: int = 1
    dba_num_trigger_parts: int = 4
    dba_trigger_size: int = 4
    dba_trigger_gap: int = 2
    dba_trigger_location: str = "top_left"
    dba_trigger_value: float = 1.0

    candidate_scales: List[float] = field(default_factory=lambda: [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.0])

    # Backward-compatible aliases read by older scripts; GeoChokeConfig drives the actual gate.
    gate_risk_threshold: float = 1.0
    gate_kappa: float = 1.0


CONFIG = ExperimentConfig()
