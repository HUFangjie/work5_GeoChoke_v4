from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

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
class DatasetConfig:
    name: str = "mnist"
    data_dir: str = "./data_cache"
    download: bool = True
    num_classes: int = 10
    input_channels: int = 1
    image_size: int = 28
    quick_data_limit: int | None = 600
    proxy_size: int = 64
    test_size: int = 256
    partition_type: str = "iid"
    dirichlet_alpha: float = 0.5
    batch_size: int = 32
    model_name: str = "mnist_cnn"

@dataclass
class AttackConfig:
    name: str = "dba"  # none, alie, fang_mean, dba
    malicious_client_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
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
    dba_attack_mode: str = "multi_shot"
    dba_attack_start_round: int = 10
    dba_attack_end_round: int = 19
    dba_poison_interval: int = 1
    dba_num_trigger_parts: int = 4
    dba_trigger_size: int = 4
    dba_trigger_gap: int = 2
    dba_trigger_location: str = "top_left"
    dba_trigger_value: float = 1.0

@dataclass
class DefenseConfig:
    name: str = "geochoke"

@dataclass
class CryptoConfig:
    backend_name: str = "ckks"
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=lambda: CKKS_PROFILES)

@dataclass
class TrainingConfig:
    num_clients: int = 5
    clients_per_round: int = 5
    min_clients_per_round: int = 2
    num_rounds: int = 50
    local_epochs: int = 1
    local_lr: float = 0.01
    server_lr: float = 1.0
    aggregation: str = "weighted_mean"

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
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    geochoke: GeoChokeConfig = field(default_factory=GeoChokeConfig)
    output_dir: str = "./outputs"
    log_level: str = "INFO"
    enable_plaintext_reference_metrics: bool = True
    pipeline_validation_rtol: float = 5e-2
    pipeline_validation_atol: float = 5e-2
    pipeline_validation_norm_ratio_tolerance: float = 5e-2

    def __init__(self, **kwargs: Any) -> None:
        # Keep the historic flat constructor working while storing new configs structurally.
        self.seed = kwargs.pop("seed", 7)
        self.device = kwargs.pop("device", "cpu")
        self.training = kwargs.pop("training", TrainingConfig())
        self.dataset = kwargs.pop("dataset", DatasetConfig())
        self.attack = kwargs.pop("attack", AttackConfig())
        self.defense = kwargs.pop("defense", DefenseConfig())
        self.crypto = kwargs.pop("crypto", CryptoConfig())
        self.geochoke = kwargs.pop("geochoke", GeoChokeConfig())
        self.output_dir = kwargs.pop("output_dir", "./outputs")
        self.log_level = kwargs.pop("log_level", "INFO")
        self.enable_plaintext_reference_metrics = kwargs.pop("enable_plaintext_reference_metrics", True)
        self.pipeline_validation_rtol = kwargs.pop("pipeline_validation_rtol", 5e-2)
        self.pipeline_validation_atol = kwargs.pop("pipeline_validation_atol", 5e-2)
        self.pipeline_validation_norm_ratio_tolerance = kwargs.pop("pipeline_validation_norm_ratio_tolerance", 5e-2)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

_ALIAS_MAP = {
    **{k: ("training", k) for k in TrainingConfig.__dataclass_fields__},
    "batch_size": ("dataset", "batch_size"), "partition_type": ("dataset", "partition_type"),
    "dirichlet_alpha": ("dataset", "dirichlet_alpha"), "dataset_name": ("dataset", "name"),
    "num_classes": ("dataset", "num_classes"), "data_dir": ("dataset", "data_dir"),
    "download_data": ("dataset", "download"), "quick_data_limit": ("dataset", "quick_data_limit"),
    "proxy_size": ("dataset", "proxy_size"), "test_size": ("dataset", "test_size"),
    "model_name": ("dataset", "model_name"), "input_channels": ("dataset", "input_channels"),
    "image_size": ("dataset", "image_size"), "malicious_client_ids": ("attack", "malicious_client_ids"),
    "attack_name": ("attack", "name"), "defense_name": ("defense", "name"),
    "crypto_backend_name": ("crypto", "backend_name"), "ckks_profiles": ("crypto", "profiles"),
    **{k: ("attack", k) for k in AttackConfig.__dataclass_fields__ if k != "name"},
}

def _install_alias(name: str, target: tuple[str, str]) -> None:
    section_name, field_name = target
    def getter(self: ExperimentConfig) -> Any:
        return getattr(getattr(self, section_name), field_name)
    def setter(self: ExperimentConfig, value: Any) -> None:
        getattr(self, section_name).__setattr__(field_name, value)
    setattr(ExperimentConfig, name, property(getter, setter))

for _alias, _target in _ALIAS_MAP.items():
    _install_alias(_alias, _target)


def dataset_config(name: str) -> DatasetConfig:
    presets = {
        "mnist": DatasetConfig(name="mnist", input_channels=1, image_size=28, model_name="mnist_cnn"),
        "fashion_mnist": DatasetConfig(name="fashion_mnist", input_channels=1, image_size=28, model_name="mnist_cnn"),
        "cifar10": DatasetConfig(name="cifar10", input_channels=3, image_size=32, model_name="cifar10_cnn"),
    }
    if name not in presets:
        raise ValueError(f"Unknown dataset preset: {name}. Available: {sorted(presets)}")
    return presets[name]

def strong_geochoke_config(dataset_name: str = "mnist") -> ExperimentConfig:
    return ExperimentConfig(dataset=dataset_config(dataset_name))

def mild_geochoke_config(dataset_name: str = "mnist") -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_config(dataset_name),
        geochoke=GeoChokeConfig(
            initial_profile_id="ckks_s40", reference_profile_id="ckks_s30", calibration_vectors=8,
            perturbation_count=8, perturbation_scale=1.0, lambda_=1.0, gamma=1.0, rho=1.0,
        ),
    )

CONFIG = strong_geochoke_config()
