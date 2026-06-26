from dataclasses import dataclass, field, asdict, replace
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

VALID_DATASETS = {"mnist", "fashion_mnist", "cifar10"}
VALID_ATTACKS = {"none", "alie", "fang_mean", "dba"}
VALID_DEFENSES = {"none", "no_defense", "geochoke"}


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
    num_rounds: int = 20
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
    "batch_size": ("dataset", "batch_size"),
    "partition_type": ("dataset", "partition_type"),
    "dirichlet_alpha": ("dataset", "dirichlet_alpha"),
    "dataset_name": ("dataset", "name"),
    "num_classes": ("dataset", "num_classes"),
    "data_dir": ("dataset", "data_dir"),
    "download_data": ("dataset", "download"),
    "quick_data_limit": ("dataset", "quick_data_limit"),
    "proxy_size": ("dataset", "proxy_size"),
    "test_size": ("dataset", "test_size"),
    "model_name": ("dataset", "model_name"),
    "input_channels": ("dataset", "input_channels"),
    "image_size": ("dataset", "image_size"),
    "malicious_client_ids": ("attack", "malicious_client_ids"),
    "attack_name": ("attack", "name"),
    "defense_name": ("defense", "name"),
    "crypto_backend_name": ("crypto", "backend_name"),
    "ckks_profiles": ("crypto", "profiles"),
    **{k: ("attack", k) for k in AttackConfig.__dataclass_fields__ if k != "name"},
}


def _install_alias(name: str, target: tuple[str, str]) -> None:
    section_name, field_name = target

    def getter(self: ExperimentConfig) -> Any:
        return getattr(getattr(self, section_name), field_name)

    def setter(self: ExperimentConfig, value: Any) -> None:
        setattr(getattr(self, section_name), field_name, value)

    setattr(ExperimentConfig, name, property(getter, setter))


for _alias, _target in _ALIAS_MAP.items():
    _install_alias(_alias, _target)


def _replace_with_overrides(config: Any, overrides: Dict[str, Any]) -> Any:
    try:
        return replace(config, **overrides)
    except TypeError as exc:
        raise ValueError(f"Invalid override for {type(config).__name__}: {exc}") from exc


def dataset_config(name: str, **overrides: Any) -> DatasetConfig:
    # Dataset presets include shape/model metadata and a sensible default data
    # budget so switching datasets usually does not require dataset_overrides.
    presets = {
        "mnist": DatasetConfig(
            name="mnist",
            input_channels=1,
            image_size=28,
            model_name="mnist_cnn",
            quick_data_limit=600,
            proxy_size=64,
            test_size=256,
        ),
        "fashion_mnist": DatasetConfig(
            name="fashion_mnist",
            input_channels=1,
            image_size=28,
            model_name="mnist_cnn",
            quick_data_limit=600,
            proxy_size=64,
            test_size=256,
        ),
        "cifar10": DatasetConfig(
            name="cifar10",
            input_channels=3,
            image_size=32,
            model_name="cifar10_cnn",
            quick_data_limit=6000,
            proxy_size=256,
            test_size=2000,
            partition_type="dirichlet",
            dirichlet_alpha=0.5,
        ),
    }
    if name not in presets:
        raise ValueError(f"Unknown dataset preset: {name}. Available: {sorted(presets)}")
    return _replace_with_overrides(presets[name], overrides)


def training_config(scale: str = "debug", **overrides: Any) -> TrainingConfig:
    presets = {
        "debug": TrainingConfig(
            num_clients=5,
            clients_per_round=5,
            min_clients_per_round=2,
            num_rounds=20,
            local_epochs=1,
            local_lr=0.01,
            server_lr=1.0,
            aggregation="weighted_mean",
        ),
        "standard": TrainingConfig(
            num_clients=10,
            clients_per_round=10,
            min_clients_per_round=5,
            num_rounds=50,
            local_epochs=1,
            local_lr=0.01,
            server_lr=1.0,
            aggregation="weighted_mean",
        ),
        "paper": TrainingConfig(
            num_clients=20,
            clients_per_round=10,
            min_clients_per_round=5,
            num_rounds=100,
            local_epochs=1,
            local_lr=0.01,
            server_lr=1.0,
            aggregation="weighted_mean",
        ),
    }
    if scale not in presets:
        raise ValueError(f"Unknown training scale: {scale}. Available: {sorted(presets)}")
    return _replace_with_overrides(presets[scale], overrides)


def attack_config(name: str = "none", **overrides: Any) -> AttackConfig:
    presets = {
        "none": AttackConfig(name="none", malicious_client_ids=[]),
        "dba_multi": AttackConfig(
            name="dba",
            malicious_client_ids=[1, 2, 3, 4],
            dba_target_label=2,
            dba_poison_ratio=0.3125,
            dba_local_epochs=10,
            dba_local_lr=0.05,
            dba_attack_mode="multi_shot",
            dba_attack_start_round=10,
            dba_attack_end_round=19,
            dba_poison_interval=1,
            dba_num_trigger_parts=4,
            dba_trigger_size=4,
            dba_trigger_gap=2,
            dba_trigger_location="top_left",
            dba_trigger_value=1.0,
            dba_multi_shot_scale_factor=1.0,
            dba_single_shot_scale_factor=20.0,
        ),
        "dba_single": AttackConfig(
            name="dba",
            malicious_client_ids=[1, 2, 3, 4],
            dba_target_label=2,
            dba_poison_ratio=0.5,
            dba_local_epochs=10,
            dba_local_lr=0.05,
            dba_attack_mode="single_shot",
            dba_attack_start_round=10,
            dba_attack_end_round=20,
            dba_poison_interval=1,
            dba_num_trigger_parts=4,
            dba_trigger_size=4,
            dba_trigger_gap=2,
            dba_trigger_location="top_left",
            dba_trigger_value=1.0,
            dba_multi_shot_scale_factor=1.0,
            dba_single_shot_scale_factor=20.0,
        ),
        "alie": AttackConfig(
            name="alie",
            malicious_client_ids=[1, 2],
            alie_z=None,
            alie_oracle_all_updates=False,
        ),
        "fang_mean": AttackConfig(
            name="fang_mean",
            malicious_client_ids=[1, 2],
            fang_max_norm=5.0,
            fang_search_steps=6,
        ),
    }
    if name not in presets:
        raise ValueError(f"Unknown attack preset: {name}. Available: {sorted(presets)}")
    return _replace_with_overrides(presets[name], overrides)


def geochoke_config(name: str = "strong", **overrides: Any) -> GeoChokeConfig:
    presets = {
        "off": GeoChokeConfig(
            initial_profile_id="ckks_s40",
            reference_profile_id="ckks_s40",
            tangent_commitment_enabled=False,
        ),
        "mild": GeoChokeConfig(
            initial_profile_id="ckks_s40",
            reference_profile_id="ckks_s30",
            calibration_vectors=8,
            perturbation_count=8,
            perturbation_scale=1.0,
            lambda_=1.0,
            gamma=1.0,
            rho=1.0,
            tangent_commitment_enabled=True,
            tangent_basis_rank=8,
            tangent_tau_max=0.50,
            tangent_tau_min=0.05,
            tangent_lambda=2.0,
        ),
        "strong": GeoChokeConfig(
            initial_profile_id="ckks_s40",
            reference_profile_id="ckks_s28",
            calibration_vectors=16,
            perturbation_count=16,
            perturbation_scale=1.0,
            lambda_=50.0,
            gamma=0.01,
            rho=0.01,
            tangent_commitment_enabled=True,
            tangent_basis_rank=16,
            tangent_max_proxy_batches=16,
            tangent_tau_max=0.20,
            tangent_tau_min=0.02,
            tangent_lambda=10.0,
            tangent_refresh_interval=1,
        ),
    }
    if name not in presets:
        raise ValueError(f"Unknown GeoChoke preset: {name}. Available: {sorted(presets)}")
    return _replace_with_overrides(presets[name], overrides)


def defense_config(name: str = "none") -> DefenseConfig:
    mapping = {
        "none": "none",
        "no_defense": "none",
        "geochoke": "geochoke",
        "geochoke_mild": "geochoke",
        "geochoke_strong": "geochoke",
    }
    if name not in mapping:
        raise ValueError(f"Unknown defense preset: {name}. Available: {sorted(mapping)}")
    return DefenseConfig(name=mapping[name])


def _geochoke_preset_for_defense(defense: str) -> str:
    if defense in {"geochoke", "geochoke_strong"}:
        return "strong"
    if defense == "geochoke_mild":
        return "mild"
    if defense in {"none", "no_defense"}:
        return "off"
    raise ValueError(f"Unknown defense preset: {defense}")


def make_config(
    dataset: str = "mnist",
    attack: str = "none",
    defense: str = "none",
    scale: str = "debug",
    seed: int = 7,
    device: str = "cpu",
    output_dir: str = "./outputs",
    dataset_overrides: Dict[str, Any] | None = None,
    training_overrides: Dict[str, Any] | None = None,
    attack_overrides: Dict[str, Any] | None = None,
    geochoke_overrides: Dict[str, Any] | None = None,
) -> ExperimentConfig:
    cfg = ExperimentConfig(
        seed=seed,
        device=device,
        dataset=dataset_config(dataset, **(dataset_overrides or {})),
        training=training_config(scale, **(training_overrides or {})),
        attack=attack_config(attack, **(attack_overrides or {})),
        defense=defense_config(defense),
        geochoke=geochoke_config(_geochoke_preset_for_defense(defense), **(geochoke_overrides or {})),
        crypto=CryptoConfig(),
        output_dir=f"{output_dir}/{dataset}",
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.attack.name not in VALID_ATTACKS:
        raise ValueError(f"Invalid attack.name: {cfg.attack.name}. Expected one of {sorted(VALID_ATTACKS)}")
    if cfg.defense.name not in VALID_DEFENSES:
        raise ValueError(f"Invalid defense.name: {cfg.defense.name}. Expected one of {sorted(VALID_DEFENSES)}")
    if cfg.dataset.name not in VALID_DATASETS:
        raise ValueError(f"Invalid dataset.name: {cfg.dataset.name}. Expected one of {sorted(VALID_DATASETS)}")
    if cfg.training.clients_per_round > cfg.training.num_clients:
        raise ValueError("clients_per_round must be <= num_clients")
    if cfg.training.min_clients_per_round > cfg.training.clients_per_round:
        raise ValueError("min_clients_per_round must be <= clients_per_round")
    invalid_clients = [cid for cid in cfg.attack.malicious_client_ids if cid < 0 or cid >= cfg.training.num_clients]
    if invalid_clients:
        raise ValueError(f"malicious_client_ids must be in [0, {cfg.training.num_clients - 1}], got {invalid_clients}")
    if cfg.attack.name == "none" and cfg.attack.malicious_client_ids:
        raise ValueError("malicious_client_ids must be empty when attack.name == 'none'")
    if cfg.attack.name == "dba":
        if cfg.attack.dba_attack_mode not in {"multi_shot", "single_shot"}:
            raise ValueError("dba_attack_mode must be 'multi_shot' or 'single_shot'")
        if not 0 <= cfg.attack.dba_attack_start_round <= cfg.attack.dba_attack_end_round < cfg.training.num_rounds:
            raise ValueError("DBA schedule must satisfy 0 <= start_round <= end_round < num_rounds")
        if len(cfg.attack.malicious_client_ids) < cfg.attack.dba_num_trigger_parts:
            raise ValueError("DBA requires at least dba_num_trigger_parts malicious clients")
        if not 0 <= cfg.attack.dba_target_label < cfg.dataset.num_classes:
            raise ValueError("dba_target_label must satisfy 0 <= target_label < dataset.num_classes")
        if not 0.0 < cfg.attack.dba_poison_ratio < 1.0:
            raise ValueError("dba_poison_ratio must satisfy 0.0 < ratio < 1.0")
        required_width = (
            cfg.attack.dba_num_trigger_parts * cfg.attack.dba_trigger_size
            + (cfg.attack.dba_num_trigger_parts - 1) * cfg.attack.dba_trigger_gap
        )
        if required_width > cfg.dataset.image_size:
            raise ValueError(
                f"DBA trigger width {required_width} does not fit image_size={cfg.dataset.image_size}"
            )
    if cfg.geochoke.initial_profile_id not in cfg.crypto.profiles:
        raise ValueError(f"Unknown geochoke.initial_profile_id: {cfg.geochoke.initial_profile_id}")
    if cfg.geochoke.reference_profile_id not in cfg.crypto.profiles:
        raise ValueError(f"Unknown geochoke.reference_profile_id: {cfg.geochoke.reference_profile_id}")
    if cfg.defense.name == "none" and cfg.geochoke.tangent_commitment_enabled:
        raise ValueError("geochoke.tangent_commitment_enabled must be False when defense.name == 'none'")


def strong_geochoke_config(dataset_name: str = "mnist") -> ExperimentConfig:
    return make_config(dataset=dataset_name, attack="dba_multi", defense="geochoke_strong", scale="standard", seed=7)


def mild_geochoke_config(dataset_name: str = "mnist") -> ExperimentConfig:
    return make_config(dataset=dataset_name, attack="dba_multi", defense="geochoke_mild", scale="standard", seed=7)


# Common experiment templates:
# 1. Clean baseline:
# CONFIG = make_config(dataset="mnist", attack="none", defense="none", scale="standard", seed=7)
#
# 2. DBA without defense:
# CONFIG = make_config(dataset="mnist", attack="dba_multi", defense="none", scale="standard", seed=7)
#
# 3. DBA with GeoChoke:
# CONFIG = make_config(dataset="mnist", attack="dba_multi", defense="geochoke_strong", scale="standard", seed=7)
#
# 4. Fashion-MNIST:
# CONFIG = make_config(dataset="fashion_mnist", attack="dba_multi", defense="geochoke_strong", scale="standard", seed=7)
#
# 5. CIFAR-10:
# CONFIG = make_config(
#     dataset="cifar10",
#     attack="dba_multi",
#     defense="geochoke_strong",
#     scale="standard",
#     seed=7,
#     training_overrides={
#         "num_clients": 10,
#         "clients_per_round": 10,
#         "num_rounds": 80,
#     },
# )

CONFIG = make_config(
    dataset="mnist",
    attack="dba_multi",
    defense="geochoke_strong",
    scale="standard",
    seed=7,
)
