import numpy as np

from config import CKKS_PROFILES, ExperimentConfig, GeoChokeConfig
from core.coordinator import FederatedCoordinator
from factories import defaults
from factories.attack_factory import create_attack
from factories.crypto_factory import create_crypto_backend
from factories.dataset_factory import create_dataset_provider
from factories.defense_factory import create_defense
from factories.model_factory import create_model_factory
from utils.logger import setup_logger
from utils.seed import set_seed
import tests.test_factories  # registers dummy dataset/defense


def test_end_to_end_smoke(tmp_path):
    cfg = ExperimentConfig(
        num_clients=2,
        clients_per_round=2,
        min_clients_per_round=2,
        malicious_client_ids=[],
        num_rounds=2,
        quick_data_limit=64,
        proxy_size=8,
        test_size=16,
        output_dir=str(tmp_path),
        download_data=False,
        ckks_profiles=CKKS_PROFILES,
        geochoke=GeoChokeConfig(calibration_vectors=1, perturbation_count=1),
    )
    cfg.dataset_name = "dummy"
    cfg.defense_name = "dummy_defense"
    cfg.attack_name = "none"
    cfg.attack_type = "none"
    set_seed(cfg.seed)
    crypto_bundle = create_crypto_backend(cfg)
    coordinator = FederatedCoordinator(
        cfg=cfg,
        dataset_provider=create_dataset_provider(cfg),
        model_factory=create_model_factory(cfg),
        crypto_backend=crypto_bundle.public_backend,
        decryption_service=crypto_bundle.decryption_service,
        defense=create_defense(cfg, crypto_bundle),
        attack=create_attack(cfg),
        logger=setup_logger("WARNING"),
    )
    rows = coordinator.run()
    assert len(rows) == 2
    assert rows[0]["ciphertext_block_count"] >= 1
    assert rows[0]["selected_next_profile"] in CKKS_PROFILES
    assert np.isfinite(rows[0]["aggregate_update_norm"])
