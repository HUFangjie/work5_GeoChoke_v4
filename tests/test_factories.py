import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import CKKS_PROFILES, ExperimentConfig
from data.mnist import MNISTSplits
from factories import defaults
from factories.crypto_factory import create_crypto_backend
from factories.dataset_factory import create_dataset_provider, register_dataset
from factories.defense_factory import create_defense, register_defense
from factories.model_factory import create_model_factory


@register_dataset("dummy")
class DummyDatasetProvider:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self):
        x = torch.zeros(12, 1, 28, 28)
        y = torch.arange(12) % 10
        loaders = [DataLoader(TensorDataset(x[:6], y[:6]), batch_size=3), DataLoader(TensorDataset(x[6:], y[6:]), batch_size=3)]
        proxy = DataLoader(TensorDataset(x[:4], torch.full((4,), -1)), batch_size=2)
        test = DataLoader(TensorDataset(x[:4], y[:4]), batch_size=2)
        metadata = [
            {"client_id": 0, "num_samples": 6, "label_distribution": {0: 1}, "malicious": False},
            {"client_id": 1, "num_samples": 6, "label_distribution": {1: 1}, "malicious": False},
        ]
        return MNISTSplits(loaders, proxy, test, metadata)


@register_defense("dummy_defense")
class DummyDefense:
    def __init__(self, cfg):
        self.profile = cfg.geochoke.initial_profile_id

    def initialize(self, model, crypto_backend, proxy_loader, decrypt_aggregate_fn=None):
        return None

    def get_profile_for_round(self, round_id):
        return self.profile

    def after_aggregate(self, previous_model, candidate_model, current_profile_id, round_id):
        return {
            "previous_cfi": 0.0,
            "candidate_cfi": 0.0,
            "fragility_injection_score": 0.0,
            "current_calibrated_error_energy": 0.0,
            "target_next_error_energy": 0.0,
            "selected_next_profile": self.profile,
            "profile_switching_indicator": False,
        }


def test_factories_create_builtin_components():
    cfg = ExperimentConfig(num_clients=2, clients_per_round=2, malicious_client_ids=[], ckks_profiles=CKKS_PROFILES)
    assert create_dataset_provider(cfg) is not None
    assert create_model_factory(cfg).create() is not None
    assert create_crypto_backend(cfg).public_backend is not None
    assert create_defense(cfg) is not None


def test_unknown_factory_names_raise_value_error():
    cfg = ExperimentConfig()
    cfg.dataset_name = "unknown_dataset"
    with pytest.raises(ValueError):
        create_dataset_provider(cfg)
    cfg = ExperimentConfig()
    cfg.defense_name = "unknown_defense"
    with pytest.raises(ValueError):
        create_defense(cfg)


def test_dummy_dataset_and_defense_registered_without_coordinator_changes():
    cfg = ExperimentConfig(num_clients=2, clients_per_round=2, malicious_client_ids=[], ckks_profiles=CKKS_PROFILES)
    cfg.dataset_name = "dummy"
    cfg.defense_name = "dummy_defense"
    assert isinstance(create_dataset_provider(cfg), DummyDatasetProvider)
    assert isinstance(create_defense(cfg), DummyDefense)


def test_no_defense_factory_keeps_initial_profile():
    cfg = ExperimentConfig(ckks_profiles=CKKS_PROFILES)
    cfg.defense_name = "none"
    defense = create_defense(cfg)
    assert defense.get_profile_for_round(0) == cfg.geochoke.initial_profile_id
    metrics = defense.after_aggregate(None, None, cfg.geochoke.initial_profile_id, 0)
    assert metrics["defense_enabled"] is False
    assert metrics["selected_next_profile"] == cfg.geochoke.initial_profile_id
