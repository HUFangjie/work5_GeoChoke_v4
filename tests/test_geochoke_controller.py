import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import GeoChokeConfig
from crypto.update_codec import ModelUpdateCodec
from defenses.geochoke.cfi_estimator import CFIEstimator
from defenses.geochoke.controller import GeoChokeController
from defenses.geochoke.defense import GeoChokeDefense
from models.mnist_cnn import MNISTCNN


def test_controller_next_round_only_formula_outputs():
    cfg = GeoChokeConfig(initial_profile_id="a")
    calibration = {"a": {"mse": 0.0}, "b": {"mse": 1.0}}
    controller = GeoChokeController(cfg, calibration)
    next_profile, metrics = controller.select(0.1, 0.3, "a")
    assert next_profile in calibration
    assert metrics["selected_next_profile"] == next_profile
    assert "unconstrained_target_error_energy" in metrics


def test_defense_applies_selected_profile_to_next_round_only():
    cfg = GeoChokeConfig(initial_profile_id="a", warmup_rounds=0)
    defense = GeoChokeDefense(cfg, {"a": {}, "b": {}})
    defense.next_profile = "a"

    class DummyEstimator:
        def estimate(self, model):
            return 0.0 if getattr(model, "tag", "previous") == "previous" else 10.0

        def functional_drift(self, previous_model, candidate_model):
            return 0.0

    class DummyController:
        def select(self, prev_cfi, cand_cfi, current_profile_id, **kwargs):
            return "b", {
                "previous_cfi": prev_cfi,
                "candidate_cfi": cand_cfi,
                "fragility_injection_score": max(0.0, cand_cfi - prev_cfi),
                "current_calibrated_error_energy": 0.0,
                "target_next_error_energy": 1.0,
                "selected_next_profile": "b",
                "profile_switching_indicator": True,
            }

    defense.estimator = DummyEstimator()
    defense.controller = DummyController()
    previous = type("DummyModel", (), {"tag": "previous"})()
    candidate = type("DummyModel", (), {"tag": "candidate"})()
    assert defense.get_profile_for_round(0) == "a"
    defense.after_aggregate(previous, candidate, "a", 0)
    assert defense.get_profile_for_round(1) == "b"


def test_cfi_estimator_does_not_mutate_model():
    model = MNISTCNN()
    before = copy.deepcopy(model.state_dict())
    codec = ModelUpdateCodec(model)
    proxy_x = torch.zeros(2, 1, 28, 28)
    proxy_loader = DataLoader(TensorDataset(proxy_x, torch.full((2,), -1)), batch_size=1)
    perturbation = torch.zeros(codec.total_dimension).numpy()
    estimator = CFIEstimator(codec, proxy_loader, [perturbation], "cpu")
    estimator.estimate(model)
    after = model.state_dict()
    for name in before:
        assert torch.equal(before[name], after[name])
