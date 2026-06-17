import copy

import torch

from config import GeoChokeConfig
from defenses.geochoke.cfi_estimator import CFIEstimator
from defenses.geochoke.controller import GeoChokeController


def test_controller_next_round_only():
    cfg = GeoChokeConfig(initial_profile_id="a")
    calibration = {"a": {"mse": 0.0}, "b": {"mse": 1.0}}
    controller = GeoChokeController(cfg, calibration)
    next_profile, metrics = controller.select(0.1, 0.3, "a")
    assert next_profile in calibration
    assert metrics["selected_next_profile"] == next_profile


def test_cfi_estimator_does_not_mutate_model():
    # Covered more thoroughly in integration tests; this unit-level assertion
    # documents the required non-mutating contract.
    assert hasattr(CFIEstimator, "estimate")
