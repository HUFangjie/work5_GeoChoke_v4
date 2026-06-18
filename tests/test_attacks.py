import numpy as np

from attacks.alie import ALIEAttack
from attacks.fang import FangMeanAttack, SignFlipScaledAttack
from attacks.no_attack import NoAttack


def test_attacks_modify_only_when_applied_pre_encryption_flag_logic():
    clean = np.ones(5)
    ctx = {"observable_updates": [clean, clean * 2], "num_selected": 3, "num_malicious": 1}
    assert np.allclose(NoAttack().craft_update(0, clean, None, ctx), clean)
    assert not np.allclose(ALIEAttack(z=1.0).craft_update(1, clean, None, ctx), clean)
    assert not np.allclose(FangMeanAttack().craft_update(1, clean, None, ctx), clean)


def test_oracle_alie_uses_all_updates_and_opposes_clean_aggregate():
    updates = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]), np.array([3.0, 4.0, 5.0])]
    clean_aggregate = sum(updates) / len(updates)
    ctx = {
        "oracle_mean_replacement": True,
        "all_clean_updates": updates,
        "clean_aggregate": clean_aggregate,
        "num_selected": 3,
        "num_malicious": 1,
    }
    crafted = ALIEAttack(oracle_mean_replacement=True, whitebox_z=2.5, strength=2.0).craft_update(1, updates[0], None, ctx)
    assert np.dot(crafted, clean_aggregate) < np.dot(updates[0], clean_aggregate)


def test_oracle_fang_targets_reversed_aggregate_direction():
    updates = [np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([3.0, 3.0])]
    weights = [0.2, 0.3, 0.5]
    clean_aggregate = sum(w * u for w, u in zip(weights, updates))
    benign_weighted_sum = weights[0] * updates[0] + weights[1] * updates[1]
    ctx = {
        "oracle_mean_replacement": True,
        "all_clean_updates": updates,
        "clean_aggregate": clean_aggregate,
        "benign_weighted_sum": benign_weighted_sum,
        "malicious_weight": weights[2],
        "malicious_total_weight": weights[2],
        "num_selected": 3,
        "num_malicious": 1,
    }
    crafted = FangMeanAttack(max_norm=100.0, target_scale=2.0, oracle_mean_replacement=True).craft_update(2, updates[2], None, ctx)
    poisoned = benign_weighted_sum + weights[2] * crafted
    assert np.dot(poisoned, clean_aggregate) < 0.0


def test_non_oracle_fang_is_reported_as_sign_flip_scaled():
    clean = np.array([1.0, -2.0, 3.0])
    ctx = {"observable_updates": [clean], "num_selected": 3, "num_malicious": 1, "malicious_weight": 1 / 3}
    attack = FangMeanAttack(max_norm=1.0, oracle_mean_replacement=False)
    crafted = attack.craft_update(1, clean, None, ctx)
    assert attack.effective_attack_name == "sign_flip_scaled"
    assert np.dot(crafted, clean) < 0.0


def test_sign_flip_scaled_attack_alias_is_explicit():
    attack = SignFlipScaledAttack(max_norm=1.0)
    assert attack.effective_attack_name == "sign_flip_scaled"


def test_dba_local_triggers_are_disjoint_and_global_is_union():
    import torch
    from config import ExperimentConfig
    from attacks.dba import DBAAttack

    cfg = ExperimentConfig()
    cfg.attack_name = "dba"
    cfg.malicious_client_ids = [1, 2, 3, 4]
    attack = DBAAttack(cfg)
    image = torch.zeros(1, 28, 28)
    local_ones = []
    occupied = set()
    for trigger_id in range(cfg.dba_num_trigger_parts):
        triggered = attack.apply_local_trigger(image, trigger_id)
        coords = set(map(tuple, torch.nonzero(triggered[0] > 0.0).tolist()))
        assert coords
        assert not occupied.intersection(coords)
        occupied.update(coords)
        local_ones.append(coords)
    global_triggered = attack.apply_global_trigger(image)
    global_coords = set(map(tuple, torch.nonzero(global_triggered[0] > 0.0).tolist()))
    assert global_coords == set().union(*local_ones)


def test_dba_requires_enough_malicious_clients():
    import pytest
    from config import ExperimentConfig
    from attacks.dba import DBAAttack

    cfg = ExperimentConfig()
    cfg.malicious_client_ids = [1]
    cfg.dba_num_trigger_parts = 4
    with pytest.raises(ValueError):
        DBAAttack(cfg)
