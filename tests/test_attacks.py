import numpy as np

from attacks.alie import ALIEAttack
from attacks.fang import FangMeanAttack
from attacks.no_attack import NoAttack


def test_attacks_modify_only_when_applied_pre_encryption_flag_logic():
    clean = np.ones(5)
    ctx = {"observable_updates": [clean, clean * 2], "num_selected": 3, "num_malicious": 1}
    assert np.allclose(NoAttack().craft_update(0, clean, None, ctx), clean)
    assert not np.allclose(ALIEAttack(z=1.0).craft_update(1, clean, None, ctx), clean)
    assert not np.allclose(FangMeanAttack().craft_update(1, clean, None, ctx), clean)


def test_whitebox_alie_uses_all_updates_and_opposes_clean_aggregate():
    updates = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]), np.array([3.0, 4.0, 5.0])]
    clean_aggregate = sum(updates) / len(updates)
    ctx = {
        "whitebox": True,
        "all_clean_updates": updates,
        "clean_aggregate": clean_aggregate,
        "num_selected": 3,
        "num_malicious": 1,
    }
    crafted = ALIEAttack(whitebox=True, whitebox_z=2.5, strength=2.0).craft_update(1, updates[0], None, ctx)
    assert np.dot(crafted, clean_aggregate) < np.dot(updates[0], clean_aggregate)


def test_whitebox_fang_targets_reversed_aggregate_direction():
    updates = [np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([3.0, 3.0])]
    weights = [0.2, 0.3, 0.5]
    clean_aggregate = sum(w * u for w, u in zip(weights, updates))
    benign_weighted_sum = weights[0] * updates[0] + weights[1] * updates[1]
    ctx = {
        "whitebox": True,
        "all_clean_updates": updates,
        "clean_aggregate": clean_aggregate,
        "benign_weighted_sum": benign_weighted_sum,
        "malicious_weight": weights[2],
        "malicious_total_weight": weights[2],
        "num_selected": 3,
        "num_malicious": 1,
    }
    crafted = FangMeanAttack(max_norm=100.0, target_scale=2.0, whitebox=True).craft_update(2, updates[2], None, ctx)
    poisoned = benign_weighted_sum + weights[2] * crafted
    assert np.dot(poisoned, clean_aggregate) < 0.0
