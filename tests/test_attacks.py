import numpy as np
from attacks.no_attack import NoAttack
from attacks.alie import ALIEAttack
from attacks.fang import FangMeanAttack

def test_attacks_modify_only_when_applied_pre_encryption_flag_logic():
    clean=np.ones(5); ctx={'observable_updates':[clean,clean*2],'num_selected':3,'num_malicious':1}
    assert np.allclose(NoAttack().craft_update(0,clean,None,ctx),clean)
    assert not np.allclose(ALIEAttack(z=1.0).craft_update(1,clean,None,ctx),clean)
    assert not np.allclose(FangMeanAttack().craft_update(1,clean,None,ctx),clean)


def test_dba_asr_debiases_clean_target_predictions():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from evaluation.evaluator import Evaluator

    class TargetBiasedModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros((len(x), 3), device=x.device)
            logits[:, 2] = 1.0
            return logits

    loader = DataLoader(TensorDataset(torch.zeros(4, 1, 2, 2), torch.tensor([0, 0, 1, 1])), batch_size=2)
    evaluator = Evaluator(loader, "cpu")
    metrics = evaluator._asr_metrics(TargetBiasedModel(), lambda x: x + 1.0, target_label=2)

    assert metrics["raw_asr"] == 1.0
    assert metrics["clean_target_rate"] == 1.0
    assert metrics["eligible_test_count"] == 0
    assert metrics["asr"] == 0.0
    assert metrics["hits"] == 0


def test_dba_asr_counts_trigger_only_successes():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from evaluation.evaluator import Evaluator

    class TriggerSensitiveModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros((len(x), 3), device=x.device)
            triggered = x.flatten(1).sum(dim=1) > 0
            logits[~triggered, 0] = 1.0
            logits[triggered, 2] = 1.0
            return logits

    loader = DataLoader(TensorDataset(torch.zeros(4, 1, 2, 2), torch.tensor([0, 0, 1, 1])), batch_size=2)
    evaluator = Evaluator(loader, "cpu")
    metrics = evaluator._asr_metrics(TriggerSensitiveModel(), lambda x: x + 1.0, target_label=2)

    assert metrics["raw_asr"] == 1.0
    assert metrics["clean_target_rate"] == 0.0
    assert metrics["eligible_test_count"] == 4
    assert metrics["asr"] == 1.0
    assert metrics["hits"] == 4


def test_new_attack_factories_instantiate():
    from config import make_config
    from factories.attack_factory import create_attack

    expected = {
        "neurotoxin": "NeurotoxinAttack",
        "a3fl": "A3FLAttack",
        "three_dfed": "ThreeDFedAttack",
        "adaptive_geochoke": "AdaptiveGeoChokeAttack",
    }
    for preset, class_name in expected.items():
        cfg = make_config(dataset="mnist", attack=preset, defense="none", scale="debug", seed=7)
        assert create_attack(cfg).__class__.__name__ == class_name


def test_new_vector_attack_craft_outputs_are_valid():
    import numpy as np

    from config import make_config
    from factories.attack_factory import create_attack

    clean_update = np.linspace(-1.0, 1.0, 20, dtype=np.float64)
    observable = [np.linspace(0.0, 2.0, 20, dtype=np.float64), np.linspace(0.0, 1.0, 20, dtype=np.float64)]
    for preset in ["neurotoxin", "a3fl", "three_dfed", "adaptive_geochoke"]:
        cfg = make_config(dataset="mnist", attack=preset, defense="none", scale="debug", seed=7)
        attack = create_attack(cfg)
        crafted = attack.craft_update(1, clean_update.copy(), None, {"observable_updates": observable})
        assert crafted.shape == clean_update.shape
        assert crafted.ndim == 1
        assert np.all(np.isfinite(crafted))


def test_required_attack_presets_still_construct():
    from config import make_config
    from factories.attack_factory import create_attack

    for preset in ["none", "dba_multi", "neurotoxin", "a3fl", "three_dfed", "adaptive_geochoke"]:
        cfg = make_config(dataset="mnist", attack=preset, defense="none", scale="debug", seed=7)
        assert create_attack(cfg) is not None
