import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from defenses.geochoke.cfi_estimator import CFIEstimator
from defenses.geochoke.controller import GeoChokeController
from config import GeoChokeConfig


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x.view(x.shape[0], -1))


class TinyCodec:
    def flatten_state_dict(self, state):
        return np.concatenate([tensor.detach().cpu().numpy().reshape(-1) for tensor in state.values()]).astype(np.float64)

    def apply_update_to_state_dict(self, state, update, step_size=1.0):
        out = {}
        cursor = 0
        for name, tensor in state.items():
            width = tensor.numel()
            delta = torch.tensor(update[cursor:cursor + width].reshape(tensor.shape), dtype=tensor.dtype)
            out[name] = tensor + step_size * delta
            cursor += width
        return out


def test_js_divergence_nonnegative():
    base = torch.tensor([[2.0, -1.0], [0.0, 1.0]])
    perturbed = torch.tensor([[-1.0, 2.0], [1.0, 0.0]])
    js = CFIEstimator.js_divergence_from_logits(base, perturbed)
    assert torch.all(js >= 0.0)


def test_cfi_zero_perturbation_is_zero():
    model = TinyModel()
    loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.zeros(4, dtype=torch.long)), batch_size=2)
    codec = TinyCodec()
    zero = np.zeros_like(codec.flatten_state_dict(model.state_dict()))
    cfi = CFIEstimator(codec, loader, [zero], "cpu").estimate(model)
    assert cfi >= 0.0
    assert cfi < 1e-10


def test_cfi_random_perturbation_nonnegative():
    model = TinyModel()
    loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.zeros(4, dtype=torch.long)), batch_size=2)
    codec = TinyCodec()
    rng = np.random.default_rng(7)
    perturbation = rng.normal(0.0, 0.01, size=codec.flatten_state_dict(model.state_dict()).shape)
    cfi = CFIEstimator(codec, loader, [perturbation], "cpu").estimate(model)
    assert cfi >= 0.0


def test_geochoke_metrics_never_negative():
    cfg = GeoChokeConfig()
    calibration = {
        "ckks_s40": {"mse": 1e-10, "valid_profile": True},
        "ckks_s30": {"mse": 1e-6, "valid_profile": True},
    }
    _, metrics = GeoChokeController(cfg, calibration).select(0.0, 0.1, "ckks_s40")
    assert metrics["previous_cfi"] >= 0.0
    assert metrics["candidate_cfi"] >= 0.0
    assert metrics["fragility_injection_score"] >= 0.0
    assert metrics["cfi_nonnegative_check"] is True
