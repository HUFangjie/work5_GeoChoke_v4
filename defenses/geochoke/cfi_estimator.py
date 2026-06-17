from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F


class CFIEstimator:
    """Online unlabeled CFI estimator with fixed proxy loader and perturbation bank."""

    def __init__(self, codec: Any, proxy_loader: Any, perturbation_bank: list[Any], device: str) -> None:
        self.codec = codec
        self.proxy_loader = proxy_loader
        self.perturbation_bank = perturbation_bank
        self.device = device

    @staticmethod
    def _js_divergence(probabilities: torch.Tensor, perturbed_probabilities: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        p = probabilities.clamp_min(eps)
        q = perturbed_probabilities.clamp_min(eps)
        midpoint = 0.5 * (p + q)
        return 0.5 * (p * (p.log() - midpoint.log())).sum(dim=1) + 0.5 * (q * (q.log() - midpoint.log())).sum(dim=1)

    def estimate(self, model: torch.nn.Module) -> float:
        base_model = copy.deepcopy(model).to(self.device).eval()
        base_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        divergence_sum = 0.0
        comparison_count = 0
        with torch.no_grad():
            for perturbation in self.perturbation_bank:
                perturbed_model = copy.deepcopy(model).to(self.device).eval()
                perturbed_state = self.codec.apply_update_to_state_dict(base_state, perturbation, step_size=1.0)
                perturbed_model.load_state_dict(perturbed_state)
                for images, _unused_labels in self.proxy_loader:
                    images = images.to(self.device)
                    base_probs = F.softmax(base_model(images), dim=1)
                    perturbed_probs = F.softmax(perturbed_model(images), dim=1)
                    divergence = self._js_divergence(base_probs, perturbed_probs)
                    divergence_sum += float(divergence.sum().cpu())
                    comparison_count += int(images.shape[0])
        return divergence_sum / max(1, comparison_count)
