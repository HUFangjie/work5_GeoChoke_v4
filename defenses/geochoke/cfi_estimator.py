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
        eps = 1e-12
        p = probabilities.to(torch.float64).clamp_min(eps)
        q = perturbed_probabilities.to(torch.float64).clamp_min(eps)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
        midpoint = (0.5 * (p + q)).clamp_min(eps)
        midpoint = midpoint / midpoint.sum(dim=1, keepdim=True).clamp_min(eps)
        js = 0.5 * (p * (p.log() - midpoint.log())).sum(dim=1) + 0.5 * (q * (q.log() - midpoint.log())).sum(dim=1)
        return torch.clamp(js, min=0.0)

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
        return max(0.0, divergence_sum / max(1, comparison_count))

    def functional_drift(self, previous_model: torch.nn.Module, candidate_model: torch.nn.Module) -> float:
        previous = copy.deepcopy(previous_model).to(self.device).eval()
        candidate = copy.deepcopy(candidate_model).to(self.device).eval()
        divergence_sum = 0.0
        sample_count = 0
        with torch.no_grad():
            for images, _unused_labels in self.proxy_loader:
                images = images.to(self.device)
                previous_probs = F.softmax(previous(images), dim=1)
                candidate_probs = F.softmax(candidate(images), dim=1)
                divergence = self._js_divergence(previous_probs, candidate_probs)
                divergence_sum += float(divergence.sum().cpu())
                sample_count += int(images.shape[0])
        return max(0.0, divergence_sum / max(1, sample_count))
