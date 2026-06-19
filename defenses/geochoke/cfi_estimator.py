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
    def js_divergence_from_logits(base_logits: torch.Tensor, perturbed_logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        base_log_probs = F.log_softmax(base_logits, dim=1)
        perturbed_log_probs = F.log_softmax(perturbed_logits, dim=1)
        p = base_log_probs.exp()
        q = perturbed_log_probs.exp()
        midpoint = 0.5 * (p + q)
        midpoint_log = midpoint.clamp_min(eps).log()
        js = 0.5 * (p * (base_log_probs - midpoint_log)).sum(dim=1) + 0.5 * (q * (perturbed_log_probs - midpoint_log)).sum(dim=1)
        return js.clamp_min(0.0)

    @staticmethod
    def _js_divergence(probabilities: torch.Tensor, perturbed_probabilities: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        p = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(eps)
        q = perturbed_probabilities / perturbed_probabilities.sum(dim=1, keepdim=True).clamp_min(eps)
        midpoint = 0.5 * (p + q)
        js = 0.5 * (p * (p.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(dim=1) + 0.5 * (q * (q.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(dim=1)
        return js.clamp_min(0.0)

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
                    base_logits = base_model(images)
                    perturbed_logits = perturbed_model(images)
                    divergence = self.js_divergence_from_logits(base_logits, perturbed_logits)
                    divergence_sum += float(divergence.sum().cpu())
                    comparison_count += int(images.shape[0])
        cfi = divergence_sum / max(1, comparison_count)
        if cfi < -1e-12:
            raise ValueError(f"CFI must be nonnegative, got {cfi}")
        return max(0.0, cfi)
