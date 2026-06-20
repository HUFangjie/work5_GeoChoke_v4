from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class TangentCommitter:
    """Fisher tangent-cone projection for aggregate update commitment.

    The basis is estimated only from server-side unlabeled proxy inputs and the
    current global model's pseudo-labels. It never inspects individual client
    updates.
    """

    def __init__(self, codec: Any, proxy_loader: Any, cfg: Any, device: str) -> None:
        self.codec = codec
        self.proxy_loader = proxy_loader
        self.cfg = cfg
        self.device = torch.device(device)
        self._basis: torch.Tensor | None = None
        self._basis_round: int | None = None

    def build_basis(self, model: torch.nn.Module, round_id: int) -> torch.Tensor:
        refresh_interval = max(1, int(getattr(self.cfg, "tangent_refresh_interval", 1)))
        if self._basis is not None and self._basis_round is not None and (round_id - self._basis_round) < refresh_interval:
            return self._basis

        was_training = model.training
        model.to(self.device)
        model.eval()
        gradients: list[torch.Tensor] = []
        max_vectors = max(0, int(getattr(self.cfg, "tangent_basis_rank", 16)))
        max_batches = max(0, int(getattr(self.cfg, "tangent_max_proxy_batches", 16)))
        eps = float(getattr(self.cfg, "tangent_eps", 1e-12))

        try:
            for batch_idx, batch in enumerate(self.proxy_loader):
                if batch_idx >= max_batches or len(gradients) >= max_vectors:
                    break
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(self.device)
                model.zero_grad(set_to_none=True)
                logits = model(x)
                pseudo_y = logits.argmax(dim=1)
                loss = F.cross_entropy(logits, pseudo_y)
                params = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.is_floating_point()]
                grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
                flat_parts: list[torch.Tensor] = []
                for parameter, grad in zip(params, grads):
                    if grad is None:
                        flat_parts.append(torch.zeros(parameter.numel(), device=self.device, dtype=torch.float64))
                    else:
                        flat_parts.append(grad.detach().reshape(-1).to(self.device, dtype=torch.float64))
                if not flat_parts:
                    continue
                vector = torch.cat(flat_parts)
                if vector.numel() != self.codec.total_dimension:
                    raise ValueError(f"tangent gradient dimension {vector.numel()} != update dimension {self.codec.total_dimension}")
                norm = torch.linalg.vector_norm(vector)
                if torch.isfinite(norm) and float(norm.item()) > eps:
                    gradients.append(vector / norm.clamp_min(eps))
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        if not gradients:
            basis = torch.empty((self.codec.total_dimension, 0), dtype=torch.float64, device=self.device)
        else:
            matrix = torch.stack(gradients, dim=1)
            q, r = torch.linalg.qr(matrix, mode="reduced")
            diag = torch.abs(torch.diagonal(r))
            keep = diag > eps
            basis = q[:, keep]

        self._basis = basis.detach()
        self._basis_round = round_id
        return self._basis

    def commit(self, update_vector: Any, model: torch.nn.Module, round_id: int, tau: float):
        g_np = np.asarray(update_vector, dtype=np.float64)
        original_norm = float(np.linalg.norm(g_np))
        eps = float(getattr(self.cfg, "tangent_eps", 1e-12))
        basis = self.build_basis(model, round_id)
        rank = int(basis.shape[1]) if basis is not None else 0
        if rank == 0:
            metrics = self._metrics(True, rank, tau, 1.0, original_norm, 0.0, original_norm, original_norm, eps)
            return g_np.copy(), metrics

        g = torch.as_tensor(g_np, dtype=torch.float64, device=basis.device)
        g_parallel = basis @ (basis.T @ g)
        g_perp = g - g_parallel
        norm_parallel = float(torch.linalg.vector_norm(g_parallel).item())
        norm_perp = float(torch.linalg.vector_norm(g_perp).item())
        rho = min(1.0, float(tau) * norm_parallel / (norm_perp + eps))
        committed = g_parallel + rho * g_perp
        committed_np = committed.detach().cpu().numpy().astype(np.float64, copy=False)
        committed_norm = float(np.linalg.norm(committed_np))
        metrics = self._metrics(True, rank, tau, rho, norm_parallel, norm_perp, original_norm, committed_norm, eps)
        return committed_np, metrics

    def _metrics(self, enabled: bool, rank: int, tau: float, rho: float, parallel_norm: float, perp_norm: float, original_norm: float, committed_norm: float, eps: float) -> dict[str, Any]:
        return {
            "tangent_commitment_enabled": bool(enabled),
            "tangent_basis_rank_actual": int(rank),
            "tangent_tau": float(tau),
            "tangent_rho": float(rho),
            "tangent_parallel_norm": float(parallel_norm),
            "tangent_perp_norm": float(perp_norm),
            "tangent_null_ratio": float(perp_norm / (original_norm + eps)),
            "tangent_committed_update_norm": float(committed_norm),
            "tangent_original_update_norm": float(original_norm),
            "tangent_update_shrink_ratio": float(committed_norm / (original_norm + eps)),
        }
