from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from attacks.base import AttackStrategy
from attacks.neurotoxin import PatchTrigger, flatten_update
from core.local_trainer import LocalTrainer
from core.types import LocalUpdateRecord


class AdaptiveGeoChokeTrainer(LocalTrainer):
    def __init__(self, cfg: Any, trigger: PatchTrigger) -> None:
        super().__init__(int(cfg.adaptive_geochoke_local_epochs), float(cfg.adaptive_geochoke_lr), cfg.device)
        self.cfg = cfg
        self.trigger = trigger
        self.poisoned_sample_count = 0
        self.seen_sample_count = 0

    def train(self, model, loader):
        model.to(self.device)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        total = 0.0
        n = 0
        poison_ratio = float(self.cfg.adaptive_geochoke_poison_ratio)
        target_label = int(self.cfg.adaptive_geochoke_target_label)
        for _ in range(self.epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                batch_n = len(x)
                opt.zero_grad()
                clean_loss = loss_fn(model(x), y)
                poison_n = max(0, min(batch_n, int(round(batch_n * poison_ratio))))
                if poison_n > 0:
                    idx = torch.randperm(batch_n, device=self.device)[:poison_n]
                    x_poison = self.trigger.apply_global(x[idx])
                    y_poison = torch.full((poison_n,), target_label, device=self.device, dtype=torch.long)
                    backdoor_loss = loss_fn(model(x_poison), y_poison)
                    self.poisoned_sample_count += poison_n
                else:
                    backdoor_loss = torch.zeros((), device=self.device)
                loss = clean_loss + backdoor_loss
                loss.backward()
                opt.step()
                self.seen_sample_count += batch_n
                total += float(loss.item()) * batch_n
                n += batch_n
        return total / max(1, n)


class AdaptiveGeoChokeAttack(AttackStrategy):
    """Tangent-aware white-box baseline against GeoChoke.

    The attacker approximates GeoChoke's pseudo-label tangent basis from local
    unlabeled batches, projects the poisoned update into that basis, and shrinks
    the orthogonal component before CKKS encryption.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.trigger = PatchTrigger(
            cfg.adaptive_geochoke_trigger_size,
            cfg.adaptive_geochoke_trigger_location,
            cfg.adaptive_geochoke_trigger_value,
        )
        self._last_metadata: dict[int, dict[str, Any]] = {}

    def should_poison(self, client_id: int, round_id: int) -> bool:
        return int(self.cfg.adaptive_geochoke_attack_start_round) <= round_id <= int(self.cfg.adaptive_geochoke_attack_end_round)

    def _flatten_gradients(self, model) -> np.ndarray:
        chunks = []
        for param in model.parameters():
            grad = param.grad
            if grad is None:
                chunks.append(torch.zeros_like(param).detach().cpu().reshape(-1))
            else:
                chunks.append(grad.detach().cpu().reshape(-1))
        if not chunks:
            return np.zeros(0, dtype=np.float64)
        return torch.cat(chunks).numpy().astype(np.float64, copy=False)

    def _build_tangent_basis(self, model, loader, update_dim: int) -> np.ndarray:
        model.to(self.cfg.device)
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        grads: list[np.ndarray] = []
        max_batches = int(self.cfg.adaptive_geochoke_proxy_batches)
        max_rank = int(self.cfg.adaptive_geochoke_basis_rank)
        for batch_idx, (x, _y) in enumerate(loader):
            if batch_idx >= max_batches or len(grads) >= max_rank:
                break
            x = x.to(self.cfg.device)
            with torch.no_grad():
                pseudo = model(x).argmax(1)
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), pseudo)
            loss.backward()
            grad = self._flatten_gradients(model)
            norm = float(np.linalg.norm(grad))
            if norm > 1e-12 and grad.size == update_dim:
                grads.append(grad / norm)
        model.zero_grad(set_to_none=True)
        if not grads:
            return np.zeros((update_dim, 0), dtype=np.float64)
        matrix = np.stack(grads, axis=1)
        q, _ = np.linalg.qr(matrix, mode="reduced")
        rank = min(q.shape[1], max_rank)
        return q[:, :rank].astype(np.float64, copy=False)

    def _project_update(self, update: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        before_norm = float(np.linalg.norm(update))
        if basis.size == 0 or basis.shape[1] == 0:
            parallel = np.zeros_like(update)
            perp = update.copy()
        else:
            parallel = basis @ (basis.T @ update)
            perp = update - parallel
        parallel_norm = float(np.linalg.norm(parallel))
        perp_norm_before = float(np.linalg.norm(perp))
        rho_adv = 1.0 / (1.0 + max(0.0, float(self.cfg.adaptive_geochoke_orth_lambda)))
        update_adv = parallel + rho_adv * perp
        update_adv = update_adv * float(self.cfg.adaptive_geochoke_scale_factor)
        perp_after = update_adv - (basis @ (basis.T @ update_adv) if basis.size and basis.shape[1] else np.zeros_like(update_adv))
        after_norm = float(np.linalg.norm(update_adv))
        perp_norm_after = float(np.linalg.norm(perp_after))
        metadata = {
            "adaptive_basis_rank_actual": int(basis.shape[1]) if basis.ndim == 2 else 0,
            "adaptive_parallel_norm": parallel_norm,
            "adaptive_perp_norm_before": perp_norm_before,
            "adaptive_perp_norm_after": perp_norm_after,
            "adaptive_null_ratio_before": perp_norm_before / max(before_norm, 1e-12),
            "adaptive_null_ratio_after": perp_norm_after / max(after_norm, 1e-12),
            "adaptive_update_norm_before": before_norm,
            "adaptive_update_norm_after": after_norm,
            "adaptive_projection_applied": True,
        }
        return update_adv, metadata

    def _estimate_cfi_js(self, global_model, candidate_model, loader) -> float:
        js_values = []
        global_model.to(self.cfg.device).eval()
        candidate_model.to(self.cfg.device).eval()
        with torch.no_grad():
            for batch_idx, (x, _y) in enumerate(loader):
                if batch_idx >= int(self.cfg.adaptive_geochoke_proxy_batches):
                    break
                x = x.to(self.cfg.device)
                p = torch.softmax(global_model(x), dim=1).clamp_min(1e-12)
                q = torch.softmax(candidate_model(x), dim=1).clamp_min(1e-12)
                m = 0.5 * (p + q)
                js = 0.5 * (p * (p.log() - m.log())).sum(1) + 0.5 * (q * (q.log() - m.log())).sum(1)
                js_values.append(float(js.mean().item()))
        return float(np.mean(js_values)) if js_values else 0.0

    def train_local_update(self, client_id: int, loader: Any, model_factory: Any, codec_factory: Any, global_state: dict[str, Any], round_id: int) -> LocalUpdateRecord:
        global_model = model_factory()
        global_model.load_state_dict(copy.deepcopy(global_state))
        local_model = model_factory()
        local_model.load_state_dict(copy.deepcopy(global_state))
        trainer = AdaptiveGeoChokeTrainer(self.cfg, self.trigger)
        train_loss = trainer.train(local_model, loader)
        local_flat, global_flat = flatten_update(local_model, codec_factory, global_state)
        raw_update = local_flat - global_flat
        basis = self._build_tangent_basis(global_model, loader, raw_update.size)
        projected_update, projection_meta = self._project_update(raw_update.astype(np.float64, copy=False), basis)
        if not np.all(np.isfinite(projected_update)) or projected_update.ndim != 1 or projected_update.shape != raw_update.shape:
            raise ValueError("AdaptiveGeoChoke produced an invalid update")
        cfi_proxy_js = 0.0
        if float(self.cfg.adaptive_geochoke_cfi_lambda) > 0.0:
            cfi_proxy_js = self._estimate_cfi_js(global_model, local_model, loader)
        meta = {
            "attack_name": "adaptive_geochoke",
            "update_type": "poisoned",
            "backdoor_attack_active": True,
            "poisoned_sample_count": trainer.poisoned_sample_count,
            "effective_poison_ratio": trainer.poisoned_sample_count / max(1, trainer.seen_sample_count),
            "poison_ratio": trainer.poisoned_sample_count / max(1, trainer.seen_sample_count),
            "adaptive_align_lambda": float(self.cfg.adaptive_geochoke_align_lambda),
            "adaptive_orth_lambda": float(self.cfg.adaptive_geochoke_orth_lambda),
            "adaptive_cfi_lambda": float(self.cfg.adaptive_geochoke_cfi_lambda),
            "adaptive_cfi_proxy_js": cfi_proxy_js,
            "poisoned_update_norm": float(np.linalg.norm(projected_update)),
            **projection_meta,
        }
        self._last_metadata[client_id] = meta
        return LocalUpdateRecord(client_id, len(loader.dataset), projected_update, float(train_loss), metadata=meta.copy())

    def craft_update(self, client_id, clean_update, global_model, attacker_context):
        update = np.asarray(clean_update).copy()
        if update.ndim != 1 or not np.all(np.isfinite(update)):
            raise ValueError("AdaptiveGeoChoke produced an invalid crafted update")
        return update

    def pop_last_metadata(self, client_id: int) -> dict[str, Any]:
        return self._last_metadata.pop(client_id, {})
