from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from attacks.base import AttackStrategy
from attacks.neurotoxin import PatchTrigger, flatten_update
from core.local_trainer import LocalTrainer
from core.types import LocalUpdateRecord


class ThreeDFedTrainer(LocalTrainer):
    def __init__(self, cfg: Any, trigger: PatchTrigger, decoy_role: bool) -> None:
        super().__init__(int(cfg.three_dfed_local_epochs), float(cfg.three_dfed_local_lr), cfg.device)
        self.cfg = cfg
        self.trigger = trigger
        self.decoy_role = decoy_role
        self.poisoned_sample_count = 0
        self.seen_sample_count = 0

    def train(self, model, loader, global_state):
        model.to(self.device)
        model.train()
        global_params = {name: tensor.detach().to(self.device).clone() for name, tensor in global_state.items() if torch.is_floating_point(tensor)}
        opt = torch.optim.SGD(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        total = 0.0
        n = 0
        for _ in range(self.epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                batch_n = len(x)
                opt.zero_grad()
                clean_loss = loss_fn(model(x), y)
                if self.decoy_role:
                    loss = clean_loss
                else:
                    poison_n = max(0, min(batch_n, int(round(batch_n * float(self.cfg.three_dfed_poison_ratio)))))
                    if poison_n > 0:
                        idx = torch.randperm(batch_n, device=self.device)[:poison_n]
                        x_p = self.trigger.apply_global(x[idx])
                        y_p = torch.full((poison_n,), int(self.cfg.three_dfed_target_label), device=self.device, dtype=torch.long)
                        backdoor_loss = loss_fn(model(x_p), y_p)
                        self.poisoned_sample_count += poison_n
                    else:
                        backdoor_loss = torch.zeros((), device=self.device)
                    penalty = torch.zeros((), device=self.device)
                    for name, param in model.named_parameters():
                        if name in global_params:
                            penalty = penalty + torch.sum((param - global_params[name]) ** 2)
                    loss = clean_loss + backdoor_loss + float(self.cfg.three_dfed_constrain_beta) * torch.sqrt(penalty + 1e-12)
                self.seen_sample_count += batch_n
                loss.backward()
                opt.step()
                total += float(loss.item()) * batch_n
                n += batch_n
        return total / max(1, n)


class ThreeDFedAttack(AttackStrategy):
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.trigger = PatchTrigger(cfg.three_dfed_trigger_size, cfg.three_dfed_trigger_location, cfg.three_dfed_trigger_value)
        self.indicator_enabled = bool(cfg.three_dfed_indicator_enabled)
        self.indicator_status = "not_implemented"
        self._last_metadata: dict[int, dict[str, Any]] = {}

    def should_poison(self, client_id: int, round_id: int) -> bool:
        return int(self.cfg.three_dfed_attack_start_round) <= round_id <= int(self.cfg.three_dfed_attack_end_round)

    def _is_decoy(self, client_id: int) -> bool:
        decoys = set(getattr(self.cfg, "three_dfed_decoy_client_ids", []) or [])
        if decoys:
            return client_id in decoys
        malicious = list(getattr(self.cfg, "malicious_client_ids", []))
        decoy_count = int(round(len(malicious) * float(self.cfg.three_dfed_decoy_fraction)))
        return client_id in set(malicious[:decoy_count])

    def train_local_update(self, client_id: int, loader: Any, model_factory: Any, codec_factory: Any, global_state: dict[str, Any], round_id: int) -> LocalUpdateRecord:
        decoy_role = self._is_decoy(client_id)
        local_model = model_factory()
        local_model.load_state_dict(copy.deepcopy(global_state))
        trainer = ThreeDFedTrainer(self.cfg, self.trigger, decoy_role)
        train_loss = trainer.train(local_model, loader, global_state)
        local_flat, global_flat = flatten_update(local_model, codec_factory, global_state)
        update = local_flat - global_flat
        if decoy_role:
            rng = np.random.default_rng(int(self.cfg.seed) + client_id + round_id)
            k = max(1, int(round(update.size * float(self.cfg.three_dfed_decoy_coordinate_ratio))))
            idx = rng.choice(update.size, size=min(k, update.size), replace=False)
            benign_norm = max(float(np.linalg.norm(update)), 1e-12)
            update[idx] += rng.normal(0.0, float(self.cfg.three_dfed_decoy_std) * benign_norm / np.sqrt(k), size=len(idx))
        else:
            rng = np.random.default_rng(int(self.cfg.seed) + 7919 * (client_id + 1) + round_id)
            noise = rng.normal(0.0, 1.0, size=update.shape)
            noise_norm = max(float(np.linalg.norm(noise)), 1e-12)
            update_norm = float(np.linalg.norm(update))
            max_noise_norm = float(self.cfg.three_dfed_noise_alpha) * update_norm
            noise = noise / noise_norm * max_noise_norm
            update = update + noise
            cap = float(self.cfg.three_dfed_norm_cap)
            final_norm = float(np.linalg.norm(update))
            if cap > 0.0 and final_norm > cap:
                update = update * (cap / max(final_norm, 1e-12))
        update = update * float(self.cfg.three_dfed_scale_factor)
        if not np.all(np.isfinite(update)):
            raise ValueError("3DFed produced a non-finite update")
        role = "decoy" if decoy_role else "backdoor"
        meta = {
            "attack_name": "three_dfed",
            "update_type": "poisoned" if not decoy_role else "decoy",
            "backdoor_attack_active": not decoy_role,
            "three_dfed_role": role,
            "decoy_role": decoy_role,
            "indicator_enabled": self.indicator_enabled,
            "indicator_status": self.indicator_status,
            "noise_alpha": float(self.cfg.three_dfed_noise_alpha),
            "constrain_beta": float(self.cfg.three_dfed_constrain_beta),
            "three_dfed_constrain_beta": float(self.cfg.three_dfed_constrain_beta),
            "three_dfed_noise_alpha": float(self.cfg.three_dfed_noise_alpha),
            "poisoned_sample_count": trainer.poisoned_sample_count,
            "effective_poison_ratio": trainer.poisoned_sample_count / max(1, trainer.seen_sample_count),
            "poison_ratio": trainer.poisoned_sample_count / max(1, trainer.seen_sample_count),
            "poisoned_update_norm": float(np.linalg.norm(update)),
        }
        self._last_metadata[client_id] = meta
        return LocalUpdateRecord(client_id, len(loader.dataset), update, float(train_loss), metadata=meta.copy())

    def craft_update(self, client_id, clean_update, global_model, attacker_context):
        update = np.asarray(clean_update).copy()
        if not np.all(np.isfinite(update)):
            raise ValueError("3DFed produced an invalid update")
        return update

    def pop_last_metadata(self, client_id: int) -> dict[str, Any]:
        return self._last_metadata.pop(client_id, {})
