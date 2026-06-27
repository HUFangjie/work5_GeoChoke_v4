from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from attacks.base import AttackStrategy
from attacks.neurotoxin import PoisonedLocalTrainer, flatten_update
from core.types import LocalUpdateRecord


class LearnableTrigger:
    def __init__(self, trigger: torch.Tensor, mask: torch.Tensor) -> None:
        self.trigger = trigger.detach().cpu().clone()
        self.mask_tensor = mask.detach().cpu().clone()

    def apply_global(self, images: torch.Tensor) -> torch.Tensor:
        trigger = self.trigger.to(images.device, dtype=images.dtype)
        mask = self.mask_tensor.to(images.device, dtype=images.dtype)
        while trigger.dim() < images.dim():
            trigger = trigger.unsqueeze(0)
            mask = mask.unsqueeze(0)
        return torch.clamp(images * (1.0 - mask) + trigger * mask, 0.0, 1.0)

    def apply_global_trigger(self, images: torch.Tensor) -> torch.Tensor:
        return self.apply_global(images)

    def mask(self, channels: int, height: int, width: int) -> torch.Tensor:
        return self.mask_tensor.clone()


class A3FLAttack(AttackStrategy):
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.trigger = self._initial_trigger()
        self._last_metadata: dict[int, dict[str, Any]] = {}

    def _patch_mask(self) -> torch.Tensor:
        channels = int(getattr(self.cfg, "input_channels", 1))
        image_size = int(getattr(self.cfg, "image_size", 28))
        size = int(self.cfg.a3fl_trigger_size)
        mask = torch.zeros(channels, image_size, image_size)
        loc = str(self.cfg.a3fl_trigger_location)
        if loc == "top_left":
            r0, c0 = 0, 0
        elif loc == "top_right":
            r0, c0 = 0, image_size - size
        elif loc == "bottom_left":
            r0, c0 = image_size - size, 0
        elif loc == "bottom_right":
            r0, c0 = image_size - size, image_size - size
        else:
            raise ValueError(f"Unsupported a3fl_trigger_location: {loc}")
        mask[:, r0 : r0 + size, c0 : c0 + size] = 1.0
        return mask

    def _initial_trigger(self) -> LearnableTrigger:
        mask = self._patch_mask()
        trigger = torch.zeros_like(mask) + float(self.cfg.a3fl_trigger_init)
        return LearnableTrigger(trigger, mask)

    def should_poison(self, client_id: int, round_id: int) -> bool:
        return int(self.cfg.a3fl_attack_start_round) <= round_id <= int(self.cfg.a3fl_attack_end_round)

    def _optimize_trigger(self, theta_t, theta_adv, loader) -> LearnableTrigger:
        device = self.cfg.device
        theta_t.to(device).eval()
        theta_adv.to(device).train()
        mask = self.trigger.mask_tensor.to(device)
        trigger = self.trigger.trigger.to(device).clone().detach().requires_grad_(True)
        loss_fn = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(theta_adv.parameters(), lr=float(self.cfg.a3fl_adv_lr))
        target_label = int(self.cfg.a3fl_target_label)
        loader_iter = iter(loader)
        for step in range(int(self.cfg.a3fl_trigger_steps)):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)
            x, y = x.to(device), y.to(device)
            poison_n = max(1, min(len(x), int(round(len(x) * float(self.cfg.a3fl_poison_ratio)))))
            x = x[:poison_n]
            y = y[:poison_n]
            trigger_b = trigger.unsqueeze(0)
            mask_b = mask.unsqueeze(0)
            x_trig = torch.clamp(x * (1.0 - mask_b) + trigger_b * mask_b, 0.0, 1.0)
            target = torch.full((len(x_trig),), target_label, dtype=torch.long, device=device)
            loss = loss_fn(theta_t(x_trig), target) + float(self.cfg.a3fl_lambda) * loss_fn(theta_adv(x_trig), target)
            if trigger.grad is not None:
                trigger.grad.zero_()
            loss.backward()
            with torch.no_grad():
                trigger -= float(self.cfg.a3fl_trigger_lr) * trigger.grad.sign()
                trigger.clamp_(0.0, 1.0)
            trigger.requires_grad_(True)
            for _ in range(int(self.cfg.a3fl_adv_steps)):
                x_adv = torch.clamp(x * (1.0 - mask_b) + trigger.detach().unsqueeze(0) * mask_b, 0.0, 1.0)
                adv_opt.zero_grad()
                adv_loss = loss_fn(theta_adv(x_adv), y)
                adv_loss.backward()
                adv_opt.step()
        return LearnableTrigger(trigger.detach().cpu(), mask.detach().cpu())

    def train_local_update(self, client_id: int, loader: Any, model_factory: Any, codec_factory: Any, global_state: dict[str, Any], round_id: int) -> LocalUpdateRecord:
        theta_t = model_factory(); theta_t.load_state_dict(copy.deepcopy(global_state))
        theta_adv = model_factory(); theta_adv.load_state_dict(copy.deepcopy(global_state))
        self.trigger = self._optimize_trigger(theta_t, theta_adv, loader)
        local_model = model_factory(); local_model.load_state_dict(copy.deepcopy(global_state))
        trainer = PoisonedLocalTrainer(
            self.cfg.a3fl_local_epochs,
            self.cfg.a3fl_local_lr,
            self.cfg.device,
            self.trigger,
            self.cfg.a3fl_target_label,
            self.cfg.a3fl_poison_ratio,
        )
        train_loss = trainer.train(local_model, loader)
        local_flat, global_flat = flatten_update(local_model, codec_factory, global_state)
        update = (local_flat - global_flat) * float(self.cfg.a3fl_scale_factor)
        if not np.all(np.isfinite(update)):
            raise ValueError("A3FL produced a non-finite update")
        trigger_linf = float((self.trigger.trigger * self.trigger.mask_tensor).abs().max().item())
        meta = {
            "attack_name": "a3fl",
            "update_type": "poisoned",
            "backdoor_attack_active": True,
            "a3fl_trigger_steps": int(self.cfg.a3fl_trigger_steps),
            "a3fl_adv_steps": int(self.cfg.a3fl_adv_steps),
            "a3fl_lambda": float(self.cfg.a3fl_lambda),
            "poisoned_sample_count": trainer.poisoned_sample_count,
            "effective_poison_ratio": trainer.poisoned_sample_count / max(1, trainer.seen_sample_count),
            "poison_ratio": trainer.poisoned_sample_count / max(1, trainer.seen_sample_count),
            "trigger_linf_norm": trigger_linf,
            "poisoned_update_norm": float(np.linalg.norm(update)),
        }
        self._last_metadata[client_id] = meta
        return LocalUpdateRecord(client_id, len(loader.dataset), update, float(train_loss), metadata=meta.copy())

    def craft_update(self, client_id, clean_update, global_model, attacker_context):
        update = np.asarray(clean_update).copy()
        if not np.all(np.isfinite(update)):
            raise ValueError("A3FL produced an invalid update")
        return update

    def pop_last_metadata(self, client_id: int) -> dict[str, Any]:
        return self._last_metadata.pop(client_id, {})
