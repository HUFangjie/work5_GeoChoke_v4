from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from attacks.base import AttackStrategy
from core.local_trainer import LocalTrainer
from core.types import LocalUpdateRecord


class DBATrigger:
    """Non-overlapping DBA trigger parts for MNIST-like [C,H,W] tensors."""

    def __init__(self, cfg: Any) -> None:
        self.num_parts = int(cfg.dba_num_trigger_parts)
        self.size = int(cfg.dba_trigger_size)
        self.gap = int(cfg.dba_trigger_gap)
        self.location = str(cfg.dba_trigger_location)
        self.value = float(cfg.dba_trigger_value)
        if self.num_parts <= 0 or self.size <= 0:
            raise ValueError("dba_num_trigger_parts and dba_trigger_size must be positive")

    def _origin(self, height: int, width: int) -> tuple[int, int]:
        total_width = self.num_parts * self.size + (self.num_parts - 1) * self.gap
        total_height = self.size
        if total_width > width or total_height > height:
            raise ValueError("DBA trigger layout does not fit in the input image")
        if self.location == "top_left":
            return 0, 0
        if self.location == "top_right":
            return 0, width - total_width
        if self.location == "bottom_left":
            return height - total_height, 0
        if self.location == "bottom_right":
            return height - total_height, width - total_width
        raise ValueError(f"Unsupported dba_trigger_location: {self.location}")

    def region(self, trigger_id: int, height: int, width: int) -> tuple[int, int, int, int]:
        if trigger_id < 0 or trigger_id >= self.num_parts:
            raise ValueError(f"trigger_id must be in [0, {self.num_parts - 1}], got {trigger_id}")
        row, col = self._origin(height, width)
        col += trigger_id * (self.size + self.gap)
        return row, row + self.size, col, col + self.size

    def apply_local(self, images: torch.Tensor, trigger_id: int) -> torch.Tensor:
        out = images.clone()
        h, w = out.shape[-2], out.shape[-1]
        r0, r1, c0, c1 = self.region(trigger_id, h, w)
        out[..., r0:r1, c0:c1] = self.value
        return out

    def apply_global(self, images: torch.Tensor) -> torch.Tensor:
        out = images.clone()
        for trigger_id in range(self.num_parts):
            out = self.apply_local(out, trigger_id)
        return out


def apply_local_trigger(image: torch.Tensor, trigger_id: int, cfg: Any | None = None) -> torch.Tensor:
    if cfg is None:
        from config import CONFIG
        cfg = CONFIG
    return DBATrigger(cfg).apply_local(image, trigger_id)


def apply_global_trigger(image: torch.Tensor, cfg: Any | None = None) -> torch.Tensor:
    if cfg is None:
        from config import CONFIG
        cfg = CONFIG
    return DBATrigger(cfg).apply_global(image)


class DBALocalTrainer(LocalTrainer):
    def __init__(self, cfg: Any, trigger_id: int) -> None:
        super().__init__(int(cfg.dba_local_epochs), float(cfg.dba_local_lr), cfg.device)
        self.cfg = cfg
        self.trigger_id = trigger_id
        self.trigger = DBATrigger(cfg)
        self.poisoned_sample_count = 0
        self.seen_sample_count = 0

    def train(self, model, loader):
        model.to(self.device)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        total = 0.0
        n = 0
        ratio = float(self.cfg.dba_poison_ratio)
        target = int(self.cfg.dba_target_label)
        for _ in range(self.epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                batch_n = len(x)
                poison_n = int(round(batch_n * ratio))
                poison_n = max(0, min(batch_n, poison_n))
                if poison_n > 0:
                    idx = torch.randperm(batch_n, device=self.device)[:poison_n]
                    x = x.clone()
                    y = y.clone()
                    x[idx] = self.trigger.apply_local(x[idx], self.trigger_id)
                    y[idx] = target
                    self.poisoned_sample_count += poison_n
                self.seen_sample_count += batch_n
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                total += float(loss.item()) * batch_n
                n += batch_n
        return total / max(1, n)


class DBAAttack(AttackStrategy):
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.trigger = DBATrigger(cfg)
        if len(cfg.malicious_client_ids) < int(cfg.dba_num_trigger_parts):
            raise ValueError("DBA requires at least dba_num_trigger_parts malicious clients to avoid centralized backdoor behavior")
        self.client_to_trigger = {client_id: i for i, client_id in enumerate(cfg.malicious_client_ids[: int(cfg.dba_num_trigger_parts)])}

    def is_attack_round(self, round_id: int) -> bool:
        return int(self.cfg.dba_attack_start_round) <= round_id <= int(self.cfg.dba_attack_end_round) and (round_id - int(self.cfg.dba_attack_start_round)) % int(self.cfg.dba_poison_interval) == 0

    def active_malicious_clients(self, round_id: int) -> list[int]:
        if not self.is_attack_round(round_id):
            return []
        if self.cfg.dba_attack_mode == "multi_shot":
            return list(self.client_to_trigger)
        if self.cfg.dba_attack_mode == "single_shot":
            ordered = list(self.client_to_trigger)
            return [ordered[(round_id - int(self.cfg.dba_attack_start_round)) % len(ordered)]]
        raise ValueError(f"Unsupported dba_attack_mode: {self.cfg.dba_attack_mode}")

    def should_poison(self, client_id: int, round_id: int) -> bool:
        return client_id in self.active_malicious_clients(round_id)

    def train_local_update(self, client_id: int, loader: Any, model_factory: Any, codec_factory: Any, global_state: dict[str, Any], round_id: int) -> LocalUpdateRecord:
        if client_id not in self.client_to_trigger:
            raise ValueError(f"malicious client {client_id} has no DBA trigger binding")
        local_model = model_factory()
        local_model.load_state_dict(copy.deepcopy(global_state))
        trainer = DBALocalTrainer(self.cfg, self.client_to_trigger[client_id])
        train_loss = trainer.train(local_model, loader)
        codec = codec_factory(local_model)
        local_flat = codec.flatten_state_dict(local_model.state_dict())
        global_flat = codec.flatten_state_dict(global_state)
        update = (local_flat - global_flat) * float(self.cfg.dba_scale_factor)
        return LocalUpdateRecord(client_id, len(loader.dataset), update, float(train_loss), metadata={
            "dba_attack_active": True,
            "dba_trigger_id": self.client_to_trigger[client_id],
            "poisoned_sample_count": trainer.poisoned_sample_count,
            "dba_seen_sample_count": trainer.seen_sample_count,
            "poison_ratio": trainer.poisoned_sample_count / max(1, trainer.seen_sample_count),
            "dba_scale_factor": float(self.cfg.dba_scale_factor),
        })

    def craft_update(self, client_id, clean_update, global_model, attacker_context):
        return clean_update.copy()
