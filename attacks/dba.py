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

    def validate_layout(self, height: int, width: int) -> None:
        regions = [self.region(trigger_id, height, width) for trigger_id in range(self.num_parts)]
        occupied: set[tuple[int, int]] = set()
        for r0, r1, c0, c1 in regions:
            if r0 < 0 or c0 < 0 or r1 > height or c1 > width:
                raise ValueError("DBA trigger part exceeds input image bounds")
            for row in range(r0, r1):
                for col in range(c0, c1):
                    pixel = (row, col)
                    if pixel in occupied:
                        raise ValueError("DBA trigger parts must not overlap")
                    occupied.add(pixel)

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
        self._validate_config()
        self.trigger = DBATrigger(cfg)
        bound_clients = list(cfg.malicious_client_ids[: int(cfg.dba_num_trigger_parts)])
        self.client_to_trigger = {client_id: i for i, client_id in enumerate(bound_clients)}
        self.extra_malicious_clients = [client_id for client_id in cfg.malicious_client_ids if client_id not in self.client_to_trigger]

    def _validate_config(self) -> None:
        num_classes = int(getattr(self.cfg, "num_classes", 10))
        if not 0 <= int(self.cfg.dba_target_label) < num_classes:
            raise ValueError("dba_target_label must satisfy 0 <= dba_target_label < num_classes")
        if not 0.0 < float(self.cfg.dba_poison_ratio) < 1.0:
            raise ValueError("dba_poison_ratio must satisfy 0 < ratio < 1")
        for field in ["dba_local_epochs", "dba_num_trigger_parts"]:
            if int(getattr(self.cfg, field)) <= 0:
                raise ValueError(f"{field} must be > 0")
        for field in ["dba_local_lr", "dba_scale_factor", "dba_multi_shot_scale_factor", "dba_single_shot_scale_factor"]:
            if float(getattr(self.cfg, field)) <= 0.0:
                raise ValueError(f"{field} must be > 0")
        if self.cfg.dba_attack_mode not in {"multi_shot", "single_shot"}:
            raise ValueError("dba_attack_mode must be 'multi_shot' or 'single_shot'")
        if int(self.cfg.dba_poison_interval) < 0:
            raise ValueError("dba_poison_interval must be >= 0")
        if not int(self.cfg.dba_attack_start_round) <= int(self.cfg.dba_attack_end_round) < int(self.cfg.num_rounds):
            raise ValueError("DBA schedule must satisfy start_round <= end_round < num_rounds")
        if len(self.cfg.malicious_client_ids) < int(self.cfg.dba_num_trigger_parts):
            raise ValueError("DBA requires at least dba_num_trigger_parts malicious clients to avoid centralized backdoor behavior")
        image_size = int(getattr(self.cfg, "image_size", 28))
        DBATrigger(self.cfg).validate_layout(image_size, image_size)

    def scale_factor_for_mode(self) -> float:
        if self.cfg.dba_attack_mode == "single_shot":
            return float(self.cfg.dba_single_shot_scale_factor)
        return float(self.cfg.dba_multi_shot_scale_factor)

    def active_malicious_clients(self, round_id: int) -> list[int]:
        start = int(self.cfg.dba_attack_start_round)
        end = int(self.cfg.dba_attack_end_round)
        if round_id < start or round_id > end:
            return []
        if self.cfg.dba_attack_mode == "multi_shot":
            return list(self.client_to_trigger)
        ordered = list(self.client_to_trigger)
        interval = int(self.cfg.dba_poison_interval)
        if interval == 0:
            return ordered if round_id == start else []
        delta = round_id - start
        if delta % interval != 0:
            return []
        attack_event_index = delta // interval
        if attack_event_index >= len(ordered):
            return []
        return [ordered[attack_event_index]]

    def is_attack_round(self, round_id: int) -> bool:
        return bool(self.active_malicious_clients(round_id))

    def should_poison(self, client_id: int, round_id: int) -> bool:
        return client_id in self.active_malicious_clients(round_id)

    def train_local_update(self, client_id: int, loader: Any, model_factory: Any, codec_factory: Any, global_state: dict[str, Any], round_id: int) -> LocalUpdateRecord:
        if client_id not in self.client_to_trigger:
            raise ValueError(f"malicious client {client_id} is not bound to a DBA trigger and will not run DBA poisoning")
        local_model = model_factory()
        local_model.load_state_dict(copy.deepcopy(global_state))
        trainer = DBALocalTrainer(self.cfg, self.client_to_trigger[client_id])
        train_loss = trainer.train(local_model, loader)
        codec = codec_factory(local_model)
        local_flat = codec.flatten_state_dict(local_model.state_dict())
        global_flat = codec.flatten_state_dict(global_state)
        unscaled_update = local_flat - global_flat
        scale = self.scale_factor_for_mode()
        update = unscaled_update * scale
        before_norm = float(np.linalg.norm(unscaled_update))
        after_norm = float(np.linalg.norm(update))
        return LocalUpdateRecord(client_id, len(loader.dataset), update, float(train_loss), metadata={
            "update_type": "poisoned",
            "dba_attack_active": True,
            "dba_trigger_id": self.client_to_trigger[client_id],
            "poisoned_sample_count": trainer.poisoned_sample_count,
            "dba_seen_sample_count": trainer.seen_sample_count,
            "effective_poison_ratio": trainer.poisoned_sample_count / max(1, trainer.seen_sample_count),
            "poison_ratio": trainer.poisoned_sample_count / max(1, trainer.seen_sample_count),
            "dba_scale_factor": scale,
            "update_norm_before_scale": before_norm,
            "update_norm_after_scale": after_norm,
            "poisoned_update_norm": after_norm,
        })

    def craft_update(self, client_id, clean_update, global_model, attacker_context):
        return clean_update.copy()
