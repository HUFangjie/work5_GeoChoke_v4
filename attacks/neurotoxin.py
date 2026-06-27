from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from attacks.base import AttackStrategy
from core.local_trainer import LocalTrainer
from core.types import LocalUpdateRecord


class PatchTrigger:
    def __init__(self, size: int = 4, location: str = "top_left", value: float = 1.0) -> None:
        self.size = int(size)
        self.location = str(location)
        self.value = float(value)

    def _origin(self, height: int, width: int) -> tuple[int, int]:
        if self.size > height or self.size > width:
            raise ValueError("trigger patch does not fit in image")
        if self.location == "top_left":
            return 0, 0
        if self.location == "top_right":
            return 0, width - self.size
        if self.location == "bottom_left":
            return height - self.size, 0
        if self.location == "bottom_right":
            return height - self.size, width - self.size
        raise ValueError(f"Unsupported trigger location: {self.location}")

    def mask(self, channels: int, height: int, width: int) -> torch.Tensor:
        mask = torch.zeros(channels, height, width)
        r0, c0 = self._origin(height, width)
        mask[:, r0 : r0 + self.size, c0 : c0 + self.size] = 1.0
        return mask

    def apply_global(self, images: torch.Tensor) -> torch.Tensor:
        out = images.clone()
        h, w = out.shape[-2], out.shape[-1]
        r0, c0 = self._origin(h, w)
        out[..., r0 : r0 + self.size, c0 : c0 + self.size] = self.value
        return out

    def apply_global_trigger(self, images: torch.Tensor) -> torch.Tensor:
        return self.apply_global(images)


class PoisonedLocalTrainer(LocalTrainer):
    def __init__(self, epochs: int, lr: float, device: str, trigger: Any, target_label: int, poison_ratio: float) -> None:
        super().__init__(int(epochs), float(lr), device)
        self.trigger = trigger
        self.target_label = int(target_label)
        self.poison_ratio = float(poison_ratio)
        self.poisoned_sample_count = 0
        self.seen_sample_count = 0

    def train(self, model, loader):
        model.to(self.device)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        total = 0.0
        n = 0
        for _ in range(self.epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                batch_n = len(x)
                poison_n = max(0, min(batch_n, int(round(batch_n * self.poison_ratio))))
                if poison_n > 0:
                    idx = torch.randperm(batch_n, device=self.device)[:poison_n]
                    x = x.clone()
                    y = y.clone()
                    x[idx] = self.trigger.apply_global(x[idx])
                    y[idx] = self.target_label
                    self.poisoned_sample_count += poison_n
                self.seen_sample_count += batch_n
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                total += float(loss.item()) * batch_n
                n += batch_n
        return total / max(1, n)


def flatten_update(model, codec_factory, global_state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    codec = codec_factory(model)
    local_flat = codec.flatten_state_dict(model.state_dict())
    global_flat = codec.flatten_state_dict(global_state)
    return local_flat, global_flat


class NeurotoxinAttack(AttackStrategy):
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.trigger = PatchTrigger(cfg.neurotoxin_trigger_size, cfg.neurotoxin_trigger_location, cfg.neurotoxin_trigger_value)
        self._last_metadata: dict[int, dict[str, Any]] = {}

    def should_poison(self, client_id: int, round_id: int) -> bool:
        return int(self.cfg.neurotoxin_attack_start_round) <= round_id <= int(self.cfg.neurotoxin_attack_end_round)

    def train_local_update(self, client_id: int, loader: Any, model_factory: Any, codec_factory: Any, global_state: dict[str, Any], round_id: int) -> LocalUpdateRecord:
        local_model = model_factory()
        local_model.load_state_dict(copy.deepcopy(global_state))
        trainer = PoisonedLocalTrainer(
            self.cfg.neurotoxin_local_epochs,
            self.cfg.neurotoxin_local_lr,
            self.cfg.device,
            self.trigger,
            self.cfg.neurotoxin_target_label,
            self.cfg.neurotoxin_poison_ratio,
        )
        train_loss = trainer.train(local_model, loader)
        local_flat, global_flat = flatten_update(local_model, codec_factory, global_state)
        update = (local_flat - global_flat) * float(self.cfg.neurotoxin_scale_factor)
        if not np.all(np.isfinite(update)):
            raise ValueError("Neurotoxin produced a non-finite local update")
        self._last_metadata[client_id] = {
            "attack_name": "neurotoxin",
            "update_type": "poisoned",
            "backdoor_attack_active": True,
            "poisoned_sample_count": trainer.poisoned_sample_count,
            "effective_poison_ratio": trainer.poisoned_sample_count / max(1, trainer.seen_sample_count),
            "poison_ratio": trainer.poisoned_sample_count / max(1, trainer.seen_sample_count),
        }
        return LocalUpdateRecord(client_id, len(loader.dataset), update, float(train_loss), metadata=self._last_metadata[client_id].copy())

    def craft_update(self, client_id, clean_update, global_model, attacker_context):
        update = np.asarray(clean_update, dtype=np.float64).copy()
        before_norm = float(np.linalg.norm(update))
        observable = attacker_context.get("observable_updates") or []
        if observable:
            benign_mean = np.mean(np.stack(observable, axis=0), axis=0)
            k = int(round(update.size * float(self.cfg.neurotoxin_topk_ratio)))
            k = max(0, min(update.size, k))
            masked = np.zeros(update.size, dtype=bool)
            if k > 0:
                top_idx = np.argpartition(np.abs(benign_mean), -k)[-k:]
                masked[top_idx] = True
                update[masked] *= float(self.cfg.neurotoxin_mask_decay)
        else:
            masked = np.zeros(update.size, dtype=bool)
        if not np.all(np.isfinite(update)) or update.shape != np.asarray(clean_update).shape:
            raise ValueError("Neurotoxin produced an invalid update")
        after_norm = float(np.linalg.norm(update))
        meta = self._last_metadata.get(client_id, {}).copy()
        meta.update({
            "attack_name": "neurotoxin",
            "neurotoxin_mask_ratio": float(masked.mean()) if masked.size else 0.0,
            "masked_coordinate_count": int(masked.sum()),
            "update_norm_before_mask": before_norm,
            "update_norm_after_mask": after_norm,
            "poisoned_update_norm": after_norm,
        })
        self._last_metadata[client_id] = meta
        return update

    def pop_last_metadata(self, client_id: int) -> dict[str, Any]:
        return self._last_metadata.pop(client_id, {})
