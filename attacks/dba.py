from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from attacks.base import AttackStrategy
from core.local_trainer import LocalTrainer
from core.types import LocalUpdateRecord


class PoisonedBatchList(list):
    poisoned_sample_count: int = 0


@dataclass(frozen=True)
class TriggerPatch:
    row_start: int
    row_end: int
    col_start: int
    col_end: int


class DBAAttack(AttackStrategy):
    """Distributed Backdoor Attack with disjoint local trigger parts.

    Each malicious client is bound to one local trigger part. During evaluation,
    all local trigger parts are combined into the global trigger.
    """

    effective_attack_name = "dba"

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.target_label = int(cfg.dba_target_label)
        self.num_trigger_parts = int(cfg.dba_num_trigger_parts)
        if len(cfg.malicious_client_ids) < self.num_trigger_parts:
            raise ValueError(
                "DBA requires at least dba_num_trigger_parts malicious clients; "
                f"got {len(cfg.malicious_client_ids)} malicious clients for {self.num_trigger_parts} trigger parts"
            )
        self.client_to_trigger = {
            int(client_id): index % self.num_trigger_parts for index, client_id in enumerate(cfg.malicious_client_ids)
        }
        self.patches = self._build_patches()

    def craft_update(self, client_id: int, clean_update: np.ndarray, global_model: Any, attacker_context: dict[str, Any]) -> np.ndarray:
        return np.asarray(clean_update, dtype=np.float64)

    def is_active(self, client_id: int, round_id: int) -> bool:
        if client_id not in self.client_to_trigger:
            return False
        if not (int(self.cfg.dba_attack_start_round) <= round_id <= int(self.cfg.dba_attack_end_round)):
            return False
        interval = max(1, int(self.cfg.dba_poison_interval))
        if (round_id - int(self.cfg.dba_attack_start_round)) % interval != 0:
            return False
        mode = str(self.cfg.dba_attack_mode)
        if mode == "multi_shot":
            return True
        if mode == "single_shot":
            slot = ((round_id - int(self.cfg.dba_attack_start_round)) // interval) % self.num_trigger_parts
            return self.client_to_trigger[client_id] == slot
        raise ValueError(f"Unknown dba_attack_mode: {self.cfg.dba_attack_mode}")

    def craft_local_update(
        self,
        client_id: int,
        global_state: dict[str, torch.Tensor],
        model_factory: Any,
        codec_factory: Any,
        loader: Any,
        device: str,
        round_id: int,
    ) -> LocalUpdateRecord:
        if not self.is_active(client_id, round_id):
            raise ValueError("craft_local_update called for inactive DBA client")
        local_model = model_factory()
        local_model.load_state_dict({name: tensor.detach().cpu().clone() for name, tensor in global_state.items()})
        poisoned_loader = self._poisoned_batches(loader, self.client_to_trigger[client_id], device)
        trainer = LocalTrainer(int(self.cfg.dba_local_epochs), float(self.cfg.dba_local_lr), device)
        train_loss = trainer.train(local_model, poisoned_loader)
        codec = codec_factory(local_model)
        local_flat = codec.flatten_state_dict(local_model.state_dict())
        global_flat = codec.flatten_state_dict(global_state)
        update = (local_flat - global_flat) * float(self.cfg.dba_scale_factor)
        return LocalUpdateRecord(
            client_id=client_id,
            num_samples=len(loader.dataset),
            clean_update=update,
            train_loss=float(train_loss),
            metadata={
                "dba_active": True,
                "dba_trigger_id": self.client_to_trigger[client_id],
                "poisoned_sample_count": int(getattr(poisoned_loader, "poisoned_sample_count", 0)),
                "poison_ratio": float(self.cfg.dba_poison_ratio),
                "dba_scale_factor": float(self.cfg.dba_scale_factor),
            },
        )

    def apply_local_trigger(self, image: torch.Tensor, trigger_id: int) -> torch.Tensor:
        patched = image.clone()
        patch = self.patches[int(trigger_id)]
        if patched.dim() == 3:
            patched[:, patch.row_start : patch.row_end, patch.col_start : patch.col_end] = float(self.cfg.dba_trigger_value)
        elif patched.dim() == 4:
            patched[:, :, patch.row_start : patch.row_end, patch.col_start : patch.col_end] = float(self.cfg.dba_trigger_value)
        else:
            raise ValueError(f"unsupported image tensor shape for DBA trigger: {tuple(patched.shape)}")
        return patched

    def apply_global_trigger(self, image: torch.Tensor) -> torch.Tensor:
        patched = image.clone()
        for trigger_id in range(self.num_trigger_parts):
            patched = self.apply_local_trigger(patched, trigger_id)
        return patched

    def evaluate_asr(self, model: torch.nn.Module, loader: Any, device: str) -> dict[str, float]:
        model.to(device).eval()
        metrics: dict[str, float] = {}
        metrics["global_trigger_asr"] = self._asr_for_trigger(model, loader, device, trigger_id=None)
        for trigger_id in range(self.num_trigger_parts):
            metrics[f"local_trigger_{trigger_id + 1}_asr"] = self._asr_for_trigger(model, loader, device, trigger_id=trigger_id)
        return metrics

    def _asr_for_trigger(self, model: torch.nn.Module, loader: Any, device: str, trigger_id: int | None) -> float:
        success = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                mask = labels != self.target_label
                if int(mask.sum()) == 0:
                    continue
                images = images[mask].to(device)
                if trigger_id is None:
                    poisoned = self.apply_global_trigger(images)
                else:
                    poisoned = self.apply_local_trigger(images, trigger_id)
                predictions = model(poisoned).argmax(dim=1).detach().cpu()
                success += int((predictions == self.target_label).sum())
                total += int(images.shape[0])
        return success / max(1, total)

    def _poisoned_batches(self, loader: Any, trigger_id: int, device: str) -> PoisonedBatchList:
        poisoned_batches = PoisonedBatchList()
        poisoned_count = 0
        generator = torch.Generator().manual_seed(int(self.cfg.seed) + 1000 + int(trigger_id))
        for images, labels in loader:
            images = images.clone()
            labels = labels.clone()
            batch_size = int(images.shape[0])
            poison_count = int(round(batch_size * float(self.cfg.dba_poison_ratio)))
            poison_count = max(0, min(batch_size, poison_count))
            if poison_count > 0:
                indices = torch.randperm(batch_size, generator=generator)[:poison_count]
                images[indices] = self.apply_local_trigger(images[indices], trigger_id)
                labels[indices] = self.target_label
                poisoned_count += poison_count
            poisoned_batches.append((images.to(device), labels.to(device)))
        setattr(poisoned_batches, "poisoned_sample_count", poisoned_count)
        return poisoned_batches

    def _build_patches(self) -> list[TriggerPatch]:
        size = int(self.cfg.dba_trigger_size)
        gap = int(self.cfg.dba_trigger_gap)
        location = str(self.cfg.dba_trigger_location)
        if size <= 0:
            raise ValueError("dba_trigger_size must be positive")
        if location == "top_left":
            base_row = gap
            base_col = gap
            row_step = 0
            col_step = size + gap
        elif location == "bottom_right":
            base_row = 28 - gap - size
            base_col = 28 - gap - self.num_trigger_parts * size - (self.num_trigger_parts - 1) * gap
            row_step = 0
            col_step = size + gap
        else:
            raise ValueError(f"Unsupported dba_trigger_location: {location}")
        patches: list[TriggerPatch] = []
        occupied: set[tuple[int, int]] = set()
        for trigger_id in range(self.num_trigger_parts):
            row_start = base_row + trigger_id * row_step
            col_start = base_col + trigger_id * col_step
            row_end = row_start + size
            col_end = col_start + size
            if row_start < 0 or col_start < 0 or row_end > 28 or col_end > 28:
                raise ValueError("DBA trigger parts do not fit into MNIST 28x28 images")
            cells = {(row, col) for row in range(row_start, row_end) for col in range(col_start, col_end)}
            if occupied.intersection(cells):
                raise ValueError("DBA local trigger parts overlap; increase dba_trigger_gap or reduce dba_trigger_size")
            occupied.update(cells)
            patches.append(TriggerPatch(row_start, row_end, col_start, col_end))
        return patches
