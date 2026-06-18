import torch


class Evaluator:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def evaluate(self, model):
        model.to(self.device).eval()
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = 0.0
        correct = 0
        n = 0
        with torch.no_grad():
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                loss += float(loss_fn(logits, y))
                correct += int((logits.argmax(1) == y).sum())
                n += len(x)
        return loss / max(1, n), correct / max(1, n)

    def _asr(self, model, transform, target_label: int) -> float:
        model.to(self.device).eval()
        hits = 0
        n = 0
        with torch.no_grad():
            for x, y in self.loader:
                mask = y != target_label
                if not bool(mask.any()):
                    continue
                x = x[mask].to(self.device)
                x = transform(x)
                logits = model(x)
                hits += int((logits.argmax(1).cpu() == target_label).sum())
                n += len(x)
        return hits / max(1, n)

    def evaluate_dba(self, model, attack, cfg, profile_id, uploads):
        if cfg.attack_name != "dba" or not hasattr(attack, "trigger"):
            return {}
        target = int(cfg.dba_target_label)
        metrics = {
            "global_trigger_asr": self._asr(model, attack.trigger.apply_global, target),
            "attack_active": any(upload.metadata.get("dba_attack_active", False) for upload in uploads),
            "active_malicious_clients": [upload.client_id for upload in uploads if upload.metadata.get("dba_attack_active", False)],
            "poisoned_sample_count": int(sum(upload.metadata.get("poisoned_sample_count", 0) for upload in uploads)),
            "poison_ratio": float(sum(upload.metadata.get("poisoned_sample_count", 0) for upload in uploads)) / max(1, sum(upload.metadata.get("dba_seen_sample_count", 0) for upload in uploads if upload.metadata.get("dba_attack_active", False))),
            "malicious_update_norm": float(sum(upload.metadata.get("malicious_update_norm_after", 0.0) for upload in uploads if upload.metadata.get("dba_attack_active", False))),
            "dba_scale_factor": float(cfg.dba_scale_factor),
            "current_ckks_profile": profile_id,
        }
        for trigger_id in range(int(cfg.dba_num_trigger_parts)):
            metrics[f"local_trigger_{trigger_id + 1}_asr"] = self._asr(
                model,
                lambda x, trigger_id=trigger_id: attack.trigger.apply_local(x, trigger_id),
                target,
            )
        return metrics
