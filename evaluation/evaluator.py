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

    def _asr_counts(self, model, transform, target_label: int) -> tuple[int, int, float]:
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
        if n == 0:
            raise ValueError("DBA ASR denominator is zero after excluding target-label test samples")
        return hits, n, hits / n

    def evaluate_dba(self, model, attack, cfg, profile_id, uploads):
        if cfg.attack_name != "dba" or not hasattr(attack, "trigger"):
            return {}
        target = int(cfg.dba_target_label)
        global_hits, global_count, global_asr = self._asr_counts(model, attack.trigger.apply_global, target)
        poisoned = int(sum(upload.metadata.get("poisoned_sample_count", 0) for upload in uploads))
        seen = int(sum(upload.metadata.get("dba_seen_sample_count", 0) for upload in uploads if upload.metadata.get("dba_attack_active", False)))
        metrics = {
            "global_trigger_asr": global_asr,
            "global_trigger_hits": global_hits,
            "global_trigger_test_count": global_count,
            "attack_active": any(upload.metadata.get("dba_attack_active", False) for upload in uploads),
            "active_malicious_clients": [upload.client_id for upload in uploads if upload.metadata.get("dba_attack_active", False)],
            "poisoned_sample_count": poisoned,
            "effective_poison_ratio": poisoned / seen if seen > 0 else 0.0,
            "poison_ratio": poisoned / seen if seen > 0 else 0.0,
            "malicious_update_norm": float(sum(upload.metadata.get("poisoned_update_norm", 0.0) for upload in uploads if upload.metadata.get("dba_attack_active", False))),
            "dba_scale_factor": float(max([upload.metadata.get("dba_scale_factor", 0.0) for upload in uploads if upload.metadata.get("dba_attack_active", False)] or [0.0])),
            "current_ckks_profile": profile_id,
        }
        for trigger_id in range(int(cfg.dba_num_trigger_parts)):
            hits, count, asr = self._asr_counts(
                model,
                lambda x, trigger_id=trigger_id: attack.trigger.apply_local(x, trigger_id),
                target,
            )
            metrics[f"local_trigger_{trigger_id + 1}_asr"] = asr
            metrics[f"local_trigger_{trigger_id + 1}_hits"] = hits
            metrics[f"local_trigger_{trigger_id + 1}_test_count"] = count
        return metrics
