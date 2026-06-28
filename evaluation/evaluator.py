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

    def _asr_metrics(self, model, transform, target_label: int) -> dict[str, int | float]:
        """Measure trigger success while auditing clean target-class bias.

        Raw ASR counts non-target-label samples whose triggered prediction is the
        target label. This can be inflated when the clean model already predicts
        many samples as the target class. The debiased ASR therefore only counts
        samples that were *not* predicted as the target before applying the
        trigger, and reports the clean target bias separately.
        """

        model.to(self.device).eval()
        raw_hits = 0
        clean_target_hits = 0
        debiased_hits = 0
        clean_non_target_count = 0
        n = 0
        with torch.no_grad():
            for x, y in self.loader:
                mask = y != target_label
                if not bool(mask.any()):
                    continue
                x = x[mask].to(self.device)
                clean_pred = model(x).argmax(1)
                triggered_pred = model(transform(x)).argmax(1)
                clean_target = clean_pred == target_label
                triggered_target = triggered_pred == target_label
                raw_hits += int(triggered_target.sum().item())
                clean_target_hits += int(clean_target.sum().item())
                debiased_hits += int((triggered_target & ~clean_target).sum().item())
                clean_non_target_count += int((~clean_target).sum().item())
                n += len(x)
        if n == 0:
            raise ValueError("DBA ASR denominator is zero after excluding target-label test samples")
        return {
            "hits": debiased_hits,
            "test_count": n,
            "eligible_test_count": clean_non_target_count,
            "asr": debiased_hits / max(1, clean_non_target_count),
            "raw_hits": raw_hits,
            "raw_asr": raw_hits / n,
            "clean_target_hits": clean_target_hits,
            "clean_target_rate": clean_target_hits / n,
            "asr_lift": (raw_hits - clean_target_hits) / n,
        }

    def evaluate_backdoor(self, model, attack, cfg, profile_id, uploads):
        transform = None
        if hasattr(attack, "trigger") and hasattr(attack.trigger, "apply_global"):
            transform = attack.trigger.apply_global
        elif hasattr(attack, "apply_global_trigger"):
            transform = attack.apply_global_trigger
        if transform is None:
            return {}
        target = int(getattr(cfg, f"{cfg.attack_name}_target_label", getattr(cfg, "dba_target_label", 0)))
        global_metrics = self._asr_metrics(model, transform, target)
        poisoned = int(sum(upload.metadata.get("poisoned_sample_count", 0) for upload in uploads))
        seen = int(sum(upload.metadata.get("dba_seen_sample_count", 0) for upload in uploads if upload.metadata.get("dba_attack_active", False)))
        metrics = {
            "global_trigger_asr": global_metrics["asr"],
            "global_trigger_hits": global_metrics["hits"],
            "global_trigger_test_count": global_metrics["test_count"],
            "global_trigger_eligible_test_count": global_metrics["eligible_test_count"],
            "global_trigger_raw_asr": global_metrics["raw_asr"],
            "global_trigger_raw_hits": global_metrics["raw_hits"],
            "global_clean_target_rate": global_metrics["clean_target_rate"],
            "global_clean_target_hits": global_metrics["clean_target_hits"],
            "global_trigger_asr_lift": global_metrics["asr_lift"],
            "attack_active": any(upload.metadata.get("dba_attack_active", False) for upload in uploads),
            "active_malicious_clients": [upload.client_id for upload in uploads if upload.metadata.get("dba_attack_active", False)],
            "poisoned_sample_count": poisoned,
            "effective_poison_ratio": poisoned / seen if seen > 0 else 0.0,
            "poison_ratio": poisoned / seen if seen > 0 else 0.0,
            "malicious_update_norm": float(sum(upload.metadata.get("poisoned_update_norm", 0.0) for upload in uploads if upload.metadata.get("dba_attack_active", False))),
            "dba_scale_factor": float(max([upload.metadata.get("dba_scale_factor", 0.0) for upload in uploads if upload.metadata.get("dba_attack_active", False)] or [0.0])),
            "current_ckks_profile": profile_id,
        }
        if cfg.attack_name == "dba" and hasattr(attack, "trigger") and hasattr(attack.trigger, "apply_local"):
            for trigger_id in range(int(cfg.dba_num_trigger_parts)):
                local_metrics = self._asr_metrics(
                    model,
                    lambda x, trigger_id=trigger_id: attack.trigger.apply_local(x, trigger_id),
                    target,
                )
                prefix = f"local_trigger_{trigger_id + 1}"
                metrics[f"{prefix}_asr"] = local_metrics["asr"]
                metrics[f"{prefix}_hits"] = local_metrics["hits"]
                metrics[f"{prefix}_test_count"] = local_metrics["test_count"]
                metrics[f"{prefix}_eligible_test_count"] = local_metrics["eligible_test_count"]
                metrics[f"{prefix}_raw_asr"] = local_metrics["raw_asr"]
                metrics[f"{prefix}_raw_hits"] = local_metrics["raw_hits"]
                metrics[f"{prefix}_clean_target_rate"] = local_metrics["clean_target_rate"]
                metrics[f"{prefix}_clean_target_hits"] = local_metrics["clean_target_hits"]
                metrics[f"{prefix}_asr_lift"] = local_metrics["asr_lift"]
        return metrics

    def evaluate_dba(self, model, attack, cfg, profile_id, uploads):
        return self.evaluate_backdoor(model, attack, cfg, profile_id, uploads)
