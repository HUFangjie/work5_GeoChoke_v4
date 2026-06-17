import random, time, numpy as np, torch
class AggregationServer:
    def __init__(self,cfg,model,codec,crypto_backend,decryption_service,defense):
        self.cfg=cfg; self.model=model; self.codec=codec; self.crypto_backend=crypto_backend; self.decryption_service=decryption_service; self.defense=defense
        if hasattr(self,'secret_key'): raise RuntimeError('server must not hold secret key')
    def sample_clients(self,round_id):
        rng=random.Random(self.cfg.seed+round_id); ids=list(range(self.cfg.num_clients)); rng.shuffle(ids); return ids[:self.cfg.clients_per_round]
    def aggregate_encrypted(self,uploads):
        if len(uploads)<self.cfg.min_clients_per_round: raise ValueError('too few aggregation participants')
        pid=uploads[0].profile_id
        if any(u.profile_id!=pid for u in uploads): raise ValueError('profile mismatch among client uploads')
        total=sum(u.num_samples for u in uploads); weighted=[]
        t0=time.perf_counter()
        for u in uploads: weighted.append(self.crypto_backend.multiply_plain(u.encrypted_update,u.num_samples/total))
        agg=self.crypto_backend.add_ciphertexts(weighted); return agg, time.perf_counter()-t0
    def apply_round(self,uploads,round_id):
        prev={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}; agg,agg_time=self.aggregate_encrypted(uploads); t0=time.perf_counter(); dec=self.decryption_service.decrypt_aggregate(agg,uploads[0].profile_id); dec_time=time.perf_counter()-t0
        cand=self.codec.update_to_state_dict(prev, self.cfg.server_lr*dec); candidate=type(self.model)(); candidate.load_state_dict(cand); geo=self.defense.after_aggregate(self.model,candidate,uploads[0].profile_id,round_id); self.model.load_state_dict(cand)
        plain_ref=None; mse=maxerr=None
        if self.cfg.enable_plaintext_reference_metrics and all(u.plaintext_reference is not None for u in uploads):
            total=sum(u.num_samples for u in uploads); plain_ref=sum((u.num_samples/total)*u.plaintext_reference for u in uploads); diff=dec-plain_ref; mse=float(np.mean(diff*diff)); maxerr=float(np.max(np.abs(diff)))
        return dec, {'encrypted_aggregation_time':agg_time,'decryption_time':dec_time,'ciphertext_block_count':agg.block_count,'serialized_ciphertext_bytes':sum(len(c.payload.serialize()) for c in agg.chunks),'aggregate_update_norm':float(np.linalg.norm(dec)),'aggregate_ckks_mse':mse,'aggregate_maximum_absolute_error':maxerr, **geo}
