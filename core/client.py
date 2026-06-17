import copy, time, numpy as np, torch
from core.types import ClientUpload
from core.local_trainer import LocalTrainer
class Client:
    def __init__(self,client_id,loader,model_factory,codec_factory,crypto_backend,attack_strategy,malicious,cfg):
        self.client_id=client_id; self.loader=loader; self.model_factory=model_factory; self.codec_factory=codec_factory; self.crypto_backend=crypto_backend; self.attack_strategy=attack_strategy; self.malicious=malicious; self.cfg=cfg
    def run_round(self, global_state, profile_id, attacker_context):
        model=self.model_factory(); model.load_state_dict(copy.deepcopy(global_state)); trainer=LocalTrainer(self.cfg.local_epochs,self.cfg.local_lr,self.cfg.device); loss=trainer.train(model,self.loader)
        codec=self.codec_factory(model); new=codec.flatten_state_dict(model.state_dict()); old=codec.flatten_state_dict(global_state); clean=new-old
        before=clean.copy(); t0=time.perf_counter(); update=self.attack_strategy.craft_update(self.client_id,clean,model,attacker_context) if self.malicious else clean; attack_time=time.perf_counter()-t0
        enc=self.crypto_backend.encrypt_update(update,profile_id)
        meta={'train_loss':loss,'attack_time':attack_time,'malicious_update_norm_before':float(np.linalg.norm(before)),'malicious_update_norm_after':float(np.linalg.norm(update)),'cosine_before_after':float(np.dot(before,update)/(np.linalg.norm(before)*np.linalg.norm(update)+1e-12)),'attack_applied_before_encryption':bool(self.malicious)}
        return ClientUpload(self.client_id,len(self.loader.dataset),profile_id,enc,meta, update.copy() if self.cfg.enable_plaintext_reference_metrics else None)
