from defenses.base import DefenseStrategy
from crypto.update_codec import ModelUpdateCodec
from defenses.geochoke.calibration import ProfileCalibrator
from defenses.geochoke.cfi_estimator import CFIEstimator
from defenses.geochoke.controller import GeoChokeController
class GeoChokeDefense(DefenseStrategy):
    def __init__(self,cfg,profiles): self.cfg=cfg; self.profiles=profiles; self.next_profile=cfg.initial_profile_id; self.history=[]
    def initialize(self, model, crypto_backend, proxy_loader):
        self.codec=ModelUpdateCodec(model); self.calibration,self.perturbation_bank=ProfileCalibrator(crypto_backend,self.profiles,self.cfg).calibrate(self.codec.total_dimension); self.estimator=CFIEstimator(self.codec,proxy_loader,self.perturbation_bank,'cpu'); self.controller=GeoChokeController(self.cfg,self.calibration)
    def get_profile_for_round(self, round_id): return self.cfg.initial_profile_id if round_id<self.cfg.warmup_rounds else self.next_profile
    def after_aggregate(self, previous_model, candidate_model, current_profile_id, round_id):
        prev=self.estimator.estimate(previous_model); cand=self.estimator.estimate(candidate_model)
        next_pid,metrics=self.controller.select(prev,cand,current_profile_id)
        if round_id>=self.cfg.warmup_rounds: self.next_profile=next_pid
        self.history.append({'round':round_id,'current_profile':current_profile_id,**metrics}); return metrics
