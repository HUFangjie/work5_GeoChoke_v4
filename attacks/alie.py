import math, numpy as np
from statistics import NormalDist
from attacks.base import AttackStrategy
class ALIEAttack(AttackStrategy):
    def __init__(self,z=None,std_floor=1e-6,oracle_all_updates=False): self.z=z; self.std_floor=std_floor; self.oracle_all_updates=oracle_all_updates
    def craft_update(self, client_id, clean_update, global_model, attacker_context):
        obs=attacker_context.get('observable_updates',[clean_update]); arr=np.vstack(obs); mu=arr.mean(0); sigma=np.maximum(arr.std(0),self.std_floor)
        m=max(1,attacker_context.get('num_malicious',1)); n=max(m+1,attacker_context.get('num_selected',m+1))
        z=self.z if self.z is not None else NormalDist().inv_cdf(max(0.51, min(0.99, (n-m)/n)))
        direction=np.sign(mu); direction[direction==0]=1.0
        return mu - z*sigma*direction
