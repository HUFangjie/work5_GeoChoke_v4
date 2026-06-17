import numpy as np
from attacks.base import AttackStrategy
class FangMeanAttack(AttackStrategy):
    def __init__(self,aggregation='weighted_mean',max_norm=5.0,search_steps=6):
        if aggregation!='weighted_mean': raise ValueError('FangMeanAttack is only compatible with weighted_mean aggregation')
        self.max_norm=max_norm; self.search_steps=search_steps
    def craft_update(self, client_id, clean_update, global_model, attacker_context):
        benign=np.vstack(attacker_context.get('observable_updates',[clean_update])); center=benign.mean(0); direction=-(center/(np.linalg.norm(center)+1e-12))
        clean_norm=np.linalg.norm(clean_update)+1e-12; best=clean_update; best_score=-1
        for mag in np.linspace(clean_norm, self.max_norm*clean_norm, self.search_steps):
            cand=direction*mag; score=np.linalg.norm(cand-center)
            if score>best_score: best, best_score=cand, score
        return best
FangAttack=FangMeanAttack
