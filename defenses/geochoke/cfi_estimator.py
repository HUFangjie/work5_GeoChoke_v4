import copy, torch
import torch.nn.functional as F
class CFIEstimator:
    def __init__(self,codec,proxy_loader,perturbation_bank,device): self.codec=codec; self.proxy_loader=proxy_loader; self.perturbation_bank=perturbation_bank; self.device=device
    def _js(self,p,q):
        eps=1e-8; p=p.clamp_min(eps); q=q.clamp_min(eps); m=0.5*(p+q); return 0.5*(p*(p.log()-m.log())).sum(1)+0.5*(q*(q.log()-m.log())).sum(1)
    def estimate(self,model):
        base=copy.deepcopy(model).to(self.device).eval(); base_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; total=0.0; n=0
        with torch.no_grad():
            for xi in self.perturbation_bank:
                pert=copy.deepcopy(model).to(self.device).eval(); pert.load_state_dict(self.codec.update_to_state_dict(base_state, xi))
                for x,_ in self.proxy_loader:
                    x=x.to(self.device); p=F.softmax(base(x),dim=1); q=F.softmax(pert(x),dim=1); total+=float(self._js(p,q).sum().cpu()); n+=x.shape[0]
        return total/max(1,n*len(self.perturbation_bank))
