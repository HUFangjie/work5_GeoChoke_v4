from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np, torch

@dataclass
class ParamSpec:
    name:str; shape:Tuple[int,...]; dtype:torch.dtype; numel:int

class ModelUpdateCodec:
    def __init__(self, model:torch.nn.Module):
        self.specs=[ParamSpec(n,tuple(p.shape),p.dtype,p.numel()) for n,p in model.named_parameters() if p.requires_grad and p.is_floating_point()]
        self.total_dimension=sum(s.numel for s in self.specs)
    def flatten_state_dict(self, state_dict:Dict[str,torch.Tensor])->np.ndarray:
        parts=[]
        for s in self.specs:
            if s.name not in state_dict: raise KeyError(s.name)
            t=state_dict[s.name].detach().cpu().to(torch.float64).reshape(-1); parts.append(t.numpy())
        return np.concatenate(parts) if parts else np.array([],dtype=np.float64)
    def unflatten_to_state_dict(self, flat, reference_state_dict:Dict[str,torch.Tensor])->Dict[str,torch.Tensor]:
        arr=np.asarray(flat); out={k:v.clone() for k,v in reference_state_dict.items()}; pos=0
        for s in self.specs:
            vals=torch.tensor(arr[pos:pos+s.numel], dtype=s.dtype).reshape(s.shape); out[s.name]=vals; pos+=s.numel
        if pos!=len(arr): raise ValueError('flat vector length mismatch')
        return out
    def update_to_state_dict(self, base_state, flat_update):
        upd=self.unflatten_to_state_dict(flat_update, base_state); return {k:(base_state[k]+upd[k] if k in upd and torch.is_floating_point(base_state[k]) else v) for k,v in base_state.items()}
