import numpy as np, torch

def ensure_finite_array(x, name:str="array"):
    arr=np.asarray(x)
    if not np.all(np.isfinite(arr)): raise ValueError(f"{name} contains non-finite values")

def model_l2_norm(model: torch.nn.Module)->float:
    return float(torch.sqrt(sum((p.detach()**2).sum() for p in model.parameters())).cpu())
