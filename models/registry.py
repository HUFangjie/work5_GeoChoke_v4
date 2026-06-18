from typing import Callable, Dict
import torch
_REGISTRY:Dict[str,Callable[[],torch.nn.Module]]={}
def register_model(name:str):
    def deco(fn): _REGISTRY[name]=fn; return fn
    return deco
def create_model(name:str)->torch.nn.Module:
    if name not in _REGISTRY: raise KeyError(f'unknown model {name}')
    return _REGISTRY[name]()
