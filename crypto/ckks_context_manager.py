from dataclasses import dataclass
from typing import Dict, Any
import tenseal as ts

@dataclass
class CKKSContextBundle:
    profile_id:str; secret_context:Any; public_context:Any; slot_count:int

class CKKSContextManager:
    def __init__(self, profiles:Dict[str,Dict[str,Any]]): self.profiles=profiles; self.bundles:{}={}
    def initialize(self)->None:
        for pid,cfg in self.profiles.items():
            ctx=ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=cfg['poly_modulus_degree'], coeff_mod_bit_sizes=cfg['coeff_mod_bit_sizes'])
            ctx.global_scale=2**cfg['global_scale_bits']; ctx.generate_galois_keys()
            pub=ts.context_from(ctx.serialize(save_secret_key=False))
            self.bundles[pid]=CKKSContextBundle(pid, ctx, pub, cfg['poly_modulus_degree']//2)
    def secret(self,pid): return self.bundles[pid].secret_context
    def public(self,pid): return self.bundles[pid].public_context
    def slot_count(self,pid)->int: return self.bundles[pid].slot_count
