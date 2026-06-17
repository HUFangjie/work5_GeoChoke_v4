import numpy as np
class ProfileCalibrator:
    def __init__(self,crypto_backend,profiles,cfg): self.crypto_backend=crypto_backend; self.profiles=profiles; self.cfg=cfg
    def calibrate(self,dimension:int):
        rng=np.random.default_rng(12345); out={}; residual_bank=[]
        for pid in self.profiles:
            mses=[]; maxes=[]; residuals=[]
            for _ in range(self.cfg.calibration_vectors):
                v=rng.normal(0,0.01,size=dimension); enc=self.crypto_backend.encrypt_update(v,pid); from core.decryption_service import AuthorizedDecryptionService
                dec=AuthorizedDecryptionService(self.crypto_backend.context_manager).decrypt_aggregate(enc,pid); res=dec-v; mses.append(float(np.mean(res*res))); maxes.append(float(np.max(abs(res)))); residuals.append(res)
            out[pid]={'mse':float(np.mean(mses)),'max_abs_error':float(np.max(maxes)),'residual_samples':residuals}
        ref=list(self.profiles.keys())[0]; samples=out[ref]['residual_samples'];
        while len(residual_bank)<self.cfg.perturbation_count: residual_bank.append(samples[len(residual_bank)%len(samples)].copy()*self.cfg.perturbation_scale)
        return out,residual_bank
