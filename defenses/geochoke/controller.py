class GeoChokeController:
    def __init__(self,cfg,calibration): self.cfg=cfg; self.calibration=calibration
    def select(self,prev_cfi,cand_cfi,current_profile_id):
        delta=max(0.0,cand_cfi-prev_cfi); u=self.calibration[current_profile_id]['mse']; raw=(self.cfg.lambda_*delta-prev_cfi+2*self.cfg.rho*u)/(2*(self.cfg.gamma+self.cfg.rho)); umax=max(v['mse'] for v in self.calibration.values()); target=min(max(raw,0.0),umax)
        next_pid=min(self.calibration, key=lambda p: abs(self.calibration[p]['mse']-target))
        return next_pid, {'previous_cfi':prev_cfi,'candidate_cfi':cand_cfi,'fragility_injection_score':delta,'current_calibrated_error_energy':u,'target_next_error_energy':target,'selected_next_profile':next_pid,'profile_switching_indicator': next_pid!=current_profile_id}
