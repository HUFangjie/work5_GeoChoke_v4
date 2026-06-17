import copy, torch
from config import GeoChokeConfig
from defenses.geochoke.controller import GeoChokeController

def test_controller_next_round_only():
    cfg=GeoChokeConfig(initial_profile_id='a'); cal={'a':{'mse':0.0},'b':{'mse':1.0}}
    ctl=GeoChokeController(cfg,cal); next_pid,m=ctl.select(0.1,0.3,'a')
    assert next_pid in cal and m['selected_next_profile']==next_pid
