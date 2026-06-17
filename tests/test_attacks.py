import numpy as np
from attacks.no_attack import NoAttack
from attacks.alie import ALIEAttack
from attacks.fang import FangMeanAttack

def test_attacks_modify_only_when_applied_pre_encryption_flag_logic():
    clean=np.ones(5); ctx={'observable_updates':[clean,clean*2],'num_selected':3,'num_malicious':1}
    assert np.allclose(NoAttack().craft_update(0,clean,None,ctx),clean)
    assert not np.allclose(ALIEAttack(z=1.0).craft_update(1,clean,None,ctx),clean)
    assert not np.allclose(FangMeanAttack().craft_update(1,clean,None,ctx),clean)
