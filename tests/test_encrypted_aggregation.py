import numpy as np, pytest
from config import CKKS_PROFILES
from crypto.ckks_context_manager import CKKSContextManager
from crypto.ckks_backend import CKKSBackend
from core.decryption_service import AuthorizedDecryptionService

def test_encrypted_weighted_aggregation_and_profile_mismatch():
    cm=CKKSContextManager(CKKS_PROFILES); b=CKKSBackend(cm,CKKS_PROFILES); b.initialize_profiles(); d=AuthorizedDecryptionService(cm)
    v1=np.array([1.,2.,3.]); v2=np.array([4.,5.,6.]); e1=b.encrypt_update(v1,'high_precision'); e2=b.encrypt_update(v2,'high_precision')
    agg=b.add_ciphertexts([b.multiply_plain(e1,0.25),b.multiply_plain(e2,0.75)]); dec=d.decrypt_aggregate(agg,'high_precision')
    assert np.allclose(dec,0.25*v1+0.75*v2,atol=1e-3)
    with pytest.raises(ValueError): b.add_ciphertexts([e1,b.encrypt_update(v2,'medium_precision')])
