import numpy as np
from config import CKKS_PROFILES
from crypto.ckks_context_manager import CKKSContextManager
from crypto.ckks_backend import CKKSBackend
from core.decryption_service import AuthorizedDecryptionService

def test_ckks_roundtrip_and_chunking():
    cm=CKKSContextManager(CKKS_PROFILES); b=CKKSBackend(cm,CKKS_PROFILES); b.initialize_profiles(); d=AuthorizedDecryptionService(cm)
    v=np.linspace(-1,1,9000); enc=b.encrypt_update(v,'high_precision'); assert enc.block_count>1
    dec=d.decrypt_aggregate(enc,'high_precision'); assert np.mean((dec-v)**2)<1e-6
