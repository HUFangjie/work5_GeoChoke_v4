import numpy as np

from config import CKKS_PROFILES
from core.decryption_service import AuthorizedDecryptionService
from crypto.ckks_backend import CKKSBackend
from crypto.ckks_context_manager import CKKSContextManager


def _build_crypto():
    manager = CKKSContextManager(CKKS_PROFILES)
    manager.initialize()
    backend = CKKSBackend(manager.public_bundles())
    backend.initialize_profiles()
    return backend, AuthorizedDecryptionService(manager)


def test_ckks_roundtrip_and_chunking():
    backend, service = _build_crypto()
    vector = np.linspace(-1, 1, 9000)
    encrypted = backend.encrypt_update(vector, "high_precision")
    assert encrypted.block_count > 1
    decrypted = service.decrypt_aggregate(encrypted, "high_precision")
    assert np.mean((decrypted - vector) ** 2) < 1e-6
