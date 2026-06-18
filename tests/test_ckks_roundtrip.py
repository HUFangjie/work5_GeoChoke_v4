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
    return manager, backend, AuthorizedDecryptionService(manager)


def test_ckks_decryption_uses_secret_key():
    manager, backend, service = _build_crypto()
    vector = np.array([1.0, -2.0, 0.25])
    encrypted = backend.encrypt_update(vector, "high_precision")
    direct = encrypted.chunks[0].payload.decrypt(manager.secret_key("high_precision"))[:3]
    decrypted = service.decrypt_aggregate(encrypted, "high_precision")
    np.testing.assert_allclose(decrypted, direct, rtol=1e-3, atol=1e-3)
    assert not hasattr(backend, "secret_key")


def test_ckks_roundtrip_for_every_profile():
    _manager, backend, service = _build_crypto()
    vector = np.array([1.0, -2.0, 0.0, 1e-6, 3.5])
    for profile_id in CKKS_PROFILES:
        encrypted = backend.encrypt_update(vector, profile_id)
        decrypted = service.decrypt_aggregate(encrypted, profile_id)
        np.testing.assert_allclose(decrypted, vector, rtol=1e-3, atol=1e-3)


def test_ckks_roundtrip_chunking():
    _manager, backend, service = _build_crypto()
    vector = np.linspace(-1, 1, 9000)
    encrypted = backend.encrypt_update(vector, "high_precision")
    assert encrypted.block_count > 1
    decrypted = service.decrypt_aggregate(encrypted, "high_precision")
    assert decrypted.shape == vector.shape
    np.testing.assert_allclose(decrypted, vector, rtol=1e-3, atol=1e-3)
