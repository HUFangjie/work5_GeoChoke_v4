import numpy as np
import pytest

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


def test_encrypted_weighted_aggregation_and_profile_mismatch():
    backend, service = _build_crypto()
    left = np.array([1.0, 2.0, 3.0])
    right = np.array([4.0, 5.0, 6.0])
    encrypted_left = backend.encrypt_update(left, "high_precision")
    encrypted_right = backend.encrypt_update(right, "high_precision")
    aggregate = backend.add_ciphertexts(
        [backend.multiply_plain(encrypted_left, 0.25), backend.multiply_plain(encrypted_right, 0.75)]
    )
    decrypted = service.decrypt_aggregate(aggregate, "high_precision")
    assert np.allclose(decrypted, 0.25 * left + 0.75 * right, atol=1e-3)
    with pytest.raises(ValueError):
        backend.add_ciphertexts([encrypted_left, backend.encrypt_update(right, "medium_precision")])
