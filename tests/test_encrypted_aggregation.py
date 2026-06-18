import numpy as np
import pytest

from config import CKKS_PROFILES
from core.decryption_service import AuthorizedDecryptionService
from crypto.aggregation_pipeline_validator import AggregationPipelineValidator
from crypto.ckks_backend import CKKSBackend
from crypto.ckks_context_manager import CKKSContextManager


def _build_crypto():
    manager = CKKSContextManager(CKKS_PROFILES)
    manager.initialize()
    backend = CKKSBackend(manager.public_bundles())
    backend.initialize_profiles()
    return backend, AuthorizedDecryptionService(manager)


def _aggregate(backend, service, profile_id, vectors, weights):
    encrypted = [backend.encrypt_update(vector, profile_id) for vector in vectors]
    weighted = [backend.multiply_plain(ciphertext, weight) for ciphertext, weight in zip(encrypted, weights)]
    aggregate = backend.add_ciphertexts(weighted)
    return service.decrypt_aggregate(aggregate, profile_id), aggregate


def test_ckks_weighted_aggregation_for_every_profile():
    backend, service = _build_crypto()
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])
    weights = [0.25, 0.75]
    expected = 0.25 * v1 + 0.75 * v2
    for profile_id in CKKS_PROFILES:
        decrypted, _aggregate_ciphertext = _aggregate(backend, service, profile_id, [v1, v2], weights)
        np.testing.assert_allclose(decrypted, expected, rtol=1e-2, atol=1e-2)
        ratio = np.linalg.norm(decrypted) / np.linalg.norm(expected)
        assert abs(ratio - 1.0) < 5e-2


def test_single_multi_client_signed_and_chunked_aggregation():
    backend, service = _build_crypto()
    profile_id = "high_precision"
    slot_count = backend.slot_count(profile_id)
    vectors = [
        np.linspace(-1.0, 1.0, slot_count + 17),
        np.linspace(0.5, -0.5, slot_count + 17),
        np.full(slot_count + 17, 1e-6),
    ]
    sample_counts = np.array([2.0, 3.0, 5.0])
    weights = (sample_counts / sample_counts.sum()).tolist()
    expected = sum(weight * vector for weight, vector in zip(weights, vectors))
    decrypted, aggregate = _aggregate(backend, service, profile_id, vectors, weights)
    assert aggregate.block_count == 2
    assert decrypted.shape == expected.shape
    np.testing.assert_allclose(decrypted, expected, rtol=1e-2, atol=1e-2)
    assert abs(np.linalg.norm(decrypted) / np.linalg.norm(expected) - 1.0) < 5e-2


def test_profile_mismatch_rejected():
    backend, _service = _build_crypto()
    left = backend.encrypt_update(np.array([1.0]), "high_precision")
    right = backend.encrypt_update(np.array([1.0]), "medium_precision")
    with pytest.raises(ValueError):
        backend.add_ciphertexts([left, right])


def test_pipeline_validator_rejects_scaled_decryptor():
    backend, service = _build_crypto()

    def scaled_decrypt(ciphertext, profile_id):
        return service.decrypt_aggregate(ciphertext, profile_id) / 32.0

    validator = AggregationPipelineValidator(backend, scaled_decrypt, rtol=1e-2, atol=1e-2, norm_ratio_tolerance=5e-2)
    result = validator.validate_profile(
        "high_precision",
        [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
        [0.25, 0.75],
    )
    assert result.passed is False
