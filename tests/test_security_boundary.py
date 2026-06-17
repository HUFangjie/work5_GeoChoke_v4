from core.decryption_service import AuthorizedDecryptionService
from core.server import AggregationServer
from crypto.ckks_backend import CKKSBackend
from crypto.ckks_context_manager import CKKSContextManager
from config import CKKS_PROFILES


def test_decryption_service_api_boundary():
    assert hasattr(AuthorizedDecryptionService, "decrypt_aggregate")
    assert not hasattr(AuthorizedDecryptionService, "decrypt_client_update")
    assert not hasattr(AuthorizedDecryptionService, "decrypt_ciphertext_list")


def test_public_backend_has_no_secret_accessors():
    manager = CKKSContextManager(CKKS_PROFILES)
    manager.initialize()
    backend = CKKSBackend(manager.public_bundles())
    assert not hasattr(backend, "secret")
    assert not hasattr(backend, "secret_context")
    assert not hasattr(backend, "context_manager")


def test_server_class_no_secret_or_decrypt_storage():
    assert not hasattr(AggregationServer, "decrypt_client_update")
    assert "secret_key" not in AggregationServer.__dict__
    assert "decryption_service" not in AggregationServer.__dict__
    assert "_decrypt_aggregate" not in AggregationServer.__dict__
