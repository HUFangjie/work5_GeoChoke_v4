from core.decryption_service import AuthorizedDecryptionService
from core.server import AggregationServer

def test_decryption_service_api_boundary():
    assert hasattr(AuthorizedDecryptionService,'decrypt_aggregate')
    assert not hasattr(AuthorizedDecryptionService,'decrypt_client_update')
    assert not hasattr(AuthorizedDecryptionService,'decrypt_ciphertext_list')

def test_server_class_no_secret_api():
    assert not hasattr(AggregationServer,'decrypt_client_update')
    assert 'secret_key' not in AggregationServer.__dict__
