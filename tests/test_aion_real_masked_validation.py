import numpy as np

from config import AionConfig
from defenses.aion.mgf import MaskedGradientFilter


class Upload:
    def __init__(self, cid, vector, reported_norm):
        self.client_id = cid
        self.metadata = {"aion_masked_update_norm": reported_norm, "is_malicious": cid == 1}
        self.protocol_payload = {"aion_masked_update_vector": np.asarray(vector, dtype=np.float64)}


def test_mgf_uses_real_masked_vector_not_reported_norm():
    cfg = AionConfig(initial_bound_quantile=0.5, initial_bound_multiplier=1.0)
    mgf = MaskedGradientFilter(cfg)
    uploads = [Upload(0, [1.0], 1.0), Upload(1, [100.0], 0.01), Upload(2, [1.1], 1.1)]
    result = mgf.filter(uploads, round_id=0, min_clients_per_round=1)
    assert 1 in result.metrics["aion_filtered_client_ids"]
    assert result.metrics["aion_filtered_malicious_count"] == 1
