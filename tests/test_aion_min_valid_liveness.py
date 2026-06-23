import numpy as np

from config import AionConfig
from defenses.aion.mgf import MaskedGradientFilter


class Upload:
    def __init__(self, cid, norm):
        self.client_id = cid
        self.metadata = {"is_malicious": False, "aion_masked_update_norm": 0.0}
        self.protocol_payload = {"aion_masked_update_vector": np.array([norm], dtype=np.float64)}


def test_mgf_promotes_smallest_finite_updates_to_min_valid_instead_of_zero_valid_crash():
    cfg = AionConfig(initial_bound_quantile=0.0, initial_bound_multiplier=0.1)
    mgf = MaskedGradientFilter(cfg)
    uploads = [Upload(0, 10.0), Upload(1, 11.0), Upload(2, 1000.0)]
    result = mgf.filter(uploads, round_id=0, min_clients_per_round=2)
    assert result.metrics["aion_valid_client_count"] == 2
    assert result.metrics["aion_valid_client_ids"] == [0, 1]
    assert result.metrics["aion_min_valid_promoted_count"] >= 1
