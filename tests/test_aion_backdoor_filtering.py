import numpy as np

from config import AionConfig
from defenses.aion.mgf import MaskedGradientFilter


class Upload:
    def __init__(self, cid, norm, malicious=False):
        self.client_id = cid
        self.metadata = {"is_malicious": malicious, "aion_masked_update_norm": 0.0}
        self.protocol_payload = {"aion_masked_update_vector": np.array([norm], dtype=np.float64)}


def test_boosted_backdoor_like_updates_are_filtered():
    cfg = AionConfig(initial_bound_quantile=0.8, initial_bound_multiplier=1.0)
    mgf = MaskedGradientFilter(cfg)
    uploads = [Upload(i, 1.0 + 0.02 * i, False) for i in range(5)] + [Upload(10, 20.0, True), Upload(11, 25.0, True)]
    result = mgf.filter(uploads, round_id=0, min_clients_per_round=1)
    assert result.metrics["aion_filtered_malicious_count"] > 0
    assert any(cid in result.metrics["aion_filtered_client_ids"] for cid in [10, 11])
    assert not set(result.metrics["aion_filtered_client_ids"]).isdisjoint({10, 11})
