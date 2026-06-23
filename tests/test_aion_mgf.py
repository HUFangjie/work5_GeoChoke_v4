import numpy as np

from config import AionConfig
from defenses.aion.defense import AionDefense
from defenses.aion.mgf import MaskedGradientFilter


class Upload:
    def __init__(self, client_id, norm, malicious=False):
        self.client_id = client_id
        self.num_samples = 1
        self.metadata = {"aion_masked_update_norm": 0.0, "is_malicious": malicious}
        self.protocol_payload = {"aion_masked_update_vector": np.array([norm], dtype=np.float64)}


def test_mgf_filters_obviously_amplified_masked_update():
    cfg = AionConfig(initial_bound_quantile=0.8)
    mgf = MaskedGradientFilter(cfg)
    uploads = [Upload(i, norm) for i, norm in enumerate([1.0, 1.1, 0.9, 1.2, 100.0])]
    result = mgf.filter(uploads, round_id=0)
    assert 4 in result.metrics["aion_filtered_client_ids"]
    assert result.metrics["aion_filtered_client_count"] == 1


def test_aion_amr_recovers_aggregate_mask():
    cfg = AionConfig(hprf_key_dim=4)
    defense = AionDefense(cfg, {"ckks_s40": {}})
    dim = 7
    ids = [0, 1, 2]
    masks = [defense.protocol.client_mask(cid, round_id=3, dimension=dim) for cid in ids]
    aggregate_mask, metrics = defense.protocol.reconstruct_aggregated_mask(ids, 3, dim, mean=True)
    np.testing.assert_allclose(aggregate_mask, np.mean(np.stack(masks), axis=0), atol=1e-8)
    assert metrics["aion_reconstruction_success"] is True
