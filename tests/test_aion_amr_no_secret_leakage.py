import numpy as np

from config import AionConfig
from defenses.aion.defense import AionDefense


def test_amr_mask_matches_direct_without_secret_flag():
    cfg = AionConfig(hprf_key_dim=4, reconstruction_mode="amr")
    defense = AionDefense(cfg, {"ckks_s40": {}})
    ids = [0, 1, 2]
    dim = 5
    direct = np.mean(np.stack([defense.protocol.client_raw_mask(cid, 1, dim) for cid in ids]), axis=0)
    mask, metrics = defense.protocol.reconstruct_aggregated_mask(ids, 1, dim, mean=True, mode="amr")
    np.testing.assert_allclose(mask, direct, atol=1e-8)
    assert metrics["aion_secret_recovered"] is False


def test_asr_and_amr_same_mask_but_asr_marks_secret_recovered():
    cfg = AionConfig(hprf_key_dim=4, reconstruction_mode="amr", strict_protocol_checks=False)
    defense = AionDefense(cfg, {"ckks_s40": {}})
    ids = [0, 1, 2]
    amr, amr_metrics = defense.protocol.reconstruct_aggregated_mask(ids, 2, 5, mean=True, mode="amr")
    asr, asr_metrics = defense.protocol.reconstruct_aggregated_mask(ids, 2, 5, mean=True, mode="asr")
    np.testing.assert_allclose(amr, asr, atol=1e-8)
    assert amr_metrics["aion_secret_recovered"] is False
    assert asr_metrics["aion_secret_recovered"] is True
