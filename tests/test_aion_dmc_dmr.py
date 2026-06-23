import numpy as np

from config import AionConfig
from defenses.aion.defense import AionDefense
from defenses.aion.mgf import MaskedGradientFilter


def test_extra_digits_uses_two_q():
    cfg = AionConfig()
    mgf = MaskedGradientFilter(cfg)
    assert mgf.extra_digits(5) == 1  # ceil(log10(2*5))
    assert mgf.extra_digits(6) == 2  # ceil(log10(12))


def test_dmc_scales_mask_and_dmr_rounds_after_unmask():
    cfg = AionConfig(decimal_places=4, hprf_key_dim=2)
    defense = AionDefense(cfg, {"ckks_s40": {}})
    update = np.array([1.234567, -2.345678])
    masked, md = defense.prepare_client_upload(0, update, 0, cfg.profile_id, 1, {"num_selected": 1})
    payload = md["aion_protocol_payload"]
    assert np.linalg.norm(masked - update, ord=np.inf) < 1e-3
    upload = type("U", (), {"client_id": 0, "metadata": md, "protocol_payload": payload})()
    unmasked, metrics = defense.unmask_aggregate_update(masked, [upload], 0, [1.0])
    np.testing.assert_allclose(unmasked, np.round(update, cfg.decimal_places), atol=10 ** -cfg.decimal_places)
    assert metrics["aion_dmr_enabled"] is True
