from config import AionConfig
from defenses.aion.mgf import MaskedGradientFilter


def test_paper_bound_can_decrease_and_mu_clips():
    cfg = AionConfig(mu_min=0.5, mu_max=1.2, warmup_rounds_for_bound=0)
    mgf = MaskedGradientFilter(cfg)
    mgf.previous_bound = 10.0
    mgf.note_round_result(global_l2_norm=10.0, alpha=1.0, aggregated_mask_linf=0.0)
    mgf.note_round_result(global_l2_norm=4.0, alpha=1.0, aggregated_mask_linf=0.0)
    bound, mu_raw, mu_clipped, numerator, denominator = mgf._compute_bound(__import__("numpy").array([1.0, 2.0]), round_id=2)
    assert mu_raw < 1.0
    assert mu_clipped == 0.5
    assert bound == 5.0


def test_mu_max_prevents_unbounded_growth():
    cfg = AionConfig(mu_min=0.5, mu_max=1.2, warmup_rounds_for_bound=0)
    mgf = MaskedGradientFilter(cfg)
    mgf.previous_bound = 10.0
    mgf.note_round_result(global_l2_norm=1.0, alpha=1.0, aggregated_mask_linf=0.0)
    mgf.note_round_result(global_l2_norm=100.0, alpha=1.0, aggregated_mask_linf=0.0)
    bound, mu_raw, mu_clipped, numerator, denominator = mgf._compute_bound(__import__("numpy").array([1.0, 2.0]), round_id=2)
    assert mu_raw > 1.2
    assert mu_clipped == 1.2
    assert bound == 12.0
