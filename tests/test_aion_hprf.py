import numpy as np

from defenses.aion.hprf import KeyHomomorphicPRF


def test_hprf_homomorphic_deterministic_and_round_separated():
    hprf = KeyHomomorphicPRF(key_dim=3, hmax=1.0)
    keys = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    err = hprf.assert_homomorphic_property(keys, round_id=1, dimension=11)
    assert err < 1e-9
    assert hprf.mask(keys[0], 1, 11).shape == (11,)
    np.testing.assert_allclose(hprf.mask(keys[0], 1, 11), hprf.mask(keys[0], 1, 11))
    assert not np.allclose(hprf.mask(keys[0], 1, 11), hprf.mask(keys[0], 2, 11))
