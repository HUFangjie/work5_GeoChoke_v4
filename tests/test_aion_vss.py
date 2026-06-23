import pytest

from defenses.aion.vss import FeldmanVSS, Share


def test_vss_share_verify_and_reconstruct():
    vss = FeldmanVSS(aggregator_count=4, malicious_aggregator_count=1, threshold=2)
    sharing = vss.share_secret(12345)
    assert all(vss.verify_share(share, sharing.commitments) for share in sharing.shares)
    assert vss.reconstruct(sharing.shares[:2]) == 12345
    with pytest.raises(ValueError):
        vss.reconstruct(sharing.shares[:1])


def test_vss_tampered_share_fails_verification():
    vss = FeldmanVSS(aggregator_count=4, malicious_aggregator_count=1, threshold=2)
    sharing = vss.share_secret(12345)
    bad = Share(sharing.shares[0].x, sharing.shares[0].y + 1)
    assert not vss.verify_share(bad, sharing.commitments)


def test_aggregated_share_verification():
    vss = FeldmanVSS(aggregator_count=4, malicious_aggregator_count=1, threshold=2)
    a = vss.share_secret(10)
    b = vss.share_secret(20)
    shares = vss.aggregate_shares([a.shares, b.shares])
    commitments = vss.aggregate_commitments([a.commitments, b.commitments])
    assert all(vss.verify_share(share, commitments) for share in shares)
    assert vss.reconstruct(shares[:2]) == 30
