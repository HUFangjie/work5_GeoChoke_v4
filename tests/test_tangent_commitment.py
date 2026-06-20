import numpy as np
import pytest
import torch
from torch import nn

from config import ExperimentConfig, GeoChokeConfig
from core.server import AggregationServer
from core.types import ClientUpload
from crypto.ciphertext_payload import CiphertextChunk, EncryptedUpdate
from crypto.update_codec import ModelUpdateCodec
from defenses.geochoke.defense import GeoChokeDefense
from defenses.geochoke.tangent_commitment import TangentCommitter


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.linear(x.view(x.shape[0], -1))


def _manual_committer(basis):
    model = TinyModel()
    codec = ModelUpdateCodec(model)
    cfg = GeoChokeConfig(tangent_basis_rank=1)
    committer = TangentCommitter(codec, [], cfg, "cpu")
    committer._basis = torch.as_tensor(basis, dtype=torch.float64)
    committer._basis_round = 0
    committer.build_basis = lambda model, round_id: committer._basis
    return committer, model


def test_disabled_tangent_commitment_returns_decrypted_update_unchanged():
    cfg = GeoChokeConfig(tangent_commitment_enabled=False)
    defense = GeoChokeDefense(cfg, {"ckks_s40": {}, "ckks_s28": {}})
    update = np.array([1.0, -2.0, 3.0], dtype=np.float64)

    committed, metrics = defense.commit_update(update, None, 0, 0.2)

    assert committed is update
    np.testing.assert_array_equal(committed, update)
    assert metrics["tangent_commitment_enabled"] is False
    assert metrics["tangent_rho"] == 1.0


def test_update_inside_tangent_basis_is_committed_unchanged_with_rho_one():
    basis = np.array([[1.0], [0.0], [0.0], [0.0]])
    committer, model = _manual_committer(basis)
    update = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64)

    committed, metrics = committer.commit(update, model, 0, tau=0.2)

    np.testing.assert_allclose(committed, update, atol=1e-12)
    assert metrics["tangent_rho"] == 1.0
    assert metrics["tangent_update_shrink_ratio"] == pytest.approx(1.0, abs=1e-12)


def test_orthogonal_heavy_update_is_shrunk():
    basis = np.array([[1.0], [0.0], [0.0], [0.0]])
    committer, model = _manual_committer(basis)
    update = np.array([1.0, 10.0, 0.0, 0.0], dtype=np.float64)

    committed, metrics = committer.commit(update, model, 0, tau=0.2)

    assert metrics["tangent_rho"] < 1.0
    assert metrics["tangent_update_shrink_ratio"] < 1.0
    assert np.linalg.norm(committed) < np.linalg.norm(update)


class FakeBackend:
    def multiply_plain(self, encrypted_update, weight):
        return encrypted_update

    def add_ciphertexts(self, ciphertexts):
        return list(ciphertexts)[0]

    def serialize(self, aggregate_ciphertext):
        return {"chunks": [{"payload": b"x"}]}


class HalvingDefense:
    def after_aggregate(self, previous_model, candidate_model, current_profile_id, round_id):
        return {"tangent_tau": 0.2, "candidate_weight_sum": float(candidate_model.linear.weight.sum())}

    def commit_update(self, update_vector, previous_model, round_id, tangent_tau):
        committed = 0.5 * update_vector
        return committed, {"tangent_rho": 0.5, "tangent_commitment_enabled": True}


def test_server_apply_round_loads_committed_update_not_candidate_state():
    cfg = ExperimentConfig(num_clients=1, clients_per_round=1, min_clients_per_round=1, server_lr=1.0)
    model = TinyModel()
    with torch.no_grad():
        model.linear.weight.zero_()
    codec = ModelUpdateCodec(model)
    server = AggregationServer(cfg, model, codec, FakeBackend(), HalvingDefense(), TinyModel)
    payload = EncryptedUpdate([CiphertextChunk(0, 4, "ckks_s40", 4, b"x")], "ckks_s40", 4)
    upload = ClientUpload(0, 1, "ckks_s40", payload)
    decrypted = np.ones(codec.total_dimension, dtype=np.float64)

    _, metrics = server.apply_round([upload], 0, lambda aggregate, profile: decrypted)

    final_vector = codec.flatten_state_dict(server.model.state_dict())
    np.testing.assert_allclose(final_vector, 0.5 * decrypted)
    assert metrics["aggregate_update_norm"] == np.linalg.norm(decrypted)
    assert metrics["committed_aggregate_update_norm"] == np.linalg.norm(0.5 * decrypted)
