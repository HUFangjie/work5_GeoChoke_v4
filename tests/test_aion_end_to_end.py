import numpy as np
import torch
from torch import nn

from config import ExperimentConfig
from core.client import Client
from core.server import AggregationServer
from core.types import LocalUpdateRecord
from crypto.ciphertext_payload import CiphertextChunk, EncryptedUpdate
from crypto.update_codec import ModelUpdateCodec
from defenses.aion.defense import AionDefense
from torch.utils.data import DataLoader, TensorDataset


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.linear(x.view(x.shape[0], -1))


class FakeCrypto:
    def __init__(self):
        self.last_encryption_time = 0.0

    def encrypt_update(self, update, profile_id):
        return EncryptedUpdate([CiphertextChunk(0, len(update), profile_id, len(update), np.asarray(update, dtype=np.float64))], profile_id, len(update))

    def multiply_plain(self, encrypted_update, weight):
        chunk = encrypted_update.chunks[0]
        return EncryptedUpdate([CiphertextChunk(0, chunk.valid_length, chunk.profile_id, chunk.total_dimension, chunk.payload * weight)], chunk.profile_id, chunk.total_dimension)

    def add_ciphertexts(self, ciphertexts):
        items = list(ciphertexts)
        payload = sum(item.chunks[0].payload for item in items)
        first = items[0].chunks[0]
        return EncryptedUpdate([CiphertextChunk(0, first.valid_length, first.profile_id, first.total_dimension, payload)], first.profile_id, first.total_dimension)

    def serialize(self, aggregate_ciphertext):
        return {"chunks": [{"payload": b"x"}]}


def test_aion_two_round_dummy_flow():
    cfg = ExperimentConfig(defense_name="aion", aggregation="mean", num_clients=2, clients_per_round=2, min_clients_per_round=1, malicious_client_ids=[])
    model = TinyModel()
    codec = ModelUpdateCodec(model)
    defense = AionDefense(cfg.aion, cfg.ckks_profiles)
    defense.initialize(model, FakeCrypto(), None)
    server = AggregationServer(cfg, model, codec, FakeCrypto(), defense, TinyModel)
    for round_id in range(2):
        uploads = []
        for cid in [0, 1]:
            update = np.ones(codec.total_dimension) * (cid + 1)
            masked, md = defense.prepare_client_upload(cid, update, round_id, cfg.aion.profile_id, 1, {})
            encrypted = FakeCrypto().encrypt_update(masked, cfg.aion.profile_id)
            payload = md.pop("aion_protocol_payload")
            uploads.append(type("Upload", (), {"client_id": cid, "num_samples": 1, "profile_id": cfg.aion.profile_id, "encrypted_update": encrypted, "metadata": {"train_loss": 0.0, "is_malicious": False, **md}, "protocol_payload": payload})())
        _, metrics = server.apply_round(uploads, round_id, lambda aggregate, profile: aggregate.chunks[0].payload)
        assert metrics["defense_enabled"] is True
        assert metrics["secure_aggregation_scheme"] == "aion_single_mask"
