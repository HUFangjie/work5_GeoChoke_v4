import numpy as np
import torch

import models.mnist_cnn
from crypto.update_codec import ModelUpdateCodec
from models.registry import create_model


def test_codec_roundtrip_and_apply_update():
    model = create_model("mnist_cnn")
    codec = ModelUpdateCodec(model)
    flat = codec.flatten_state_dict(model.state_dict())
    update_state = codec.unflatten_to_update_state_dict(flat * 0.0)
    assert set(update_state) == set(codec.trainable_names)
    new_state = codec.apply_update_to_state_dict(model.state_dict(), np.zeros_like(flat))
    for name, tensor in model.state_dict().items():
        assert torch.allclose(new_state[name], tensor.cpu())
