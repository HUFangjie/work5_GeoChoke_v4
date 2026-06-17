import models.mnist_cnn
from models.registry import create_model
from crypto.update_codec import ModelUpdateCodec
import numpy as np, torch

def test_codec_roundtrip():
    m=create_model('mnist_cnn'); c=ModelUpdateCodec(m); flat=c.flatten_state_dict(m.state_dict()); sd=c.unflatten_to_state_dict(flat,m.state_dict())
    for s in c.specs: assert torch.allclose(sd[s.name], m.state_dict()[s.name])
