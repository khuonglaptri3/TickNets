"""A packaging-only SE refactor must preserve checkpoint and gradient behavior."""
import importlib
import random

import numpy as np
import pytest
import torch

from models import ticknet_l
from models.SE_Attention import SE as OriginalSE


@pytest.mark.parametrize("channels,size", [(24, 32), (144, 16), (512, 4)])
def test_inline_se_preserves_state_forward_and_gradients(channels, size):
    assert ticknet_l.SE.__module__ == "models.ticknet_l"
    old, new = OriginalSE(channels), ticknet_l.SE(channels)
    new.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(2, channels, size, size, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a, b = old(x), new(y)
    assert torch.equal(a, b)
    a.sum().backward()
    b.sum().backward()
    assert torch.equal(x.grad, y.grad)
    for (_, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
        assert torch.equal(p.grad, q.grad)


def test_seed_helpers_are_independent_and_repeat_all_rng_streams():
    assert importlib.util.find_spec("models.reproducibility") is not None
    seeds = importlib.import_module("models.reproducibility")
    seeds.seed_everything(42)
    expected = (random.random(), np.random.random(), torch.rand(4))
    seeds.seed_everything(42)
    actual = (random.random(), np.random.random(), torch.rand(4))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    torch.manual_seed(99)
    seeds.seed_worker(0)
    expected_worker = (random.random(), np.random.random())
    seeds.seed_worker(7)
    assert (random.random(), np.random.random()) == expected_worker
    assert torch.are_deterministic_algorithms_enabled()
