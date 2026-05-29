"""Tester för opt-in uniform-bias-init i PolicyNetwork.

Verifierar att:
  - Default (uniform_bias_init=False) lämnar output-lagret med standard
    PyTorch-init (icke-noll bias, icke-skalade vikter).
  - När uniform_bias_init=True är output-bias=0, output-vikter nedskalade,
    och softmax är ~uniform även vid systematiskt biased input.
"""
import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.runners.policy import PolicyNetwork  # noqa: E402


@pytest.mark.parametrize("in_dim,out_dim", [(10, 7), (20, 11), (40, 15)])
def test_default_keeps_standard_init(in_dim, out_dim):
    """Utan uniform_bias_init=True ska output-lagret ha standard
    PyTorch-init: bias är inte exakt 0 och vikter inte nedskalade."""
    torch.manual_seed(0)
    p = PolicyNetwork(in_dim, out_dim)
    out = p.net[-1]
    # Standard Kaiming init på Linear ger weights med std ~ 1/sqrt(3*fan_in).
    # För hidden_dim=30 är det ~0.10–0.13. Säkert >0.02.
    assert out.weight.detach().numpy().std() > 0.02
    # Bias från standard init är uniform(-1/sqrt(fan_in), +1/sqrt(fan_in)),
    # alltså inte exakt 0.
    assert not np.allclose(out.bias.detach().numpy(), 0.0)


@pytest.mark.parametrize("in_dim,out_dim", [(10, 7), (20, 11), (40, 15)])
def test_opt_in_zero_bias(in_dim, out_dim):
    torch.manual_seed(0)
    p = PolicyNetwork(in_dim, out_dim, uniform_bias_init=True)
    assert np.allclose(p.net[-1].bias.detach().numpy(), 0.0)


@pytest.mark.parametrize("in_dim,out_dim", [(10, 7), (20, 11), (40, 15)])
def test_opt_in_weights_downscaled(in_dim, out_dim):
    torch.manual_seed(0)
    p = PolicyNetwork(in_dim, out_dim, uniform_bias_init=True)
    assert p.net[-1].weight.detach().numpy().std() < 0.02


@pytest.mark.parametrize("in_dim,out_dim", [(10, 7), (20, 11), (40, 15)])
def test_opt_in_softmax_uniform_on_random_input(in_dim, out_dim):
    torch.manual_seed(0)
    p = PolicyNetwork(in_dim, out_dim, uniform_bias_init=True)
    x = torch.randn(500, in_dim)
    with torch.no_grad():
        probs = p.forward(x).numpy()
    mean_probs = probs.mean(axis=0)
    expected = 1.0 / out_dim
    assert np.abs(mean_probs - expected).max() < 0.01 * expected + 1e-3


@pytest.mark.parametrize("in_dim,out_dim", [(10, 7), (20, 11), (40, 15)])
def test_opt_in_softmax_uniform_on_biased_input(in_dim, out_dim):
    torch.manual_seed(0)
    p = PolicyNetwork(in_dim, out_dim, uniform_bias_init=True)
    offsets = torch.linspace(-1.0, 1.0, in_dim)
    x = torch.randn(500, in_dim) * 0.1 + offsets
    with torch.no_grad():
        probs = p.forward(x).numpy()
    mean_probs = probs.mean(axis=0)
    expected = 1.0 / out_dim
    assert np.abs(mean_probs - expected).max() < 0.02


def test_opt_in_hidden_layers_keep_standard_init():
    torch.manual_seed(0)
    p = PolicyNetwork(20, 11, uniform_bias_init=True)
    for i in (0, 2):
        assert p.net[i].weight.detach().numpy().std() > 0.05
