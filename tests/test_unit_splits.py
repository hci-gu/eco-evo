"""Section 159: no split hands off less than one ``min_split_biomass`` unit."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env.decisions import merge_subunit_fractions
from test_gpu_ecosystem import DEVICES, _assert_parity, make_env


def stub_env(biomass, unit=0.05, unit_splits=True):
    """One DM, one cell per entry of ``biomass`` (t), 2 eat actions."""
    b = np.asarray(biomass, dtype=np.float32).reshape(1, 1, -1)
    return SimpleNamespace(
        unit_splits=unit_splits, dtype=np.float32,
        dm_min_split=np.array([unit], dtype=np.float32),
        dm_ids=["porp"], fgs={"porp": SimpleNamespace(biomass=b[0])})


def probs_for(*cells):
    """Each cell: (N, E, S, W, rest, eat1, eat2) -> (1, 7, 1, n)."""
    p = np.asarray(cells, dtype=np.float32).T
    return p.reshape(1, p.shape[0], 1, p.shape[1])


def spatial(p):
    return np.concatenate([p[0, :4, 0], p[0, 4:, 0].sum(axis=0, keepdims=True)])


def test_subunit_fractions_go_to_the_unit_fractions():
    # 200 kg: N 60 kg and stay 120 kg hold an animal, E 20 kg does not.
    env = stub_env([0.2])
    p = probs_for((0.3, 0.1, 0.0, 0.0, 0.2, 0.3, 0.1))
    out = merge_subunit_fractions(env, p)
    np.testing.assert_allclose(spatial(out)[:, 0], [1 / 3, 0, 0, 0, 2 / 3], rtol=1e-6)
    # eat/rest mix inside stay kept: 0.2 : 0.3 : 0.1
    np.testing.assert_allclose(out[0, 4:, 0, 0] / out[0, 4:, 0, 0].sum(),
                               [1 / 3, 1 / 2, 1 / 6], rtol=1e-6)


def test_no_unit_fraction_collapses_to_the_largest():
    # 70 kg: N 21, E 7, stay 42 kg - none holds an animal, stay is largest.
    env = stub_env([0.07, 0.07])
    p = probs_for((0.3, 0.1, 0.0, 0.0, 0.2, 0.4, 0.0),
                  (0.6, 0.0, 0.0, 0.0, 0.1, 0.2, 0.1))
    out = merge_subunit_fractions(env, p)
    np.testing.assert_allclose(spatial(out)[:, 0], [0, 0, 0, 0, 1], atol=1e-7)
    np.testing.assert_allclose(out[0, 4:, 0, 0], [1 / 3, 2 / 3, 0], rtol=1e-6)
    # Second cell: N 42 kg is the largest fraction, the cell moves whole.
    np.testing.assert_allclose(spatial(out)[:, 1], [1, 0, 0, 0, 0], atol=1e-7)


def test_sub_unit_cells_and_the_opt_out_are_untouched():
    p = probs_for((0.3, 0.1, 0.0, 0.0, 0.2, 0.3, 0.1),
                  (0.3, 0.1, 0.0, 0.0, 0.2, 0.3, 0.1))
    out = merge_subunit_fractions(stub_env([0.03, 0.0]), p)
    np.testing.assert_array_equal(out, p)
    one = p[..., :1]
    for env in (stub_env([0.2], unit_splits=False), stub_env([0.2], unit=0.0)):
        np.testing.assert_array_equal(merge_subunit_fractions(env, one), one)


def test_random_distributions_keep_every_fraction_at_a_unit():
    rng = np.random.default_rng(5)
    n = 4000
    biomass = rng.uniform(0.05, 0.4, n)
    raw = rng.dirichlet(np.full(7, 0.6), n)
    env = stub_env(biomass)
    out = merge_subunit_fractions(env, probs_for(*raw))
    frac = spatial(out)                                   # (5, n)
    np.testing.assert_allclose(out[0, :, 0].sum(axis=0), 1.0, atol=2e-6)
    mass = frac * biomass[None].astype(np.float32)
    assert np.all((frac == 0) | (mass >= 0.05 * (1 - 1e-6)))
    # A fraction that already held a unit is never dropped.
    before = spatial(probs_for(*raw)) * biomass[None]
    assert np.all(frac[before >= 0.05] > 0)


def test_env_default_is_on():
    env = make_env()
    assert env.unit_splits is True


@pytest.mark.parametrize("migration", [False, True])
def test_unit_splits_match_reference(device, migration):
    """CPU and tensor engines apply the same merge.

    min_split 1000 kg (1 t) against biomass 0.01-3 t puts cells on both
    sides of one unit, with a 10 kg sweep so splitting cells survive.
    """
    env = make_env(migration=migration, mortality=True,
                   min_split=1000.0, extinction_factor=0.01)
    b = np.stack([env.fgs[f].biomass for f in env.dm_ids])
    assert np.any(b >= 1.0) and np.any((b > 0) & (b < 1.0))
    model = _assert_parity(env, device, ticks=8)
    assert model.unit_splits


@pytest.mark.parametrize("migration", [False, True])
def test_opt_out_matches_reference(device, migration):
    env = make_env(migration=migration, mortality=True,
                   min_split=1000.0, extinction_factor=0.01)
    env.unit_splits = False
    model = _assert_parity(env, device, ticks=4)
    assert not model.unit_splits


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def test_tensor_merge_matches_numpy():
    rng = np.random.default_rng(9)
    n = 500
    biomass = rng.uniform(0.0, 0.3, n).astype(np.float32)
    raw = rng.dirichlet(np.full(7, 0.6), n).astype(np.float32)
    expected = merge_subunit_fractions(stub_env(biomass), probs_for(*raw))
    model = SimpleNamespace(min_split=torch.tensor([[[0.05]]]),
                            fraction_indices=torch.arange(5)[None, None, :, None])
    from lib.gpu.ecosystem import TensorEcosystem
    probs = torch.from_numpy(probs_for(*raw)[:, :, 0])            # [1, A, n] -> E=1
    probs = probs[None]                                           # [E=1, D=1, A, n]
    out = TensorEcosystem.merge_subunit_fractions(
        model, probs, torch.from_numpy(biomass)[None, None])
    np.testing.assert_allclose(out[0].numpy(), expected[:, :, 0], atol=1e-6)
