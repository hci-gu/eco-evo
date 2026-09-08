import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.gpu.config import EnvironmentBuilder, ProjectSpec
from lib.gpu.ecosystem import TensorEcosystem
from lib.gpu.random import fold_in, uniform, normal
from lib.gpu.spawn import WorldSpawner, distribute_with_floor, legacy_clusters
from lib.spawn.allocator import distribute_with_floor as reference_allocate


@pytest.mark.parametrize("floor", [0, 0.001, 1, 10, 1000])
def test_allocator_matches_greedy_reference(floor):
    rng = np.random.default_rng(123)
    weights = rng.random((20, 35))
    weights[weights < 0.3] = 0
    weights[0] = 0
    weights[1] = 1  # ties
    totals = rng.random(20) * 100
    totals[2] = 0
    actual = distribute_with_floor(torch.tensor(weights), torch.tensor(totals), floor)
    expected = np.stack([reference_allocate(w.reshape(5, 7), t, floor).ravel() for w, t in zip(weights, totals)])
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("floor,cells", [(0, 100), (0.1, 3600), (5, 30), (1000, 30), (1, 1)])
def test_legacy_stick_breaking_conserves_biomass(floor, cells):
    keys = torch.arange(20)
    totals = torch.arange(20, dtype=torch.float64) * 3.1
    result = legacy_clusters(keys, totals, floor, cells)
    torch.testing.assert_close(result.sum(-1), totals)
    assert torch.isfinite(result).all() and (result >= 0).all()
    if floor > 0:
        assert ((result == 0) | (result >= torch.minimum(totals[:, None], torch.tensor(float(floor))) - 1e-8)).all()


def test_random_fields_are_chunk_independent():
    keys = fold_in(torch.arange(6), 17)
    for fn in (uniform, normal):
        full = fn(keys, (3, 9), 25)
        chunks = torch.cat([fn(keys[:2], (3, 9), 25), fn(keys[2:], (3, 9), 25)])
        torch.testing.assert_close(full, chunks, rtol=0, atol=0)
        assert not torch.equal(full, fn(keys, (3, 9), 26))
    draws = uniform(torch.arange(64), (1000,), 5)
    assert 0.49 < draws.mean() < 0.51


def test_project_spawn_and_reserves():
    spec = ProjectSpec(EnvironmentBuilder(project_path="mareld2.yaml", grid=(8, 9)))
    model = TensorEcosystem(spec.env, "cpu")
    spawner = WorldSpawner(spec, model)
    keys = torch.arange(3)
    biomass = spawner.biomass(keys)
    assert biomass.shape == (3, 6, 72)
    assert torch.isfinite(biomass).all() and (biomass >= 0).all()
    for j, fid in enumerate(model.ids):
        total = biomass[:, j].double().sum(-1)
        lo, hi = spec.ranges[fid]
        assert (total >= lo - 1e-4).all() and (total <= hi + 1e-4).all()
    torch.testing.assert_close(biomass, spawner.biomass(keys), rtol=0, atol=0)
    assert not torch.equal(biomass[0], biomass[1])
    reserves = spawner.reserves(biomass, keys)
    assert (reserves >= 0).all() and (reserves <= biomass * model.max_reserve + 1e-5).all()
