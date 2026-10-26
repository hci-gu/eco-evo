"""Depth habitat of the benthos (section 207).

Contracts asserted:

1. ``capacity_depth_table`` parses to sorted rows, refuses bad rows, and
   K(c) interpolates linearly in depth, held at the end rows outside them.
2. B*(c) = K(c) max(0, 1 - (F(c) + M2)/r): zero where the losses reach
   r, zeros for r <= 0; F(c) is the yearly loss of the group's impact
   tables at the baseline maps, at the library tick length.
3. maps/depth.npz is the 60x60 box, 40-540 m, finite.
4. The mareld2 load: the benthos carries K per cell from the table at the
   depth map and spawns in proportion to B*; the manifest's inference
   total is the 60x60 sum of B* (the two cannot drift apart). Without a
   depth map the scalar K stays and the pattern is the scalar-K B*.
5. The equilibrium strategy is uniform without a pattern, and an
   env_driven ref to ``depth`` reads the depth map on both engines.
6. GPU parity with a per-cell K, and the GPU spawner puts the benthos on
   the same pattern as the CPU loader.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config.config_loader import load_project_config
from lib.spawn import StrategySpec, make_weights
from lib.world import habitat
from test_gpu_ecosystem import DEVICES, _assert_parity, device, make_env  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
BENTHOS = "benthic_infauna"


def _load(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8-sig"))


def test_capacity_table_parses_and_interpolates():
    table = habitat.parse_capacity_table({"capacity_depth_table": [
        {"depth_m": 300, "capacity": 70}, {"depth_m": 100, "capacity": 140}]})
    np.testing.assert_array_equal(table[0], [100, 300])
    k = habitat.capacity_map(table, np.array([50.0, 100.0, 200.0, 300.0, 500.0]))
    np.testing.assert_allclose(k, [140, 140, 105, 70, 70])
    assert habitat.parse_capacity_table({}) is None
    for rows in ([{"depth_m": 1}], [{"depth_m": 1, "capacity": -1}],
                 [{"depth_m": 1, "capacity": 1}, {"depth_m": 1, "capacity": 2}]):
        with pytest.raises(ValueError):
            habitat.parse_capacity_table({"capacity_depth_table": rows})


def test_equilibrium_formula():
    k = np.array([100.0, 100.0, 50.0])
    loss = np.array([0.0, 0.5, 2.0])
    np.testing.assert_allclose(habitat.equilibrium_map(k, loss, 0.1, 1.0), [90, 40, 0])
    np.testing.assert_array_equal(habitat.equilibrium_map(k, loss, 0.0, 0.0), 0)


def test_impact_loss_is_yearly_at_the_library_tick():
    library = {"interaction_definitions": {f"{BENTHOS}_impacted_by_trawl": {
        "impact_affects": True, "impact_table": [
            {"value": 0, "biomass_factor": 0, "energy_factor": 0},
            {"value": 100, "biomass_factor": 1e-4, "energy_factor": 0}]}}}
    field = np.full((2, 3), 50.0)
    loss = habitat.annual_impact_loss(BENTHOS, library, {"trawl": field, "other": field}, (2, 3))
    np.testing.assert_allclose(loss, 1460 * 5e-5, rtol=1e-5)


def test_depth_map_file():
    depth = np.load(ROOT / "maps" / "depth.npz")["depth"]
    assert depth.shape == (60, 60)
    assert np.isfinite(depth).all() and 40 < depth.min() and depth.max() < 540
    assert 160 < depth.mean() < 175           # 199.2: mean 168 m


def _mareld2(grid=(60, 60), project=None, **kw):
    project = project if project is not None else _load("mareld2.yaml")
    groups, *_ = load_project_config(str(ROOT / "mareld2.yaml"), grid_size=grid, seed=1,
                                     mode="inference", project_config=project,
                                     library_path=str(ROOT / "fgconfig" / "fg_library.yaml"), **kw)
    return groups


def test_mareld2_benthos_carries_the_depth_habitat():
    groups = _mareld2()
    fg = groups[BENTHOS]
    library = _load("fgconfig/fg_library.yaml")
    depth = np.load(ROOT / "maps" / "depth.npz")["depth"].astype(np.float64)
    table = habitat.parse_capacity_table(library["species_definitions"][BENTHOS])
    np.testing.assert_allclose(fg.capacity_map, habitat.capacity_map(table, depth), rtol=1e-6)
    eq = fg.equilibrium_map
    assert eq is not None and (eq > 0).all() and (eq <= fg.capacity_map + 1e-6).all()
    # the trawled cells start lower than their K (section 221: kill 0.136
    # /yr at mean effort, up to ~0.43 /yr in the most fished cells)
    assert (eq / fg.capacity_map).min() < 0.6 < (eq / fg.capacity_map).max()
    np.testing.assert_allclose(fg.biomass / fg.biomass.sum(), eq / eq.sum(), rtol=1e-5, atol=1e-12)
    # the other groups keep a scalar K, but for the mobile epifauna
    # (section 221) and the Norway lobster (223.15), which have their own
    # depth tables
    benthic = (BENTHOS, "mobile_epifauna", "norway_lobster")
    assert all(groups[f].capacity_map is not None for f in benthic)
    assert all(g.capacity_map is None for f, g in groups.items() if f not in benthic)


@pytest.mark.parametrize("fid", [BENTHOS, "mobile_epifauna", "norway_lobster"])
def test_manifest_total_is_the_equilibrium_sum(fid):
    groups = _mareld2()
    project = _load("mareld2.yaml")
    entry = next(g for g in project["non_decision_makers"] if g["group_id"] == fid)
    total = float(groups[fid].equilibrium_map.sum())
    assert entry["inference_initial_biomass"] == pytest.approx(total, rel=0.02)
    assert entry["initial_biomass_min"] < total < entry["initial_biomass_max"]


def test_without_a_depth_map_the_scalar_k_stays():
    project = _load("mareld2.yaml")
    del project["simulation_settings"]["depth_map"]
    fg = _mareld2(project=project)[BENTHOS]
    assert fg.capacity_map is None
    k = float(fg.params["max_carrying_capacity"])
    assert fg.equilibrium_map.max() <= k + 1e-9 and fg.equilibrium_map.min() < k


def test_equilibrium_strategy_and_depth_refs():
    spec = StrategySpec(mode="equilibrium")
    w = make_weights(spec, (3, 4), project_seed=0)
    np.testing.assert_allclose(w, 1 / 12)
    pattern = np.arange(12, dtype=float).reshape(3, 4)
    w = make_weights(spec, (3, 4), project_seed=0, context={"equilibrium": pattern})
    np.testing.assert_allclose(w, pattern / pattern.sum())
    depth = np.linspace(50, 300, 12).reshape(3, 4)
    ref = StrategySpec(mode="env_driven", params={"refs": [{"name": "depth", "weight": 1.0}]})
    w = make_weights(ref, (3, 4), project_seed=0, context={"env_fields": {"depth": depth}})
    np.testing.assert_allclose(w, depth / depth.sum())


@pytest.mark.parametrize("seed", [0, 3])
def test_per_cell_capacity_matches_reference(device, seed):
    env = make_env(min_split=0.0, extinction_factor=0.0, mortality=True)
    rng = np.random.default_rng(seed)
    cap = rng.uniform(0.5, 3.0, env.fgs["c"].biomass.shape).astype(np.float32)
    env.fgs["c"].capacity_map = cap
    env.fgs["c"].biomass = (cap * rng.uniform(0.3, 1.2, cap.shape)).astype(np.float32)
    model = _assert_parity(env, device, ticks=8)
    assert model.capacity.shape[-1] == model.C
    assert (env.fgs["c"].biomass <= cap + 1e-6).all()


def test_gpu_spawner_uses_the_equilibrium_pattern():
    from lib.gpu.config import EnvironmentBuilder, ProjectSpec
    from lib.gpu.ecosystem import TensorEcosystem
    from lib.gpu.random import fold_in
    from lib.gpu.spawn import WorldSpawner
    spec = ProjectSpec(EnvironmentBuilder(project_path=str(ROOT / "mareld2.yaml"), grid=(6, 5)))
    model = TensorEcosystem(spec.env, "cpu")
    pattern = spec.env.fgs[BENTHOS].equilibrium_map.reshape(-1)
    keys = fold_in(torch.zeros(3, dtype=torch.int64), torch.arange(3))
    b = spawner_b = WorldSpawner(spec, model).biomass(keys)[:, model.ids.index(BENTHOS)].double()
    np.testing.assert_allclose((b / b.sum(-1, keepdim=True)).numpy(),
                               np.broadcast_to(pattern / pattern.sum(), (3, pattern.size)),
                               rtol=1e-5)
    assert spawner_b.sum(-1).min() > 0
    np.testing.assert_allclose(model.capacity[0, model.ids.index(BENTHOS)].numpy(),
                               spec.env.fgs[BENTHOS].capacity_map.reshape(-1), rtol=1e-6)
    # env_driven 'depth' reads the depth map on the tensor engine too
    weights = WorldSpawner(spec, model).weights(
        {"mode": "env_driven", "refs": [{"name": "depth", "weight": 1.0}]}, keys, {})
    depth = spec.depth.reshape(-1)
    np.testing.assert_allclose(weights.numpy(), np.broadcast_to(depth / depth.sum(), weights.shape),
                               rtol=1e-6)
