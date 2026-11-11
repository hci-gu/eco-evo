"""Open-population fish groups (section 250; stages.exchange, the staged
open_pool.migrate, the GPU _stage_exchange / _migrate, growth_budget's
somatic_production).

Contracts asserted:

1. ``juvenile_stage.recruit_import`` (t/km2/yr, >= 0) and ``emigration``
   ([e_J, e_A], >= 0) are checked; with an import the group's own
   recruitment weight rho is 0 and the recruitment months shape the import
   (annual mean = I / ticks per year per cell).
2. One exchange step: J <- J exp(-e_J dt) + I_t, A <- A exp(-e_A dt), each
   cell's reserve pro rata with its biomass, the arrivals at fill x ME; the
   books and the tick's net exchange (the reward ledger) follow.
3. A staged DM migrates on its adults: an export leaves J untouched and
   scales A, an import enters A.
4. GPU parity: a staged, migrating DM with an import and a staged DM with
   emigration, on the daylight calendar.
5. ``somatic_production`` P: g = P / ((s* - u) ticks per year), refused with
   r_max or a feeding season, and the loader refuses a broken budget.
6. The reward ledger covers the open-population groups; mareld2 pins.
7. Section 270: ``recruit_depth_table`` (rows {depth_m, weight}) needs a
   ``recruit_import``; the loader's map (equal-cell mean 1) spreads the
   import over the cells, the booked import total is unchanged, the GPU
   mirrors it, and without a map the import stays uniform.
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.environments.ecosystem_env import open_pool, stages
from lib.runners.energy_ledger import migrating
from lib.world import daylight as daylight_module
from lib.world.functional_group import FunctionalGroup
from lib.world.growth_budget import TICKS_PER_YEAR, growth_budget
from test_gpu_ecosystem import _assert_parity, device, make_env  # noqa: F401

DAYLIGHT = {"latitude_deg": 58.15, "tick_hours": 6, "start_tick": 600, "random_start": False}
PER_YEAR = daylight_module.ticks_per_year(6)
RHO = [0, 0, 0, 0, 0, 1.0, 2.0, 2.0, 1.0, 0, 0, 0]


def block(rho=RHO, start=0.3, recruit=None, emigration=None):
    out = {"growth_ratio": 4.0, "graduation_rate": 1.0, "recruitment": list(rho),
           "start_share": start}
    if recruit is not None:
        out["recruit_import"] = recruit
    if emigration is not None:
        out["emigration"] = list(emigration)
    return out


def env_with(stages_, regen=None, daylight=True, **kw):
    kw.setdefault("min_split", 0.0)
    kw.setdefault("extinction_factor", 0.0)
    if daylight:
        kw.setdefault("daylight", dict(DAYLIGHT))
    env = make_env(stages=stages_, regen=regen, **kw)
    env.build_static_caches()
    return env


def test_settings_are_checked():
    good = {"is_decision_maker": True, "juvenile_stage": block(recruit=0.5, emigration=[1.0, 0.2])}
    fg = FunctionalGroup("a", good)
    spec = stages.settings(fg)
    assert spec["import"] == 0.5 and spec["emigration"] == [1.0, 0.2]
    for bad in (block(recruit=-1.0), block(recruit="x"), block(emigration=[1.0]),
                block(emigration=[1.0, -0.1]), block(emigration="x")):
        with pytest.raises(ValueError):
            stages.settings(FunctionalGroup("a", {"is_decision_maker": True,
                                                  "juvenile_stage": bad}))
    plain = stages.settings(FunctionalGroup("a", {"is_decision_maker": True,
                                                  "juvenile_stage": block()}))
    assert plain["import"] is None and plain["emigration"] is None


def test_tables_replace_the_own_recruitment():
    env = env_with({"a": block(recruit=2.0, emigration=[1.0, 0.5]), "b": block()})
    ja, jb = env.global_fg_order.index("a"), env.global_fg_order.index("b")
    assert env.stage_exchange_ids == ["a"]
    assert np.all(env.stage_rho[:, ja] == 0.0) and env.stage_rho[:, jb].max() > 0.0
    assert float(env.stage_import[:, ja].mean()) == pytest.approx(2.0 / PER_YEAR, rel=1e-3)
    assert env.stage_import[:, ja].max() > 0.0 and float(env.stage_import[:, jb].sum()) == 0.0
    assert env.stage_emig_keep[0, ja] == pytest.approx(math.exp(-1.0 / PER_YEAR), rel=1e-6)
    assert env.stage_emig_keep[1, ja] == pytest.approx(math.exp(-0.5 / PER_YEAR), rel=1e-6)
    assert np.allclose(env.stage_fill[:, ja], 0.7)         # no start_fill: 0.7


def test_one_exchange_step_is_exact_and_booked():
    env = env_with({"a": block(rho=[1.0] * 12, recruit=3.0, emigration=[2.0, 0.5])})
    a = env.fgs["a"]
    rng = np.random.default_rng(0)
    a.biomass = rng.uniform(1.0, 3.0, a.biomass.shape).astype(np.float32)
    a.juvenile = (a.biomass * rng.uniform(0.1, 0.6, a.biomass.shape)).astype(np.float32)
    a.energy_reserve = (a.biomass * rng.uniform(100.0, 400.0, a.biomass.shape)).astype(np.float32)
    b0, j0, r0 = (x.astype(np.float64).copy() for x in (a.biomass, a.juvenile, a.energy_reserve))
    env.loss_exchange, env.gain_exchange = {}, {}
    stages.exchange(env)
    kj, ka = math.exp(-2.0 / PER_YEAR), math.exp(-0.5 / PER_YEAR)
    per_cell = 3.0 / PER_YEAR
    new_j, new_b = j0 * kj, j0 * kj + (b0 - j0) * ka
    me, ec = float(a.max_energy_reserve), float(a.params["energy_content"])
    np.testing.assert_allclose(a.juvenile, new_j + per_cell, rtol=1e-5)
    np.testing.assert_allclose(a.biomass, new_b + per_cell, rtol=1e-5)
    np.testing.assert_allclose(a.energy_reserve, r0 * new_b / b0 + per_cell * 0.7 * me, rtol=1e-5)
    export, imported = float((b0 - new_b).sum()), per_cell * a.biomass.size
    leaving = float((r0 - r0 * new_b / b0).sum())
    assert env.loss_exchange["a"] == pytest.approx(export, rel=1e-4)
    assert env.gain_exchange["a"] == pytest.approx(imported, rel=1e-6)
    assert env.tick_exchange_biomass["a"] == pytest.approx(imported - export, rel=1e-4)
    assert env.tick_exchange_energy["a"] == pytest.approx(
        (imported - export) * ec + imported * 0.7 * me - leaving, rel=1e-4)


OPEN = {"outside_density": 2.0, "outside_exchange_rate": 0.2,
        "outside_fill": [0.6] * 12, "outside_entry": [{"edge": "west"}]}


def test_a_staged_dm_migrates_on_its_adults():
    env = env_with({"a": block()}, regen={"a": OPEN}, daylight=False)
    a = env.fgs["a"]
    a.biomass = np.full(a.biomass.shape, 5.0, dtype=np.float32)    # T = 2 x 20 = 40 adults
    a.juvenile = np.full(a.biomass.shape, 2.0, dtype=np.float32)    # adults 60 > 40: export
    a.energy_reserve = (a.biomass * np.float32(300.0)).astype(np.float32)
    open_pool.migrate(env)
    adults = 40.0 + (60.0 - 40.0) * math.exp(-0.2)
    np.testing.assert_allclose(a.juvenile, 2.0)
    assert float((a.biomass - a.juvenile).sum()) == pytest.approx(adults, rel=1e-5)
    np.testing.assert_allclose(a.energy_reserve / a.biomass, 300.0, rtol=1e-5)  # one fill
    a.biomass = np.full(a.biomass.shape, 2.5, dtype=np.float32)    # adults 10 < 40: import
    open_pool.migrate(env)
    np.testing.assert_allclose(a.juvenile, 2.0)
    net = (40.0 - 10.0) * (1 - math.exp(-0.2))
    assert float(a.biomass.sum()) == pytest.approx(50.0 + net, rel=1e-5)


@pytest.mark.parametrize("density", [2.0, 0.01])      # import / export
def test_gpu_matches_reference(device, density):
    env = env_with({"a": block(recruit=4.0, emigration=[1.5, 0.3]),
                    "b": block(emigration=[0.8, 0.4])},
                   regen={"a": dict(OPEN, outside_density=density)}, mortality=True,
                   handling=0.3)
    model = _assert_parity(env, device, ticks=8, base_tol=1e-5)
    assert model.stage_exchange_on and model.open_dm_on and model.ledger_on


SPEC = {"is_decision_maker": True, "maintenance_level": 0.1, "appetite": "reserve_room",
        "somatic_production": 0.6, "natural_mortality": 0.0002}


def test_somatic_production_sets_g():
    b = growth_budget("f", {"f": SPEC}, {})
    assert b.error is None and b.somatic == 0.6
    assert b.growth_rate == pytest.approx(0.6 / (b.gate * TICKS_PER_YEAR))
    assert "P 0.6" in b.describe()
    assert growth_budget("f", {"f": dict(SPEC, r_max=0.4)}, {}).error
    assert growth_budget("f", {"f": dict(SPEC, material_growth_share=0.4,
                                         feeding_season_months=8)}, {}).error
    assert "somatic_production" in growth_budget("f", {"f": dict(SPEC, somatic_production=-1)}, {}).error


def _library():
    return yaml.safe_load(open(ROOT / "fgconfig" / "fg_library.yaml"))


def test_loader_refuses_a_broken_open_population(tmp_path):
    from lib.config.config_loader import load_project_config
    lib = _library()
    lib["species_definitions"]["whiting"]["r_max"] = 0.49
    path = tmp_path / "lib.yaml"
    path.write_text(yaml.safe_dump(lib))
    with pytest.raises(ValueError, match="somatic_production replaces r_max"):
        load_project_config(str(ROOT / "mareld2.yaml"), mode="inference", spawn_seed=1,
                            grid_size=(6, 7), library_path=str(path))


def test_mareld2_open_population():
    sd = _library()["species_definitions"]
    expected = {"planktivorous_fish": (0.66, 0.527, [0.1, 0.0]),
                "benthivorous_fish": (0.74, 0.183, [1.1, 0.17]),
                "whiting": (0.61, 0.172, [0.2, 0.3])}
    for fid, (p, imp, em) in expected.items():
        assert sd[fid]["somatic_production"] == pytest.approx(p), fid
        assert "r_max" not in sd[fid] and "feeding_season_months" not in sd[fid], fid
        assert sd[fid]["juvenile_stage"]["recruit_import"] == pytest.approx(imp), fid
        assert sd[fid]["juvenile_stage"]["emigration"] == em, fid
    wc = sd["predatory_fish"]
    assert wc["somatic_production"] == pytest.approx(0.42) and "r_max" not in wc
    for fid in ("planktivorous_fish", "predatory_fish"):
        assert sum(sd[fid]["outside_season"]) / 12 == pytest.approx(1.0, abs=1e-3)
    from lib.config.config_loader import load_project_config
    fgs = load_project_config(str(ROOT / "mareld2.yaml"), mode="inference", spawn_seed=1,
                              grid_size=(6, 7))[0]
    for fid, (p, _, _) in expected.items():
        fg = fgs[fid]
        assert fg.growth_rate == pytest.approx(p / ((1.0 - fg.params["maintenance_level"])
                                                    * TICKS_PER_YEAR), rel=1e-9), fid


def test_ledger_covers_the_open_population_groups():
    class Env:
        open_dm_ids = ["m"]
        stage_exchange_ids = ["w"]
    assert migrating(Env(), ["m", "w", "x"]) == ["m", "w"]


ROWS = [{"depth_m": 99.9, "weight": 2.71}, {"depth_m": 100.0, "weight": 1.0},
        {"depth_m": 149.9, "weight": 1.0}, {"depth_m": 150.0, "weight": 0.21},
        {"depth_m": 249.9, "weight": 0.21}, {"depth_m": 250.0, "weight": 0.05}]


def test_recruit_depth_table_is_checked():
    def spec(b):
        return stages.settings(FunctionalGroup("a", {"is_decision_maker": True,
                                                     "juvenile_stage": b}))
    depths, weights = spec(dict(block(recruit=0.5), recruit_depth_table=ROWS))["depth_table"]
    assert list(depths) == [r["depth_m"] for r in ROWS]
    assert list(weights) == [r["weight"] for r in ROWS]
    assert spec(block(recruit=0.5))["depth_table"] is None
    for bad in (dict(block(), recruit_depth_table=ROWS),               # no import
                dict(block(recruit=0.5), recruit_depth_table=[{"depth_m": 1.0}]),
                dict(block(recruit=0.5), recruit_depth_table=[{"depth_m": 1.0, "weight": -1.0}])):
        with pytest.raises(ValueError):
            spec(bad)


def _shaped_env(weight, **kw):
    env = env_with({"a": block(rho=[1.0] * 12, recruit=3.0, emigration=[2.0, 0.5])}, **kw)
    if weight is not None:
        env.fgs["a"].recruit_weight_map = weight
        env.build_static_caches()
    return env


def test_the_import_follows_the_recruit_map_and_keeps_its_total():
    rng = np.random.default_rng(1)
    flat = _shaped_env(None)
    weight = rng.uniform(0.1, 3.0, flat.fgs["a"].biomass.shape)
    env = _shaped_env(weight)
    ja = env.global_fg_order.index("a")
    np.testing.assert_allclose(flat.stage_import_map[ja], 1.0)
    np.testing.assert_allclose(env.stage_import_map[ja], weight / weight.mean(), rtol=1e-6)
    for e in (flat, env):
        a = e.fgs["a"]
        a.biomass = np.full(a.biomass.shape, 2.0, dtype=np.float32)
        a.juvenile = np.full(a.biomass.shape, 0.5, dtype=np.float32)
        a.energy_reserve = (a.biomass * np.float32(300.0)).astype(np.float32)
        e.loss_exchange, e.gain_exchange = {}, {}
        stages.exchange(e)
    per_cell = 3.0 / PER_YEAR
    kj = math.exp(-2.0 / PER_YEAR)
    np.testing.assert_allclose(env.fgs["a"].juvenile, 0.5 * kj + per_cell * weight / weight.mean(),
                               rtol=1e-5)
    np.testing.assert_allclose(flat.fgs["a"].juvenile, 0.5 * kj + per_cell, rtol=1e-5)
    assert env.gain_exchange["a"] == pytest.approx(flat.gain_exchange["a"], rel=1e-6)
    assert env.tick_exchange_biomass["a"] == pytest.approx(flat.tick_exchange_biomass["a"], rel=1e-5)
    assert env.tick_exchange_energy["a"] == pytest.approx(flat.tick_exchange_energy["a"], rel=1e-5)


def test_a_wrong_sized_recruit_map_is_refused():
    with pytest.raises(ValueError):
        _shaped_env(np.ones((2, 2)))


def test_gpu_matches_reference_with_a_recruit_map(device):
    rng = np.random.default_rng(2)
    weight = rng.uniform(0.1, 3.0, (4, 5))
    env = env_with({"a": block(recruit=4.0, emigration=[1.5, 0.3]),
                    "b": block(emigration=[0.8, 0.4])}, mortality=True, handling=0.3)
    env.fgs["a"].recruit_weight_map = weight
    env.build_static_caches()
    model = _assert_parity(env, device, ticks=8, base_tol=1e-5)
    assert model.stage_exchange_on
    assert float(model.stage_import_map.min()) < 0.5 < 1.5 < float(model.stage_import_map.max())


def test_habitat_builds_the_recruit_weight():
    from lib.world import habitat
    lib = {"species_definitions": {
        "a": {"is_decision_maker": True,
              "juvenile_stage": dict(block(recruit=0.5), recruit_depth_table=ROWS)},
        "b": {"is_decision_maker": True, "juvenile_stage": block(recruit=0.5)}}}
    _depth, maps = habitat.build(None, None, lib, (3, 4), ["a", "b"])
    np.testing.assert_allclose(maps["a"]["recruit_weight"], 1.0)    # no depth map: uniform
    assert "b" not in maps
    depth = np.array([[50.0, 120.0], [200.0, 300.0]])
    table = stages.settings(FunctionalGroup("a", lib["species_definitions"]["a"]))["depth_table"]
    weight = open_pool.depth_weight_map(table, depth, depth.shape)
    assert weight.mean() == pytest.approx(1.0)
    raw = np.array([[2.71, 1.0], [0.21, 0.05]])
    np.testing.assert_allclose(weight, raw / raw.mean())


def test_mareld2_benthivores_import_on_shallow_beds():
    from lib.world import habitat
    project_path = ROOT / "mareld2.yaml"
    project = yaml.safe_load(open(project_path, encoding="utf-8-sig"))
    if habitat.depth_map_path(project, str(project_path)) is None or not Path(
            habitat.depth_map_path(project, str(project_path))).exists():
        pytest.skip("maps/ is untracked")
    library = yaml.safe_load(open(ROOT / "fgconfig" / "fg_library.yaml", encoding="utf-8-sig"))
    ids = [g["group_id"] for key in ("decision_makers", "non_decision_makers")
           for g in project[key] if not g.get("muted")]
    depth, maps = habitat.build(project, str(project_path), library, (60, 60), ids)
    weight = maps["benthivorous_fish"]["recruit_weight"]
    assert weight.mean() == pytest.approx(1.0)
    assert weight[depth < 99.9].mean() > 2.5 and weight[depth >= 250.0].mean() < 0.1
    assert "recruit_weight" not in maps.get("planktivorous_fish", {})
