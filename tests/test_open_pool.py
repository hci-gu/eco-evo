"""Open pool and per-cell pair depth multiplier (section 245;
lib/environments/ecosystem_env/open_pool.py, interactions.build_pair_depth_maps).

Contracts asserted:

1. The keys are checked: an open pool is a non decision maker with a
   positive outside density and exchange rate, 12 season factors; the
   other keys need ``outside_density``.
2. One tick is B <- T + (B (1 + g) - T) exp(-k) after M1, with no logistic
   and no clip at K; without predation the pool settles at the exact
   fixed point T (1 - x) / (1 - (1 + g) x), x = exp(-k).
3. The season table interpolates the 12 monthly factors on the daylight
   calendar and the target follows it; the depth shape is renormalised to
   an equal-cell mean of 1.
4. The stability refusal: g >= exp(k) - 1 is refused.
5. The growth budget of an open pool is derived with r_max 0:
   g = (M1 + sum M2) / 1460.
6. A pair's depth map multiplies a per cell only: a constant map c is a
   pair with a x c, and the visible prey is untouched.
7. GPU parity with an open pool, its season and a non-uniform pair map.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env import open_pool, population_change, predation
from lib.environments.ecosystem_env.state import ActionProbabilities
from lib.world import growth_budget, habitat
from lib.world.functional_group import FunctionalGroup
from test_gpu_ecosystem import DEVICES, _assert_parity, device, make_env  # noqa: F401

K_TICK, G_TICK, DENSITY = 0.2, 0.01, 2.0
OPEN = {"outside_density": DENSITY, "outside_exchange_rate": K_TICK, "growth_rate": G_TICK}
SEASON = [1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 1.1, 1.1, 1.1, 1.3, 1.3, 1.3]
DAYLIGHT = {"latitude_deg": 58.15, "tick_hours": 6, "start_tick": 0, "random_start": False}


def _env(keys=OPEN, weight=None, **kw):
    env = make_env(min_split=0.0, extinction_factor=0.0, refuge=keys, **kw)
    if weight is not None:
        env.fgs["c"].outside_weight_map = weight
    env.build_static_caches()
    return env


def test_keys_are_checked():
    ok = dict(outside_density=5.0, outside_exchange_rate=0.01)
    assert FunctionalGroup("x", ok).open_pool == {"density": 5.0, "rate": 0.01, "season": None}
    for bad in ({"outside_density": 0.0, "outside_exchange_rate": 0.01},
                {"outside_density": 5.0, "outside_exchange_rate": 0.0},
                {"outside_density": 5.0},
                {"outside_exchange_rate": 0.01},
                {"outside_season": SEASON},
                dict(ok, outside_season=[1.0] * 11),
                dict(ok, outside_season=[1.0] * 11 + [-1.0]),
                dict(ok, is_decision_maker=True),
                dict(ok, exposed_share=0.2)):
        with pytest.raises(ValueError):
            FunctionalGroup("x", bad)


def test_one_tick_and_no_clip_at_k():
    env = _env(dict(OPEN, outside_density=30.0))    # above the fixture's K 12
    c = env.fgs["c"]
    b0 = c.biomass.astype(np.float64).copy()
    population_change._apply_non_decision_maker_population_change(env, "c", c)
    expected = 30.0 + (b0 * (1 + G_TICK) - 30.0) * math.exp(-K_TICK)
    np.testing.assert_allclose(c.biomass, expected, rtol=1e-5)
    for _ in range(40):
        population_change._apply_non_decision_maker_population_change(env, "c", c)
    assert c.biomass.min() > 12.0


def test_fixed_point_without_predation():
    env = _env()
    c = env.fgs["c"]
    for _ in range(400):
        population_change._apply_non_decision_maker_population_change(env, "c", c)
    x = math.exp(-K_TICK)
    fixed = DENSITY * (1 - x) / (1 - (1 + G_TICK) * x)
    np.testing.assert_allclose(c.biomass, fixed, rtol=1e-4)


def test_depth_shape_and_season():
    depth = np.array([[50.0, 100.0], [150.0, 300.0]])
    table = open_pool.parse_depth_table(
        {"outside_depth_table": [{"depth_m": 100, "weight": 1.0},
                                 {"depth_m": 200, "weight": 3.0}]})
    w = open_pool.depth_weight_map(table, depth, depth.shape)
    assert w.mean() == pytest.approx(1.0)
    np.testing.assert_allclose(w / w[0, 1], [[1.0, 1.0], [2.0, 3.0]])
    assert np.all(open_pool.depth_weight_map(table, None, (2, 2)) == 1.0)
    season = open_pool.season_table(SEASON, 1460, 6)
    assert season.mean() == pytest.approx(np.mean(SEASON), rel=2e-3)
    assert season.min() >= 0.6 - 1e-9 and season.max() <= 1.3 + 1e-9
    weight = np.linspace(0.5, 1.5, 20).reshape(4, 5).astype(np.float32)
    env = _env(dict(OPEN, outside_season=SEASON), weight=weight, daylight=DAYLIGHT)
    from lib.environments.ecosystem_env import interactions
    expected = DENSITY * weight * env.open_season["c"][interactions.light_index(env)]
    np.testing.assert_allclose(open_pool.target(env, "c"), expected, rtol=1e-6)


def test_stability_refusal():
    assert open_pool.stability_error(0.009, 0.01) is None
    assert open_pool.stability_error(math.expm1(0.01), 0.01) is not None


def test_budget_is_derived_with_r_max_zero():
    species = {"k": {"outside_density": 5.0, "outside_exchange_rate": 0.0137,
                     "natural_mortality": 0.0},
               "f": {"is_decision_maker": True}}
    inter = {"f_preys_on_k": {"preys_on": True, "predation_mortality": 2.07}}
    assert growth_budget.has_budget(species["k"])
    g = growth_budget.derived_growth_rate("k", species, inter)
    assert g == pytest.approx(2.07 / 1460)


def test_habitat_gives_the_weight_and_the_start_pattern():
    lib = {"species_definitions": {"k": {"outside_density": 5.0, "outside_exchange_rate": 0.01,
                                         "spawn": {"mode": "equilibrium"}}},
           "interaction_definitions": {}}
    _, maps = habitat.build(None, None, lib, (3, 4), ["k"])
    np.testing.assert_allclose(maps["k"]["outside_weight"], 1.0)
    np.testing.assert_allclose(maps["k"]["equilibrium"], 5.0)
    depth = np.array([[50.0, 300.0]])
    m = habitat.pair_depth_map({"depth_access_table": [{"depth_m": 99, "factor": 0.16},
                                                       {"depth_m": 100, "factor": 0.0}]},
                               depth)
    assert m.mean() == pytest.approx(1.0) and m[0, 1] == 0.0
    assert habitat.pair_depth_map({}, depth) is None


def _eat_c(env):
    n = env.N_all
    eat = np.zeros((env.N_dm, n) + env.fgs["c"].biomass.shape, dtype=np.float32)
    eat[env.dm_ids.index("a"), env.global_fg_order.index("c")] = 0.5
    zeros = np.zeros((env.N_dm,) + env.fgs["c"].biomass.shape, dtype=np.float32)
    move = np.zeros((env.N_dm, 4) + env.fgs["c"].biomass.shape, dtype=np.float32)
    return ActionProbabilities(move, zeros, eat)


def test_constant_pair_map_is_a_scaled_attack_rate():
    shape = (4, 5)
    scaled = _env(keys=None, handling=2.0,
                  pair_keys={"a_preys_on_c": {"max_intake_rate": 1.6}})
    mapped = _env(keys=None, handling=2.0,
                  pair_keys={"a_preys_on_c": {"max_intake_rate": 0.8,
                                              "_depth_access_map": np.full(shape, 2.0)}})
    assert mapped._has_pair_depth and not scaled._has_pair_depth
    for env in (scaled, mapped):
        predation.apply_predation(env, _eat_c(env))
    np.testing.assert_allclose(mapped.fgs["a"].temp_energy_gains,
                               scaled.fgs["a"].temp_energy_gains, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(mapped.fgs["c"].biomass, scaled.fgs["c"].biomass,
                               rtol=1e-5, atol=1e-9)


def test_open_pool_and_pair_map_match_reference(device):
    field = np.linspace(0.2, 1.8, 20).reshape(4, 5)
    env = make_env(min_split=0.0, extinction_factor=0.0, mortality=True,
                   refuge=dict(OPEN, outside_season=SEASON), daylight=DAYLIGHT,
                   pair_keys={"a_preys_on_c": {"_depth_access_map": field / field.mean()}})
    env.fgs["c"].outside_weight_map = np.linspace(0.5, 1.5, 20).reshape(4, 5)
    env.build_static_caches()
    model = _assert_parity(env, device, ticks=8)
    assert model.open_on and model.open_season is not None and model.pair_depth is not None


def test_mareld2_krill():
    """Section 245 in the library: the open krill pool, its five pairs, the
    zooplankton without krill and whiting's re-booked ration."""
    import yaml
    from lib.config.config_loader import load_config, load_project_config
    root = Path(__file__).resolve().parents[1]
    lib = load_config(str(root / "fgconfig" / "fg_library.yaml"))
    spec, inter = lib["species_definitions"]["krill"], lib["interaction_definitions"]
    assert spec["outside_density"] == 5.0
    assert spec["outside_exchange_rate"] == pytest.approx(20.0 / 1460, rel=1e-6)
    assert np.mean(spec["outside_season"]) == pytest.approx(1.0)
    assert float(spec["r_max"]) == 0.0 and float(spec["natural_mortality"]) == 0.0
    preds = ("planktivorous_fish", "benthivorous_fish", "predatory_fish", "mackerel", "whiting")
    m2 = {p: float(inter[f"{p}_preys_on_krill"]["predation_mortality"]) for p in preds}
    # The booked fish predation equals the field P/B 2 (1.2-3.5) within 4 %.
    assert sum(m2.values()) == pytest.approx(2.0, rel=0.04)
    for p in preds:
        assert inter[f"{p}_preys_on_krill"]["depth_access_table"]
        assert inter[f"{p}_observes_krill"]["observes"] is True
    names = [m["name"] for m in lib["species_definitions"]["zooplankton"]["member_species"]]
    assert "krill" not in names
    # Whiting's booked ration at the manifest start: 2.63 -> 3.77 t/t/yr.
    with open(root / "mareld2.yaml", encoding="utf-8-sig") as fh:
        manifest = yaml.safe_load(fh)
    start = {e["group_id"]: float(e["inference_initial_biomass"])
             for key in ("decision_makers", "non_decision_makers") for e in manifest[key]}
    ration = sum(float(v.get("predation_mortality") or 0.0) * start[k[len("whiting_preys_on_"):]]
                 for k, v in inter.items()
                 if k.startswith("whiting_preys_on_") and v.get("preys_on"))
    assert ration / start["whiting"] == pytest.approx(3.77, abs=0.02)
    fgs = load_project_config("mareld2.yaml", mode="inference")[0]
    krill = fgs["krill"]
    assert krill.open_pool is not None
    assert krill.outside_weight_map.mean() == pytest.approx(1.0)
    assert float(krill.biomass.sum()) == pytest.approx(18000.0, rel=1e-4)
    for p in preds:
        field = fgs[p].params["interaction"][f"{p}_preys_on_krill"]["_depth_access_map"]
        assert field.mean() == pytest.approx(1.0)


def test_exchange_is_booked_per_direction():
    """Section 245: B_new - B (1 + g) = import - export, the export a loss
    cause and the import its own book."""
    from lib.environments.ecosystem_env.loss_accounting import exchange_import, loss_shares
    weight = np.linspace(0.2, 1.8, 20).reshape(4, 5)     # cells above and below T
    env = _env(dict(OPEN, outside_density=1.5), weight=weight)
    c = env.fgs["c"]
    grown = c.biomass.astype(np.float64) * (1 + G_TICK)
    population_change._apply_non_decision_maker_population_change(env, "c", c)
    flow = c.biomass.astype(np.float64) - grown
    assert env.loss_exchange["c"] == pytest.approx(-flow[flow < 0].sum(), rel=1e-5)
    assert exchange_import(env, "c") == pytest.approx(flow[flow > 0].sum(), rel=1e-5)
    assert env.loss_exchange["c"] > 0 and env.gain_exchange["c"] > 0
    assert loss_shares(env, "c")["exchange"] == pytest.approx(1.0)  # its only loss here
