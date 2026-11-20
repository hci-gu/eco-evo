"""Seasonal krill forcing (section 295): an open pool's monthly layout by bed
depth (``outside_depth_season``, open_pool.py) and a prey's monthly energy
content (``energy_season``, prey_energy.py).

Contracts asserted:

1. Both keys are checked: 12 factors with a mean of ~1 (per row for the depth
   season), non decision makers only, the depth season needs an open pool.
2. Each month's layout is the annual shape x the month's depth multiplier at
   an equal-cell mean of 1, so the box total is unchanged; a tick mixes two
   months with the ``daylight.monthly_value`` weights.
3. The open pool's target follows the mixed layout (x the stock season).
4. A prey's energy factor f is exactly a prey with energy x f and handling
   time x f for every predator, in the gain and in the reserve-room cap.
5. Without the keys nothing changes; GPU parity with both on.
6. The library: krill carries both, with annual means 1.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env import interactions, open_pool, predation, prey_energy
from lib.world import daylight, habitat
from lib.world.functional_group import FunctionalGroup
from test_gpu_ecosystem import DEVICES, _assert_parity, device, make_env  # noqa: F401
from test_open_pool import DAYLIGHT, OPEN, SEASON, _eat_c

ROOT = Path(__file__).resolve().parents[1]
SHALLOW = [1.4176] * 4 + [0.1648] * 4 + [1.4176] * 4
DEPTH_SEASON = [{"depth_m": 99.9, "season": SHALLOW}, {"depth_m": 100.0, "season": [1.0] * 12}]
# Dec and Jan far apart so the first ticks (between the two mid-months) mix them.
ENERGY = [1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7]
DEPTH = np.array([[50.0, 80.0, 120.0, 200.0, 300.0]] * 4)


def test_keys_are_checked():
    ok = dict(outside_density=5.0, outside_exchange_rate=0.01)
    FunctionalGroup("x", dict(ok, outside_depth_season=DEPTH_SEASON, energy_season=ENERGY))
    for bad in (dict(outside_depth_season=DEPTH_SEASON),                       # no open pool
                dict(ok, outside_depth_season=[{"depth_m": 50, "season": [1.0] * 11}]),
                dict(ok, outside_depth_season=[{"depth_m": 50, "season": [2.0] * 12}]),
                dict(ok, outside_depth_season=[{"depth_m": 50, "season": [-1.0] + [1.1] * 11}]),
                dict(ok, outside_depth_season=[{"depth_m": 50, "season": [0.0] + [12 / 11] * 11}]),
                dict(ok, energy_season=[1.0] * 11),
                dict(ok, energy_season=[2.0] * 12),
                dict(ok, energy_season=[0.0] + [12 / 11] * 11),
                {"is_decision_maker": True, "energy_season": ENERGY}):
        with pytest.raises(ValueError):
            FunctionalGroup("x", bad)


def test_monthly_layouts_and_weights():
    base = open_pool.depth_weight_map(
        open_pool.parse_depth_table({"outside_depth_table": [{"depth_m": 50, "weight": 0.4},
                                                             {"depth_m": 300, "weight": 1.4}]}),
        DEPTH, DEPTH.shape)
    table = open_pool.parse_depth_season({"outside_depth_season": DEPTH_SEASON})
    maps = open_pool.monthly_weight_maps(base, table, DEPTH)
    assert maps.shape == (12,) + DEPTH.shape
    np.testing.assert_allclose(maps.mean(axis=(1, 2)), 1.0, rtol=1e-12)
    # Shallow cells (< 100 m) carry the month's factor relative to the deep ones.
    for m in (0, 5):
        ratio = (maps[m] / base)[:, :2] / (maps[m] / base)[:, 2:3]
        np.testing.assert_allclose(ratio, SHALLOW[m], rtol=1e-9)
    assert open_pool.monthly_weight_maps(base, None, DEPTH) is None
    assert open_pool.monthly_weight_maps(base, table, None) is None
    w = open_pool.month_weights(1460, 6)
    # Elementwise, not w @ SEASON: numpy's BLAS product is intermittently
    # wrong in this environment (see open_pool.target).
    np.testing.assert_allclose((w * np.array(SEASON)[None]).sum(axis=1),
                               open_pool.season_table(SEASON, 1460, 6), rtol=1e-12)
    _, maps_by_habitat = habitat.build(None, None, {"species_definitions": {"k": dict(
        OPEN, outside_depth_season=DEPTH_SEASON)}, "interaction_definitions": {}}, (3, 4), ["k"])
    assert maps_by_habitat["k"]["outside_weight_months"] is None       # no depth map


def _pool_env(keys, months=None, **kw):
    env = make_env(min_split=0.0, extinction_factor=0.0, refuge=keys, **kw)
    weight = np.linspace(0.5, 1.5, 20).reshape(4, 5)
    env.fgs["c"].outside_weight_map = weight
    if months is not None:
        env.fgs["c"].outside_weight_months = months
    env.build_static_caches()
    return env


def _months(weight):
    depth = np.linspace(40.0, 300.0, 20).reshape(4, 5)
    return open_pool.monthly_weight_maps(
        weight, open_pool.parse_depth_season({"outside_depth_season": DEPTH_SEASON}), depth)


def test_target_follows_the_monthly_layout():
    weight = np.linspace(0.5, 1.5, 20).reshape(4, 5)
    months = _months(weight)
    env = _pool_env(dict(OPEN, outside_season=SEASON, outside_depth_season=DEPTH_SEASON),
                    months=months, daylight=dict(DAYLIGHT, start_tick=600))
    li = interactions.light_index(env)
    w = open_pool.month_weights(env.light_period, 6)[li]
    expected = DENSITY_T((w[:, None, None] * months).sum(axis=0)) * env.open_season["c"][li]
    np.testing.assert_allclose(open_pool.target(env, "c"), expected, rtol=1e-5)
    assert open_pool.target(env, "c").mean() == pytest.approx(
        OPEN["outside_density"] * env.open_season["c"][li], rel=1e-5)
    # Without the daylight calendar the layout is the annual shape.
    plain = _pool_env(dict(OPEN, outside_depth_season=DEPTH_SEASON), months=months)
    assert not plain.open_target_months
    np.testing.assert_allclose(open_pool.target(plain, "c"), OPEN["outside_density"] * weight,
                               rtol=1e-6)


def DENSITY_T(layout):
    return OPEN["outside_density"] * layout


def test_energy_factor_is_scaled_energy_and_handling():
    seasonal = _pool_env(dict(OPEN, energy_season=ENERGY), daylight=DAYLIGHT, handling=2.0)
    f = float(prey_energy.factor(seasonal)[seasonal.global_fg_order.index("c")])
    assert f == pytest.approx(daylight.monthly_value(ENERGY, 0.125), rel=1e-6) and f != 1.0
    scaled = _pool_env(dict(OPEN), daylight=DAYLIGHT, handling=2.0)
    j = scaled.global_fg_order.index("c")
    scaled.energy_gain_mat[:, j] *= np.float32(f)
    scaled.handling_time_mat[:, j] *= np.float32(f)
    for env in (seasonal, scaled):
        predation.apply_predation(env, _eat_c(env))
    np.testing.assert_allclose(seasonal.fgs["a"].temp_energy_gains,
                               scaled.fgs["a"].temp_energy_gains, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(seasonal.fgs["c"].biomass, scaled.fgs["c"].biomass,
                               rtol=1e-5, atol=1e-9)
    # No key, or no calendar: no table, the static matrices.
    assert prey_energy.factor(_pool_env(dict(OPEN), daylight=DAYLIGHT)) is None
    assert prey_energy.factor(_pool_env(dict(OPEN, energy_season=ENERGY))) is None


def test_both_terms_match_reference(device):
    weight = np.linspace(0.5, 1.5, 20).reshape(4, 5)
    env = make_env(min_split=0.0, extinction_factor=0.0, mortality=True, daylight=DAYLIGHT,
                   refuge=dict(OPEN, outside_season=SEASON, outside_depth_season=DEPTH_SEASON,
                               energy_season=ENERGY))
    env.fgs["c"].outside_weight_map = weight
    env.fgs["c"].outside_weight_months = _months(weight)
    env.build_static_caches()
    model = _assert_parity(env, device, ticks=8)
    assert model.open_months is not None and model.prey_energy is not None


def test_library_krill():
    lib = yaml.safe_load(open(ROOT / "fgconfig" / "fg_library.yaml", encoding="utf-8"))
    krill = lib["species_definitions"]["krill"]
    assert np.mean(krill["energy_season"]) == pytest.approx(1.0, abs=1e-3)
    table = open_pool.parse_depth_season(krill, "krill")
    assert table is not None
    depths, values = table
    np.testing.assert_allclose(values.mean(axis=1), 1.0, atol=1e-3)
    shallow = values[depths < 100.0]
    assert (shallow[:, 5] < 0.2).all() and (shallow[:, 0] > 1.3).all()   # Jun out, Jan in
    assert (values[depths >= 100.0] == 1.0).all()
