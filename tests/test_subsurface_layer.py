"""The two-layer water column (section 180).

Contracts asserted:

1. The manifest block is validated: every key is required and the
   layer's bottom must lie below the deepest mixed layer.
2. ``layer_tables`` conserves water and algae: the biomass map is a pure
   exchange between the layers; the pools lose only the reservoir share
   and gain only the reservoir supply. Entrainment moves subsurface water
   up only while the mixed layer deepens, detrainment only while it
   shoals.
3. With the layer, each producer's own table is inert (keep 1, supply
   0) with the half-saturation over its own layer thickness.
4. The subsurface light factor is below the mixed layer's, zero at
   night, and larger for a shade-adapted (lower I_k) producer.
5. A concentration-feeding grazer dilutes a subsurface prey over the
   layer thickness H = bottom - M, not over M.
6. The environment refuses an inconsistent producer set, and the tick
   applies the map exactly.

GPU parity: tests/test_gpu_ecosystem.py::test_subsurface_layer_matches_reference.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.environments.ecosystem import EcosystemEnvironment  # noqa: E402
from lib.environments.ecosystem_env import interactions, population_change  # noqa: E402
from lib.world import daylight, nutrients  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402

DEPTHS = [13, 9, 6, 6, 6, 6, 6, 7, 10, 10, 17, 15]
CLIMATE = {"light_attenuation_per_m": 0.29, "mixed_layer_depth_m": DEPTHS,
           "cloud_transmission": [0.7] * 12}
LAYER = {"bottom_m": 30.0, "light_attenuation_per_m": 0.17,
         "reservoir_umol_l": [6.3, 7.4, 7.5, 7.9, 7.2, 7.2, 6.4, 4.7, 4.4,
                              2.5, 4.6, 4.5],
         "reservoir_exchange_m_per_day": 0.3}


def _project(layer=None):
    nut = {"enabled": True,
           "sub_mixed_layer_umol_l": [5.7, 6.6, 6.8, 7.0, 4.9, 4.4, 2.8, 1.4,
                                      0.9, 0.9, 2.5, 3.9],
           "surface_umol_l": [5.1, 4.8, 2.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1,
                              0.3, 1.3, 3.6],
           "exchange_m_per_day": [0.3, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05,
                                  0.1, 0.2, 0.3, 0.3]}
    if layer is not None:
        nut["subsurface_layer"] = layer
    return {"simulation_settings": {
        "daylight": {"enabled": True, "latitude_deg": 58.15, **CLIMATE},
        "nutrients": nut}}


def _config(layer=LAYER):
    return nutrients.runtime_config(
        nutrients.parse_settings(_project(dict(layer))), 6)


def test_block_is_validated():
    assert nutrients.parse_settings(_project())["subsurface"] is None
    assert _config()["subsurface"]["bottom_m"] == 30.0
    for key in LAYER:
        broken = {k: v for k, v in LAYER.items() if k != key}
        with pytest.raises(ValueError, match=key):
            nutrients.parse_settings(_project(broken))
    with pytest.raises(ValueError, match="deepest mixed layer"):
        nutrients.parse_settings(_project(dict(LAYER, bottom_m=17.0)))


def test_layer_tables_conserve_water_and_algae():
    tab = nutrients.layer_tables(_config(), 0.795)
    np.testing.assert_allclose(tab["b_mm"] + tab["b_dm"], 1.0, atol=1e-12)
    np.testing.assert_allclose(tab["b_md"] + tab["b_dd"], 1.0, atol=1e-12)
    # Without the reservoir the pools conserve exactly.
    closed = nutrients.layer_tables(
        _config(dict(LAYER, reservoir_exchange_m_per_day=0.0)), 0.795)
    np.testing.assert_allclose(closed["n_mm"] + closed["n_dm"], 1.0, atol=1e-12)
    np.testing.assert_allclose(closed["n_md"] + closed["n_dd"], 1.0, atol=1e-12)
    np.testing.assert_allclose(closed["n_s"], 0.0)
    # With it, only the reservoir share leaves.
    assert np.all(tab["n_mm"] + tab["n_dm"] <= 1.0 + 1e-12)
    assert np.all(tab["n_md"] + tab["n_dd"] < 1.0)
    assert np.all(tab["n_s"] > 0.0)
    # Entrainment only while deepening (autumn), detrainment only while
    # shoaling (spring); never both in one tick.
    assert not np.any((tab["b_md"] > 0) & (tab["b_dm"] > 0))
    autumn = slice(4 * 290, 4 * 300)      # Oct -> Nov, 10 -> 17 m
    spring = slice(4 * 20, 4 * 40)        # Jan -> Feb, 13 -> 9 m
    assert np.all(tab["b_md"][autumn] > 0) and np.all(tab["b_dm"][autumn] == 0)
    assert np.all(tab["b_dm"][spring] > 0) and np.all(tab["b_md"][spring] == 0)


def test_producer_tables_are_inert_with_the_layer():
    cfg = _config()
    hours = (np.arange(1460) * 6 + 3.0)
    for layer, thickness in (("mixed", nutrients._depth(cfg, hours)),
                             ("subsurface", 30.0 - nutrients._depth(cfg, hours))):
        tab = nutrients.tick_tables(cfg, 0.795, 0.5, layer)
        np.testing.assert_allclose(tab["keep"], 1.0)
        np.testing.assert_allclose(tab["supply"], 0.0)
        np.testing.assert_allclose(tab["half"], 0.5 * 0.795 * thickness)
    with pytest.raises(ValueError, match="subsurface_layer"):
        nutrients.tick_tables(nutrients.runtime_config(
            nutrients.parse_settings(_project()), 6), 0.795, 0.5, "subsurface")
    assert nutrients.initial_pool(cfg, 0.795, 4 * 190, "subsurface") == \
        pytest.approx(0.795 * (30.0 - float(nutrients._depth(cfg, 4 * 190 * 6)))
                      * float(daylight.monthly_value(
                          cfg["sub_mixed_layer_umol_l"], 190)))


def test_subsurface_light_is_dimmer_and_follows_shade_adaptation():
    top = daylight.growth_light_schedule(58.15, 6, CLIMATE, 50.0)
    deep = daylight.subsurface_light_schedule(58.15, 6, CLIMATE, 50.0, 30.0, 0.17)
    shaded = daylight.subsurface_light_schedule(58.15, 6, CLIMATE, 30.0, 30.0, 0.17)
    june = slice(4 * 152, 4 * 182)
    assert deep[june].mean() < top[june].mean()
    assert shaded[june].mean() > deep[june].mean()
    assert deep[4 * 355] == 0.0                     # December, 00-06 h


def _env(layer=True, deep_layer="subsurface", start_tick=4 * 190):
    project = _project(dict(LAYER) if layer else None)
    nut = nutrients.runtime_config(nutrients.parse_settings(project), 6)
    light = {"latitude_deg": 58.15, "tick_hours": 6, "start_tick": start_tick,
             "random_start": False, "light_climate": CLIMATE}
    shape = (2, 3)
    groups = {}
    producer = dict(is_decision_maker=False, movement_speed=0.0,
                    growth_rate=0.1675, max_carrying_capacity=24.0,
                    energy_content=4800.0, min_split_biomass=0.0,
                    extinction_threshold_factor=0.0, daylight=light,
                    nutrients=nut, nutrient_half_saturation=0.5,
                    nutrient_yield=0.795, grazing_regeneration=0.3)
    groups["phyto"] = FunctionalGroup("phyto", dict(
        producer, light_saturation=150.0))
    groups["deep"] = FunctionalGroup("deep", dict(
        producer, light_saturation=50.0, water_layer=deep_layer))
    grazer_pair = {"preys_on": True, "max_intake_rate": 0.06,
                   "handling_time": 2.0, "mixed_layer_reference_m": 10.0}
    groups["micro"] = FunctionalGroup("micro", dict(
        producer, nutrient_half_saturation=0.0, light_saturation=0.0,
        growth_rate=0.0, growth_efficiency=0.3,
        interaction={"micro_preys_on_deep": grazer_pair}))
    rng = np.random.default_rng(3)
    for fg in groups.values():
        fg.initialize_state(shape, initial_biomass=rng.uniform(0.5, 3, shape))
    return EcosystemEnvironment(dict(height=2, width=3), groups,
                                apply_natural_mortality=False)


def test_grazer_dilutes_a_subsurface_prey_over_the_layer():
    env = _env()
    entry = env.ndm_grazers[0]
    hours = np.arange(1460) * 6 + 3.0
    m = daylight.monthly_value(DEPTHS, hours / 24.0)
    np.testing.assert_allclose(entry["scale"], 10.0 / (30.0 - m), rtol=1e-6)


def test_environment_validates_and_applies_the_map():
    with pytest.raises(ValueError, match="exactly one"):
        _env(deep_layer="mixed")
    with pytest.raises(ValueError, match="subsurface_layer"):
        _env(layer=False)
    env = _env(start_tick=4 * 295)          # deepening: entrainment
    ex = env.layer_exchange
    assert (ex["mixed"], ex["subsurface"]) == ("phyto", "deep")
    t = interactions.light_index(env)
    top, deep = env.fgs["phyto"], env.fgs["deep"]
    b, bd = top.biomass.copy(), deep.biomass.copy()
    n, nd = top.energy_reserve.copy(), deep.energy_reserve.copy()
    population_change.apply_layer_exchange(env)
    np.testing.assert_allclose(top.biomass, ex["b_mm"][t] * b + ex["b_md"][t] * bd,
                               rtol=1e-6)
    np.testing.assert_allclose(deep.energy_reserve,
                               ex["n_dm"][t] * n + ex["n_dd"][t] * nd + ex["n_s"][t],
                               rtol=1e-6)
    assert ex["b_md"][t] > 0.0
    np.testing.assert_allclose(top.biomass.sum() + deep.biomass.sum(),
                               b.sum() + bd.sum(), rtol=1e-6)
