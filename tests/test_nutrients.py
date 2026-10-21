"""Nutrient pool of light-limited producers (section 172).

Contracts asserted:

1. The manifest block is off unless enabled, refuses to run without the
   daylight light climate or a required key, and carries the mixed-layer
   depths along.
2. A producer's library keys are validated; without
   ``nutrient_half_saturation`` it stays logistic.
3. The per-tick tables are the Evans & Parslow supply: with a constant
   mixed layer the pool relaxes to N0 x Y x M, a deepening layer
   entrains N0 water, a shoaling one leaves its own water behind.
4. In the tick: growth is paid for by the pool (biomass + pool change by
   the supply only), the logistic factor is gone, K stays a ceiling,
   an empty cell keeps its pool, grazing returns its fraction, and an
   impact on the producer does not touch the pool.
5. Without the manifest block the library keys are inert: the tick is
   exactly the logistic one.
6. The loader puts the settings on every group and the environment
   starts the pool at the surface climatology of its start day.

GPU parity: tests/test_gpu_ecosystem.py::test_nutrient_pool_matches_reference.
"""
import copy
import os
import sys

import numpy as np
import pytest
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.environments.ecosystem import EcosystemEnvironment  # noqa: E402
from lib.environments.ecosystem_env import (  # noqa: E402
    interactions, population_change, predation)
from lib.environments.ecosystem_env.state import ActionProbabilities  # noqa: E402
from lib.world import nutrients  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402

DEPTHS = [13, 9, 6, 6, 6, 6, 6, 7, 10, 10, 17, 15]
CLIMATE = {"light_attenuation_per_m": 0.29, "mixed_layer_depth_m": DEPTHS,
           "cloud_transmission": [0.7] * 12}
BLOCK = {"enabled": True,
         "sub_mixed_layer_umol_l": [5.7, 6.6, 6.8, 7.0, 4.9, 4.4, 2.8, 1.4,
                                    0.9, 0.9, 2.5, 3.9],
         "surface_umol_l": [5.1, 4.8, 2.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1,
                            0.3, 1.3, 3.6],
         "exchange_m_per_day": 0.2}
SPECIES = {"nutrient_half_saturation": 0.5, "nutrient_yield": 0.8,
           "grazing_regeneration": 0.5}


def _project(nutrient_block=BLOCK, climate=True):
    daylight = {"enabled": True, "latitude_deg": 58.15,
                "start_day_of_year": 100}
    if climate:
        daylight.update(CLIMATE)
    settings = {"daylight": daylight}
    if nutrient_block is not None:
        settings["nutrients"] = dict(nutrient_block)
    return {"simulation_settings": settings}


def _config(exchange=0.2, depths=DEPTHS, n0=None):
    block = dict(BLOCK, exchange_m_per_day=exchange)
    if n0 is not None:
        block["sub_mixed_layer_umol_l"] = [n0] * 12
    project = _project(block)
    project["simulation_settings"]["daylight"]["mixed_layer_depth_m"] = depths
    return nutrients.runtime_config(nutrients.parse_settings(project), 6)


# ------------------------------------------------------------------ 1.


def test_block_is_off_unless_enabled():
    assert nutrients.parse_settings({}) is None
    assert nutrients.parse_settings(_project(None)) is None
    assert nutrients.parse_settings(
        _project(dict(BLOCK, enabled=False))) is None


def test_block_needs_the_light_climate_and_its_keys():
    with pytest.raises(ValueError, match="light climate"):
        nutrients.parse_settings(_project(climate=False))
    for key in ("sub_mixed_layer_umol_l", "surface_umol_l",
                "exchange_m_per_day"):
        block = {k: v for k, v in BLOCK.items() if k != key}
        with pytest.raises(ValueError, match=key):
            nutrients.parse_settings(_project(block))
    with pytest.raises(ValueError, match="12"):
        nutrients.parse_settings(_project(dict(BLOCK, surface_umol_l=[1, 2])))
    with pytest.raises(ValueError, match=">= 0"):
        nutrients.parse_settings(_project(dict(BLOCK, exchange_m_per_day=-1)))


def test_exchange_may_be_monthly():
    """Section 173: weak summer exchange under the stratification, strong
    winter storms - one value or 12."""
    monthly = [0.3, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.3, 0.3]
    config = _config(depths=[10.0] * 12, n0=6.0)
    config["exchange_m_per_day"] = monthly
    tables = nutrients.tick_tables(config, 0.8, 0.5)
    jan, jul = 4 * 15, 4 * 196                     # mid-January, mid-July
    assert tables["supply"][jan] == pytest.approx(0.8 * 6.0 * 0.3 * 0.25,
                                                  rel=1e-3)
    assert tables["supply"][jul] == pytest.approx(0.8 * 6.0 * 0.05 * 0.25,
                                                  rel=2e-2)
    with pytest.raises(ValueError, match="12"):
        nutrients.parse_settings(_project(dict(BLOCK,
                                               exchange_m_per_day=[0.1, 0.2])))
    assert nutrients.parse_settings(_project(dict(
        BLOCK, exchange_m_per_day=monthly)))["exchange_m_per_day"] == monthly


def test_block_carries_the_mixed_layer_depths():
    settings = nutrients.parse_settings(_project())
    assert settings["mixed_layer_depth_m"] == [float(d) for d in DEPTHS]
    assert nutrients.runtime_config(settings, 6)["tick_hours"] == 6


# ------------------------------------------------------------------ 2.


def test_species_keys_are_validated():
    assert nutrients.species_settings({}) is None
    assert nutrients.species_settings({"nutrient_half_saturation": 0}) is None
    assert nutrients.species_settings(SPECIES) == (0.5, 0.8, 0.5)
    with pytest.raises(ValueError, match="nutrient_yield"):
        nutrients.species_settings({"nutrient_half_saturation": 0.5})
    with pytest.raises(ValueError, match="grazing_regeneration"):
        nutrients.species_settings(dict(SPECIES, grazing_regeneration=1.5))
    # Regeneration defaults to none.
    assert nutrients.species_settings(
        {"nutrient_half_saturation": 0.5, "nutrient_yield": 1.0})[2] == 0.0


# ------------------------------------------------------------------ 3.


def test_constant_layer_relaxes_to_the_reservoir():
    config = _config(exchange=0.5, depths=[10.0] * 12, n0=6.0)
    tables = nutrients.tick_tables(config, 0.8, 0.5)
    assert np.allclose(tables["keep"], 1.0 - 0.5 * 0.25 / 10.0)
    assert np.allclose(tables["supply"], 0.8 * 6.0 * 0.5 * 0.25)
    assert np.allclose(tables["half"], 0.5 * 0.8 * 10.0)
    pool = 0.0
    for t in range(4 * 365):
        pool = pool * tables["keep"][t] + tables["supply"][t]
    assert pool == pytest.approx(6.0 * 0.8 * 10.0, rel=1e-3)


def test_tables_are_tick_length_agnostic():
    """A 3 h calendar has twice the ticks and half the supply per tick,
    and relaxes to the same pool (AGENTS: the pipeline is per tick)."""
    six = _config(exchange=0.5, depths=[10.0] * 12, n0=6.0)
    three = dict(six, tick_hours=3)
    t6 = nutrients.tick_tables(six, 0.8, 0.5)
    t3 = nutrients.tick_tables(three, 0.8, 0.5)
    assert len(t3["keep"]) == 2 * len(t6["keep"])
    # Constant depths: the supply per tick is constant, half at 3 h.
    assert np.allclose(t3["supply"], t6["supply"][0] / 2.0)
    pool = 0.0
    for t in range(len(t3["keep"])):
        pool = pool * t3["keep"][t] + t3["supply"][t]
    assert pool == pytest.approx(6.0 * 0.8 * 10.0, rel=1e-3)
    # Seasonal depths: a day's supply is the same at either tick length.
    s6 = nutrients.tick_tables(_config(), 0.8, 0.5)["supply"]
    s3 = nutrients.tick_tables(dict(_config(), tick_hours=3), 0.8, 0.5)["supply"]
    assert np.allclose(s3.reshape(-1, 8).sum(1), s6.reshape(-1, 4).sum(1),
                       rtol=1e-6)


def test_deepening_entrains_and_shoaling_detrains():
    config = _config(exchange=0.0, n0=5.0)
    tables = nutrients.tick_tables(config, 1.0, 0.5)
    hours = np.arange(len(tables["keep"])) * 6
    depth = lambda h: nutrients._depth(config, h)
    deepening = depth(hours + 6) > depth(hours) + 1e-9
    shoaling = depth(hours + 6) < depth(hours) - 1e-9
    assert deepening.any() and shoaling.any()
    # Entrainment adds N0 x dM and keeps the concentration's share.
    d_m = depth(hours + 6) - depth(hours)
    assert np.allclose(tables["supply"][deepening], 5.0 * d_m[deepening])
    assert np.all(tables["keep"][deepening] == 1.0)
    # Shoaling leaves dM / M of the pool below, and adds nothing.
    assert np.all(tables["supply"][shoaling] == 0.0)
    assert np.all(tables["keep"][shoaling] < 1.0)


# ------------------------------------------------------------------ 4.


def _env(nutrient_on=True, species=SPECIES, start_tick=4 * 100,
         shape=(3, 4), capacity=50.0):
    config = _config()
    groups = {}
    rng = np.random.default_rng(3)
    for fid, dm in (("producer", False), ("grazer", True)):
        params = dict(is_decision_maker=dm, movement_speed=0.0,
                      max_energy_reserve=100.0, resting_metabolism=0.5,
                      feeding_cost=1.5, movement_cost=1.2, growth_rate=0.1,
                      maintenance_level=0.3, natural_mortality=0.0,
                      max_carrying_capacity=capacity, energy_content=4000.0,
                      max_intake_rate=0.05, min_split_biomass=0.0,
                      extinction_threshold_factor=0.0,
                      observes=["producer"] if dm else [],
                      menu=["producer"] if dm else [],
                      daylight={"latitude_deg": 58.15, "tick_hours": 6,
                                "start_tick": start_tick,
                                "random_start": False,
                                "light_climate": CLIMATE})
        if dm:
            params["interaction"] = {"grazer_preys_on_producer": dict(
                preys_on=True, assimilation_factor=0.7, handling_time=2.0)}
        else:
            params["light_saturation"] = 150.0
            params.update(species)
        if nutrient_on:
            params["nutrients"] = dict(config)
        fg = FunctionalGroup(fid, params)
        fg.initialize_state(shape, initial_biomass=rng.uniform(1, 5, shape),
                            rng=rng)
        groups[fid] = fg
    groups["producer"].biomass[0, 0] = 0.0
    return EcosystemEnvironment(dict(height=shape[0], width=shape[1]), groups,
                                apply_natural_mortality=False)


def _noon(env):
    """Advance the calendar to a daylight tick so the light factor is > 0."""
    while interactions.growth_light(env, "producer") <= 0.0:
        env.tick_count += 1


def test_pool_starts_at_the_surface_climatology():
    env = _env()
    expected = nutrients.initial_pool(env.nutrient_config, 0.8, env.light_start)
    assert np.allclose(env.fgs["producer"].energy_reserve, expected)
    assert env.fgs["producer"].energy_reserve[0, 0] > 0.0   # empty cell too


def test_growth_is_paid_by_the_pool():
    env = _env()
    _noon(env)
    fg = env.fgs["producer"]
    tables = env.nutrient_tables["producer"]
    t = interactions.light_index(env)
    b0, n0 = fg.biomass.copy(), fg.energy_reserve.copy()
    population_change._apply_non_decision_maker_population_change(
        env, "producer", fg)
    supplied = n0 * tables["keep"][t] + tables["supply"][t]
    # Biomass + pool change by the supply alone.
    assert np.allclose(fg.biomass + fg.energy_reserve, b0 + supplied,
                       rtol=1e-5)
    # The rate is r L B N / (N + half): no logistic factor.
    rate = fg.growth_rate * interactions.growth_light(env, "producer")
    expected = rate * b0 * supplied / (supplied + tables["half"][t])
    assert np.allclose(fg.biomass - b0, expected, rtol=1e-4)
    assert fg.biomass[0, 0] == 0.0 and fg.energy_reserve[0, 0] > 0.0


def test_capacity_is_a_ceiling_and_the_pool_a_limit():
    env = _env(capacity=3.0)
    _noon(env)
    fg = env.fgs["producer"]
    fg.energy_reserve[...] = 1e-3                 # nearly empty pool
    population_change._apply_non_decision_maker_population_change(
        env, "producer", fg)
    assert np.all(fg.biomass <= 3.0 + 1e-6)
    assert np.all(fg.energy_reserve >= 0.0)


def test_grazing_returns_its_fraction_and_impacts_spare_the_pool():
    env = _env()
    fg = env.fgs["producer"]
    pool0, b0 = fg.energy_reserve.copy(), fg.biomass.copy()
    shape = (env.N_dm, env.H, env.W)
    eat = np.ones((env.N_dm, env.N_all) + shape[1:], dtype=env.dtype)
    eat[:, [j for j, f in enumerate(env.global_fg_order) if f != "producer"]] = 0
    actions = ActionProbabilities(
        move=np.zeros((env.N_dm, 4) + shape[1:], dtype=env.dtype),
        rest=np.zeros(shape, dtype=env.dtype), eat=eat)
    predation.apply_predation(env, actions)
    grazed = b0 - fg.biomass
    assert grazed.sum() > 0
    assert np.allclose(fg.energy_reserve, pool0 + 0.5 * grazed, rtol=1e-5)

    population_change._zero_energy_in_empty_cells(env)
    assert fg.energy_reserve[0, 0] > 0.0


def test_stress_loss_follows_the_ersem_shape():
    """Section 175: 0 when replete, L_s when exhausted, and ERSEM's
    lysis curve in between (0.045 / 0.125 / 0.5 per day at f = 1 / 0.3 /
    0, minus the replete 0.045)."""
    assert nutrients.stress_settings({}) is None
    assert nutrients.stress_settings({"stress_loss_max": 0.1}) == (0.1, 0.1)
    with pytest.raises(ValueError):
        nutrients.stress_settings({"stress_loss_max": 1.5})
    assert nutrients.stress_loss(1.0, 0.4, 0.1) == pytest.approx(0.0)
    assert nutrients.stress_loss(0.0, 0.4, 0.1) == pytest.approx(0.4)
    ersem = (0.125 - 0.045) / (0.5 - 0.045)
    assert nutrients.stress_loss(0.3, 1.0, 0.1) == pytest.approx(ersem,
                                                                 rel=0.02)


def test_stress_loss_acts_before_growth_and_spares_the_pool():
    env = _env(species=dict(SPECIES, stress_loss_max=0.2))
    _noon(env)
    fg = env.fgs["producer"]
    fg.energy_reserve[...] = 1e-4                  # exhausted pool
    tables = env.nutrient_tables["producer"]
    t = interactions.light_index(env)
    b0, n0 = fg.biomass.copy(), fg.energy_reserve.copy()
    population_change._apply_non_decision_maker_population_change(
        env, "producer", fg)
    pool = n0 * tables["keep"][t] + tables["supply"][t]
    f = pool / (pool + tables["half"][t])
    stressed = b0 * (1.0 - nutrients.stress_loss(f, 0.2, 0.1))
    rate = fg.growth_rate * interactions.growth_light(env, "producer")
    grown = np.minimum(rate * stressed * f, pool)
    assert np.allclose(fg.biomass, stressed + grown, rtol=1e-4)
    assert np.allclose(fg.energy_reserve, pool - grown, rtol=1e-4, atol=1e-7)
    assert env.loss_natural["producer"] > 0.0


# ------------------------------------------------------------------ 5.


def test_without_the_block_the_keys_are_inert():
    env = _env(nutrient_on=False)
    assert not env._has_nutrients
    assert not interactions.is_nutrient_producer(env, "producer")
    _noon(env)
    fg = env.fgs["producer"]
    b0 = fg.biomass.copy()
    population_change._apply_non_decision_maker_population_change(
        env, "producer", fg)
    rate = fg.growth_rate * interactions.growth_light(env, "producer")
    assert np.allclose(fg.biomass, b0 + rate * b0 * (1 - b0 / 50.0),
                       rtol=1e-5)


def test_a_block_without_any_producer_is_refused():
    with pytest.raises(ValueError, match="no group has"):
        _env(species={})


# ------------------------------------------------------------------ 6.


def test_loader_carries_the_settings_and_the_env_fills_the_pool(tmp_path):
    from inference import build_env

    with open(os.path.join(_ROOT, "mareld2.yaml")) as handle:
        project = yaml.safe_load(handle)
    project["simulation_settings"]["nutrients"] = dict(BLOCK)
    # A single mixed layer (BLOCK has no subsurface_layer), so without
    # the subsurface producer of section 180.
    project["non_decision_makers"] = [
        e for e in project["non_decision_makers"]
        if e["group_id"] != "subsurface_phytoplankton"]
    for section in ("decision_makers", "non_decision_makers"):
        for entry in project.get(section) or []:
            for key, value in list(entry.items()):
                if isinstance(value, str) and value.endswith(".npz"):
                    entry[key] = os.path.join(_ROOT, value)
    for iv in project.get("impact_variables") or []:
        if isinstance(iv.get("baseline_map"), str):
            iv["baseline_map"] = os.path.join(_ROOT, iv["baseline_map"])
    maps = (project.get("inference") or {}).get("impact_maps") or {}
    for key, value in maps.items():
        maps[key] = os.path.join(_ROOT, value)
    with open(os.path.join(_ROOT, "fgconfig", "fg_library.yaml")) as handle:
        library = yaml.safe_load(handle)
    library["species_definitions"]["phytoplankton"].update(SPECIES)
    manifest = tmp_path / "project.yaml"
    library_path = tmp_path / "library.yaml"
    manifest.write_text(yaml.safe_dump(project))
    library_path.write_text(yaml.safe_dump(library))

    env = build_env(str(manifest), (8, 8), seed=4, verbose=False,
                    apply_natural_mortality=True, mass_balance=True,
                    library_path=str(library_path))
    assert env._has_nutrients and env.nutrient_ids == ["phytoplankton"]
    for fg in env.fgs.values():
        assert fg.params["nutrients"]["exchange_m_per_day"] == 0.2
    pool = nutrients.initial_pool(env.nutrient_config, 0.8, env.light_start)
    assert np.allclose(env.fgs["phytoplankton"].energy_reserve, pool)
    env.step(env.calculate_decisions())
    assert np.all(env.fgs["phytoplankton"].energy_reserve >= 0.0)
