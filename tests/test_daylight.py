"""Daylight calendar: light-dependent attack rates (section 137)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config.config_loader import load_project_config
from lib.environments.ecosystem_env import interactions, observations, predation
from lib.environments.ecosystem_env.environment import EcosystemEnvironment
from lib.environments.ecosystem_env.state import ActionProbabilities
from lib.world import daylight
from lib.world.functional_group import FunctionalGroup

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "mareld2.yaml"
LIBRARY = ROOT / "fgconfig" / "fg_library.yaml"
LATITUDE = 58.15


def _project(enabled=True, start="random"):
    project = yaml.safe_load(PROJECT.read_text(encoding="utf-8-sig"))
    # Keep the manifest's light climate: the library's phytoplankton is
    # light-limited and refuses to run without one (section 138).
    block = dict((project.get("simulation_settings") or {}).get(
        "daylight") or {})
    block.update(enabled=enabled, latitude_deg=LATITUDE,
                 start_day_of_year=start)
    project["simulation_settings"] = {"daylight": block}
    return project


def _load(project, **kwargs):
    kwargs.setdefault("grid_size", (6, 6))
    kwargs.setdefault("seed", 1)
    return load_project_config(str(PROJECT), library_path=str(LIBRARY),
                               project_config=project, **kwargs)


def _env(project, **kwargs):
    fgs, _iv, _ir, observable = _load(project, **kwargs)
    grid = kwargs.get("grid_size", (6, 6))
    return EcosystemEnvironment(dict(height=grid[0], width=grid[1]), fgs,
                                observable_impact_vars=observable)


# ---------------------------------------------------------------- the sun

def test_solar_elevation_at_the_solstices():
    # Noon elevation = 90 - latitude +- tilt; midnight at midsummer is
    # the same angle below the horizon on the other side.
    midsummer, midwinter = 171 * 24, 354 * 24
    noon = daylight.solar_elevation_deg(LATITUDE, [midsummer + 12,
                                                   midwinter + 12])
    np.testing.assert_allclose(noon, [90 - LATITUDE + 23.44,
                                      90 - LATITUDE - 23.44], atol=0.1)
    midnight = daylight.solar_elevation_deg(LATITUDE, midsummer)
    assert midnight == pytest.approx(-(90 - LATITUDE - 23.44), abs=0.1)


@pytest.mark.parametrize("hours", range(1, 7))
def test_the_year_is_a_whole_number_of_ticks(hours):
    assert daylight.ticks_per_year(hours) * hours == 8760
    assert daylight.light_schedule(LATITUDE, hours).shape == (
        8760 // hours,)


@pytest.mark.parametrize("hours", range(1, 7))
@pytest.mark.parametrize("latitude", [0.0, LATITUDE, 70.0])
@pytest.mark.parametrize("ratio,threshold", [(0.0, -6.0), (0.1, -8.0),
                                             (0.5, 0.0)])
def test_multiplier_averages_to_one_over_the_year(hours, latitude, ratio,
                                                  threshold):
    """The library's a is the annual mean; the calendar only moves it."""
    m = daylight.attack_multiplier(latitude, hours, ratio, threshold)
    assert m.mean() == pytest.approx(1.0, abs=1e-12)
    assert np.all(m >= 0.0)
    # Darkest over lightest is the dark ratio: 1 h ticks resolve a fully
    # dark and a fully light hour at every one of these latitudes.
    if ratio > 0 and hours == 1:
        assert m.min() / m.max() == pytest.approx(ratio, rel=0.05)


def test_unmodulated_pair_is_exactly_one():
    for ratio in (1.0, 1.5, float("nan")):
        assert np.all(daylight.attack_multiplier(LATITUDE, 6, ratio) == 1.0)
    assert daylight.pair_settings({"dark_ratio": 1.0}) is None
    assert daylight.pair_settings({}) is None
    assert daylight.pair_settings({"dark_ratio": 0.2}) == (
        0.2, daylight.DEFAULT_THRESHOLD_DEG)


def test_seasons_follow_latitude():
    """Summer ticks are lighter than winter ticks at 58 N, not at 0 N."""
    def season_mean(latitude, first_day):
        light = daylight.light_schedule(latitude, 6).reshape(365, 4)
        return light[first_day:first_day + 30].mean()
    assert season_mean(LATITUDE, 160) > 0.85 > 0.5 > season_mean(LATITUDE, 340)
    assert season_mean(0.0, 160) == pytest.approx(season_mean(0.0, 340),
                                                  abs=0.02)


# ----------------------------------------------------------- the manifest

def test_disabled_or_absent_settings_are_none():
    assert daylight.parse_settings({}) is None
    assert daylight.parse_settings({"simulation_settings": {}}) is None
    assert daylight.parse_settings(_project(enabled=False)) is None


@pytest.mark.parametrize("block", [
    {"enabled": True},
    {"enabled": True, "latitude_deg": 95},
    {"enabled": True, "latitude_deg": 58, "start_day_of_year": 0},
    {"enabled": True, "latitude_deg": 58, "start_day_of_year": "june"},
])
def test_an_enabled_block_that_cannot_be_honoured_raises(block):
    with pytest.raises(ValueError):
        daylight.parse_settings({"simulation_settings": {"daylight": block}})


@pytest.mark.parametrize("hours", range(1, 7))
def test_fixed_start_day_begins_at_its_midnight(hours):
    tick = daylight.start_tick(172, hours)
    assert tick * hours <= 171 * 24 < (tick + 1) * hours


# ------------------------------------------------------------- the loader

def test_loader_puts_the_same_calendar_on_every_fg():
    fgs = _load(_project(start=100), tick_hours=3)[0]
    configs = [fg.params.get("daylight") for fg in fgs.values()]
    assert all(c == configs[0] for c in configs)
    assert configs[0]["tick_hours"] == 3
    assert configs[0]["start_tick"] == 99 * 8
    assert configs[0]["random_start"] is False


def test_random_start_is_shared_by_a_spawn_seed():
    """CRN: every delta of a generation must see the same season."""
    def start(spawn_seed, seed):
        fgs = _load(_project(), spawn_seed=spawn_seed, seed=seed)[0]
        return next(iter(fgs.values())).params["daylight"]["start_tick"]
    assert start(7, 1) == start(7, 2)
    assert len({start(s, 1) for s in range(8)}) > 1


def test_disabled_loader_adds_nothing():
    fgs = _load(_project(enabled=False))[0]
    assert not any("daylight" in fg.params for fg in fgs.values())


# ---------------------------------------------------- environment + policy

def test_light_channel_is_appended_last_for_every_decision_maker():
    import train
    off, on = _env(_project(enabled=False)), _env(_project())
    np.testing.assert_array_equal(on.per_dm_in_dim, off.per_dm_in_dim + 1)
    for env in (off, on):
        dims = train.get_dynamic_policy_params(
            env.fgs, len(env.observable_impact_vars))
        assert [dims[f][0] for f in env.dm_ids] == list(env.per_dm_in_dim)

    off_obs = observations.build_observation_batch(off)
    on_obs = observations.build_observation_batch(on)
    level = interactions.light_level(on)
    for i in range(on.N_dm):
        n = int(off.per_dm_in_dim[i])
        # Same world (same seeds), so everything before the light slot
        # is the old layout, unchanged.
        np.testing.assert_array_equal(on_obs[i, :, :n], off_obs[i, :, :n])
        np.testing.assert_array_equal(on_obs[i, :, n], level)


def test_disabled_tick_is_unchanged_by_dormant_pair_values():
    """With daylight off, the library's dark_ratio must have no effect."""
    env = _env(_project(enabled=False))
    assert not env._has_daylight and not env._has_light_pairs
    assert interactions.attack_rate(env) is env.max_intake_mat


def _pair_env(start_tick, dark_ratio=0.1, threshold=-8.0):
    """A predator ``p`` on an NDM prey ``q``, 1 t each, h = 0.

    One prey makes ``p`` a specialist, i.e. Type III; with h = 0 the
    intake is a * B_q^2 * B_p = a at unit biomasses.
    """
    groups = {}
    for fid, dm in (("p", True), ("q", False)):
        params = dict(is_decision_maker=dm, movement_speed=0.0,
                      max_energy_reserve=10.0, resting_metabolism=0.0,
                      growth_rate=0.0, maintenance_level=0.3,
                      max_carrying_capacity=1e6, energy_content=5.0,
                      max_intake_rate=0.02, menu=["q"] if dm else [],
                      daylight={"latitude_deg": LATITUDE, "tick_hours": 6,
                                "start_tick": start_tick,
                                "random_start": False})
        params["interaction"] = ({"p_preys_on_q": dict(
            preys_on=True, handling_time=0.0, dark_ratio=dark_ratio,
            light_threshold_deg=threshold)} if dm else {})
        fg = FunctionalGroup(fid, params)
        fg.initialize_state((1, 1), initial_biomass=np.full((1, 1), 1.0))
        fg.energy_reserve[:] = 0.0          # full appetite (hunger = 1)
        groups[fid] = fg
    return EcosystemEnvironment(dict(height=1, width=1), groups,
                                apply_natural_mortality=False)


@pytest.mark.parametrize("start_tick", [0, 1, 4 * 171, 4 * 354 + 2])
def test_predation_scales_with_the_multiplier_of_the_tick(start_tick):
    """Intake is exactly a * m(t) at unit biomasses (see ``_pair_env``)."""
    env = _pair_env(start_tick)
    m = daylight.attack_multiplier(LATITUDE, 6, 0.1, -8.0)[start_tick]
    actions = ActionProbabilities(
        move=np.zeros((1, 4, 1, 1), np.float32),
        rest=np.zeros((1, 1, 1), np.float32),
        eat=np.array([[[[0.0]], [[1.0]]]], np.float32))
    before = float(env.fgs["q"].biomass.sum())
    predation.apply_predation(env, actions)
    eaten = before - float(env.fgs["q"].biomass.sum())
    assert eaten == pytest.approx(0.02 * m, rel=1e-5)


# ------------------------------------------- light-limited growth (138)

CLIMATE = {"light_attenuation_per_m": 0.15,
           "mixed_layer_depth_m": [40, 40, 30, 15, 12, 10,
                                   10, 12, 15, 25, 35, 40],
           "cloud_transmission": [0.5] * 12}


@pytest.mark.parametrize("hours", range(1, 7))
def test_growth_factor_is_one_on_the_reference_day(hours):
    """growth_rate is the April rate, so April 15's daily mean is 1."""
    g = daylight.growth_light_schedule(LATITUDE, hours, CLIMATE, 150.0, 105)
    ticks = 24 // hours if 24 % hours == 0 else None
    if ticks:
        day = g.reshape(365, ticks)[104]
        assert day.mean() == pytest.approx(1.0, abs=1e-3)
    assert g.min() == 0.0                  # nights: no photosynthesis
    assert np.all(np.isfinite(g)) and g.max() < 4.0


def test_deeper_mixing_and_clouds_darken_the_algae():
    def winter_mean(climate):
        g = daylight.growth_light_schedule(LATITUDE, 6, climate, 150.0)
        return g.reshape(365, 4)[340:].mean()
    # Winter months only: the April reference must stay where it is,
    # or the normalisation moves with it.
    winter = (0, 1, 10, 11)
    deeper = dict(CLIMATE, mixed_layer_depth_m=[
        80 if m in winter else v
        for m, v in enumerate(CLIMATE["mixed_layer_depth_m"])])
    cloudier = dict(CLIMATE, cloud_transmission=[
        0.25 if m in winter else v
        for m, v in enumerate(CLIMATE["cloud_transmission"])])
    assert winter_mean(deeper) < winter_mean(CLIMATE)
    assert winter_mean(cloudier) < winter_mean(CLIMATE)


def test_monthly_values_interpolate_circularly():
    values = list(range(12))
    assert daylight.monthly_value(values, 15.5) == pytest.approx(0.0)
    # Between mid-December and mid-January, the wrap.
    between = daylight.monthly_value(values, 0.0)
    assert 0.0 < between < 11.0


@pytest.mark.parametrize("bad", [
    {"light_attenuation_per_m": 0.15},
    dict(CLIMATE, light_attenuation_per_m=0),
    dict(CLIMATE, mixed_layer_depth_m=[10] * 11),
    dict(CLIMATE, cloud_transmission=[1.2] * 12),
])
def test_an_invalid_light_climate_raises(bad):
    block = dict(bad, enabled=True, latitude_deg=LATITUDE)
    with pytest.raises(ValueError):
        daylight.parse_settings({"simulation_settings": {"daylight": block}})


def test_light_limited_producer_needs_a_light_climate():
    project = _project()
    project["simulation_settings"]["daylight"].pop("mixed_layer_depth_m",
                                                   None)
    for key in ("light_attenuation_per_m", "cloud_transmission"):
        project["simulation_settings"]["daylight"].pop(key, None)
    library = yaml.safe_load(LIBRARY.read_text(encoding="utf-8-sig"))
    library["species_definitions"]["phytoplankton"]["light_saturation"] = 150
    fgs, _iv, _ir, observable = load_project_config(
        str(PROJECT), library_config=library, project_config=project,
        grid_size=(6, 6), seed=1)
    with pytest.raises(ValueError, match="light climate"):
        EcosystemEnvironment(dict(height=6, width=6), fgs,
                             observable_impact_vars=observable)


@pytest.mark.parametrize("start_tick", [0, 2, 4 * 171 + 2, 4 * 354 + 2])
def test_producer_growth_follows_the_factor_of_the_tick(start_tick):
    """Logistic NDM growth is growth_rate * factor(t) * B (1 - B/K)."""
    cfg = {"latitude_deg": LATITUDE, "tick_hours": 6,
           "start_tick": start_tick, "random_start": False,
           "light_climate": CLIMATE}
    groups = {}
    for fid, dm in (("p", True), ("q", False)):
        params = dict(is_decision_maker=dm, movement_speed=0.0,
                      max_energy_reserve=10.0, resting_metabolism=0.0,
                      growth_rate=0.05, maintenance_level=0.3,
                      max_carrying_capacity=100.0, energy_content=5.0,
                      menu=[], daylight=cfg)
        if not dm:
            params["light_saturation"] = 150.0
        fg = FunctionalGroup(fid, params)
        fg.initialize_state((1, 1), initial_biomass=np.full((1, 1), 10.0))
        groups[fid] = fg
    env = EcosystemEnvironment(dict(height=1, width=1), groups,
                               apply_natural_mortality=False)
    env.ordered_fg_ids = list(env.fgs)
    factor = daylight.growth_light_schedule(LATITUDE, 6, CLIMATE,
                                            150.0)[start_tick]
    env._apply_growth()
    grown = float(env.fgs["q"].biomass.sum()) - 10.0
    # float32 biomass of ~10 t: one ulp is ~1e-6 t.
    assert grown == pytest.approx(0.05 * factor * 10.0 * 0.9, abs=2e-6)
