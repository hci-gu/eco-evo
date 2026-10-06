"""Water temperature on the daylight calendar: Q10 metabolism (section 143)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config.config_loader import load_project_config
from lib.environments.ecosystem_env import interactions, movement
from lib.environments.ecosystem_env.environment import EcosystemEnvironment
from lib.environments.ecosystem_env.state import ActionProbabilities
from lib.world import daylight, temperature
from lib.world.functional_group import FunctionalGroup

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "mareld2.yaml"
LIBRARY = ROOT / "fgconfig" / "fg_library.yaml"
LATITUDE = 58.15
LAYER = [6.3, 5.4, 5.2, 6.1, 7.8, 9.9, 12.1, 13.4, 13.8, 12.8, 10.5, 8.4]


def _manifest():
    return yaml.safe_load(PROJECT.read_text(encoding="utf-8-sig"))


def _project(temperature_block=None, daylight_on=True):
    project = _manifest()
    sim = project.get("simulation_settings") or {}
    block = dict(sim.get("daylight") or {})
    block.update(enabled=daylight_on, latitude_deg=LATITUDE)
    settings = {"daylight": block}
    if temperature_block is not None:
        settings["temperature"] = temperature_block
    project["simulation_settings"] = settings
    return project


def _block(**overrides):
    block = {"enabled": True, "layers": {"upper": LAYER, "deep": LAYER[::-1]},
             "group_layers": {"pelagic_fish": "upper", "zooplankton": "upper",
                              "gadoids": "deep"}}
    block.update(overrides)
    return block


def _load(project, **kwargs):
    kwargs.setdefault("grid_size", (6, 6))
    kwargs.setdefault("seed", 1)
    return load_project_config(str(PROJECT), library_path=str(LIBRARY),
                               project_config=project, **kwargs)


# ---------------------------------------------------------------- settings

def test_disabled_or_absent_settings_are_none():
    assert temperature.parse_settings({}) is None
    assert temperature.parse_settings(_project()) is None
    assert temperature.parse_settings(
        _project(_block(enabled=False))) is None


@pytest.mark.parametrize("block", [
    _block(layers={"upper": LAYER[:11]}),
    _block(layers={"upper": ["warm"] * 12}),
    _block(layers={"upper": [60.0] * 12}),
    _block(layers={}),
    _block(group_layers={"pelagic_fish": "missing"}),
])
def test_an_enabled_block_that_cannot_be_honoured_raises(block):
    with pytest.raises(ValueError):
        temperature.parse_settings(_project(block))


def test_temperature_needs_the_daylight_calendar():
    with pytest.raises(ValueError, match="daylight"):
        temperature.parse_settings(_project(_block(), daylight_on=False))


@pytest.mark.parametrize("params,expected", [
    ({}, None),
    ({"metabolism_q10": 1.0, "metabolism_t_ref": 10.0}, None),
    ({"metabolism_q10": 0.0}, None),
    ({"metabolism_q10": 2.2, "metabolism_t_ref": 14}, (2.2, 14.0)),
    ({"metabolism_q10": 2.0, "metabolism_t_ref": "annual_mean"},
     (2.0, "annual_mean")),
])
def test_species_settings(params, expected):
    assert temperature.species_settings(params) == expected


@pytest.mark.parametrize("params", [
    {"metabolism_q10": 2.2},
    {"metabolism_q10": -1.0, "metabolism_t_ref": 10.0},
    {"metabolism_q10": 2.2, "metabolism_t_ref": "summer"},
])
def test_invalid_species_settings_raise(params):
    with pytest.raises(ValueError):
        temperature.species_settings(params)


# -------------------------------------------------------------- multiplier

@pytest.mark.parametrize("hours", range(1, 7))
def test_annual_mean_reference_averages_to_one(hours):
    m = temperature.metabolism_multiplier(LAYER, hours, 2.0, "annual_mean")
    assert m.shape == (daylight.ticks_per_year(hours),)
    assert float(m.mean()) == pytest.approx(1.0, abs=1e-12)


def test_numeric_reference_is_the_q10_law():
    m = temperature.metabolism_multiplier(LAYER, 6, 2.2, 14.0)
    temps = temperature.tick_temperatures(LAYER, 6)
    np.testing.assert_allclose(m, 2.2 ** ((temps - 14.0) / 10.0))
    # Mid-March (tick of day 74, noon) is the coldest month, 5.2 C.
    mid_march = 74 * 4 + 2
    assert temps[mid_march] == pytest.approx(5.2, abs=0.05)
    assert m[mid_march] == pytest.approx(2.2 ** ((5.2 - 14.0) / 10.0),
                                         rel=1e-2)


def test_tick_temperature_is_continuous_over_new_year():
    temps = temperature.tick_temperatures(LAYER, 6)
    assert abs(temps[-1] - temps[0]) < 0.05


# ------------------------------------------------------------------ loader

def test_loader_puts_the_layer_on_the_configured_groups_only():
    fgs, *_ = _load(_project(_block()))
    assert fgs["pelagic_fish"].params["temperature"] == {
        "layer": "upper", "monthly_c": LAYER}
    assert fgs["gadoids"].params["temperature"] == {
        "layer": "deep", "monthly_c": LAYER[::-1]}
    for fid in ("porpoises", "phytoplankton", "benthic_community"):
        assert "temperature" not in fgs[fid].params


def test_disabled_loader_adds_nothing():
    fgs, *_ = _load(_project(_block(enabled=False)))
    assert not any("temperature" in fg.params for fg in fgs.values())


def test_a_q10_group_without_a_layer_raises():
    library = yaml.safe_load(LIBRARY.read_text(encoding="utf-8-sig"))
    assert library["species_definitions"]["gadoids"].get("metabolism_q10")
    with pytest.raises(ValueError, match="gadoids"):
        _load(_project(_block(group_layers={"pelagic_fish": "upper",
                                            "zooplankton": "upper"})))


def test_a_q10_on_a_non_decision_maker_raises():
    library = yaml.safe_load(LIBRARY.read_text(encoding="utf-8-sig"))
    library["species_definitions"]["benthic_community"].update(
        metabolism_q10=2.0, metabolism_t_ref="annual_mean")
    block = _block()
    block["group_layers"]["benthic_community"] = "upper"
    with pytest.raises(ValueError, match="benthic_community"):
        load_project_config(str(PROJECT), library_config=library,
                            project_config=_project(block),
                            grid_size=(6, 6), seed=1)


def test_the_manifest_assigns_every_q10_group_a_layer():
    """mareld2.yaml must load with its own temperature block."""
    project = _manifest()
    if temperature.parse_settings(project) is None:
        pytest.skip("temperature disabled in mareld2.yaml")
    fgs, *_ = _load(project)
    for fid, fg in fgs.items():
        if temperature.species_settings(fg.params) is not None:
            assert fg.params.get("temperature"), fid


# ------------------------------------------------------------------ engine

def _rest_env(start_tick, q10=2.2, t_ref=14.0, with_temperature=True):
    """One resting DM ``p`` in one cell, 1 t, a full reserve.

    Resting costs ``resting_metabolism * resting_cost * m(t)`` per tonne.
    """
    calendar = {"latitude_deg": LATITUDE, "tick_hours": 6,
                "start_tick": start_tick, "random_start": False}
    params = dict(is_decision_maker=True, movement_speed=0.0,
                  max_energy_reserve=100.0, resting_metabolism=4.0,
                  resting_cost=1.5, feeding_cost=2.0, movement_cost=2.0,
                  growth_rate=0.0, maintenance_level=0.3,
                  max_carrying_capacity=1e6, energy_content=5.0,
                  max_intake_rate=0.0, menu=[], daylight=calendar,
                  metabolism_q10=q10, metabolism_t_ref=t_ref)
    params["interaction"] = {}
    if with_temperature:
        params["temperature"] = {"layer": "upper", "monthly_c": LAYER}
    fg = FunctionalGroup("p", params)
    fg.initialize_state((1, 1), initial_biomass=np.full((1, 1), 1.0))
    fg.energy_reserve[:] = 50.0
    return EcosystemEnvironment(dict(height=1, width=1), {"p": fg},
                                apply_natural_mortality=False)


def _resting_cost(env):
    actions = ActionProbabilities(
        move=np.zeros((1, 4, 1, 1), np.float32),
        rest=np.ones((1, 1, 1), np.float32),
        eat=np.zeros((1, 1, 1, 1), np.float32))
    env.fgs["p"].temp_energy_gains = np.zeros((1, 1))
    settlement = movement.apply_energy_costs(env, actions)
    return 50.0 - float(settlement.stationary_reserve.sum())


@pytest.mark.parametrize("start_tick", [0, 4 * 74 + 2, 4 * 250 + 1])
def test_resting_cost_scales_with_the_multiplier_of_the_tick(start_tick):
    env = _rest_env(start_tick)
    env.build_static_caches()
    assert env._has_temperature
    m = temperature.metabolism_multiplier(LAYER, 6, 2.2, 14.0)[start_tick]
    assert interactions.metabolism_temperature(env)[0] == pytest.approx(m)
    assert _resting_cost(env) == pytest.approx(4.0 * 1.5 * m, rel=1e-5)
    # The cached vector itself is never scaled in place.
    assert float(env.dm_resting_metabolism[0]) == 4.0


@pytest.mark.parametrize("q10,with_temperature", [(1.0, True), (2.2, False)])
def test_inert_settings_leave_the_cost_unchanged(q10, with_temperature):
    env = _rest_env(4 * 74, q10=q10, with_temperature=with_temperature)
    env.build_static_caches()
    assert not env._has_temperature
    assert env.metabolism_temp_table is None
    assert _resting_cost(env) == pytest.approx(4.0 * 1.5, rel=1e-6)


def test_temperature_without_the_calendar_raises():
    env = _rest_env(0)
    env.fgs["p"].params.pop("daylight")
    with pytest.raises(ValueError, match="daylight"):
        env.build_static_caches()


def test_temperature_adds_no_observation_channel():
    on = _load(_project(_block()))[0]
    off = _load(_project())[0]
    dims = []
    for fgs in (on, off):
        env = EcosystemEnvironment(dict(height=6, width=6), fgs)
        env.build_static_caches()
        dims.append(list(env.per_dm_in_dim))
    assert dims[0] == dims[1]
