"""Q10 on a decision maker's whole intake (section 181).

Contracts asserted:

1. ``ingestion_q10`` parses like the other Q10s: 0, 1 and empty are
   inert, a negative or non-numeric value raises.
2. The multiplier is normalised to an annual mean of exactly 1 and is
   the Q10 law between any two ticks.
3. The loader refuses it on a non decision maker and on a group without
   a temperature layer.
4. Predation scales the WHOLE response by f(t): with the prey far below
   and far above the half-saturation alike, the intake is f(t) times the
   intake without the term; a DM without the key is untouched.

GPU parity: tests/test_gpu_ecosystem.py::test_ingestion_q10_matches_reference.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config.config_loader import load_project_config  # noqa: E402
from lib.environments.ecosystem_env import interactions, predation  # noqa: E402
from lib.environments.ecosystem_env.environment import EcosystemEnvironment  # noqa: E402
from lib.environments.ecosystem_env.state import ActionProbabilities  # noqa: E402
from lib.world import temperature  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "mareld2.yaml"
LIBRARY = ROOT / "fgconfig" / "fg_library.yaml"
LAYER = [6.3, 5.4, 5.2, 6.1, 7.8, 9.9, 12.1, 13.4, 13.8, 12.8, 10.5, 8.4]


@pytest.mark.parametrize("params,expected", [
    ({}, None), ({"ingestion_q10": ""}, None), ({"ingestion_q10": 0.0}, None),
    ({"ingestion_q10": 1.0}, None), ({"ingestion_q10": 2.0}, 2.0),
])
def test_ingestion_q10_parses(params, expected):
    assert temperature.ingestion_q10(params) == expected


@pytest.mark.parametrize("raw", [-2.0, "warm"])
def test_invalid_ingestion_q10_raises(raw):
    with pytest.raises(ValueError, match="ingestion_q10"):
        temperature.ingestion_q10({"ingestion_q10": raw})


def _project(group_layers):
    project = yaml.safe_load(PROJECT.read_text(encoding="utf-8-sig"))
    block = project["simulation_settings"]["temperature"]
    block["group_layers"] = group_layers
    return project


def _library(**zooplankton):
    library = yaml.safe_load(LIBRARY.read_text(encoding="utf-8-sig"))
    library["species_definitions"]["zooplankton"].update(zooplankton)
    return library


def _load(library, project):
    return load_project_config(str(PROJECT), library_config=library,
                               project_config=project, grid_size=(6, 6),
                               seed=1)


def test_loader_refuses_inconsistent_settings():
    project = yaml.safe_load(PROJECT.read_text(encoding="utf-8-sig"))
    layers = dict(project["simulation_settings"]["temperature"]["group_layers"])
    # No layer for the zooplankton (and no metabolism Q10 either, so the
    # ingestion term is what the loader must catch).
    no_layer = {k: v for k, v in layers.items() if k != "zooplankton"}
    with pytest.raises(ValueError, match="ingestion_q10"):
        _load(_library(ingestion_q10=2.0, metabolism_q10=None),
              _project(no_layer))
    library = _library()
    library["species_definitions"]["benthic_community"]["ingestion_q10"] = 2.0
    with pytest.raises(ValueError, match="benthic_community"):
        _load(library, _project(dict(layers, benthic_community="mean_0_50m")))


def _env(start_tick, q10, prey):
    """DM ``p`` (with ``q10``) and DM ``o`` (without) both eat NDM ``q``."""
    calendar = {"latitude_deg": 58.15, "tick_hours": 6,
                "start_tick": start_tick, "random_start": False}
    groups = {}
    for fid in ("p", "o"):
        params = dict(is_decision_maker=True, movement_speed=0.0,
                      max_energy_reserve=100.0, resting_metabolism=1.0,
                      growth_rate=0.0, maintenance_level=0.3,
                      max_carrying_capacity=1e6, energy_content=5.0,
                      max_intake_rate=0.05, menu=["q"], daylight=calendar)
        params["interaction"] = {f"{fid}_preys_on_q": dict(
            preys_on=True, handling_time=4.0, assimilation_factor=0.8)}
        if fid == "p":
            params.update(ingestion_q10=q10, temperature={
                "layer": "upper", "monthly_c": LAYER})
        groups[fid] = FunctionalGroup(fid, params)
        groups[fid].initialize_state((1, 1), initial_biomass=np.full((1, 1), 1e-3))
        groups[fid].energy_reserve[:] = 0.0
    groups["q"] = FunctionalGroup("q", dict(
        is_decision_maker=False, movement_speed=0.0, growth_rate=0.0,
        max_carrying_capacity=1e6, energy_content=5.0, daylight=calendar))
    groups["q"].initialize_state((1, 1), initial_biomass=np.full((1, 1), prey))
    env = EcosystemEnvironment(dict(height=1, width=1), groups,
                               apply_natural_mortality=False)
    env.build_static_caches()
    return env


def _intake(env):
    n_dm, n_all = env.N_dm, len(env.global_fg_order)
    eat = np.zeros((n_dm, n_all, 1, 1), np.float32)
    eat[:, env.global_fg_order.index("q")] = 1.0
    actions = ActionProbabilities(
        move=np.zeros((n_dm, 4, 1, 1), np.float32),
        rest=np.zeros((n_dm, 1, 1), np.float32), eat=eat)
    predation.apply_predation(env, actions)
    return {fid: env.intake_by_pred_prey[fid]["q"] for fid in env.dm_ids}


@pytest.mark.parametrize("start_tick", [4 * 74 + 2, 4 * 250 + 1])
@pytest.mark.parametrize("prey", [0.05, 500.0])       # 1/(a h) = 5 t
def test_the_whole_response_scales_with_the_tick(start_tick, prey):
    m = temperature.metabolism_multiplier(LAYER, 6, 2.0, "annual_mean")
    assert float(m.mean()) == pytest.approx(1.0, abs=1e-12)
    temps = temperature.tick_temperatures(LAYER, 6)
    assert m[start_tick] / m[0] == pytest.approx(
        2.0 ** ((temps[start_tick] - temps[0]) / 10.0), rel=1e-9)
    hot = _intake(_env(start_tick, 2.0, prey))
    flat = _intake(_env(start_tick, 1.0, prey))
    env = _env(start_tick, 2.0, prey)
    assert env._has_ingestion_temperature
    i_p = env.dm_ids.index("p")
    assert interactions.ingestion_temperature(env)[i_p] == pytest.approx(
        m[start_tick], rel=1e-6)
    assert hot["p"] == pytest.approx(m[start_tick] * flat["p"], rel=1e-5)
    assert hot["o"] == pytest.approx(flat["o"], rel=1e-6)
    assert not _env(start_tick, 1.0, prey)._has_ingestion_temperature
