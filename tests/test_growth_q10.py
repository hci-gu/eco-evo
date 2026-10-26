"""Q10 on a decision maker's growth rate (section 183).

Contracts asserted:

1. ``growth_q10`` parses like the other annual-mean Q10s: 0, 1 and
   empty are inert, a negative or non-numeric value raises.
2. The loader refuses it on a non decision maker and on a group without
   a temperature layer.
3. The growth branch is ``growth_rate * f(t)``, f the annual-mean-1 Q10
   multiplier of this tick; the starvation branch keeps the library rate
   (the unscaled growth_rate when starve_rate is 0); a DM without the key
   is untouched.

GPU parity: tests/test_gpu_ecosystem.py::test_growth_q10_matches_reference.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config.config_loader import load_project_config  # noqa: E402
from lib.environments.ecosystem_env import interactions, population_change  # noqa: E402
from lib.environments.ecosystem_env.environment import EcosystemEnvironment  # noqa: E402
from lib.world import temperature  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "mareld2.yaml"
LIBRARY = ROOT / "fgconfig" / "fg_library.yaml"
LAYER = [6.3, 5.4, 5.2, 6.1, 7.8, 9.9, 12.1, 13.4, 13.8, 12.8, 10.5, 8.4]


@pytest.mark.parametrize("params,expected", [
    ({}, None), ({"growth_q10": ""}, None), ({"growth_q10": 0.0}, None),
    ({"growth_q10": 1.0}, None), ({"growth_q10": 3.0}, 3.0),
])
def test_growth_q10_parses(params, expected):
    assert temperature.growth_q10(params) == expected


@pytest.mark.parametrize("raw", [-2.0, "warm"])
def test_invalid_growth_q10_raises(raw):
    with pytest.raises(ValueError, match="growth_q10"):
        temperature.growth_q10({"growth_q10": raw})


def test_loader_refuses_inconsistent_settings():
    project = yaml.safe_load(PROJECT.read_text(encoding="utf-8-sig"))
    library = yaml.safe_load(LIBRARY.read_text(encoding="utf-8-sig"))
    block = project["simulation_settings"]["temperature"]
    layers = dict(block["group_layers"])

    def load(lib, group_layers):
        block["group_layers"] = group_layers
        return load_project_config(str(PROJECT), library_config=lib,
                                   project_config=project, grid_size=(6, 6),
                                   seed=1)

    zoo = library["species_definitions"]["zooplankton"]
    zoo.update(growth_q10=3.0, metabolism_q10=None, ingestion_q10=None)
    with pytest.raises(ValueError, match="growth_q10"):
        load(library, {k: v for k, v in layers.items() if k != "zooplankton"})
    zoo["growth_q10"] = None
    library["species_definitions"]["benthic_infauna"]["growth_q10"] = 3.0
    with pytest.raises(ValueError, match="benthic_infauna"):
        load(library, dict(layers, benthic_infauna="mean_0_50m"))


def _env(start_tick, q10, reserve):
    """DMs ``p`` (with ``q10``) and ``o`` (without), reserve fill given."""
    calendar = {"latitude_deg": 58.15, "tick_hours": 6,
                "start_tick": start_tick, "random_start": False}
    groups = {}
    for fid in ("p", "o"):
        params = dict(is_decision_maker=True, movement_speed=0.0,
                      max_energy_reserve=100.0, resting_metabolism=1.0,
                      growth_rate=0.2, maintenance_level=0.3,
                      max_carrying_capacity=1e6, energy_content=5.0,
                      daylight=calendar)
        if fid == "p":
            params.update(growth_q10=q10, temperature={
                "layer": "upper", "monthly_c": LAYER})
        groups[fid] = FunctionalGroup(fid, params)
        groups[fid].initialize_state((1, 1), initial_biomass=np.full((1, 1), 2.0))
        groups[fid].energy_reserve[:] = np.float32(reserve * 100.0 * 2.0)
    env = EcosystemEnvironment(dict(height=1, width=1), groups,
                               apply_natural_mortality=False)
    env.build_static_caches()
    return env


def _delta(env, fid):
    fg = env.fgs[fid]
    before = float(fg.biomass.sum())
    population_change._apply_decision_maker_population_change(env, fid, fg)
    return float(fg.biomass.sum()) - before


@pytest.mark.parametrize("start_tick", [4 * 40 + 2, 4 * 250 + 1])
def test_growth_branch_scales_and_starvation_does_not(start_tick):
    m = temperature.metabolism_multiplier(LAYER, 6, 3.0, "annual_mean")
    env = _env(start_tick, 3.0, 0.8)
    assert env._has_growth_temperature
    i_p = env.dm_ids.index("p")
    assert interactions.growth_temperature(env, "p") == pytest.approx(
        m[start_tick], rel=1e-6)
    assert interactions.growth_temperature(env, "o") == 1.0
    assert env.growth_temp_table[start_tick, i_p] == pytest.approx(
        m[start_tick], rel=1e-6)
    hot, flat = _env(start_tick, 3.0, 0.8), _env(start_tick, 1.0, 0.8)
    assert not flat._has_growth_temperature
    assert _delta(hot, "p") == pytest.approx(m[start_tick] * _delta(flat, "p"),
                                             rel=1e-5)
    assert _delta(hot, "o") == pytest.approx(_delta(flat, "o"), rel=1e-6)
    # Below the maintenance line the starvation loss is the library one.
    starving_hot, starving_flat = _env(start_tick, 3.0, 0.1), _env(start_tick, 1.0, 0.1)
    loss = _delta(starving_flat, "p")
    assert loss < 0.0
    assert _delta(starving_hot, "p") == pytest.approx(loss, rel=1e-6)
