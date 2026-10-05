"""Exposure-weighted natural mortality M1 (section 139)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env import natural_mortality
from lib.environments.ecosystem_env.environment import EcosystemEnvironment
from lib.environments.ecosystem_env.state import ActionProbabilities
from lib.world import daylight
from lib.world.functional_group import FunctionalGroup

RATE = 0.0134        # zooplankton's per-tick M1
CALENDAR = {"latitude_deg": 58.15, "tick_hours": 6, "start_tick": 0,
            "random_start": False}


def _env(exposure=None, calendar=None, floor=0.2):
    groups = {}
    for fid, dm in (("z", True), ("p", False)):
        params = dict(is_decision_maker=dm, movement_speed=0.0,
                      max_energy_reserve=10.0, resting_metabolism=0.0,
                      growth_rate=0.0, maintenance_level=0.3,
                      natural_mortality=RATE if dm else 0.0,
                      visibility_floor=floor, max_carrying_capacity=100.0,
                      energy_content=5.0, menu=[])
        if dm and exposure:
            params.update(exposure)
        if calendar:
            params["daylight"] = dict(calendar)
        fg = FunctionalGroup(fid, params)
        fg.initialize_state((2, 2), initial_biomass=np.full((2, 2), 10.0))
        groups[fid] = fg
    env = EcosystemEnvironment(dict(height=2, width=2), groups,
                               apply_natural_mortality=True)
    env.ordered_fg_ids = list(env.fgs)
    return env


def _set_hiding(env, pi):
    hidden = np.full((env.N_dm, env.H, env.W), pi, dtype=np.float32)
    env.publish_action_probabilities(ActionProbabilities(
        move=np.zeros((env.N_dm, 4, env.H, env.W), np.float32),
        rest=hidden,
        eat=np.zeros((env.N_dm, env.N_all, env.H, env.W), np.float32)))


SPLIT = {"m1_visual_share": 0.1, "m1_tactile_share": 0.5,
         "depth_risk_ratio": 2.0}


def test_without_shares_the_scalar_path_is_kept():
    env = _env()
    assert not env._has_m1_exposure
    assert natural_mortality.components(env, "z", RATE) is None


@pytest.mark.parametrize("bad", [
    {"m1_visual_share": 0.7, "m1_tactile_share": 0.5},
    {"m1_visual_share": -0.1},
    {"m1_tactile_share": 0.5, "depth_risk_ratio": -1.0},
    {"m1_tactile_share": 0.5, "hide_reference": 1.5},
])
def test_invalid_settings_raise(bad):
    with pytest.raises(ValueError):
        natural_mortality.settings(bad)


def test_editor_zeros_mean_defaults():
    cfg = natural_mortality.settings({"m1_tactile_share": 0.5,
                                      "depth_risk_ratio": 0.0,
                                      "hide_reference": 0.0})
    assert cfg["rho"] == 1.0
    assert cfg["reference"] == natural_mortality.DEFAULT_HIDE_REFERENCE


def test_reference_hiding_without_calendar_gives_the_library_m1():
    env = _env(SPLIT)
    _set_hiding(env, 0.5)
    parts = natural_mortality.components(env, "z", RATE)
    np.testing.assert_allclose(sum(parts), RATE, rtol=1e-12)


def test_reference_hiding_averages_to_the_library_m1_over_the_year():
    """Visual part follows the light; its annual mean stays exact."""
    env = _env(SPLIT, calendar=CALENDAR)
    _set_hiding(env, 0.5)
    rates = []
    for tick in range(env.light_period):
        env.tick_count = tick
        rates.append(float(sum(natural_mortality.components(
            env, "z", RATE))[0, 0]))
    assert np.mean(rates) == pytest.approx(RATE, rel=1e-9)
    assert min(rates) < RATE < max(rates)        # day vs night


def test_hiding_trades_visual_for_tactile_risk():
    env = _env(SPLIT)
    out = {}
    for pi in (0.0, 0.5, 1.0):
        _set_hiding(env, pi)
        visual, tactile, other = natural_mortality.components(env, "z", RATE)
        out[pi] = (float(visual[0, 0]), float(tactile[0, 0]),
                   float(other[0, 0]))
    assert out[0.0][0] > out[0.5][0] > out[1.0][0]      # visual falls
    assert out[0.0][1] < out[0.5][1] < out[1.0][1]      # tactile rises
    assert out[0.0][2] == out[1.0][2]                   # rest constant
    # Full hiding leaves the floor's share of the visual exposure.
    assert out[1.0][0] / out[0.0][0] == pytest.approx(0.2)
    # Tactile: depth twice as dangerous as the surface.
    assert out[1.0][1] / out[0.0][1] == pytest.approx(2.0)


def test_rho_one_makes_tactile_risk_independent_of_hiding():
    env = _env(dict(SPLIT, depth_risk_ratio=1.0))
    tactile = []
    for pi in (0.0, 1.0):
        _set_hiding(env, pi)
        tactile.append(float(natural_mortality.components(
            env, "z", RATE)[1][0, 0]))
    assert tactile[0] == pytest.approx(tactile[1])


@pytest.mark.parametrize("pi", [0.0, 0.8])
def test_tick_applies_and_books_the_split(pi):
    env = _env(SPLIT, calendar=CALENDAR)
    _set_hiding(env, pi)
    expected = sum(natural_mortality.components(env, "z", RATE))[0, 0]
    env._apply_growth()
    after = float(env.fgs["z"].biomass[0, 0])
    assert after == pytest.approx(10.0 * (1.0 - expected), rel=1e-6)
    parts = env.loss_natural_parts["z"]
    assert sum(parts.values()) == pytest.approx(env.loss_natural["z"],
                                                rel=1e-5)
    assert parts["tactile"] > 0 and parts["other"] > 0


def test_visual_light_uses_the_daylight_multiplier():
    env = _env(dict(SPLIT, m1_visual_share=0.4, m1_tactile_share=0.0),
               calendar=CALENDAR)
    table = daylight.attack_multiplier(58.15, 6, 0.1,
                                       daylight.DEFAULT_THRESHOLD_DEG)
    _set_hiding(env, 0.5)
    for tick in (0, 1, 2):
        env.tick_count = tick
        visual = float(natural_mortality.components(env, "z", RATE)[0][0, 0])
        assert visual == pytest.approx(RATE * 0.4 * table[tick], rel=1e-9)
