"""Prey reserve eaten with the prey (section 140)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env import predation
from lib.environments.ecosystem_env.environment import EcosystemEnvironment
from lib.environments.ecosystem_env.state import ActionProbabilities
from lib.world.functional_group import FunctionalGroup

EC, MAX_RESERVE, ASSIM = 5000.0, 4000.0, 0.8


def _env(flag, fill=0.5, prey_reserve_fraction=0.75):
    groups = {}
    for fid, menu in (("p", ["q"]), ("q", [])):
        params = dict(is_decision_maker=True, movement_speed=0.0,
                      max_energy_reserve=10.0 if fid == "p" else MAX_RESERVE,
                      resting_metabolism=0.0, growth_rate=0.0,
                      maintenance_level=0.0, energy_content=EC,
                      max_intake_rate=0.01, menu=menu)
        params["interaction"] = ({"p_preys_on_q": dict(
            preys_on=True, handling_time=0.0, assimilation_factor=ASSIM)}
            if menu else {})
        if fid == "q" and flag:
            params["prey_includes_reserve"] = True
            params["reserve_reference_fill"] = fill
        fg = FunctionalGroup(fid, params)
        fg.initialize_state((1, 1), initial_biomass=np.full((1, 1), 1.0))
        groups[fid] = fg
    groups["p"].energy_reserve[:] = 0.0            # full appetite
    groups["q"].energy_reserve[:] = prey_reserve_fraction * MAX_RESERVE
    return EcosystemEnvironment(dict(height=1, width=1), groups,
                                apply_natural_mortality=False)


def _eat(env):
    actions = ActionProbabilities(
        move=np.zeros((2, 4, 1, 1), np.float32),
        rest=np.zeros((2, 1, 1), np.float32),
        eat=np.array([[[[0.0]], [[1.0]]], [[[0.0]], [[0.0]]]], np.float32))
    before = float(env.fgs["q"].biomass.sum())
    predation.apply_predation(env, actions)
    eaten = before - float(env.fgs["q"].biomass.sum())
    return eaten, float(env.fgs["p"].temp_energy_gains.sum())


def test_static_quality_is_evaluated_at_the_reference_fill():
    env = _env(True, fill=0.25)
    i, j = env.dm_ids.index("p"), env.global_fg_order.index("q")
    assert float(env.energy_gain_mat[i, j]) == pytest.approx(
        (EC + 0.25 * MAX_RESERVE) * ASSIM)


@pytest.mark.parametrize("fraction", [0.0, 0.3, 1.0])
def test_gain_uses_the_reserve_the_eaten_prey_carries(fraction):
    env = _env(True, prey_reserve_fraction=fraction)
    eaten, gained = _eat(env)
    assert eaten > 0
    assert gained == pytest.approx(
        eaten * ASSIM * (EC + fraction * MAX_RESERVE), rel=1e-5)


def test_without_the_flag_only_energy_content_counts():
    env = _env(False, prey_reserve_fraction=1.0)
    assert not env._has_reserve_food
    eaten, gained = _eat(env)
    assert gained == pytest.approx(eaten * ASSIM * EC, rel=1e-6)
