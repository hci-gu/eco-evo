"""Fixed attack choice: p = 1 on every prey of the menu (section 265)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env import interactions, predation
from lib.environments.ecosystem_env.environment import EcosystemEnvironment
from lib.environments.ecosystem_env.state import ActionProbabilities
from lib.world.functional_group import FunctionalGroup, attack_choice_form

A_Q, A_R, H_Q, H_R = 0.4, 0.2, 0.5, 1.0


def _env(attack_choice=None, b_q=2.0, b_r=3.0):
    pred = dict(is_decision_maker=True, movement_speed=0.0,
                max_energy_reserve=1000.0, resting_metabolism=0.0,
                growth_rate=0.0, maintenance_level=0.5, energy_content=4000.0,
                max_intake_rate=0.1, menu=["q", "r"],
                interaction={
                    "p_preys_on_q": dict(preys_on=True, handling_time=H_Q,
                                         assimilation_factor=0.8,
                                         max_intake_rate=A_Q),
                    "p_preys_on_r": dict(preys_on=True, handling_time=H_R,
                                         assimilation_factor=0.8,
                                         max_intake_rate=A_R)})
    if attack_choice is not None:
        pred["attack_choice"] = attack_choice
    groups = {"p": FunctionalGroup("p", pred)}
    for fid, ec in (("q", 2000.0), ("r", 4000.0)):
        groups[fid] = FunctionalGroup(fid, dict(
            is_decision_maker=False, energy_content=ec, menu=[],
            interaction={}, max_carrying_capacity=100.0, growth_rate=0.0))
    for fid, b in (("p", 1.0), ("q", b_q), ("r", b_r)):
        groups[fid].initialize_state((1, 1), initial_biomass=np.full((1, 1), b))
    groups["p"].energy_reserve[:] = 0.0
    return EcosystemEnvironment(dict(height=1, width=1), groups,
                                apply_natural_mortality=False)


def _eat(env, e_q, e_r):
    order = env.global_fg_order
    eat = np.zeros((1, len(order), 1, 1), np.float32)
    eat[0, order.index("q")] = e_q
    eat[0, order.index("r")] = e_r
    actions = ActionProbabilities(
        move=np.zeros((1, 4, 1, 1), np.float32),
        rest=np.zeros((1, 1, 1), np.float32), eat=eat)
    before = {f: float(env.fgs[f].biomass.sum()) for f in ("q", "r")}
    predation.apply_predation(env, actions)
    return {f: before[f] - float(env.fgs[f].biomass.sum()) for f in ("q", "r")}


def test_form_parsing():
    assert attack_choice_form({}) == "learned"
    assert attack_choice_form({"attack_choice": ""}) == "learned"
    assert attack_choice_form({"attack_choice": "Fixed"}) == "fixed"
    with pytest.raises(ValueError):
        attack_choice_form({"attack_choice": "greedy"})


def test_default_is_learned_and_inert():
    env = _env()
    assert not env._has_fixed_attack and not env.dm_fixed_attack.any()
    attack = np.array([[[[0.0]], [[0.4]], [[1.0]]]], np.float32)
    assert interactions.fixed_attack(env, attack) is attack


def test_fixed_attack_sets_p_one_on_the_menu_only():
    env = _env("fixed")
    assert env._has_fixed_attack
    order = env.global_fg_order
    attack = np.zeros((1, len(order), 1, 1), np.float32)
    out = interactions.fixed_attack(env, attack)
    for j, fid in enumerate(order):
        assert out[0, j, 0, 0] == (1.0 if fid in ("q", "r") else 0.0), fid


@pytest.mark.parametrize("split", [(0.6, 0.0), (0.45, 0.15), (0.1, 0.5)])
def test_only_the_foraging_time_matters(split):
    """With p fixed every split of the same T takes the encounter diet."""
    b_q, b_r, t = 2.0, 3.0, 0.6
    taken = _eat(_env("fixed", b_q, b_r), *split)
    lam_q, lam_r = A_Q * b_q, A_R * b_r
    busy = 1.0 + H_Q * lam_q + H_R * lam_r
    assert taken["q"] == pytest.approx(t * lam_q / busy, rel=1e-5)
    assert taken["r"] == pytest.approx(t * lam_r / busy, rel=1e-5)


def test_learned_corner_still_skips_the_other_prey():
    taken = _eat(_env(), 0.6, 0.0)
    assert taken["r"] == 0.0 and taken["q"] > 0.0


def test_no_foraging_no_intake():
    taken = _eat(_env("fixed"), 0.0, 0.0)
    assert taken["q"] == 0.0 and taken["r"] == 0.0
