"""Shared handling: the multi-prey functional response (section 242)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env import interactions, predation
from lib.environments.ecosystem_env.environment import EcosystemEnvironment
from lib.environments.ecosystem_env.state import ActionProbabilities
from lib.world.functional_group import FunctionalGroup

A_Q, A_R, H_Q, H_R = 0.4, 0.2, 0.5, 1.0
EC_Q, EC_R, ASSIM = 2000.0, 4000.0, 0.8     # E/h equal: 4000 per unit h


def _env(b_q=2.0, b_r=3.0, shared=True, response=None):
    groups = {}
    pred = dict(is_decision_maker=True, movement_speed=0.0,
                max_energy_reserve=1000.0, resting_metabolism=0.0,
                growth_rate=0.0, maintenance_level=0.5, energy_content=4000.0,
                max_intake_rate=0.1, menu=["q", "r"],
                interaction={
                    "p_preys_on_q": dict(preys_on=True, handling_time=H_Q,
                                         assimilation_factor=ASSIM,
                                         max_intake_rate=A_Q),
                    "p_preys_on_r": dict(preys_on=True, handling_time=H_R,
                                         assimilation_factor=ASSIM,
                                         max_intake_rate=A_R)})
    for inter_id, kind in (response or {}).items():
        pred["interaction"][inter_id]["functional_response"] = kind
    groups["p"] = FunctionalGroup("p", pred)
    for fid, ec in (("q", EC_Q), ("r", EC_R)):
        groups[fid] = FunctionalGroup(fid, dict(
            is_decision_maker=False, energy_content=ec, menu=[],
            interaction={}, max_carrying_capacity=100.0, growth_rate=0.0))
    for fid, b in (("p", 1.0), ("q", b_q), ("r", b_r)):
        groups[fid].initialize_state((1, 1), initial_biomass=np.full((1, 1), b))
    groups["p"].energy_reserve[:] = 0.0          # hunger 1
    return EcosystemEnvironment(dict(height=1, width=1), groups,
                                apply_natural_mortality=False,
                                shared_handling=shared)


def _eat(env, e_q, e_r):
    order = env.global_fg_order                   # ['p', 'q', 'r']
    eat = np.zeros((1, len(order), 1, 1), np.float32)
    eat[0, order.index("q")] = e_q
    eat[0, order.index("r")] = e_r
    actions = ActionProbabilities(
        move=np.zeros((1, 4, 1, 1), np.float32),
        rest=np.zeros((1, 1, 1), np.float32), eat=eat)
    before = {f: float(env.fgs[f].biomass.sum()) for f in ("q", "r")}
    predation.apply_predation(env, actions)
    taken = {f: before[f] - float(env.fgs[f].biomass.sum()) for f in ("q", "r")}
    return taken, float(env.fgs["p"].temp_energy_gains.sum())


def test_attack_choice():
    eat = np.array([[[[0.2]], [[0.5]], [[0.0]]]], np.float32)
    forage, attack = interactions.attack_choice(eat)
    assert forage.shape == (1, 1, 1, 1)
    assert forage.item() == pytest.approx(0.7)
    np.testing.assert_allclose(attack.ravel(), [0.4, 1.0, 0.0], rtol=1e-6)
    forage, attack = interactions.attack_choice(np.zeros_like(eat))
    assert forage.item() == 0.0 and not attack.any()


@pytest.mark.parametrize("m3", [0.0, 1.0])
@pytest.mark.parametrize("w", [0.0, 0.7])
def test_one_attacked_prey_is_the_per_pair_response(m3, w):
    a = np.array([[[[0.4]], [[0.2]]]], np.float32)
    h = np.array([[[[0.5]], [[1.0]]]], np.float32)
    bp = np.array([[[[2.0]], [[3.0]]]], np.float32)
    mask = np.full_like(a, m3)
    attack = np.array([[[[1.0]], [[0.0]]]], np.float32)
    shared = interactions.multi_prey_rate(a, h, bp, mask, attack, w)
    single = interactions.holling_a_eff(a, h, bp, mask, w)
    assert shared[0, 0].item() == pytest.approx(single[0, 0].item(), rel=1e-6)


@pytest.mark.parametrize("response", [None, {"p_preys_on_q": 3}])
def test_corner_policy_matches_the_pre_242_engine(response):
    """All eat on one prey: the engine takes exactly what it took before."""
    new, gain = _eat(_env(response=response), 0.6, 0.0)
    old, gain_old = _eat(_env(shared=False, response=response), 0.6, 0.0)
    assert new["q"] == pytest.approx(old["q"], rel=1e-6)
    assert new["r"] == old["r"] == 0.0
    assert gain == pytest.approx(gain_old, rel=1e-6)


def test_equal_shares_give_the_encounter_diet():
    """p = 1 on both: diet shares a_j B_j, one handling budget."""
    b_q, b_r, t = 2.0, 3.0, 0.6
    taken, _ = _eat(_env(b_q, b_r), t / 2, t / 2)
    lam_q, lam_r = A_Q * b_q, A_R * b_r
    busy = 1.0 + H_Q * lam_q + H_R * lam_r
    assert taken["q"] == pytest.approx(t * lam_q / busy, rel=1e-5)
    assert taken["r"] == pytest.approx(t * lam_r / busy, rel=1e-5)
    assert taken["q"] / taken["r"] == pytest.approx(lam_q / lam_r, rel=1e-5)


def test_including_an_equally_profitable_prey_pays():
    """Same foraging time: attacking both beats either corner (E/h equal)."""
    corners = [_eat(_env(), 0.6, 0.0)[1], _eat(_env(), 0.0, 0.6)[1]]
    assert _eat(_env(), 0.3, 0.3)[1] > max(corners)
    # The per-pair engine had a corner as its optimum instead (239.1).
    old = [_eat(_env(shared=False), 0.6, 0.0)[1],
           _eat(_env(shared=False), 0.0, 0.6)[1]]
    assert _eat(_env(shared=False), 0.3, 0.3)[1] < max(old)


def test_dense_prey_slows_the_other():
    """An attacked prey's density raises everyone's handling time."""
    sparse, _ = _eat(_env(b_r=0.5), 0.3, 0.3)
    dense, _ = _eat(_env(b_r=20.0), 0.3, 0.3)
    assert dense["q"] < sparse["q"]
    old_sparse, _ = _eat(_env(b_r=0.5, shared=False), 0.3, 0.3)
    old_dense, _ = _eat(_env(b_r=20.0, shared=False), 0.3, 0.3)
    assert old_dense["q"] == pytest.approx(old_sparse["q"], rel=1e-6)


def test_attack_probability_below_one_dilutes_its_prey_only_partly():
    """p_r = 0.5: r is taken at half the per-encounter rate."""
    b_q, b_r = 2.0, 3.0
    taken, _ = _eat(_env(b_q, b_r), 0.4, 0.2)      # T 0.6, p (1, 0.5)
    lam_q, lam_r = A_Q * b_q, A_R * b_r
    busy = 1.0 + H_Q * lam_q + 0.5 * H_R * lam_r
    assert taken["q"] == pytest.approx(0.6 * lam_q / busy, rel=1e-5)
    assert taken["r"] == pytest.approx(0.6 * 0.5 * lam_r / busy, rel=1e-5)
