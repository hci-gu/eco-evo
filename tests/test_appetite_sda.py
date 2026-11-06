"""Reserve-room appetite and intake-proportional SDA (section 240)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env import predation
from lib.environments.ecosystem_env.environment import EcosystemEnvironment
from lib.environments.ecosystem_env.state import ActionProbabilities
from lib.world import energy_balance, growth_budget
from lib.world.functional_group import FunctionalGroup

ROOT = Path(__file__).resolve().parents[1]
EC, ME, ASSIM = 4000.0, 1000.0, 0.8


def _env(appetite=None, sda=None, fill=0.0, prey=1.0, rate=0.5):
    groups = {}
    for fid, menu in (("p", ["q"]), ("q", [])):
        params = dict(is_decision_maker=True, movement_speed=0.0,
                      max_energy_reserve=ME, resting_metabolism=0.0,
                      growth_rate=0.0, maintenance_level=0.5,
                      energy_content=EC, max_intake_rate=rate, menu=menu)
        if fid == "p":
            if appetite:
                params["appetite"] = appetite
            if sda is not None:
                params["sda_fraction"] = sda
        params["interaction"] = ({"p_preys_on_q": dict(
            preys_on=True, handling_time=0.0, assimilation_factor=ASSIM)}
            if menu else {})
        fg = FunctionalGroup(fid, params)
        fg.initialize_state((1, 1), initial_biomass=np.full(
            (1, 1), 1.0 if fid == "p" else prey))
        groups[fid] = fg
    groups["p"].energy_reserve[:] = fill * ME
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


def test_appetite_form_parsing():
    assert energy_balance.appetite_form({}) == "gate"
    assert energy_balance.appetite_form({"appetite": "reserve_room"}) == "reserve_room"
    with pytest.raises(ValueError):
        energy_balance.appetite_form({"appetite": "lipostat"})
    assert energy_balance.sda_fraction({}) == 0.0
    with pytest.raises(ValueError):
        energy_balance.sda_fraction({"sda_fraction": 1.0})
    assert energy_balance.hunger_at(0.9, 0.5, "reserve_room") == 1.0
    assert energy_balance.hunger_at(0.9, 0.5) == pytest.approx(0.2)


def test_reserve_room_hunger_is_one_at_any_fill():
    env = _env("reserve_room", fill=0.9)
    assert np.allclose(env.fgs["p"].get_hunger(), 1.0)
    env = _env(None, fill=0.9)
    assert np.allclose(env.fgs["p"].get_hunger(), 0.2)


@pytest.mark.parametrize("fill", [0.0, 0.6, 0.95, 1.0])
def test_gain_is_capped_at_the_reserve_room(fill):
    """Plenty of food: the gain fills the store exactly, never beyond."""
    env = _env("reserve_room", fill=fill)
    eaten, gained = _eat(env)
    room = (1.0 - fill) * ME
    uncapped = 0.5 * ASSIM * EC            # rate x quality = 1600 > room
    assert gained == pytest.approx(min(room, uncapped), rel=1e-5, abs=1e-6)
    assert eaten == pytest.approx(gained / (ASSIM * EC), rel=1e-5, abs=1e-9)


def test_reserve_room_eats_freely_below_the_cap():
    """Room larger than the ration: the full ration is eaten (no gate)."""
    env = _env("reserve_room", fill=0.6, rate=0.05)
    _, gained = _eat(env)
    assert gained == pytest.approx(0.05 * ASSIM * EC, rel=1e-5)
    gate = _env(None, fill=0.6, rate=0.05)
    _, gated = _eat(gate)
    assert gated == pytest.approx(0.8 * gained, rel=1e-5)   # h(0.6) = 0.8


def test_sda_is_netted_from_the_gain():
    env = _env(None, sda=0.1, rate=0.05)
    i, j = env.dm_ids.index("p"), env.global_fg_order.index("q")
    assert float(env.energy_gain_mat[i, j]) == pytest.approx(EC * (ASSIM - 0.1))
    eaten, gained = _eat(env)
    assert gained == pytest.approx(eaten * EC * (ASSIM - 0.1), rel=1e-5)


def test_ideal_reserve_reserve_room():
    # Ample intake: the reserve stays full.
    assert growth_budget.ideal_reserve(0.5, 20.0, 4.0, 3600, 1800, 5e-4,
                                       appetite="reserve_room") == 1.0
    # Tight intake: interior root C = cost + P (ec + s ME).
    s = growth_budget.ideal_reserve(0.5, 6.5, 4.0, 3600, 1800, 5e-4,
                                    appetite="reserve_room")
    assert 0.5 < s < 1.0
    assert 6.5 == pytest.approx(4.0 + 5e-4 * (3600 + s * 1800))
    # Infeasible: P >= (C - cost) / (ec + u ME).
    assert growth_budget.ideal_reserve(0.5, 5.5, 4.0, 3600, 1800, 5e-4,
                                       appetite="reserve_room") is None


def test_library_appetite_and_benthivore_budget():
    lib = yaml.safe_load(open(ROOT / "fgconfig" / "fg_library.yaml"))
    sd, inter = lib["species_definitions"], lib["interaction_definitions"]
    forms = {fid: energy_balance.appetite_form(spec)
             for fid, spec in sd.items() if spec.get("is_decision_maker")}
    assert {f for f, a in forms.items() if a == "reserve_room"} == {
        "benthivorous_fish"}
    b = growth_budget.growth_budget("benthivorous_fish", sd, inter)
    assert b.reference_fill == pytest.approx(1.0)
    assert b.gate == pytest.approx(0.5)
    # C nets SDA: (0.8 - 0.10) x herring quality / h, below the 0.8 value.
    assert sd["benthivorous_fish"]["sda_fraction"] == pytest.approx(0.10)
    assert b.ideal_cost == pytest.approx(3.33 * 1.18)
