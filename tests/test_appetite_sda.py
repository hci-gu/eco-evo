"""Reserve-room appetite, intake-proportional SDA (section 240) and the
deposition-proportional growth cost (section 241)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env import population_change, predation
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
    # Section 244.10: every fish group; zooplankton and mammals keep the gate.
    fish = {"benthivorous_fish", "planktivorous_fish", "predatory_fish",
            "whiting", "mackerel"}
    assert {f for f, a in forms.items() if a == "reserve_room"} == fish
    for fid in fish:
        assert growth_budget.growth_budget(
            fid, sd, inter).reference_fill == pytest.approx(1.0)
    b = growth_budget.growth_budget("benthivorous_fish", sd, inter)
    assert b.reference_fill == pytest.approx(1.0)
    # Section 247: u 0.078 (the gadoid danger level), gate 1 - u.
    assert b.gate == pytest.approx(1.0 - 0.078)
    # C nets SDA: (0.8 - 0.10) x herring quality / h, below the 0.8 value.
    assert sd["benthivorous_fish"]["sda_fraction"] == pytest.approx(0.10)
    # Section 267: feeding_cost net of SDA and the growth cost's synthesis.
    assert b.ideal_cost == pytest.approx(3.33 * 1.02)
    # Section 241: the growth cost leaves s* (and g) where it was.
    assert sd["benthivorous_fish"]["growth_cost"] == pytest.approx(0.69)
    assert {fid for fid, spec in sd.items()
            if energy_balance.growth_cost(spec) > 0} == {"benthivorous_fish"}


def test_growth_cost_parsing_and_price():
    assert energy_balance.growth_cost({}) == 0.0
    assert energy_balance.growth_cost({"growth_cost": ""}) == 0.0
    assert energy_balance.growth_cost({"growth_cost": 0.69}) == 0.69
    for bad in (-0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            energy_balance.growth_cost({"growth_cost": bad})
    assert energy_balance.tonne_price(3600.0, 0.0, 0.7, 1800.0) == 3600.0
    assert energy_balance.tonne_price(3600.0, 0.5, 1.0, 1800.0) == pytest.approx(
        3600.0 + 0.5 * 5400.0)


def _grower(cost, fill=0.9, b=2.0, g=0.01, u=0.5):
    params = dict(is_decision_maker=True, movement_speed=0.0,
                  max_energy_reserve=ME, resting_metabolism=0.0,
                  growth_rate=g, starve_rate=g, maintenance_level=u,
                  energy_content=EC, max_intake_rate=0.0, menu=[],
                  interaction={})
    if cost:
        params["growth_cost"] = cost
    fg = FunctionalGroup("p", params)
    fg.initialize_state((1, 1), initial_biomass=np.full((1, 1), b))
    fg.energy_reserve[:] = fill * ME * b
    env = EcosystemEnvironment(dict(height=1, width=1), {"p": fg},
                               apply_natural_mortality=False)
    assert env.mass_balance
    return env, fg


@pytest.mark.parametrize("cost", [0.0, 0.5, 1.2])
def test_growth_cost_debits_the_reserve(cost):
    """The wish B g (s - u) is bought at ec + c (ec + s ME) per tonne."""
    fill, b, g = 0.9, 2.0, 0.01
    env, fg = _grower(cost, fill, b, g)
    r0 = float(fg.energy_reserve.sum())
    population_change._apply_decision_maker_population_change(env, "p", fg)
    db = float(fg.biomass.sum()) - b
    # float32 state: a 0.4 % step on B keeps ~4 digits of the difference.
    assert db == pytest.approx(b * g * (fill - 0.5), rel=1e-4)
    price = EC + cost * (EC + fill * ME)
    assert r0 - float(fg.energy_reserve.sum()) == pytest.approx(db * price, rel=1e-4)


def test_growth_cost_caps_at_the_reserve_above_maintenance():
    """A large wish is cut where the reserve above u runs out at the price."""
    fill, b, cost = 0.6, 1.0, 1.0
    env, fg = _grower(cost, fill, b, g=10.0)
    population_change._apply_decision_maker_population_change(env, "p", fg)
    price = EC + cost * (EC + fill * ME)
    assert float(fg.biomass.sum()) - b == pytest.approx(
        (fill - 0.5) * ME * b / price, rel=1e-4)
    assert float(fg.energy_reserve.sum()) == pytest.approx(0.5 * ME * b, rel=1e-4)


def test_ideal_reserve_with_growth_cost():
    """(1 + c) P replaces P in both branches."""
    for appetite in ("gate", "reserve_room"):
        with_cost = growth_budget.ideal_reserve(0.5, 8.0, 4.0, 3600, 1800,
                                                4e-4, appetite, growth_cost=0.25)
        scaled = growth_budget.ideal_reserve(0.5, 8.0, 4.0, 3600, 1800,
                                             5e-4, appetite)
        assert with_cost == pytest.approx(scaled)
    s = growth_budget.ideal_reserve(0.5, 6.5, 4.0, 3600, 1800, 4e-4,
                                    "reserve_room", growth_cost=0.25)
    assert 6.5 == pytest.approx(4.0 + 1.25 * 4e-4 * (3600 + s * 1800))
