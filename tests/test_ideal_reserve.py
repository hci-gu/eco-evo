"""Section 160: the growth budget is paid at the ideal-conditions reserve s*.

Under ideal conditions (unlimited best prey, eating every tick at the
Holling ceiling) a DM's reserve settles where h(s) C = cost + P (ec + s ME);
growth_budget picks g so that the production g (s* - u) there equals
r_max + M1 + sum M2. The engine test below runs the real tick and checks
both the reserve and the production it reaches.
"""
import math
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.environments.ecosystem import EcosystemEnvironment  # noqa: E402
from lib.environments.ecosystem_env.state import ActionProbabilities  # noqa: E402
from lib.world.energy_balance import hunger_at  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402
from lib.world.growth_budget import (  # noqa: E402
    TICKS_PER_YEAR, derived_growth_rate, growth_budget, ideal_reserve)


def _toy(r_max=0.6, m1=0.4, handling=50.0):
    species = {
        "pred": {"is_decision_maker": True, "r_max": r_max,
                 "natural_mortality": 1.0 - math.exp(-m1 / TICKS_PER_YEAR),
                 "maintenance_level": 0.5, "max_energy_reserve": 2000.0,
                 "energy_content": 5000.0, "resting_metabolism": 5.0,
                 "feeding_cost": 2.0, "max_intake_rate": 0.05},
        "fox": {"is_decision_maker": True},
        "food": {"is_decision_maker": False, "energy_content": 4000.0},
        "junk": {"is_decision_maker": False, "energy_content": 1000.0},
    }
    inter = {
        "pred_preys_on_food": {"preys_on": True, "assimilation_factor": 0.8,
                               "handling_time": handling},
        "pred_preys_on_junk": {"preys_on": True, "assimilation_factor": 0.8,
                               "handling_time": handling},
        "fox_preys_on_pred": {"preys_on": True, "predation_mortality": 0.3},
    }
    return species, inter


def test_production_at_s_star_is_the_gross_budget():
    species, inter = _toy()
    b = growth_budget("pred", species, inter)
    assert b.best_prey == "food"
    assert b.ideal_intake == pytest.approx(0.8 * 4000.0 / 50.0)
    assert b.ideal_cost == pytest.approx(10.0)
    assert 0.5 < b.reference_fill < 1.0
    assert b.gate == pytest.approx(b.reference_fill - 0.5)
    assert b.growth_rate * b.gate * TICKS_PER_YEAR == pytest.approx(b.gross_annual)
    # s* is the steady reserve: intake at h(s*) pays cost, debit and dilution.
    s = b.reference_fill
    production = b.gross_annual / TICKS_PER_YEAR
    assert hunger_at(s, 0.5) * b.ideal_intake - b.ideal_cost == pytest.approx(
        production * (5000.0 + s * 2000.0))


def test_g_is_above_the_full_reserve_value():
    species, inter = _toy()
    b = growth_budget("pred", species, inter)
    full = b.gross_annual / (0.5 * TICKS_PER_YEAR)
    assert b.growth_rate > full


def test_net_low_density_growth_is_still_r_max():
    species, inter = _toy()
    for checked in (False, True):
        inter["fox_preys_on_pred"]["preys_on"] = checked
        b = growth_budget("pred", species, inter)
        net = b.growth_rate * b.gate * TICKS_PER_YEAR - b.m1_annual - b.predation_total
        assert net == pytest.approx(0.6)


def test_unchecked_or_inactive_prey_is_not_the_best_prey():
    species, inter = _toy()
    inter["pred_preys_on_food"]["preys_on"] = False
    assert growth_budget("pred", species, inter).best_prey == "junk"
    inter["pred_preys_on_food"]["preys_on"] = True
    b = growth_budget("pred", species, inter,
                      active_predators=["pred", "junk", "fox"])
    assert b.best_prey == "junk"


def test_budget_above_the_energy_is_an_error():
    species, inter = _toy(r_max=40.0)
    b = growth_budget("pred", species, inter)
    assert b.error and "Pmax" in b.error and b.growth_rate is None
    assert derived_growth_rate("pred", species, inter) is None
    assert ideal_reserve(0.5, 64.0, 10.0, 5000.0, 2000.0, 1.0) is None


def test_g_above_the_mass_balance_ceiling_is_an_error():
    species, inter = _toy(r_max=5.0, handling=5.0)
    species["pred"]["max_energy_reserve"] = 20.0    # ME/ec 0.004
    b = growth_budget("pred", species, inter)
    assert b.error and "mass-balance" in b.error


def test_missing_energy_data_falls_back_to_a_full_reserve():
    species, inter = _toy()
    del species["pred"]["energy_content"]
    b = growth_budget("pred", species, inter)
    assert b.reference_fill is None and b.gate == pytest.approx(0.5)


def test_engine_settles_at_s_star_and_produces_the_budget():
    """Real tick, one cell, unlimited prey, eating every tick."""
    species, inter = _toy()
    b = growth_budget("pred", species, inter)
    groups = {}
    pred = dict(species["pred"], growth_rate=b.growth_rate, movement_speed=0.0,
                resting_cost=1.0, starve_rate=0.0, natural_mortality=0.0,
                menu=["food"], observes=["food"],
                interaction={"pred_preys_on_food": inter["pred_preys_on_food"]})
    pred.pop("r_max")
    food = dict(species["food"], is_decision_maker=False, growth_rate=0.0,
                max_carrying_capacity=1e9, menu=[])
    for fid, params in (("food", food), ("pred", pred)):
        fg = FunctionalGroup(fid, params)
        fg.initialize_state((1, 1), initial_biomass=np.full((1, 1), 1e6 if fid == "food" else 1.0))
        groups[fid] = fg
    groups["pred"].energy_reserve[:] = 0.6 * 2000.0
    env = EcosystemEnvironment(dict(height=1, width=1), groups,
                               apply_natural_mortality=False, mass_balance=True)
    j = env.global_fg_order.index("food")
    eat = np.zeros((1, env.N_all, 1, 1), np.float32)
    eat[0, j] = 1.0
    actions = ActionProbabilities(np.zeros((1, 4, 1, 1), np.float32),
                                  np.zeros((1, 1, 1), np.float32), eat)
    for _ in range(4000):
        groups["food"].biomass[:] = 1e6
        env.step(actions)
    p = groups["pred"]
    assert float(p.energy_level[0, 0]) == pytest.approx(b.reference_fill, abs=2e-3)
    b0 = float(p.biomass[0, 0])
    for _ in range(100):
        groups["food"].biomass[:] = 1e6
        env.step(actions)
    rate = math.log(float(p.biomass[0, 0]) / b0) / 100 * TICKS_PER_YEAR
    assert rate == pytest.approx(b.gross_annual, rel=0.02)
