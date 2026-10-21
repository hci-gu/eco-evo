"""A non decision maker's own losses M1 (section 173).

Contracts asserted:

1. The growth budget counts an NDM's M1: g = (r_max + M1 + sum M2) /
   1460, and leaves it out with --mortality off; without
   ``natural_mortality`` the budget is exactly the old one.
2. The tick removes M1 of the biomass before growth, scaled by
   --mortality_multiplier, only with --mortality on, books it as natural
   mortality, and leaves a producer's nutrient pool alone.

GPU parity: tests/test_gpu_ecosystem.py (the fixture's NDMs carry
natural_mortality 0.1 and the mortality fixtures compare both engines).
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
from lib.environments.ecosystem_env import population_change  # noqa: E402
from lib.world import growth_budget as gb  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402

PHYTO = {"is_decision_maker": False, "r_max": 75.2, "growth_rate": 0.067}
ZOO = {"is_decision_maker": True}
INTERACTIONS = {"zoo_preys_on_phyto": {"preys_on": True,
                                       "predation_mortality": 22.63}}


def _defs(**phyto):
    return {"phyto": dict(PHYTO, **phyto), "zoo": dict(ZOO)}


def test_budget_counts_an_ndm_m1():
    plain = gb.derived_growth_rate("phyto", _defs(), INTERACTIONS)
    assert plain == pytest.approx((75.2 + 22.63) / gb.TICKS_PER_YEAR)
    with_m1 = gb.derived_growth_rate("phyto", _defs(natural_mortality=0.01),
                                     INTERACTIONS)
    m1 = -gb.TICKS_PER_YEAR * math.log(1.0 - 0.01)
    assert with_m1 == pytest.approx((75.2 + m1 + 22.63) / gb.TICKS_PER_YEAR)
    off = gb.derived_growth_rate("phyto", _defs(natural_mortality=0.01),
                                 INTERACTIONS, include_m1=False)
    assert off == pytest.approx(plain)
    budget = gb.growth_budget("phyto", _defs(natural_mortality=0.01),
                              INTERACTIONS)
    assert "M1" in budget.describe()
    assert "M1" not in gb.growth_budget("phyto", _defs(),
                                        INTERACTIONS).describe()


def _env(mortality=True, multiplier=1.0, rate=0.05):
    shape = (3, 3)
    params = dict(is_decision_maker=False, movement_speed=0.0,
                  growth_rate=0.0, max_carrying_capacity=50.0,
                  natural_mortality=rate, energy_content=4800.0,
                  min_split_biomass=0.0, extinction_threshold_factor=0.0)
    fg = FunctionalGroup("phyto", params)
    fg.initialize_state(shape, initial_biomass=np.full(shape, 9.0))
    fg.energy_reserve[...] = 7.0
    return EcosystemEnvironment(dict(height=3, width=3), {"phyto": fg},
                                apply_natural_mortality=mortality,
                                mortality_multiplier=multiplier)


@pytest.mark.parametrize("mortality,multiplier,keep", [
    (True, 1.0, 0.95), (True, 2.0, 0.90), (False, 1.0, 1.0)])
def test_tick_removes_m1_before_growth(mortality, multiplier, keep):
    env = _env(mortality, multiplier)
    fg = env.fgs["phyto"]
    b0 = float(fg.biomass.sum())
    population_change._apply_non_decision_maker_population_change(
        env, "phyto", fg)
    # growth_rate 0: only the loss acts.
    assert float(fg.biomass.sum()) == pytest.approx(b0 * keep, rel=1e-6)
    assert env.loss_natural["phyto"] == pytest.approx(b0 * (1 - keep),
                                                      rel=1e-5, abs=1e-9)
    # The reserve slot (a nutrient pool for a producer) is not scaled.
    assert np.allclose(fg.energy_reserve, 7.0)
