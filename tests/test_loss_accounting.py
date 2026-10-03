"""Natural mortality (M1) is booked as a loss cause (section 132).

Before section 132 the loss breakdown counted starvation, predation and
impacts only, so a group whose dominant loss was M1 was shown as
"100 % predation".
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.environments.ecosystem import EcosystemEnvironment  # noqa: E402
from lib.environments.ecosystem_env import population_change  # noqa: E402
from lib.environments.ecosystem_env.loss_accounting import loss_shares  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402

GRID_CONFIG = {'width': 3, 'height': 2, 'cell_size': 1000.0, 'tick_duration': 6.0}


def _env(rate, mortality=True):
    fg = FunctionalGroup('dm', {
        'is_decision_maker': True, 'max_energy_reserve': 1000.0,
        'energy_content': 1000.0, 'resting_metabolism': 0.0,
        'maintenance_level': 0.0, 'movement_speed': 0.0, 'growth_rate': 0.0,
        'starve_rate': 0.0, 'natural_mortality': rate,
        'min_split_biomass': 0.0, 'extinction_threshold_factor': 0.0,
        'menu': []})
    fg.initialize_state((2, 3), initial_biomass=np.full((2, 3), 10.0))
    env = EcosystemEnvironment(GRID_CONFIG, {'dm': fg},
                               apply_natural_mortality=mortality)
    return env, fg


def test_m1_is_booked_as_natural_loss():
    env, fg = _env(0.01)
    before = float(fg.biomass.sum())
    population_change._apply_decision_maker_population_change(env, 'dm', fg)
    removed = before - float(fg.biomass.sum())
    assert env.loss_natural['dm'] == pytest.approx(removed, rel=1e-5)
    assert removed == pytest.approx(0.01 * before, rel=1e-5)
    shares = loss_shares(env, 'dm')
    assert shares['natural'] == pytest.approx(1.0)
    assert shares['total'] == pytest.approx(removed, rel=1e-5)


def test_no_natural_loss_when_mortality_is_off():
    env, fg = _env(0.01, mortality=False)
    population_change._apply_decision_maker_population_change(env, 'dm', fg)
    assert env.loss_natural['dm'] == 0.0
    assert loss_shares(env, 'dm')['total'] == 0.0


def test_shares_sum_to_one_over_all_four_causes():
    env, _fg = _env(0.0)
    env.loss_predation['dm'], env.loss_starvation['dm'] = 3.0, 1.0
    env.loss_impact['dm'], env.loss_natural['dm'] = 0.0, 4.0
    s = loss_shares(env, 'dm')
    assert s['natural'] == pytest.approx(0.5)
    assert s['predation'] == pytest.approx(0.375)
    assert sum(s[k] for k in ('starvation', 'predation', 'impact', 'natural')) \
        == pytest.approx(1.0)
