import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.environments.ecosystem import EcosystemEnvironment  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402


def _env_for_fg(params, energy_ratio):
    fg = FunctionalGroup("consumer", {
        "is_decision_maker": True,
        "maintenance_level": 0.5,
        "max_energy_reserve": 100.0,
        "resting_metabolism": 0.0,
        "movement_speed": 0.0,
        "growth_rate": 0.01,
        **params,
    })
    fg.initialize_state(
        (1, 1),
        initial_biomass=np.array([[100.0]], dtype=np.float32),
        initial_energy_ratio=energy_ratio,
    )
    env = EcosystemEnvironment(
        {"width": 1, "height": 1, "cell_size": 1.0, "tick_duration": 6.0},
        {"consumer": fg},
        {},
        apply_natural_mortality=False,
    )
    env.ordered_fg_ids = ["consumer"]
    env.dtype = np.float32
    return env, fg


def test_starvation_mortality_full_deficit_reduces_biomass():
    env, fg = _env_for_fg({"starvation_mortality": 0.02}, energy_ratio=0.0)

    env._apply_growth()

    assert float(fg.biomass[0, 0]) == pytest.approx(98.0)
    assert float(fg.energy_reserve[0, 0]) == pytest.approx(0.0)


def test_starvation_mortality_scales_with_maintenance_deficit():
    env, fg = _env_for_fg({"starvation_mortality": 0.02}, energy_ratio=0.25)

    env._apply_growth()

    assert float(fg.biomass[0, 0]) == pytest.approx(99.0)


def test_starvation_mortality_does_not_reduce_biomass_above_maintenance():
    env, fg = _env_for_fg({"starvation_mortality": 0.02}, energy_ratio=0.7)

    env._apply_growth()

    # growth_rate applies only to the positive surplus above maintenance:
    # 100 * 0.01 * (0.7 - 0.5) = 0.2
    assert float(fg.biomass[0, 0]) == pytest.approx(100.2, rel=1e-6)


def test_missing_starvation_mortality_keeps_legacy_negative_growth():
    env, fg = _env_for_fg({}, energy_ratio=0.0)

    env._apply_growth()

    # Legacy: negative growth is 100 * 0.01 * (0.0 - 0.5) = -0.5.
    assert float(fg.biomass[0, 0]) == pytest.approx(99.5)
