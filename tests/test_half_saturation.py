"""Pair-level attack rate: half-saturation without moving the ceiling.

Section 134: a pair may carry its own ``max_intake_rate`` (the Holling
attack rate a). The ceiling of f(B) = a B / (1 + a h B) is 1/h; a sets the
half-saturation 1/(a h). zooplankton -> phytoplankton uses a = 0.0125,
h = 4: ceiling 0.25 t/t/tick, half-saturation 20 t/km2.
"""
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.config.config_loader import setup_full_mareld_mvp  # noqa: E402
from lib.environments.ecosystem import EcosystemEnvironment  # noqa: E402
from lib.environments.ecosystem_env.interactions import holling_a_eff  # noqa: E402
from lib.world.energy_balance import intake_ceiling  # noqa: E402


def _env():
    fgs = setup_full_mareld_mvp(grid_size=(4, 4), seed=0, spawn_seed=0)
    return EcosystemEnvironment({'width': 4, 'height': 4}, fgs, {})


def test_intake_ceiling_is_one_over_handling_time():
    assert intake_ceiling(0.0125, 4.0) == pytest.approx(0.25)
    assert intake_ceiling(0.15, 0.0) == pytest.approx(0.15)  # linear response


def test_pair_attack_rate_reaches_the_runtime_matrix():
    env = _env()
    i = env.dm_ids.index('zooplankton')
    j = env.global_fg_order.index('phytoplankton')
    a, h = float(env.max_intake_mat[i, j]), float(env.handling_time_mat[i, j])
    assert a == pytest.approx(0.0125)
    assert 1.0 / (a * h) == pytest.approx(20.0, rel=1e-5)
    # The ceiling is the species max_intake_rate, unchanged.
    assert 1.0 / h == pytest.approx(
        float(env.fgs['zooplankton'].params['max_intake_rate']))
    # Half the ceiling at the half-saturation density.
    half = float(holling_a_eff(a, h, 20.0, 0.0))
    assert half == pytest.approx(0.125, rel=1e-5)


def test_pairs_without_override_keep_the_species_rate():
    env = _env()
    i = env.dm_ids.index('pelagic_fish')
    j = env.global_fg_order.index('zooplankton')
    assert float(env.max_intake_mat[i, j]) == pytest.approx(
        float(env.fgs['pelagic_fish'].params['max_intake_rate']))
