"""Pair-level attack rate: half-saturation without moving the ceiling.

Section 134: a pair may carry its own ``max_intake_rate`` (the Holling
attack rate a). The ceiling of f(B) = a B / (1 + a h B) is 1/h; a sets the
half-saturation 1/(a h). zooplankton -> phytoplankton uses a = 0.0125,
h = 4: ceiling 0.25 t/t/tick, half-saturation 20 t/km2.
"""
import os
import sys

import numpy as np
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
    # Section 202: Type III, a per (t/km2)^2 at the 10 m reference depth
    # (no calendar here, so the mixed-layer scale is inert); half-
    # saturation 1/sqrt(a h) = 2.9 t/km2. Section 177 had Type II 0.05 /
    # 5 t/km2, section 134 0.0125 / 20.
    assert a == pytest.approx(0.03)
    assert 1.0 / np.sqrt(a * h) == pytest.approx(2.887, rel=1e-3)
    # The ceiling is the species max_intake_rate, unchanged.
    assert 1.0 / h == pytest.approx(
        float(env.fgs['zooplankton'].params['max_intake_rate']))
    # Half the ceiling at the half-saturation density (Type III).
    k = 1.0 / np.sqrt(a * h)
    half = float(holling_a_eff(a, h, k, 1.0))
    assert half == pytest.approx(0.125, rel=1e-5)


def test_pairs_without_override_keep_the_species_rate():
    # Every checked DM pair carries its own a since section 220, so the
    # fallback is checked on a copy with one override removed.
    from lib.config.config_loader import load_config
    lib = load_config(os.path.join(_ROOT, 'fgconfig', 'fg_library.yaml'))
    pair = lib['interaction_definitions']['planktivorous_fish_preys_on_zooplankton']
    pair.pop('max_intake_rate')
    fgs = setup_full_mareld_mvp(grid_size=(4, 4), seed=0, spawn_seed=0,
                                library_config=lib)
    env = EcosystemEnvironment({'width': 4, 'height': 4}, fgs, {})
    i = env.dm_ids.index('planktivorous_fish')
    j = env.global_fg_order.index('zooplankton')
    assert float(env.max_intake_mat[i, j]) == pytest.approx(
        float(env.fgs['planktivorous_fish'].params['max_intake_rate']))


def test_fish_pairs_carry_the_encounter_theory_rates():
    # Section 220: a from encounter theory per fish pair, read by the engine;
    # since 223.15 the benthic pairs carry the exposure factor e (a_enc x e)
    # and the benthivores eat Nephrops (norway_lobster) at the encounter a.
    # Section 223.26: the predators split into whiting (bed) and the water
    # column, and every a is per tonne of the WHOLE predator group.
    # Section 228: mackerel and horse mackerel are their own group; the
    # water column's a are per tonne of the rest.
    env = _env()
    expected = {
        ('planktivorous_fish', 'zooplankton'): 0.017,
        ('benthivorous_fish', 'benthic_infauna'): 0.036,
        ('benthivorous_fish', 'mobile_epifauna'): 0.15,
        ('whiting', 'mobile_epifauna'): 0.27,
        ('benthivorous_fish', 'norway_lobster'): 0.44,
        ('benthivorous_fish', 'planktivorous_fish'): 0.0062,
        ('benthivorous_fish', 'whiting'): 0.0026,
        ('predatory_fish', 'planktivorous_fish'): 0.00559,
        ('mackerel', 'planktivorous_fish'): 0.0435,
        ('whiting', 'planktivorous_fish'): 0.021,
        ('predatory_fish', 'zooplankton'): 0.000241,
        ('mackerel', 'zooplankton'): 0.0023,
        ('whiting', 'zooplankton'): 0.0023,
        ('predatory_fish', 'benthivorous_fish'): 0.0651,
        ('whiting', 'benthivorous_fish'): 0.21,
        ('predatory_fish', 'whiting'): 0.013,
        ('predatory_fish', 'mackerel'): 0.00088,
    }
    for (pred, prey), a in expected.items():
        i = env.dm_ids.index(pred)
        j = env.global_fg_order.index(prey)
        assert float(env.max_intake_mat[i, j]) == pytest.approx(a, rel=1e-6)


def test_fish_pair_light_settings_keep_a_zero_threshold():
    # Section 220: 0 deg is a real threshold, not "unset" (falsy 0.0).
    from lib.config.config_loader import load_config
    from lib.world import daylight
    inter = load_config(os.path.join(_ROOT, 'fgconfig', 'fg_library.yaml'))[
        'interaction_definitions']
    expected = {
        'planktivorous_fish_preys_on_zooplankton': (0.02, 0.0),
        'benthivorous_fish_preys_on_planktivorous_fish': (0.06, 0.0),
        'benthivorous_fish_preys_on_whiting': (0.10, 0.0),
        'predatory_fish_preys_on_planktivorous_fish': (0.30, -5.0),
        'mackerel_preys_on_planktivorous_fish': (0.36, -5.0),
        'whiting_preys_on_planktivorous_fish': (0.33, -5.0),
        'predatory_fish_preys_on_zooplankton': (0.51, -5.0),
        'mackerel_preys_on_zooplankton': (0.11, -5.0),
        'whiting_preys_on_zooplankton': (0.78, -5.0),
        'predatory_fish_preys_on_benthivorous_fish': (0.48, -5.0),
        'whiting_preys_on_benthivorous_fish': (0.64, -5.0),
        'predatory_fish_preys_on_whiting': (0.48, -5.0),
        'predatory_fish_preys_on_mackerel': (0.92, -5.0),
    }
    for key, (rho, theta) in expected.items():
        assert daylight.pair_settings(inter[key]) == pytest.approx((rho, theta))
    for key in ('benthivorous_fish_preys_on_benthic_infauna',
                'benthivorous_fish_preys_on_mobile_epifauna',
                'whiting_preys_on_mobile_epifauna'):
        assert daylight.pair_settings(inter[key]) is None
    at_zero = daylight.attack_multiplier(58.1, 6, 0.02, 0.0)
    at_minus8 = daylight.attack_multiplier(58.1, 6, 0.02, -8.0)
    assert not np.allclose(at_zero, at_minus8)
