"""Beddington-DeAngelis predator interference (Section 86).

Holling II/III is purely prey dependent, so ``demand`` is exactly linear
in the predator's own biomass: 100 t of gadoids in one cell get ten times
the food of 10 t in the same cell. Packing is therefore free, aggregation
is always weakly optimal, and the fastest growing mode of that
anti-diffusive feedback is the shortest wavelength the grid can carry -
the 2-cell vertical bands of Sections 84-85.

The interference term puts ``w_X * B_X(c)`` in the denominator:

    f(B_prey, B_pred) = a*B_prey / (1 + a*h*B_prey + w*B_pred)

Contracts asserted here:

1. ``w = 0`` reduces to today's response bit-identically, both in
   ``holling_a_eff`` and over a full tick of the live engine. This is
   what keeps existing ``.pth`` checkpoints valid.
2. The Holling shape contracts of tests/test_holling_response.py survive:
   zero at zero prey, strictly increasing in prey, bounded by 1/h.
   Interference lowers the curve, it does not deform it.
3. The point of the whole exercise: *total* intake in a cell saturates as
   the predator packs in, instead of growing without bound. That is the
   queue at the dinner table, and it is what gives aggregation a finite
   optimum.
4. The unsaturated branch (h = 0, no Holling) is covered too - it does
   not go through ``holling_a_eff`` at all.
5. The live library's calibration is wired through to ``dm_interference``
   in ``dm_ids`` order (a transposed or mis-ordered vector would silently
   give the wrong FG the wrong w).
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
from lib.environments.ecosystem_env.interactions import (  # noqa: E402
    holling_a_eff,
)
from lib.world.functional_group import FunctionalGroup  # noqa: E402

H = W = 3
GRID_CONFIG = {'width': W, 'height': H, 'cell_size': 1000.0,
               'tick_duration': 6.0}

B_GRID = np.geomspace(1e-3, 1e4, 400)


def _build_env(interference=0.0, handling_time=25.0, max_intake_rate=0.04):
    """One predator on two non-decision-maker prey, no movement.

    Two prey on the menu on purpose: a single-prey predator is a
    specialist and ``state.build_static_caches`` auto-enables Type III
    for it, which would obscure the plain Type II arithmetic these tests
    verify. ``prey2`` is kept empty, so it only sets the response type.
    """
    prey_params = {
        'is_decision_maker': False,
        'max_energy_reserve': 1000.0,
        'energy_content': 2000.0,
        'resting_metabolism': 0.0,
        'maintenance_level': 0.5,
        'movement_speed': 0.0,
        'growth_rate': 0.0,
        'max_carrying_capacity': 1e9,
    }
    predator_params = {
        'is_decision_maker': True,
        'max_energy_reserve': 1000.0,
        'energy_content': 1000.0,
        'resting_metabolism': 0.0,
        'maintenance_level': 0.5,
        'movement_speed': 0.0,
        'max_intake_rate': max_intake_rate,
        'interference': interference,
        'menu': ['prey', 'prey2'],
        'interaction': {
            f'predator_preys_on_{prey}': {
                'preys_on': True,
                'assimilation_factor': 1.0,
                'handling_time': handling_time,
            } for prey in ('prey', 'prey2')
        },
    }
    fgs = {'prey': FunctionalGroup('prey', prey_params),
           'prey2': FunctionalGroup('prey2', dict(prey_params)),
           'predator': FunctionalGroup('predator', predator_params)}
    for fg in fgs.values():
        fg.initialize_state((H, W), initial_biomass=np.zeros((H, W)))
    env = EcosystemEnvironment(GRID_CONFIG, fgs, {})
    env._build_static_caches()
    env.ordered_fg_ids = list(fgs.keys())
    return env


def _intake(env, prey_biomass, predator_biomass):
    """One predation step; returns the biomass removed per cell.

    Read off ``loss_predation`` rather than differencing the prey field:
    at the large prey densities used below, float32 has less resolution
    than the intake itself, and the difference would be dominated by
    rounding.
    """
    j_prey = env.global_fg_order.index('prey')
    for fid, fg in env.fgs.items():
        b = predator_biomass if fid == 'predator' else (
            prey_biomass if fid == 'prey' else 0.0)
        fg.biomass = np.full((H, W), b, dtype=env.dtype)
        fg.energy_reserve = np.zeros((H, W), dtype=env.dtype)
        fg.temp_energy_gains = np.zeros((H, W), dtype=env.dtype)
    env.pi_move = np.zeros((env.N_dm, 4, H, W), dtype=env.dtype)
    env.pi_rest = np.zeros((env.N_dm, H, W), dtype=env.dtype)
    env.pi_eat = np.zeros((env.N_dm, env.N_all, H, W), dtype=env.dtype)
    env.pi_eat[env.dm_ids.index('predator'), j_prey] = np.float32(1.0)
    env.loss_predation = {}
    env._apply_predation()
    return float(env.loss_predation.get('prey', 0.0)) / (H * W)


# ---------------------------------------------------------------------------
# 1. w = 0 changes nothing.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m3", [0.0, 1.0], ids=["type2", "type3"])
def test_zero_interference_is_bit_identical(m3):
    a, h = np.float64(0.04), np.float64(25.0)
    base = holling_a_eff(a, h, B_GRID, np.float64(m3))
    explicit = holling_a_eff(a, h, B_GRID, np.float64(m3), 0.0)
    assert np.array_equal(base, explicit), (
        "adding a zero interference term perturbed the response")


def test_default_library_predator_without_w_is_unchanged_over_a_tick():
    """A full predation step with w = 0 must match the pre-change engine."""
    off = _build_env(interference=0.0)
    assert not off._has_interference, "w = 0 must not arm the branch"
    removed = _intake(off, prey_biomass=8.0, predator_biomass=4.0)
    # a_eff = a*B/(1 + a*h*B) with a=0.04, h=25, B=8 -> 0.0195122;
    # demand = B_pred * a_eff * hunger(=1) = 4 * 0.0195122.
    expected = 4.0 * (0.04 * 8.0) / (1.0 + 0.04 * 25.0 * 8.0)
    assert removed == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# 2. The Holling shape contracts survive.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("w_b", [0.5, 5.0, 50.0])
@pytest.mark.parametrize("m3", [0.0, 1.0], ids=["type2", "type3"])
def test_response_shape_is_preserved_under_interference(w_b, m3):
    a, h = np.float64(0.04), np.float64(25.0)
    vals = holling_a_eff(a, h, B_GRID, np.float64(m3), w_b)
    assert holling_a_eff(a, h, np.array([0.0]), np.float64(m3), w_b)[0] == 0.0
    assert np.all(np.diff(vals) > 0.0), "no longer increasing in prey density"
    assert np.all(vals <= (1.0 / h) * (1.0 + 1e-12)), "1/h ceiling breached"
    plain = holling_a_eff(a, h, B_GRID, np.float64(m3))
    assert np.all(vals < plain), "interference must lower the curve"


# ---------------------------------------------------------------------------
# 3. Total cell intake saturates in predator density.
# ---------------------------------------------------------------------------

def test_total_intake_is_linear_in_predator_density_without_interference():
    """The status quo: ten times the predators eat ten times as much."""
    env = _build_env(interference=0.0)
    small = _intake(env, prey_biomass=1000.0, predator_biomass=1.0)
    large = _intake(env, prey_biomass=1000.0, predator_biomass=10.0)
    assert large == pytest.approx(10.0 * small, rel=1e-4)


def test_total_intake_saturates_with_interference():
    """With w > 0 the cell's total intake tends to a finite ceiling.

    ``demand = B_pred * a*B_prey / (1 + a*h*B_prey + w*B_pred)`` tends to
    ``a*B_prey / w`` as ``B_pred -> inf``, so a cell cannot be made more
    productive simply by packing more predators into it. That finite
    ceiling is what gives aggregation a finite optimum.
    """
    w = 2.0
    env = _build_env(interference=w)
    assert env._has_interference
    prey = 1000.0
    ceiling = 0.04 * prey / w
    # The last point is deliberately far past any plausible density: the
    # ceiling is only reached as B_pred -> inf, and at 1e4 the saturation
    # term a*h*B_prey = 1000 still holds the value 5 % below it.
    intakes = [_intake(env, prey, b_pred) for b_pred in (1.0, 10.0, 100.0, 1e6)]
    assert all(b < a for a, b in zip(intakes[1:], intakes[:-1])), \
        "total intake must still increase with predator biomass"
    assert intakes[-1] == pytest.approx(ceiling, rel=1e-3), (
        f"total intake did not approach a*B_prey/w = {ceiling}: {intakes}")
    # Per unit of predator, returns are diminishing all the way.
    densities = (1.0, 10.0, 100.0, 1e6)
    per_unit = [i / b for i, b in zip(intakes, densities)]
    assert all(b < a for a, b in zip(per_unit, per_unit[1:])), (
        f"intake per unit predator did not fall monotonically: {per_unit}")
    assert per_unit[-1] < 0.01 * per_unit[0]


def test_interference_only_bites_where_the_predator_is_packed():
    """At low density the term is negligible; at high density it halves."""
    w = 3.0
    off, on = _build_env(0.0), _build_env(w)
    sparse = 0.02
    assert _intake(on, 8.0, sparse) == pytest.approx(
        _intake(off, 8.0, sparse), rel=0.01)
    dense = (1.0 + 0.04 * 25.0 * 8.0) / w      # interference == saturation
    assert _intake(on, 8.0, dense) == pytest.approx(
        0.5 * _intake(off, 8.0, dense), rel=1e-3)


# ---------------------------------------------------------------------------
# 4. The unsaturated branch.
# ---------------------------------------------------------------------------

def test_unsaturated_branch_applies_interference():
    """h = 0 bypasses ``holling_a_eff`` entirely; w must still apply."""
    w = 2.0
    off = _build_env(interference=0.0, handling_time=0.0, max_intake_rate=0.01)
    on = _build_env(interference=w, handling_time=0.0, max_intake_rate=0.01)
    assert not off._has_holling2, "fixture unexpectedly took the Holling path"
    b_pred = 4.0
    assert _intake(on, 50.0, b_pred) == pytest.approx(
        _intake(off, 50.0, b_pred) / (1.0 + w * b_pred), rel=1e-5)


# ---------------------------------------------------------------------------
# 5. The live library's calibration is wired through correctly.
# ---------------------------------------------------------------------------

def test_live_project_interference_is_read_per_decision_maker():
    fgs = setup_full_mareld_mvp(grid_size=(8, 8), seed=0, spawn_seed=0)
    grid_config = {'width': 8, 'height': 8, 'cell_size': 1000.0,
                   'tick_duration': 6.0}
    env = EcosystemEnvironment(grid_config, fgs, {})
    env._build_static_caches()
    if env.N_dm == 0:
        pytest.skip("No decision makers in the project configuration")

    w = env.dm_interference.reshape(-1)
    assert w.shape == (env.N_dm,)
    assert np.all(w >= 0.0), "a negative interference coefficient got through"
    for i, fid in enumerate(env.dm_ids):
        assert float(w[i]) == pytest.approx(
            float(env.fgs[fid].params.get("interference", 0.0) or 0.0)), (
            f"{fid} got the wrong w - the vector is out of dm_ids order")
    assert env._has_interference == bool(np.any(w > 0.0))
