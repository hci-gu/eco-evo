"""Structural guard for sub-threshold split suppression (Section 69).

A move action splits the moving share of a cell across up to four
directions, and ``_apply_extinction_threshold`` zeroes every cell that
ends the tick below ``thr = extinction_threshold_factor *
min_split_biomass``. Before Section 69 a diffusing decision maker
therefore bled biomass through the extinction sweep at a rate that could
dominate its starvation term entirely: measured for porpoises in Section
68.4, 25.1 of 40 ton lost went to the sweep against 14.9 ton to
starvation, i.e. ~3.6 %/tick vs. ~2.2 %/tick.

``EcosystemEnvironment._suppress_subthreshold_splits`` cancels an outflow
whose destination would still end up below ``thr`` after receiving it;
the biomass (and its energy reserve) stays in the source cell instead.

The contracts asserted here are the ones that make the fix safe rather
than merely helpful:

1. **Retention** - a cell whose split would land entirely below ``thr``
   keeps all of its biomass instead of being swept to zero.
2. **No suppression when viable** - a split large enough to clear ``thr``
   is passed through untouched, so ordinary dispersal is unaffected.
3. **Partial suppression** - viability is judged on the destination's
   TOTAL, so a small inflow into an already-populated cell is allowed
   while the same inflow into an empty cell is not.
4. **Mass conservation** - the correction moves biomass, never creates or
   destroys it (metabolic costs switched off).
5. **Monotone improvement** - versus the pre-fix path the number of
   sub-threshold cells can never increase and the surviving biomass can
   never decrease. This is the property that makes the fix strictly safe
   for every FG, not just porpoises.
6. **Opt-out** - FGs with ``min_split_biomass == 0`` or
   ``extinction_threshold_factor == 0`` (all continuous FGs) are
   untouched.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.environments.ecosystem import EcosystemEnvironment  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402

# 50 kg indivisible weight * factor 0.5 -> thr = 0.025 ton, the porpoise
# configuration from fg_library.yaml.
MIN_SPLIT_KG = 50.0
FACTOR = 0.5
THR = FACTOR * MIN_SPLIT_KG / 1000.0

H = W = 5


def _build_env(min_split_kg=MIN_SPLIT_KG, factor=FACTOR):
    """Single-DM env with all metabolic costs and interactions switched off.

    ``resting_metabolism = 0`` makes ``_apply_movement`` a pure transport
    operator, so any change in total biomass is attributable to the split
    logic alone.
    """
    fg = FunctionalGroup('mover', {
        'is_decision_maker': True,
        'max_energy_reserve': 1000.0,
        'resting_metabolism': 0.0,
        'maintenance_level': 0.5,
        'movement_speed': 1.0,
        'min_split_biomass': min_split_kg,
        'extinction_threshold_factor': factor,
        'feeding_cost': 1.0,
        'movement_cost': 1.0,
        'resting_cost': 1.0,
    })
    fg.initialize_state((H, W), initial_biomass=np.zeros((H, W)))
    env = EcosystemEnvironment(
        {'width': W, 'height': H, 'cell_size': 1000.0, 'tick_duration': 6.0},
        {'mover': fg}, {})
    env._build_static_caches()
    return env, fg


def _set_state(env, biomass, movers=None):
    """Put ``biomass`` on the grid and split ``movers`` over all 4 dirs.

    ``movers`` is a boolean mask of cells that choose move (default: all).
    The remaining probability mass goes to rest, because the action
    distribution must sum to 1 per cell: ``_apply_movement`` only carries
    over ``pi_rest + pi_eat + pi_move`` of the biomass, so an unnormalized
    distribution silently deletes the remainder. Off-grid directions are
    masked out exactly as ``_calculate_decisions`` does in legacy
    (non-migration) mode, which is why edge cells need the rest padding.
    """
    fg = env.fgs['mover']
    fg.biomass = np.asarray(biomass, dtype=env.dtype)
    fg.energy_reserve = (fg.biomass * np.float32(0.5)
                         * np.float32(fg.max_energy_reserve))
    fg.temp_energy_gains = np.zeros((H, W), dtype=env.dtype)
    mask = (np.ones((H, W), dtype=bool) if movers is None
            else np.asarray(movers, dtype=bool))
    env.pi_eat = np.zeros((1, env.N_all, H, W), dtype=env.dtype)
    env.pi_move = (np.full((1, 4, H, W), 0.25, dtype=env.dtype)
                   * env.move_mask[None, :, :, :]
                   * mask[None, None, :, :])
    env.pi_rest = (np.float32(1.0)
                   - env.pi_move.sum(axis=1)).astype(env.dtype)


def _center_only(value):
    b = np.zeros((H, W))
    b[H // 2, W // 2] = value
    return b


def _center_mask():
    m = np.zeros((H, W), dtype=bool)
    m[H // 2, W // 2] = True
    return m


def test_split_below_threshold_is_retained_in_the_source_cell():
    """Contract 1: the four-way split that used to vanish now stays put."""
    total = 4.0 * THR * 0.8   # each of the 4 shares is 0.8 * thr
    env, fg = _build_env()
    _set_state(env, _center_only(total), _center_mask())
    env._apply_movement()
    env._apply_extinction_threshold()
    assert float(fg.biomass.sum()) == pytest.approx(total, rel=1e-5)
    assert float(fg.biomass[H // 2, W // 2]) == pytest.approx(total, rel=1e-5)


def test_split_below_threshold_used_to_be_swept_away():
    """The pre-fix behaviour, to prove the test is not vacuous.

    Disabling the suppression reproduces the Section 68.4 loss channel:
    all four shares land below ``thr`` and the extinction sweep takes the
    whole population.
    """
    total = 4.0 * THR * 0.8
    env, fg = _build_env()
    env._dm_split_thr_any = False          # pre-Section-69 code path
    _set_state(env, _center_only(total), _center_mask())
    env._apply_movement()
    env._apply_extinction_threshold()
    assert float(fg.biomass.sum()) == 0.0
    assert env._extinction_events['mover'] == 4


def test_viable_split_is_passed_through_untouched():
    """Contract 2: ordinary dispersal must not be throttled."""
    total = 4.0 * THR * 10.0   # each share is 10 * thr
    env, fg = _build_env()
    _set_state(env, _center_only(total), _center_mask())
    env._apply_movement()
    cy, cx = H // 2, W // 2
    assert float(fg.biomass[cy, cx]) == pytest.approx(0.0, abs=1e-9)
    for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        assert float(fg.biomass[cy + dy, cx + dx]) == pytest.approx(
            total / 4.0, rel=1e-5)
    env._apply_extinction_threshold()
    assert float(fg.biomass.sum()) == pytest.approx(total, rel=1e-5)
    assert env._extinction_events['mover'] == 0


def test_viability_is_judged_on_the_destination_total():
    """Contract 3: a small inflow is fine if the destination is populated.

    The same outflow is blocked when it would land in an empty cell and
    allowed when the destination already sits above ``thr`` - which is
    what makes the rule "do not CREATE sub-threshold cells" rather than
    "do not send small amounts".
    """
    env, fg = _build_env()
    b = np.zeros((H, W))
    cy, cx = H // 2, W // 2
    b[cy, cx] = 4.0 * THR * 0.5      # each share = 0.5 * thr, sub-threshold
    b[cy, cx + 1] = 100.0 * THR      # east neighbour is comfortably viable
    _set_state(env, b, _center_mask())
    src_out = 4.0 * THR * 0.5 / 4.0
    env._apply_movement()
    # East accepted the small share; N/S/W were cancelled because their
    # destinations are empty, and stayed in the source.
    assert float(fg.biomass[cy, cx]) == pytest.approx(3.0 * src_out, rel=1e-5)
    assert float(fg.biomass[cy, cx + 1]) == pytest.approx(
        100.0 * THR + src_out, rel=1e-5)
    for dy, dx in ((-1, 0), (1, 0), (0, -1)):
        assert float(fg.biomass[cy + dy, cx + dx]) == 0.0


def test_correction_conserves_biomass_and_reserve():
    """Contract 4: the suppression is a transport correction, not a source."""
    rng = np.random.default_rng(0)
    for scale in (0.3, 1.0, 3.0):
        env, fg = _build_env()
        _set_state(env, rng.random((H, W)) * scale * THR * 4.0)
        b_before = float(fg.biomass.sum())
        r_before = float(fg.energy_reserve.sum())
        env._apply_movement()
        assert float(fg.biomass.sum()) == pytest.approx(b_before, rel=1e-4)
        assert float(fg.energy_reserve.sum()) == pytest.approx(
            r_before, rel=1e-4)


def test_suppression_never_loses_biomass_or_adds_subthreshold_cells():
    """Contract 5: strictly dominates the pre-fix path, on random states.

    This is the guard that generalizes the fix beyond the porpoise case:
    for every initial condition the corrected pipeline must retain at
    least as much biomass and leave no more cells below ``thr``.
    """
    rng = np.random.default_rng(12345)
    for trial in range(25):
        b0 = rng.random((H, W)) * (4.0 * THR) * float(rng.uniform(0.1, 5.0))
        b0 *= (rng.random((H, W)) > 0.4)       # sparse occupancy
        totals = {}
        subthr = {}
        for enabled in (False, True):
            env, fg = _build_env()
            env._dm_split_thr_any = enabled
            _set_state(env, b0)
            env._apply_movement()
            b = np.asarray(fg.biomass, dtype=np.float64)
            subthr[enabled] = int(np.count_nonzero((b > 0.0) & (b < THR)))
            env._apply_extinction_threshold()
            totals[enabled] = float(fg.biomass.sum())
        assert totals[True] >= totals[False] - 1e-9, (
            f"trial {trial}: suppression lost biomass "
            f"({totals[True]:.6g} < {totals[False]:.6g})")
        assert subthr[True] <= subthr[False], (
            f"trial {trial}: suppression created sub-threshold cells "
            f"({subthr[True]} > {subthr[False]})")


@pytest.mark.parametrize("min_split_kg,factor", [(0.0, 0.5), (50.0, 0.0)])
def test_continuous_fgs_opt_out(min_split_kg, factor):
    """Contract 6: thr = 0 disables the mechanism, as for the sweep itself."""
    env, fg = _build_env(min_split_kg=min_split_kg, factor=factor)
    assert not env._dm_split_thr_any
    total = 4.0 * THR * 0.8
    _set_state(env, _center_only(total), _center_mask())
    env._apply_movement()
    cy, cx = H // 2, W // 2
    assert float(fg.biomass[cy, cx]) == pytest.approx(0.0, abs=1e-9)
    assert float(fg.biomass[cy - 1, cx]) == pytest.approx(total / 4.0,
                                                          rel=1e-5)


def test_threshold_cache_matches_the_extinction_sweep_bound():
    """The cached thr must equal the bound the sweep actually applies.

    Two independent readings of the same rule are exactly how the
    Section 64 sign bug survived; pinning them together here keeps the
    suppression and the sweep from drifting apart.
    """
    env, fg = _build_env()
    expected = (fg.min_split_biomass * fg.extinction_threshold_factor)
    assert float(env._dm_split_thr[0]) == pytest.approx(expected)
    assert expected == pytest.approx(THR)
