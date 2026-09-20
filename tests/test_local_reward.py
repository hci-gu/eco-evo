"""Contracts for the per-cell source-tracked reward (``--local_reward``).

The reward scores every occupied cell separately: ``A(c,t)`` is the
cell's energy at the start of the tick and ``B(c,t+1)`` the end-of-tick
energy of exactly the population that started in ``c``, tracked through
the move/split into ``{c, N, E, S, W}``. The whole construction stands
or falls on one identity -- the allocation must be mass conserving --
so that is the first thing asserted here.

1. **Mass balance** - ``sum_c B(c,t+1) == sum_c Q(c,t+1)``: the tracked
   energy re-partitions the grid's end-of-tick energy, it does not
   create or destroy any.
2. **Pure rest** - with no movement, ``B(c,t+1)`` is the cell's own
   end-of-tick energy, so the tracking degenerates to the trivial case.
3. **Full 4-way split** - a cell that sends everything to its four
   neighbours still gets all of it back, which is the case the reward
   exists for (movement must not read as a loss).
4. **Robustness** - a cell zeroed by the extinction sweep contributes a
   finite ratio, never a NaN.
5. **Neutrality** - enabling the tracking must not change the
   simulation; ``local_reward=None`` must leave the tick untouched.
6. **theta** - ``theta=1`` with mean normalisation reproduces the
   energy-weighted (i.e. global) growth rate, which is what makes theta
   a single A/B knob between the local and the global reward.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.environments.ecosystem import EcosystemEnvironment  # noqa: E402
from lib.environments.ecosystem_env import source_tracking  # noqa: E402
from lib.environments.ecosystem_env.source_tracking import (  # noqa: E402
    LocalRewardConfig,
)
from lib.environments.ecosystem_env.state import ActionProbabilities  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402

H = W = 6
ENERGY_CONTENT = 4000.0
MAX_RESERVE = 1000.0
# Half-full reserve with maintenance_level 0.5 puts the FG exactly at
# its maintenance energy level, so ``population_change`` yields q = 0
# and the tick is a pure transport operator.
FILL = 0.5


def _build_env(local_reward=None, min_split_kg=0.0, factor=0.0):
    fg = FunctionalGroup('mover', {
        'is_decision_maker': True,
        'energy_content': ENERGY_CONTENT,
        'max_energy_reserve': MAX_RESERVE,
        'resting_metabolism': 0.0,
        'maintenance_level': FILL,
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
        {'mover': fg}, {}, local_reward=local_reward)
    env._build_static_caches()
    return env, fg


def _set_biomass(env, biomass):
    fg = env.fgs['mover']
    fg.biomass = np.asarray(biomass, dtype=env.dtype)
    fg.energy_reserve = (fg.biomass * np.float32(FILL)
                         * np.float32(MAX_RESERVE)).astype(env.dtype)
    fg.temp_energy_gains = np.zeros((H, W), dtype=env.dtype)


def _actions(env, move_weights=None, mask_offgrid=True):
    """Action distribution; ``move_weights`` is a (4,H,W) move field.

    Off-grid directions are masked exactly as ``build_action_mask``
    does, and the remaining probability mass goes to rest so that the
    distribution sums to 1 per cell (``apply_energy_costs`` only carries
    over ``rest + eat + move`` of the biomass). ``mask_offgrid=False``
    keeps the off-grid directions, which is what the migration mode
    does -- the only way biomass can leave the tracked grid.
    """
    if move_weights is None:
        move = np.zeros((1, 4, H, W), dtype=env.dtype)
    else:
        move = np.asarray(move_weights, dtype=env.dtype)[None, :, :, :]
        if mask_offgrid:
            move = move * env.move_mask[None, :, :, :]
    rest = (np.float32(1.0) - move.sum(axis=1)).astype(env.dtype)
    eat = np.zeros((1, env.N_all, H, W), dtype=env.dtype)
    return ActionProbabilities(move=move, rest=rest, eat=eat)


def _cell_energy(env):
    fg = env.fgs['mover']
    return (fg.biomass.astype(np.float64) * ENERGY_CONTENT
            + fg.energy_reserve.astype(np.float64))


def _random_biomass(seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 10.0, size=(H, W)).astype(np.float32)


def _random_moves(seed=11):
    """A move field where every cell splits differently, incl. edges."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(4, H, W)).astype(np.float32)
    # Cap the total move probability at 0.8 so some biomass always stays.
    return raw * np.float32(0.8) / raw.sum(axis=0, keepdims=True)


# ---------------------------------------------------------------- 1


def test_tracked_energy_conserves_grid_total():
    """The tracked energy re-partitions the grid total, nothing else."""
    env, _ = _build_env(local_reward={})
    _set_biomass(env, _random_biomass())
    env.step(_actions(env, _random_moves()))

    diag = env.local_reward_last
    total_tracked = float(np.asarray(diag['tracked'], dtype=np.float64).sum())
    total_grid = float(_cell_energy(env).sum())
    assert total_grid > 0.0
    assert total_tracked == pytest.approx(total_grid, rel=1e-4)


def test_mass_balance_holds_over_many_ticks():
    env, _ = _build_env(local_reward={})
    _set_biomass(env, _random_biomass())
    moves = _random_moves()
    for _ in range(5):
        env.step(_actions(env, moves))
        diag = env.local_reward_last
        total_tracked = float(
            np.asarray(diag['tracked'], dtype=np.float64).sum())
        assert total_tracked == pytest.approx(
            float(_cell_energy(env).sum()), rel=1e-4)


# ---------------------------------------------------------------- 2


def test_pure_rest_tracks_the_cell_itself():
    """With no movement the tracked energy is the cell's own energy."""
    env, _ = _build_env(local_reward={})
    _set_biomass(env, _random_biomass())
    env.step(_actions(env))

    tracked = np.asarray(env.local_reward_last['tracked'], dtype=np.float64)
    np.testing.assert_allclose(tracked[0], _cell_energy(env), rtol=1e-4)


def test_pure_rest_gives_ratio_one_and_zero_log_reward():
    """A tick that neither gains nor loses energy scores exactly 0."""
    env, _ = _build_env(local_reward={'metric': 'log'})
    _set_biomass(env, np.full((H, W), 5.0, dtype=np.float32))
    env.step(_actions(env))

    ratio = np.asarray(env.local_reward_last['ratio'], dtype=np.float64)
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-4)
    assert source_tracking.fitness(env, 'mover') == pytest.approx(0.0,
                                                                  abs=1e-5)
    assert source_tracking.occupancy(env, 'mover') == pytest.approx(H * W)


# ---------------------------------------------------------------- 3


def test_full_four_way_split_is_not_a_loss():
    """A cell that empties itself into all four neighbours keeps its score."""
    env, _ = _build_env(local_reward={'metric': 'ratio'})
    biomass = np.zeros((H, W), dtype=np.float32)
    biomass[2, 2] = 8.0
    # Interior neighbours hold biomass too, so the shares are genuinely
    # fractional rather than trivially 1.
    for y, x in ((1, 2), (3, 2), (2, 1), (2, 3)):
        biomass[y, x] = 3.0
    _set_biomass(env, biomass)

    moves = np.zeros((4, H, W), dtype=np.float32)
    moves[:, 2, 2] = 0.25            # (2,2) splits everything four ways
    env.step(_actions(env, moves))

    diag = env.local_reward_last
    tracked = np.asarray(diag['tracked'], dtype=np.float64)
    start = np.asarray(diag['start'], dtype=np.float64)
    # The source cell is empty afterwards, yet it is credited with the
    # full energy its biomass carries in the four destinations.
    assert env.fgs['mover'].biomass[2, 2] == pytest.approx(0.0)
    assert tracked[0, 2, 2] == pytest.approx(start[0, 2, 2], rel=1e-3)
    assert float(np.asarray(diag['ratio'])[0, 2, 2]) == pytest.approx(
        1.0, rel=1e-3)


def test_border_move_is_not_punished():
    """Off-grid outflow is excluded from A instead of counted as a loss.

    The default action mask already removes off-grid directions, so this
    only arises under ``--migration on``; the mask is bypassed here to
    exercise that path.
    """
    env, _ = _build_env(local_reward={'metric': 'ratio'})
    biomass = np.zeros((H, W), dtype=np.float32)
    biomass[0, 3] = 6.0
    _set_biomass(env, biomass)

    moves = np.zeros((4, H, W), dtype=np.float32)
    moves[:, 0, 3] = 0.25            # north is off-grid on row 0
    env.step(_actions(env, moves, mask_offgrid=False))

    diag = env.local_reward_last
    frac_in = np.asarray(diag['frac_in'], dtype=np.float64)
    assert frac_in[0, 0, 3] < 1.0    # a quarter of the move left the grid
    assert float(np.asarray(diag['ratio'])[0, 0, 3]) == pytest.approx(
        1.0, rel=1e-3)


# ---------------------------------------------------------------- 4


def test_swept_cell_contributes_finite_ratio():
    """A cell zeroed by the extinction sweep scores, it does not NaN."""
    env, _ = _build_env(local_reward={'metric': 'ratio', 'clip_lo': 0.01},
                        min_split_kg=50.0, factor=0.5)
    biomass = np.zeros((H, W), dtype=np.float32)
    # Well above the 0.025 ton threshold at the start, so the cell takes
    # part, but the FG dies out completely during the tick.
    biomass[2, 2] = 1.0
    _set_biomass(env, biomass)
    env.step(_actions(env))
    # Kill the population outright and run one more tick.
    env.fgs['mover'].biomass[:] = 0.0
    env.fgs['mover'].energy_reserve[:] = 0.0
    env.step(_actions(env))

    diag = env.local_reward_last
    assert np.all(np.isfinite(np.asarray(diag['ratio'])))
    assert np.all(np.isfinite(np.asarray(diag['tracked'])))
    assert np.isfinite(source_tracking.fitness(env, 'mover'))


def test_only_viable_cells_participate():
    """Cells below the extinction threshold are excluded from A."""
    env, _ = _build_env(local_reward={}, min_split_kg=50.0, factor=0.5)
    biomass = np.zeros((H, W), dtype=np.float32)
    biomass[1, 1] = 1.0              # viable
    biomass[4, 4] = 1e-6             # far below thr = 0.025 ton
    _set_biomass(env, biomass)
    env.step(_actions(env))

    active = np.asarray(env.local_reward_last['active'])
    assert bool(active[0, 1, 1])
    assert not bool(active[0, 4, 4])
    assert source_tracking.occupancy(env, 'mover') == pytest.approx(1.0)


# ---------------------------------------------------------------- 5


def test_tracking_does_not_change_the_simulation():
    """``--local_reward`` observes the tick, it must not alter it."""
    biomass = _random_biomass()
    moves = _random_moves()

    plain, _ = _build_env(local_reward=None, min_split_kg=50.0, factor=0.5)
    tracked, _ = _build_env(local_reward={}, min_split_kg=50.0, factor=0.5)
    for env in (plain, tracked):
        _set_biomass(env, biomass)
        for _ in range(4):
            env.step(_actions(env, moves))

    np.testing.assert_array_equal(plain.fgs['mover'].biomass,
                                  tracked.fgs['mover'].biomass)
    np.testing.assert_array_equal(plain.fgs['mover'].energy_reserve,
                                  tracked.fgs['mover'].energy_reserve)
    assert plain.local_reward is None
    assert plain.local_reward_ticks == 0


# ---------------------------------------------------------------- 6


def test_theta_one_reproduces_the_global_growth_rate():
    """Energy weighting collapses the local reward onto the global one."""
    env, _ = _build_env(local_reward={'metric': 'ratio', 'theta': 1.0,
                                      'norm': 'mean'})
    _set_biomass(env, _random_biomass())
    start_total = float(_cell_energy(env).sum())
    env.step(_actions(env, _random_moves()))
    end_total = float(_cell_energy(env).sum())

    # sum_c A_c * (B_c/A_c) / sum_c A_c == sum_c B_c / sum_c A_c, and the
    # numerator is the grid total by the mass-balance identity above.
    assert source_tracking.fitness(env, 'mover') == pytest.approx(
        end_total / start_total, rel=1e-3)


def test_sum_norm_scales_with_the_number_of_cells():
    """The raw sum carries a cell-count term; the mean does not."""
    sparse = np.zeros((H, W), dtype=np.float32)
    sparse[1, 1] = 5.0
    dense = np.full((H, W), 5.0, dtype=np.float32)

    values = {}
    for name, biomass in (('sparse', sparse), ('dense', dense)):
        for norm in ('sum', 'mean'):
            env, _ = _build_env(local_reward={'metric': 'ratio',
                                              'norm': norm})
            _set_biomass(env, biomass)
            env.step(_actions(env))
            values[(name, norm)] = source_tracking.fitness(env, 'mover')

    assert values[('dense', 'sum')] == pytest.approx(H * W, rel=1e-3)
    assert values[('sparse', 'sum')] == pytest.approx(1.0, rel=1e-3)
    assert values[('dense', 'mean')] == pytest.approx(
        values[('sparse', 'mean')], rel=1e-3)


def test_grid_norm_is_the_sum_over_a_constant_denominator():
    """``grid`` is ``sum`` divided by the cell count, nothing else."""
    values = {}
    for norm in ('sum', 'grid'):
        env, _ = _build_env(local_reward={'metric': 'ratio', 'norm': norm})
        _set_biomass(env, _random_biomass())
        env.step(_actions(env, _random_moves()))
        values[norm] = source_tracking.fitness(env, 'mover')

    assert values['grid'] == pytest.approx(values['sum'] / (H * W), rel=1e-9)


def test_grid_norm_with_log_is_blind_to_the_number_of_cells():
    """A neutral cell is worth 0, so the cell count cannot be gamed.

    This is the property ``sum`` and ``mean`` both lack: ``sum`` grows
    with every cell added (each neutral cell is worth 1.0 under the raw
    ratio) and ``mean`` grows when a cell is dropped from its shrinking
    denominator. Under ``grid`` + ``log`` the denominator is a constant
    and a neutral term contributes exactly nothing, so occupying one
    cell and occupying all of them score the same.
    """
    sparse = np.zeros((H, W), dtype=np.float32)
    sparse[1, 1] = 5.0
    dense = np.full((H, W), 5.0, dtype=np.float32)

    values = {}
    for name, biomass in (('sparse', sparse), ('dense', dense)):
        env, _ = _build_env(local_reward={'metric': 'log', 'norm': 'grid'})
        _set_biomass(env, biomass)
        env.step(_actions(env))
        values[name] = source_tracking.fitness(env, 'mover')

    assert values['sparse'] == pytest.approx(0.0, abs=1e-6)
    assert values['dense'] == pytest.approx(0.0, abs=1e-6)


def test_the_default_clip_is_symmetric_in_log_space():
    """A halving must cost what a doubling is worth.

    With an asymmetric range such as ``[0.2, 2.0]`` a wiped-out cell is
    priced at ``log(0.2) = -1.61`` while the best possible tick is worth
    ``log(2) = +0.69``; the policy then buys safety at almost any price
    (section 84).
    """
    config = LocalRewardConfig()
    assert config.clip_lo * config.clip_hi == pytest.approx(1.0, rel=1e-9)
    assert np.log(config.clip_hi) == pytest.approx(-np.log(config.clip_lo),
                                                   rel=1e-9)


# ---------------------------------------------------------------- config


def test_config_rejects_bad_settings():
    with pytest.raises(ValueError):
        LocalRewardConfig(metric='sqrt')
    with pytest.raises(ValueError):
        LocalRewardConfig(norm='median')
    with pytest.raises(ValueError):
        LocalRewardConfig(clip_lo=0.0)
    with pytest.raises(ValueError):
        LocalRewardConfig(clip_lo=2.0, clip_hi=1.0)


def test_config_round_trips_through_a_plain_dict():
    config = LocalRewardConfig(metric='ratio', norm='sum', theta=0.5,
                               clip_lo=0.1, clip_hi=5.0,
                               min_energy_factor=2.0)
    assert LocalRewardConfig.from_dict(config.as_dict()).as_dict() == \
        config.as_dict()
    assert LocalRewardConfig.from_dict(None) is None


def test_local_reward_is_mutually_exclusive_with_the_other_rewards():
    from lib.runners.population_stability import (PopulationStability,
                                                  validate_reward)

    validate_reward(None, True, False, LocalRewardConfig())
    with pytest.raises(ValueError):
        validate_reward(None, True, True, LocalRewardConfig())
    with pytest.raises(ValueError):
        validate_reward(PopulationStability(), True, False,
                        LocalRewardConfig())


def test_cli_options_build_a_config():
    import argparse

    args = argparse.Namespace(
        local_reward=True, local_reward_metric='ratio',
        local_reward_norm='grid', local_reward_theta=0.5,
        local_reward_clip=[0.1, 3.0], local_reward_min_energy_factor=2.0)
    config = source_tracking.local_reward_options(args)
    assert config.metric == 'ratio'
    assert config.norm == 'grid'
    assert config.theta == 0.5
    assert (config.clip_lo, config.clip_hi) == (0.1, 3.0)
    assert config.min_energy_factor == 2.0

    args.local_reward = False
    assert source_tracking.local_reward_options(args) is None
