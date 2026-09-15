"""Structural guard for the per-(predator, prey) visibility floor (Section 71.12).

``visibility_floor`` used to be a property of the PREY alone: one scalar
per FG, applied to every predator. That is wrong for the Mareld food web,
because detection is a property of the PAIR. Herring schooling is a
visual defence; porpoises hunt with biosonar and are unaffected by it
(Au 1993; Verfuss et al. 2009). With the single prey-side scalar of 0.35
the porpoise break-even requirement was 15.1 t herring/cell, against
5.3 t/cell at full detectability -- the direct cause of the porpoise
extinction arc diagnosed in Section 71.

An optional ``visibility_floor`` on the ``{pred}_preys_on_{prey}``
interaction now overrides the prey default for that pair only.

Contracts asserted:

1. **Inheritance** - an absent override resolves to the prey FG's own
   ``visibility_floor``, for every (predator, prey) cell.
2. **Legacy parity** - with no override anywhere, ``_has_pair_vis_floor``
   stays False, i.e. the cheap vector code path (and its bit-identical
   numerics) is still taken by existing projects.
3. **Override is pair-local** - a value on one interaction changes that
   row only; every other row keeps the column default.
4. **Predation follows the pair value** - against fully hidden prey the
   floor-1.0 predator still feeds while the floor-0.0 predator gets
   nothing, in the SAME tick and the same cell.
5. **No over-harvest** - a raised floor never lets the predators remove
   more than the prey biomass actually present.
6. **Observation follows the pair value** - the observed biomass of a
   hidden prey is attenuated per observer, so "what I can see" and "what
   I can eat" stay consistent.
7. **Live library** - the porpoise -> herring pair really carries the
   biosonar override, so the mechanism is wired into mareld2.
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
from lib.world.functional_group import FunctionalGroup  # noqa: E402

H = W = 4
PREY_FLOOR = 0.0          # prey default: hiding is perfect protection
GRID_CONFIG = {'width': W, 'height': H, 'cell_size': 1000.0,
               'tick_duration': 6.0}


def _predator(fg_id, floor_override):
    """Single-prey predator; ``floor_override`` None = inherit the prey."""
    inter = {'preys_on': True, 'handling_time': 0.0,
             'assimilation_factor': 1.0}
    if floor_override is not None:
        inter['visibility_floor'] = floor_override
    return FunctionalGroup(fg_id, {
        'is_decision_maker': True,
        'max_energy_reserve': 1000.0,
        'energy_content': 1000.0,
        'resting_metabolism': 0.0,
        'maintenance_level': 0.5,
        'movement_speed': 0.0,
        'max_intake_rate': 0.5,
        'menu': ['prey'],
        'interaction': {f'{fg_id}_preys_on_prey': inter},
    })


def _build_env(sonar_floor=None, visual_floor=None, prey_floor=PREY_FLOOR):
    prey = FunctionalGroup('prey', {
        'is_decision_maker': True,
        'max_energy_reserve': 1000.0,
        'energy_content': 1000.0,
        'resting_metabolism': 0.0,
        'maintenance_level': 0.5,
        'movement_speed': 0.0,
        'visibility_floor': prey_floor,
    })
    fgs = {
        'prey': prey,
        'sonar': _predator('sonar', sonar_floor),
        'visual': _predator('visual', visual_floor),
    }
    for fg in fgs.values():
        fg.initialize_state((H, W), initial_biomass=np.zeros((H, W)))
    env = EcosystemEnvironment(GRID_CONFIG, fgs, {})
    env._build_static_caches()
    return env


def _set_predation_state(env, prey_biomass=100.0, pred_biomass=10.0,
                         prey_rest=1.0):
    """Hidden prey, starving predators that spend the whole tick eating."""
    j_prey = env.global_fg_order.index('prey')
    for fid, fg in env.fgs.items():
        b = prey_biomass if fid == 'prey' else pred_biomass
        fg.biomass = np.full((H, W), b, dtype=env.dtype)
        fg.energy_reserve = np.zeros((H, W), dtype=env.dtype)
        fg.temp_energy_gains = np.zeros((H, W), dtype=env.dtype)

    env.pi_move = np.zeros((env.N_dm, 4, H, W), dtype=env.dtype)
    env.pi_rest = np.zeros((env.N_dm, H, W), dtype=env.dtype)
    env.pi_eat = np.zeros((env.N_dm, env.N_all, H, W), dtype=env.dtype)
    env.pi_rest[env.dm_ids.index('prey')] = np.float32(prey_rest)
    for fid in ('sonar', 'visual'):
        env.pi_eat[env.dm_ids.index(fid), j_prey] = np.float32(1.0)
    return j_prey


def _intake(env):
    """Per-predator intake of the prey in one predation step (ton)."""
    j_prey = env.global_fg_order.index('prey')
    before = float(env.fgs['prey'].biomass.sum())
    env._apply_predation()
    after = float(env.fgs['prey'].biomass.sum())
    gains = {fid: float(env.fgs[fid].temp_energy_gains.sum())
             for fid in ('sonar', 'visual')}
    # energy_gain = prey energy_content * assimilation_factor = 1000.0,
    # so intake in ton is the buffered gain divided by that factor.
    intake = {fid: g / 1000.0 for fid, g in gains.items()}
    intake['_removed'] = before - after
    intake['_j_prey'] = j_prey
    return intake


# ---------------------------------------------------------------------------
# Contracts 1-3: the matrix resolves the right value in the right cell.
# ---------------------------------------------------------------------------

def test_absent_override_inherits_the_prey_default():
    env = _build_env(prey_floor=0.35)
    j_prey = env.global_fg_order.index('prey')
    for i, pred_id in enumerate(env.dm_ids):
        assert float(env.vis_floor_mat[i, j_prey]) == pytest.approx(0.35), (
            f"{pred_id} did not inherit the prey's own floor")


def test_no_override_keeps_the_legacy_vector_path():
    """Existing projects must not silently switch to the pair tensor."""
    env = _build_env(prey_floor=0.35)
    assert env._has_pair_vis_floor is False
    assert np.allclose(env.vis_floor_mat,
                       np.tile(env._all_visibility_floor[None, :],
                               (env.N_dm, 1)))


def test_override_is_local_to_its_own_pair():
    env = _build_env(sonar_floor=1.0, prey_floor=0.35)
    j_prey = env.global_fg_order.index('prey')
    assert env._has_pair_vis_floor is True
    assert float(env.vis_floor_mat[env.dm_ids.index('sonar'), j_prey]) == 1.0
    assert float(
        env.vis_floor_mat[env.dm_ids.index('visual'), j_prey]
    ) == pytest.approx(0.35)


def test_override_is_clipped_to_the_unit_interval():
    env = _build_env(sonar_floor=7.5, visual_floor=-2.0)
    j_prey = env.global_fg_order.index('prey')
    assert float(env.vis_floor_mat[env.dm_ids.index('sonar'), j_prey]) == 1.0
    assert float(env.vis_floor_mat[env.dm_ids.index('visual'), j_prey]) == 0.0


# ---------------------------------------------------------------------------
# Contracts 4-5: predation actually differentiates between the predators.
# ---------------------------------------------------------------------------

def test_hidden_prey_is_eaten_only_by_the_detecting_predator():
    """The whole point of Section 71.12, in one assertion."""
    env = _build_env(sonar_floor=1.0, visual_floor=0.0)
    _set_predation_state(env)
    intake = _intake(env)
    assert intake['sonar'] > 0.0, (
        "the biosonar predator got nothing from fully hidden prey - the "
        "pair override is not reaching _apply_predation")
    assert intake['visual'] == pytest.approx(0.0, abs=1e-9), (
        "a floor-0.0 predator fed on fully hidden prey; the per-predator "
        "demand clamp is missing")


def test_partially_hiding_prey_scales_with_the_pair_floor():
    env = _build_env(sonar_floor=1.0, visual_floor=0.5)
    _set_predation_state(env, prey_rest=1.0)
    intake = _intake(env)
    assert intake['sonar'] > intake['visual'] > 0.0


def test_floors_do_not_let_predators_over_harvest():
    """Contract 5: the shared availability cap still bounds the removal."""
    env = _build_env(sonar_floor=1.0, visual_floor=1.0)
    prey_total = 4.0 * H * W          # tiny prey stock, huge demand
    _set_predation_state(env, prey_biomass=4.0, pred_biomass=1000.0)
    intake = _intake(env)
    assert intake['_removed'] <= prey_total + 1e-6
    assert float(env.fgs['prey'].biomass.min()) >= -1e-6


def test_visible_prey_is_unaffected_by_the_floor():
    """With nobody hiding the floor must be inert (regression guard)."""
    env_a = _build_env(sonar_floor=1.0, visual_floor=0.0)
    _set_predation_state(env_a, prey_rest=0.0)
    a = _intake(env_a)
    env_b = _build_env()
    _set_predation_state(env_b, prey_rest=0.0)
    b = _intake(env_b)
    assert a['sonar'] == pytest.approx(b['sonar'], rel=1e-6)
    assert a['visual'] == pytest.approx(b['visual'], rel=1e-6)


# ---------------------------------------------------------------------------
# Contract 6: the observation side uses the same pair value.
# ---------------------------------------------------------------------------

def test_observation_attenuates_hidden_prey_per_observer():
    env = _build_env(sonar_floor=1.0, visual_floor=0.0)
    _set_predation_state(env)
    j_prey = env.global_fg_order.index('prey')
    # The observation reads the PREVIOUS tick's hide fraction.
    env.prev_hidden_frac[j_prey] = np.float32(1.0)
    obs = env._build_observation_batch()

    def _prey_center(fg_id):
        i = env.dm_ids.index(fg_id)
        k = list(env.obs_others_idx[i]).index(j_prey)
        return float(obs[i, :, 2 + k].max())

    assert _prey_center('sonar') == pytest.approx(100.0, rel=1e-6), (
        "biosonar observer should see the hidden prey in full")
    assert _prey_center('visual') == pytest.approx(0.0, abs=1e-6), (
        "visual observer should not see fully hidden prey")


# ---------------------------------------------------------------------------
# Contract 7: the live library carries the fix.
# ---------------------------------------------------------------------------

def test_mareld_porpoises_detect_hiding_herring():
    result = setup_full_mareld_mvp()
    fgs = result[0] if isinstance(result, tuple) else result
    env = EcosystemEnvironment(
        {'width': 60, 'height': 60, 'cell_size': 1000.0,
         'tick_duration': 6.0}, fgs, {})
    env._build_static_caches()
    if 'porpoises' not in env.dm_ids or 'pelagic_fish' not in env.global_fg_order:
        pytest.skip("mareld2 FG set does not contain the porpoise/herring pair")
    i = env.dm_ids.index('porpoises')
    j = env.global_fg_order.index('pelagic_fish')
    herring_default = float(fgs['pelagic_fish'].visibility_floor)
    pair_floor = float(env.vis_floor_mat[i, j])
    # Deliberately NOT pinned to one exact number: the magnitude is a
    # calibration knob (1.0 = perfect biosonar detection, ~0.95 leaves a
    # small residual advantage to schooling). What must hold is that the
    # override exists and lifts detection far above the prey-side visual
    # default, which is what moves the break-even off 15.1 t/cell.
    assert pair_floor > herring_default + 0.25, (
        "porpoises lost the biosonar override; the break-even requirement "
        f"is back near 15.1 t/cell (pair floor {pair_floor}, herring "
        f"default {herring_default}) (Section 71.12)")
    assert pair_floor <= 1.0
