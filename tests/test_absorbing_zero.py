"""Guards against the absorbing zero at the base of the food web (Section 73).

Two independent mechanisms are covered.

**Fix 2 - relative harvest cap in ``_apply_predation``.**
The old cap was ``scale = B_vis / (total_demand + 1e-9)``, leaving a
residue of ``B_vis * 1e-9 / (D + 1e-9)`` behind. Relative precision in
float32 is ~1.2e-7, i.e. LARGER than that residue, so ``B_old - intake``
rounded to *exactly* 0.0 whenever demand overshot the available biomass.

That matters because zero is absorbing:

* logistic NDM growth ``r*B*(1 - B/K)`` is multiplicative in B,
* ``phytoplankton`` has ``movement_speed: 0.0``, so no diffusion from
  neighbouring cells,
* the Holling type III refuge (auto-enabled for zooplankton, the only
  single-prey specialist) is multiplicative in prey density and therefore
  cannot restore a cell from 0 either.

Overharvest is not a corner case: ``D/P = a*Z*P/(1 + a*h*P^2)`` peaks at
``P = 1`` and gives ``0.125*Z``, so any cell with Z >~ 8 t zooplankton can
zero its phytoplankton in a single tick - right inside mareld2's start
range. The cell is zeroed from a *healthy* P, jumping straight past the
low-density region where the refuge would have applied.
``MAX_HARVEST_FRAC = 0.999`` leaves a 0.1 % survivor margin, far above
float32 epsilon and far below any ecologically meaningful biomass.

**Fix 1 - ``phytoplankton.seed_rate``.**
The recolonisation floor is the only mechanism that can bring a cell back
from exact zero, and it was 0.0 in the working copy. Section 69.6:
*"Without it, any retraining teaches the policies to live in a dying
world."*

Contracts asserted:

1. Massive overharvest leaves biomass strictly > 0 (the regression).
2. It removes at most the visible biomass (no over-harvest the other way).
3. It removes at least 99 % of it (the margin is tight, not a nerf).
4. Under-demand is untouched: scale stays exactly 1.0 (legacy parity).
5. The hidden (rested) fraction is still fully protected.
6. Repeated overharvest never reaches zero, however many ticks.
7. ``phytoplankton.seed_rate > 0`` in the live library.
8. A zeroed NDM cell recovers, and stays put when ``seed_rate = 0``.
"""
import os
import sys

import numpy as np
import pytest
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.environments.ecosystem import (  # noqa: E402
    MAX_HARVEST_FRAC,
    EcosystemEnvironment,
)
from lib.world.functional_group import FunctionalGroup  # noqa: E402

H = W = 3
GRID_CONFIG = {'width': W, 'height': H, 'cell_size': 1000.0,
               'tick_duration': 6.0}
LIBRARY = os.path.join(_ROOT, "fgconfig", "fg_library.yaml")

PREY_ENERGY_CONTENT = 2000.0


def _build_env(prey_visibility_floor=0.0, handling_time=0.0,
               max_intake_rate=2.0, prey_seed_rate=0.0,
               prey_is_dm=True):
    """One grazer on one prey. ``max_intake_rate`` >> 1 forces overharvest."""
    prey_params = {
        'is_decision_maker': prey_is_dm,
        'max_energy_reserve': 1000.0,
        'energy_content': PREY_ENERGY_CONTENT,
        'resting_metabolism': 0.0,
        'maintenance_level': 0.5,
        'movement_speed': 0.0,
        'visibility_floor': prey_visibility_floor,
        'growth_rate': 0.0375,
        'max_carrying_capacity': 24.0,
        'seed_rate': prey_seed_rate,
    }
    grazer_params = {
        'is_decision_maker': True,
        'max_energy_reserve': 1000.0,
        'energy_content': 1000.0,
        'resting_metabolism': 0.0,
        'maintenance_level': 0.5,
        'movement_speed': 0.0,
        'max_intake_rate': max_intake_rate,
        'menu': ['prey'],
        'interaction': {
            'grazer_preys_on_prey': {
                'preys_on': True,
                'assimilation_factor': 1.0,
                'handling_time': handling_time,
            },
        },
    }
    fgs = {'prey': FunctionalGroup('prey', prey_params),
           'grazer': FunctionalGroup('grazer', grazer_params)}
    for fg in fgs.values():
        fg.initialize_state((H, W), initial_biomass=np.zeros((H, W)))
    env = EcosystemEnvironment(GRID_CONFIG, fgs, {})
    env._build_static_caches()
    env.ordered_fg_ids = list(fgs.keys())
    return env


def _set_state(env, prey_biomass, grazer_biomass, prey_rest=0.0):
    """Hidden-prey fraction = ``prey_rest``; the grazer is fully starving."""
    j_prey = env.global_fg_order.index('prey')
    for fid, fg in env.fgs.items():
        b = prey_biomass if fid == 'prey' else grazer_biomass
        fg.biomass = np.full((H, W), b, dtype=env.dtype)
        # Zero reserve => hunger gate fully open (h_X = 1).
        fg.energy_reserve = np.zeros((H, W), dtype=env.dtype)
        fg.temp_energy_gains = np.zeros((H, W), dtype=env.dtype)

    env.pi_move = np.zeros((env.N_dm, 4, H, W), dtype=env.dtype)
    env.pi_rest = np.zeros((env.N_dm, H, W), dtype=env.dtype)
    env.pi_eat = np.zeros((env.N_dm, env.N_all, H, W), dtype=env.dtype)
    if 'prey' in env.dm_ids:
        env.pi_rest[env.dm_ids.index('prey')] = np.float32(prey_rest)
    env.pi_eat[env.dm_ids.index('grazer'), j_prey] = np.float32(1.0)
    return j_prey


# ---------------------------------------------------------------------------
# Fix 2: the relative harvest cap.
# ---------------------------------------------------------------------------

def test_overharvest_never_reaches_exactly_zero():
    """The regression: a single tick used to zero the cell exactly."""
    env = _build_env()
    _set_state(env, prey_biomass=5.0, grazer_biomass=10.0)
    env._apply_predation()
    left = env.fgs['prey'].biomass
    assert np.all(left > 0.0), (
        "overharvest zeroed the prey cell exactly - the absorbing state is "
        f"back (min left = {float(left.min())!r})")


def test_overharvest_leaves_the_configured_margin():
    """Removal is capped at MAX_HARVEST_FRAC of the visible biomass."""
    env = _build_env()
    _set_state(env, prey_biomass=5.0, grazer_biomass=10.0)
    env._apply_predation()
    left = env.fgs['prey'].biomass
    expected = 5.0 * (1.0 - float(MAX_HARVEST_FRAC))
    assert left == pytest.approx(np.full((H, W), expected), rel=1e-3)


def test_cap_is_tight_not_a_nerf():
    """At least 99 % of the visible biomass is still harvestable."""
    env = _build_env()
    _set_state(env, prey_biomass=5.0, grazer_biomass=10.0)
    before = float(env.fgs['prey'].biomass.sum())
    env._apply_predation()
    removed = before - float(env.fgs['prey'].biomass.sum())
    assert removed <= before, "removed more biomass than was present"
    assert removed >= 0.99 * before, (
        f"cap is too conservative: only {removed / before:.4f} harvested")


def test_under_demand_is_bit_identical_to_legacy():
    """Demand below the cap must not be scaled at all."""
    env = _build_env(max_intake_rate=0.01)
    _set_state(env, prey_biomass=5.0, grazer_biomass=1.0)
    env._apply_predation()
    # The grazer is a single-prey specialist, so type III is auto-enabled
    # and a_eff = a*P (h = 0 here). Demand is therefore
    # D = B_pred * a * P * hunger * P = 1.0 * 0.01 * 25 * 1.0 = 0.25 per
    # cell, far below the 0.999 * 5.0 cap, so scale stays exactly 1.0 and
    # the intake equals D without any rescaling.
    left = env.fgs['prey'].biomass
    assert left == pytest.approx(np.full((H, W), 5.0 - 0.25), rel=1e-6)


def test_hidden_fraction_is_still_fully_protected():
    """The cap applies to the VISIBLE biomass, not the standing stock."""
    env = _build_env(prey_visibility_floor=0.0)
    _set_state(env, prey_biomass=5.0, grazer_biomass=10.0, prey_rest=0.4)
    env._apply_predation()
    # visible = 1 - 0.4 = 0.6 of 5.0 = 3.0; hidden 2.0 is untouchable, so
    # at least 2.0 must remain regardless of how large demand was.
    left = env.fgs['prey'].biomass
    assert np.all(left >= 2.0 - 1e-4), (
        f"the hidden fraction was harvested (min left = {float(left.min())})")
    assert np.all(left < 2.0 + 0.01), "the visible fraction was not harvested"


def test_repeated_overharvest_stays_positive():
    """Zero must be unreachable by predation, not merely postponed."""
    env = _build_env()
    _set_state(env, prey_biomass=5.0, grazer_biomass=10.0)
    for tick in range(200):
        env._apply_predation()
        assert np.all(env.fgs['prey'].biomass > 0.0), (
            f"prey hit exact zero at tick {tick}")
        # Keep the grazer starving so demand stays maximal every tick.
        env.fgs['grazer'].energy_reserve = np.zeros((H, W), dtype=env.dtype)


def test_holling_type3_refuge_is_reachable_after_the_fix():
    """The type III refuge only helps if the trajectory can land in it.

    Zooplankton is the single-prey specialist in the live library, so its
    response is type III: ``f(P) = a*P^2 / (1 + a*h*P^2)``, which vanishes
    faster than linearly as P falls. With ``a*h = 1`` the per-capita
    grazing is ~a*P at low P, so a positive residue always regrows.
    """
    env = _build_env(handling_time=4.0, max_intake_rate=0.25,
                     prey_is_dm=False)
    assert bool(env._has_holling3), "type III was not auto-enabled"
    _set_state(env, prey_biomass=5.0, grazer_biomass=80.0)
    for _ in range(50):
        env._apply_predation()
        env._apply_growth()
        env.fgs['grazer'].energy_reserve = np.zeros((H, W), dtype=env.dtype)
    left = env.fgs['prey'].biomass
    assert np.all(left > 0.0), (
        f"prey collapsed to zero despite the refuge (min {float(left.min())})")


# ---------------------------------------------------------------------------
# Fix 1: the recolonisation floor.
# ---------------------------------------------------------------------------

def test_phytoplankton_seed_rate_is_enabled_in_the_library():
    with open(LIBRARY, "r", encoding="utf-8") as fh:
        library = yaml.safe_load(fh) or {}
    phyto = (library.get("species_definitions", {})
             .get("phytoplankton") or {})
    seed_rate = float(phyto.get("seed_rate", 0.0) or 0.0)
    assert seed_rate > 0.0, (
        "phytoplankton.seed_rate is 0 - zero is absorbing at the base of "
        "the food web and long training runs learn a dying world "
        "(Section 69.6)")


def test_seed_rate_recovers_a_zeroed_ndm_cell():
    env = _build_env(prey_seed_rate=1e-07, prey_is_dm=False)
    env.fgs['prey'].biomass = np.zeros((H, W), dtype=env.dtype)
    env.fgs['grazer'].biomass = np.zeros((H, W), dtype=env.dtype)
    np.random.seed(0)
    for _ in range(100):
        env._apply_growth()
    assert np.all(env.fgs['prey'].biomass > 0.0), (
        "the recolonisation floor did not lift the cell off zero")


def test_seed_rate_zero_keeps_legacy_absorbing_behaviour():
    """Opt-in: FGs without a seed rate must be numerically unchanged."""
    env = _build_env(prey_seed_rate=0.0, prey_is_dm=False)
    env.fgs['prey'].biomass = np.zeros((H, W), dtype=env.dtype)
    env.fgs['grazer'].biomass = np.zeros((H, W), dtype=env.dtype)
    for _ in range(10):
        env._apply_growth()
    assert np.all(env.fgs['prey'].biomass == 0.0)
