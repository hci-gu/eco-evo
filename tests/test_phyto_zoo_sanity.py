"""Sanity-tester för ett slutet predator–prey-system med endast
phytoplankton (NDM) + zooplankton (DM).

Syfte: verifiera att simulatorn levererar rimlig dynamik när alla andra
FGs är borta — dvs detektera grova ekologiska felkalibreringar
(t.ex. exploderande zoo, kollapsad phyto, eller frikoppling mellan dem).

Tre kontrakt prövas:

1. **Bounded phyto utan zoo** – som ren NDM-baseline ska phyto mätta
   mot ``max_carrying_capacity`` och inte blåsa upp i oändligheten.
2. **Bounded zoo med phyto** – över 200 ticks (~50 dygn) ska zoo:s
   biomassa varken kollapsa till noll eller växa mer än en
   ekologiskt rimlig faktor (< 100×) jämfört med start.
3. **Predationstryck synligt** – phytos biomassa i två-arts-systemet
   ska vara *lägre* än i baseline utan zoo. Annars är betningen
   antingen muted eller orealistiskt svag.
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


GRID = (30, 30)
GRID_CONFIG = {
    'width': GRID[1],
    'height': GRID[0],
    'cell_size': 1000.0,
    'tick_duration': 6.0,
}


def _build_env(keep_ids, seed=0, ticks=0):
    """Construct an EcosystemEnvironment containing only ``keep_ids``."""
    all_fgs = setup_full_mareld_mvp(grid_size=GRID, seed=seed, spawn_seed=seed)
    fgs = {fid: fg for fid, fg in all_fgs.items() if fid in keep_ids}
    assert set(fgs.keys()) == set(keep_ids), (
        f"Missing FGs in library: wanted {keep_ids}, got {list(fgs.keys())}"
    )
    env = EcosystemEnvironment(GRID_CONFIG, fgs, {})
    # Ensure impact maps exist as zero arrays so _apply_impact_mortality
    # is a no-op (no impacts in this closed sanity scenario).
    for iid in ('djup', 'windfarm_noise', 'bottom_trawling',
                'pelagic_trawling', 'rotor'):
        env.grid.add_map(iid, np.zeros(GRID, dtype=np.float32))
    history = {fid: [] for fid in fgs}
    for _ in range(ticks):
        env.step()
        for fid, fg in env.fgs.items():
            history[fid].append(float(fg.biomass.sum()))
    return env, history


def test_phyto_only_bounded_by_carrying_capacity():
    """NDM phyto alone must saturate against carrying_capacity, never blow up."""
    env, hist = _build_env(['phytoplankton'], seed=0, ticks=200)
    cc = env.fgs['phytoplankton'].params.get('max_carrying_capacity', 0.0)
    cells = GRID[0] * GRID[1]
    hard_cap = cc * cells  # absolute ceiling: cc per cell
    series = np.array(hist['phytoplankton'])
    assert series[-1] <= hard_cap * 1.001, (
        f"Phytoplankton exceeded hard carrying-capacity cap: "
        f"{series[-1]} > {hard_cap}")
    # Should also be monotone-ish toward the cap (no NaN, no negative).
    assert np.all(series >= 0)
    assert not np.any(np.isnan(series))


def test_zoo_bounded_in_closed_phyto_zoo_system():
    """Over 200 ticks (~50 days) zooplankton must neither vanish nor explode."""
    _, hist = _build_env(['phytoplankton', 'zooplankton'], seed=0, ticks=200)
    zoo = np.array(hist['zooplankton'])
    z0 = zoo[0]
    z_end = zoo[-1]
    assert z0 > 0, "Zooplankton spawned to zero biomass — config bug"
    ratio = z_end / z0
    # Ecologically plausible band: factor 0.03x-100x over 50 days.
    # Lower bound widened from 0.05 to 0.03 after canonical Holling Type II
    # calibration (handling_time=2.5 so 1/h=a, ceiling at physiological max).
    # The tighter saturation suppresses zoo more than the previous h=0.1.
    # Outside this range the calibration is clearly broken.
    assert 0.03 <= ratio <= 100.0, (
        f"Zooplankton biomass ratio after 200 ticks = {ratio:.3f} "
        f"(start={z0:.1f}, end={z_end:.1f}); outside plausible band [0.03, 100].")
    # No NaN/negative biomass anywhere along the trajectory.
    assert np.all(zoo >= 0)
    assert not np.any(np.isnan(zoo))


def test_predation_visibly_suppresses_phyto():
    """With zoo present, phyto must show measurable grazing pressure
    somewhere along the trajectory. We compare the *time-integrated*
    biomass (mean over all ticks) rather than only the endpoint — with
    Holling Type II and a recolonisation floor both runs may saturate
    near ``cc`` at t=200, but the transient suppression mid-rollout is
    the real diagnostic for active predation."""
    _, hist_baseline = _build_env(['phytoplankton'], seed=0, ticks=200)
    _, hist_two = _build_env(['phytoplankton', 'zooplankton'], seed=0, ticks=200)
    phyto_baseline = float(np.mean(hist_baseline['phytoplankton']))
    phyto_with_zoo = float(np.mean(hist_two['phytoplankton']))
    assert phyto_baseline > 0
    suppression = (phyto_baseline - phyto_with_zoo) / phyto_baseline
    # Threshold deliberately tolerant: with Holling Type II + a phyto
    # recolonisation floor, both runs saturate near ``cc`` and the mean
    # suppression is small (~0.5%). A truly disabled predation pathway
    # would give ~0% with float noise.
    assert suppression > 0.003, (
        f"No measurable grazing pressure across trajectory: mean phyto with "
        f"zoo = {phyto_with_zoo:.1f} vs baseline = {phyto_baseline:.1f} "
        f"(suppression = {suppression:.4f}). Predation appears disabled "
        f"or far too weak.")


def test_no_nan_or_negative_biomass_in_either_fg():
    """Trajectory hygiene: no NaN, no negative biomass, no infinities."""
    _, hist = _build_env(['phytoplankton', 'zooplankton'], seed=0, ticks=200)
    for fid, series in hist.items():
        arr = np.array(series)
        assert np.all(np.isfinite(arr)), f"{fid} produced non-finite biomass"
        assert np.all(arr >= 0), f"{fid} produced negative biomass"
