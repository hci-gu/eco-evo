"""Structural guard for reference-grid spawn invariance (Section 70).

``config_loader`` scales the initial TOTAL biomass of every FG by
``biomass_scale = (H_act*W_act) / (ref_H*ref_W)``. For the density per
cell to be preserved, whatever concentrates that biomass spatially has to
scale with it. For ``mode: colony`` that is ``n_colonies``: with a fixed
colony count a 16x16 grid spread 1/14 of the herring over the full 25
schools, peak density fell from ~103 t/cell to ~11 t/cell, and NO cell in
the whole world cleared the porpoise break-even requirement of ~5.3 t
visible herring derived in Section 69. That is why a ``--grid 16*16``
training run sees porpoises die immediately while the 60x60 rollouts of
Section 69 do not.

Two defects were fixed and are both locked here:

1. ``config_loader`` never put ``biomass_scale`` into the spawn
   ``env_context``, so ``_apply_grid_scaling`` was dead code.
2. The rule itself was wrong. ``cell_size`` is a fixed physical length
   (1000 m), so a smaller grid is a SMALLER WORLD, not a coarser sampling
   of the same one. Length params (``sigma_cells``, ``scale``) are
   therefore already grid-independent, while the COUNT of colonies is
   extensive and must scale with area. The old code did the opposite:
   ``sqrt(biomass_scale)`` on the lengths, nothing on the count.

Contracts asserted:

1. **Wiring** - the real loader passes ``biomass_scale`` into the spawn
   context, and its value is the area ratio.
2. **Count scales with area** - ``n_colonies`` is multiplied by
   ``biomass_scale`` and floored at 1.
3. **Lengths are untouched** - ``sigma_cells`` / ``scale`` come out
   exactly as configured. This is the direct regression guard against the
   old ``sqrt`` rule.
4. **Reference grid is a no-op** - at ``biomass_scale == 1.0`` the params
   are returned unchanged, so 60x60 runs are bit-for-bit preserved.
5. **Density invariance (synthetic)** - the same colony spec on a small
   and a large grid puts the same FRACTION of cells above a given
   density, and a comparable peak density.
6. **Density invariance (live project)** - the same, measured on
   ``mareld2.yaml`` herring, expressed as the fraction of cells where a
   porpoise clears the Section 69 break-even requirement.
"""
import os
import sys

import numpy as np
import pytest
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import lib.config.config_loader as config_loader  # noqa: E402
import lib.spawn.strategies as strategies  # noqa: E402
from lib.spawn.strategies import (  # noqa: E402
    StrategySpec,
    _apply_grid_scaling,
    make_weights,
)

PROJECT = os.path.join(_ROOT, "mareld2.yaml")

# mareld2.yaml reference grid.
REF = (60, 60)
SMALL = (16, 16)

# Section 69 break-even for porpoises preying on herring: a*h = 0.05*20 = 1,
# so a_eff/a = B_vis/(1 + B_vis), and the required ratio is 0.84.
REQ_A_EFF_RATIO = 0.84

LIBRARY = os.path.join(_ROOT, "fgconfig", "fg_library.yaml")


def _pair_visibility_floor(predator, prey):
    """Resolve the floor exactly as EcosystemEnvironment does.

    The per-(predator, prey) override in ``interaction_definitions`` wins;
    an absent/empty value inherits the prey FG's own species-level
    ``visibility_floor``. Hardcoding the prey default here would make this
    test lie as soon as a pair override is configured.
    """
    with open(LIBRARY, "r", encoding="utf-8") as fh:
        library = yaml.safe_load(fh) or {}
    idef = (library.get("interaction_definitions", {})
            .get(f"{predator}_preys_on_{prey}") or {})
    val = idef.get("visibility_floor")
    if val in (None, ""):
        val = (library.get("species_definitions", {})
               .get(prey) or {}).get("visibility_floor", 0.0)
    return float(val or 0.0)


HERRING_VISIBILITY_FLOOR = _pair_visibility_floor("porpoises", "pelagic_fish")


def _colony_params(n_colonies=25, sigma_cells=1.0):
    """The pelagic_fish spawn block from fg_library.yaml."""
    return {
        "n_colonies": n_colonies,
        "sigma_cells": sigma_cells,
        "anchor": "free",
        "amplitude_mode": "jitter",
        "amplitude_min": 0.0,
        "amplitude_max": 1.0,
    }


def _area_ratio(grid):
    return (grid[0] * grid[1]) / float(REF[0] * REF[1])


# ---------------------------------------------------------------------------
# Contract 1: the mechanism is actually wired up.
# ---------------------------------------------------------------------------

def test_loader_passes_biomass_scale_into_the_spawn_context(monkeypatch):
    """The key was missing for the mechanism's entire lifetime."""
    seen = []
    original = strategies._apply_grid_scaling

    def spy(params, context):
        seen.append(dict(context) if context else None)
        return original(params, context)

    monkeypatch.setattr(strategies, "_apply_grid_scaling", spy)
    config_loader.load_project_config(PROJECT, grid_size=SMALL, seed=1,
                                      spawn_seed=1)

    assert seen, "_apply_grid_scaling was never reached by the loader"
    for ctx in seen:
        assert ctx is not None
        assert "biomass_scale" in ctx, (
            "biomass_scale missing from the spawn context - the grid "
            "invariance mechanism is dead code again")
        assert float(ctx["biomass_scale"]) == pytest.approx(
            _area_ratio(SMALL))


def test_loader_biomass_scale_is_one_on_the_reference_grid(monkeypatch):
    seen = []
    original = strategies._apply_grid_scaling

    def spy(params, context):
        seen.append(dict(context) if context else None)
        return original(params, context)

    monkeypatch.setattr(strategies, "_apply_grid_scaling", spy)
    config_loader.load_project_config(PROJECT, grid_size=REF, seed=1,
                                      spawn_seed=1)

    assert seen
    for ctx in seen:
        assert float(ctx["biomass_scale"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Contract 2: the colony COUNT scales with area.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid", [(16, 16), (30, 30), (60, 60), (120, 120)])
def test_n_colonies_scales_with_the_area_ratio(grid):
    bs = _area_ratio(grid)
    out = _apply_grid_scaling(_colony_params(n_colonies=25),
                              {"biomass_scale": bs})
    assert out["n_colonies"] == max(1, int(round(25 * bs)))


def test_n_colonies_is_floored_at_one():
    """A degenerate grid must still get a colony, never zero."""
    out = _apply_grid_scaling(_colony_params(n_colonies=25),
                              {"biomass_scale": 1e-6})
    assert out["n_colonies"] == 1


def test_n_colonies_scaling_is_monotone_in_the_area_ratio():
    counts = [_apply_grid_scaling(_colony_params(n_colonies=25),
                                  {"biomass_scale": _area_ratio((n, n))}
                                  )["n_colonies"]
              for n in (8, 16, 30, 60, 90, 120)]
    assert all(b >= a for a, b in zip(counts, counts[1:])), counts
    assert counts[-1] > counts[0]


# ---------------------------------------------------------------------------
# Contract 3: length parameters are NOT scaled (regression on the old rule).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid", [(16, 16), (30, 30), (120, 120)])
def test_sigma_cells_is_not_rescaled(grid):
    """Under constant cell_size a colony of sigma=1 is the same physical
    size on every grid. The old sqrt(biomass_scale) rule shrank it."""
    params = _colony_params(sigma_cells=3.0)
    out = _apply_grid_scaling(params, {"biomass_scale": _area_ratio(grid)})
    assert out["sigma_cells"] == pytest.approx(3.0)


@pytest.mark.parametrize("grid", [(16, 16), (30, 30), (120, 120)])
def test_perlin_scale_is_not_rescaled(grid):
    params = {"scale": 12.0, "octaves": 4, "persistence": 0.5,
              "lacunarity": 2.0, "threshold": 0.0}
    out = _apply_grid_scaling(params, {"biomass_scale": _area_ratio(grid)})
    assert out["scale"] == pytest.approx(12.0)
    assert out["octaves"] == 4


# ---------------------------------------------------------------------------
# Contract 4: the reference grid is untouched.
# ---------------------------------------------------------------------------

def test_reference_grid_is_a_no_op():
    params = _colony_params()
    out = _apply_grid_scaling(params, {"biomass_scale": 1.0})
    assert out == params


def test_missing_context_is_a_no_op():
    params = _colony_params()
    assert _apply_grid_scaling(params, None) == params
    assert _apply_grid_scaling(params, {}) == params


def test_non_colony_params_are_passed_through():
    params = {"floor": 0.0, "noise_amp": 0.0, "min_frac_of_max": 0.1}
    out = _apply_grid_scaling(params, {"biomass_scale": _area_ratio(SMALL)})
    assert out == params


def test_unparseable_n_colonies_does_not_raise():
    out = _apply_grid_scaling({"n_colonies": "many"},
                              {"biomass_scale": _area_ratio(SMALL)})
    assert out["n_colonies"] == "many"


# ---------------------------------------------------------------------------
# Contract 5: density invariance, synthetic.
# ---------------------------------------------------------------------------

def _density_map(grid, total_ref, seed):
    """Biomass per cell for a colony spec, with the loader's own scaling.

    ``total_ref`` is the reference-grid total; the loader multiplies it by
    ``biomass_scale``, so this mirrors the real pipeline end to end.
    """
    bs = _area_ratio(grid)
    spec = StrategySpec(mode="colony", params=_colony_params(), seed=seed)
    w = make_weights(spec, grid, project_seed=seed,
                     context={"biomass_scale": bs})
    return np.asarray(w) / max(float(np.sum(w)), 1e-300) * (total_ref * bs)


@pytest.mark.parametrize("grid", [(16, 16), (30, 30)])
def test_fraction_of_dense_cells_is_grid_invariant(grid):
    """Same spec, same density threshold -> same share of the world."""
    threshold = 5.0        # tonnes/cell
    total_ref = 12000.0    # mareld2 herring, order of magnitude
    ref_fracs, small_fracs = [], []
    for seed in range(1, 6):
        ref = _density_map(REF, total_ref, seed)
        small = _density_map(grid, total_ref, seed)
        ref_fracs.append(float((ref >= threshold).mean()))
        small_fracs.append(float((small >= threshold).mean()))
    ref_mean = float(np.mean(ref_fracs))
    small_mean = float(np.mean(small_fracs))
    assert ref_mean > 0.0
    # Colony centres are drawn on a coarse integer lattice, so exact
    # equality is not available; a factor of 2 either way still separates
    # "invariant" from the 0-viable-cells regime the bug produced.
    assert 0.5 * ref_mean <= small_mean <= 2.0 * ref_mean, (
        f"{grid}: {small_mean:.4f} vs reference {ref_mean:.4f}")


@pytest.mark.parametrize("grid", [(16, 16), (30, 30)])
def test_peak_density_is_grid_invariant(grid):
    total_ref = 12000.0
    ref_peak = float(np.mean([_density_map(REF, total_ref, s).max()
                              for s in range(1, 6)]))
    small_peak = float(np.mean([_density_map(grid, total_ref, s).max()
                                for s in range(1, 6)]))
    assert 0.4 * ref_peak <= small_peak <= 2.5 * ref_peak, (
        f"{grid}: peak {small_peak:.1f} vs reference {ref_peak:.1f}")


# ---------------------------------------------------------------------------
# Contract 6: density invariance on the live project, in the units that
# actually decide whether a porpoise starves.
# ---------------------------------------------------------------------------

def _viable_fraction(grid, seed):
    """Share of cells where visible herring clears the Section 69 need."""
    result = config_loader.load_project_config(
        PROJECT, grid_size=grid, seed=seed, spawn_seed=seed)
    fgs = result[0] if isinstance(result, tuple) else result
    herring = np.asarray(fgs["pelagic_fish"].biomass, dtype=np.float64)
    visible = HERRING_VISIBILITY_FLOOR * herring
    a_eff_ratio = visible / (1.0 + visible)     # a*h = 1 for this pair
    return float((a_eff_ratio >= REQ_A_EFF_RATIO).mean())


def test_live_project_viable_cell_fraction_is_grid_invariant():
    """The regression that made --grid 16*16 unusable for porpoises.

    Before the fix the small grid had literally zero viable cells on four
    of five seeds, against ~5 % on the reference grid.
    """
    ref = [_viable_fraction(REF, s) for s in range(1, 6)]
    small = [_viable_fraction(SMALL, s) for s in range(1, 6)]
    assert min(small) > 0.0, (
        "no cell in the whole world lets a porpoise break even: "
        f"{small}")
    ref_mean, small_mean = float(np.mean(ref)), float(np.mean(small))
    assert 0.5 * ref_mean <= small_mean <= 2.0 * ref_mean, (
        f"16x16 {small_mean:.4f} vs 60x60 {ref_mean:.4f}")


def test_live_project_reference_grid_still_has_viable_cells():
    """Guards the other direction: the fix must not disturb 60x60."""
    fracs = [_viable_fraction(REF, s) for s in range(1, 6)]
    assert min(fracs) > 0.0
    assert float(np.mean(fracs)) > 0.01
