"""Regression tests for the Holling functional response (Section 64).

Purpose: lock the SHAPE of the attack-rate curve computed by
``EcosystemEnvironment._holling_a_eff`` so the Section 64 bug (a missing
``B_prey`` factor in the numerator, which made ``a_eff`` *fall* toward 0
at high prey density) cannot be reintroduced by a future refactoring.

Four contracts are asserted per (predator, prey) pair with h > 0:

1. **Zero point** -- ``a_eff(0) == 0``. The buggy Type II gave
   ``a_eff = a`` at B=0, i.e. full "trying to eat" in empty cells.
2. **Monotonicity** -- ``a_eff(B)`` is strictly increasing in B. This is
   exactly the property the bug violated (buggy Type II was strictly
   *decreasing*).
3. **Asymptote** -- ``a_eff(B) -> 1/h`` as B grows, and ``a_eff <= 1/h``
   everywhere. Catches mis-scaling between numerator and denominator.
4. **Type III shape** -- at low B the Type III response lies BELOW Type II
   (low-density refuge) while both converge to the same 1/h ceiling.
   Catches a mix-up of ``Bp`` and ``Bp2``.

These are mathematical contracts on the implemented response, NOT an
ecological calibration check: they say nothing about whether a and h are
plausible values (that is Section 36 / tools/starve_calibration.py).
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


# Prey-density raster spanning six decades around the calibrated
# ton/cell working range (Section 63.4).
B_GRID = np.geomspace(1e-3, 1e4, 400)

# (a, h) pairs from the mareld2 calibration (Section 36 / 64.3).
AH_CASES = [
    (0.25, 4.0),    # zooplankton -> phytoplankton
    (0.04, 25.0),   # pelagic_fish -> zooplankton
    (0.02, 50.0),   # gadoids -> pelagic_fish (order of magnitude)
    (0.1, 3.0),     # off-calibration control: a*h != 1
]


def _a_eff(a, h, B, m3):
    return EcosystemEnvironment._holling_a_eff(
        np.float64(a), np.float64(h), np.asarray(B, dtype=np.float64),
        np.float64(m3))


@pytest.mark.parametrize("a,h", AH_CASES)
@pytest.mark.parametrize("m3", [0.0, 1.0], ids=["type2", "type3"])
def test_a_eff_zero_at_zero_prey(a, h, m3):
    """Contract 1: no intake attempt in an empty cell."""
    assert _a_eff(a, h, np.array([0.0]), m3)[0] == 0.0


@pytest.mark.parametrize("a,h", AH_CASES)
@pytest.mark.parametrize("m3", [0.0, 1.0], ids=["type2", "type3"])
def test_a_eff_strictly_increasing_in_prey(a, h, m3):
    """Contract 2: the response saturates upward, never downward.

    This is the direct regression guard against the Section 64 sign
    inversion.
    """
    vals = _a_eff(a, h, B_GRID, m3)
    d = np.diff(vals)
    assert np.all(d > 0.0), (
        f"a_eff not strictly increasing for a={a}, h={h}, m3={m3}: "
        f"{int(np.sum(d <= 0.0))} of {d.size} steps non-increasing "
        f"(first at B={B_GRID[int(np.argmax(d <= 0.0))]:.4g})")


@pytest.mark.parametrize("a,h", AH_CASES)
@pytest.mark.parametrize("m3", [0.0, 1.0], ids=["type2", "type3"])
def test_a_eff_bounded_by_physiological_ceiling(a, h, m3):
    """Contract 3: ``a_eff <= 1/h`` everywhere and -> 1/h at large B."""
    ceiling = 1.0 / h
    vals = _a_eff(a, h, B_GRID, m3)
    assert np.all(vals <= ceiling * (1.0 + 1e-12)), (
        f"a_eff exceeded 1/h = {ceiling} for a={a}, h={h}, m3={m3}: "
        f"max = {vals.max()}")
    far = float(_a_eff(a, h, np.array([1e9]), m3)[0])
    assert far == pytest.approx(ceiling, rel=1e-4), (
        f"a_eff did not approach 1/h = {ceiling} at B=1e9 "
        f"(got {far}) for a={a}, h={h}, m3={m3}")


@pytest.mark.parametrize("a,h", AH_CASES)
def test_type3_gives_low_density_refuge(a, h):
    """Contract 4: Type III below Type II at low B, same ceiling at high B."""
    low = 1.0 / (10.0 * a * h)  # well inside the sub-saturation regime
    ii_low = float(_a_eff(a, h, np.array([low]), 0.0)[0])
    iii_low = float(_a_eff(a, h, np.array([low]), 1.0)[0])
    assert 0.0 < iii_low < ii_low, (
        f"Type III must offer a low-density refuge at B={low:.4g} "
        f"(a={a}, h={h}): got III={iii_low:.6g} vs II={ii_low:.6g}")
    high = 1e9
    ii_high = float(_a_eff(a, h, np.array([high]), 0.0)[0])
    iii_high = float(_a_eff(a, h, np.array([high]), 1.0)[0])
    assert ii_high == pytest.approx(1.0 / h, rel=1e-4)
    assert iii_high == pytest.approx(1.0 / h, rel=1e-4)


def test_section_64_numeric_table():
    """The verification table in Section 64.3 (a=0.04, h=25).

    Note: the Type III entry at B=0.01 is 3.9996e-6, not the 4e-8 printed
    in the resume table (a*B^2 = 0.04*1e-4 = 4e-6); the resume value is a
    typo, the shape contract is unaffected.
    """
    a, h = 0.04, 25.0
    b = np.array([0.01, 1.0, 10.0, 100.0])
    ii = _a_eff(a, h, b, 0.0)
    iii = _a_eff(a, h, b, 1.0)
    np.testing.assert_allclose(
        ii, [0.00039604, 0.02, 0.0363636, 0.0396040], rtol=1e-4)
    np.testing.assert_allclose(
        iii, [3.9996e-6, 0.02, 0.0396040, 0.0399960], rtol=1e-4)


def test_mixture_mask_selects_per_predator_branch():
    """The (N_dm, 1, 1, 1) mask must pick Type III per predator row."""
    a = np.full((2, 1, 1, 1), 0.04, dtype=np.float64)
    h = np.full((2, 1, 1, 1), 25.0, dtype=np.float64)
    Bp = np.array([1.0, 10.0, 100.0], dtype=np.float64).reshape(1, 1, 1, 3)
    m3 = np.array([1.0, 0.0], dtype=np.float64).reshape(2, 1, 1, 1)
    mixed = EcosystemEnvironment._holling_a_eff(a, h, Bp, m3)
    pure_iii = _a_eff(0.04, 25.0, Bp.ravel(), 1.0)
    pure_ii = _a_eff(0.04, 25.0, Bp.ravel(), 0.0)
    np.testing.assert_allclose(mixed[0].ravel(), pure_iii)
    np.testing.assert_allclose(mixed[1].ravel(), pure_ii)


def test_live_project_matrices_satisfy_the_contracts():
    """Same contracts, but with the a/h matrices of the live project.

    Guards against a calibration edit that pushes some (pred, prey) pair
    into a degenerate corner of the response.
    """
    fgs = setup_full_mareld_mvp(grid_size=(8, 8), seed=0, spawn_seed=0)
    grid_config = {'width': 8, 'height': 8, 'cell_size': 1000.0,
                   'tick_duration': 6.0}
    env = EcosystemEnvironment(grid_config, fgs, {})
    env._build_static_caches()
    if env.N_dm == 0:
        pytest.skip("No decision makers in the project configuration")

    a_mat = np.asarray(env.max_intake_mat, dtype=np.float64)
    h_mat = np.asarray(env.handling_time_mat, dtype=np.float64)
    mask = np.asarray(env._type3_pred_mask, dtype=np.float64).reshape(-1)

    checked = 0
    for i, pred_id in enumerate(env.dm_ids):
        for j, prey_id in enumerate(env.global_fg_order):
            a, h = float(a_mat[i, j]), float(h_mat[i, j])
            if a <= 0.0 or h <= 0.0:
                continue
            checked += 1
            vals = _a_eff(a, h, B_GRID, mask[i])
            label = f"{pred_id} -> {prey_id} (a={a:g}, h={h:g})"
            assert vals[0] > 0.0, f"{label}: zero response over whole raster"
            assert np.all(np.diff(vals) > 0.0), (
                f"{label}: a_eff is not strictly increasing in B_prey")
            assert np.all(vals <= (1.0 / h) * (1.0 + 1e-12)), (
                f"{label}: a_eff exceeds the 1/h ceiling")
    assert checked > 0, (
        "No (predator, prey) pair with h > 0 found - the Holling branch "
        "is inactive, so this regression guard is vacuous")
