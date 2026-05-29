"""Tester för `lib.spawn.allocator.distribute_with_floor`.

Formaliserar de smoke-test-scenarier som validerades manuellt vid
implementationen. Täcker kontraktet:

1. Sum-konservering: output.sum() == total_b (modulo flyttalsbrus) när
   total_b >= min_per_cell.
2. Per-cell-golv: alla aktiva celler har biomass >= min_per_cell (utom i
   det dokumenterade rest-population-fallet då total_b < min_per_cell).
3. Allowed-mask: celler utanför masken är alltid 0.
4. Edge cases: total_b == 0, weights.sum() == 0, min_per_cell == 0.
5. Determinism: samma input -> identisk output.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.spawn import (  # noqa: E402
    StrategySpec,
    distribute_with_floor,
    make_weights,
    summarize,
)


# ---------------------------------------------------------------------------
# Hjälpare
# ---------------------------------------------------------------------------

def _uniform_weights(H, W, seed=0):
    """Bygg en uniform-vikt-array via spawn-strategin (summa = 1)."""
    spec = StrategySpec(mode="uniform", params={}, seed=seed)
    return make_weights(spec, (H, W), project_seed=seed)


# ---------------------------------------------------------------------------
# 1. Sum-konservering & per-cell-golv
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("total_b,min_per_cell", [
    (12.0, 0.5),     # porpoises-liknande
    (25.0, 0.8),     # seals-liknande
    (2500.0, 0.005), # gadoids-liknande
    (56000.0, 0.0),  # phytoplankton: inget golv
    (1.0, 0.5),      # gränsfall: total = 2 * golv
])
def test_sum_preserved_and_floor_respected(total_b, min_per_cell):
    w = _uniform_weights(60, 60, seed=42)
    out = distribute_with_floor(w, total_b, min_per_cell)
    # Sum-konservering.
    assert out.sum() == pytest.approx(total_b, rel=1e-9, abs=1e-9)
    # Per-cell-golv (för aktiva celler).
    active = out[out > 0.0]
    if min_per_cell > 0 and total_b >= min_per_cell:
        assert active.min() >= min_per_cell - 1e-9, (
            f"Active cell under floor: min={active.min()}, floor={min_per_cell}"
        )


# ---------------------------------------------------------------------------
# 2. Allowed-mask
# ---------------------------------------------------------------------------

def test_allowed_mask_zeros_outside():
    H, W = 20, 20
    w = _uniform_weights(H, W, seed=1)
    mask = np.zeros((H, W), dtype=bool)
    mask[5:15, 5:15] = True  # 10x10 ruta i mitten
    out = distribute_with_floor(w, total_b=50.0, min_per_cell=0.5, allowed_mask=mask)
    # Inget utanför mask.
    assert np.all(out[~mask] == 0.0)
    # Sum-konservering.
    assert out.sum() == pytest.approx(50.0, rel=1e-9)
    # Alla aktiva i masken.
    assert np.all((out > 0.0) <= mask)


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------

def test_total_zero_returns_zeros():
    w = _uniform_weights(10, 10)
    out = distribute_with_floor(w, total_b=0.0, min_per_cell=0.5)
    assert out.sum() == 0.0
    assert out.shape == (10, 10)


def test_zero_weights_returns_zeros():
    w = np.zeros((10, 10), dtype=np.float64)
    out = distribute_with_floor(w, total_b=100.0, min_per_cell=0.5)
    assert out.sum() == 0.0


def test_min_per_cell_zero_is_proportional():
    """Med min_per_cell=0 ska output vara ren proportionell skalning av vikterna."""
    w = _uniform_weights(10, 10, seed=7)
    out = distribute_with_floor(w, total_b=100.0, min_per_cell=0.0)
    # Alla celler aktiva, exakt proportionellt.
    expected = w * (100.0 / w.sum())
    assert np.allclose(out, expected, atol=1e-9)


def test_tiny_total_below_floor_rest_population():
    """total_b < min_per_cell -> allt i 1 cell, golv-violation accepteras."""
    w = _uniform_weights(10, 10, seed=3)
    out = distribute_with_floor(w, total_b=0.3, min_per_cell=0.5)
    active = out[out > 0.0]
    assert active.size == 1
    assert active[0] == pytest.approx(0.3, rel=1e-9)


def test_exact_floor_gives_single_cell():
    """total_b == min_per_cell ska kunna ge minst 1 aktiv cell vid golvet."""
    w = _uniform_weights(10, 10, seed=5)
    out = distribute_with_floor(w, total_b=1.0, min_per_cell=1.0)
    active = out[out > 0.0]
    assert active.size == 1
    assert active[0] == pytest.approx(1.0, rel=1e-9)
    assert active[0] >= 1.0 - 1e-9


# ---------------------------------------------------------------------------
# 4. n_active styrs av golv
# ---------------------------------------------------------------------------

def test_n_active_bounded_by_floor_quota():
    """Antal aktiva celler får inte överstiga floor(total_b / min_per_cell)."""
    w = _uniform_weights(60, 60, seed=11)
    total_b, min_per_cell = 12.0, 0.5
    out = distribute_with_floor(w, total_b, min_per_cell)
    n_active, _, _, _ = summarize(out, min_per_cell)
    n_max = int(np.floor(total_b / min_per_cell))
    assert n_active <= n_max, f"n_active={n_active} > n_max={n_max}"


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------

def test_deterministic_same_input():
    w = _uniform_weights(30, 30, seed=99)
    out1 = distribute_with_floor(w.copy(), 100.0, 0.5)
    out2 = distribute_with_floor(w.copy(), 100.0, 0.5)
    assert np.array_equal(out1, out2)


# ---------------------------------------------------------------------------
# 6. Output-shape & dtype
# ---------------------------------------------------------------------------

def test_output_shape_and_dtype():
    w = _uniform_weights(15, 20, seed=2)
    out = distribute_with_floor(w, 50.0, 0.5)
    assert out.shape == (15, 20)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# 7. Colony-strategi integration
# ---------------------------------------------------------------------------

def test_colony_weights_with_floor():
    """Colony-vikter + golv ska ge koncentrerad fördelning med korrekt summa."""
    spec = StrategySpec(
        mode="colony",
        params={"n_colonies": 3, "sigma_cells": 2.0, "anchor": "free"},
        seed=17,
    )
    w = make_weights(spec, (60, 60), project_seed=17)
    out = distribute_with_floor(w, total_b=12.0, min_per_cell=0.5)
    assert out.sum() == pytest.approx(12.0, rel=1e-9)
    active = out[out > 0.0]
    assert active.size > 0
    assert active.min() >= 0.5 - 1e-9
    # Koncentrationskontroll: max-cellen ska ha mycket mer än medel-aktiv.
    assert active.max() > active.mean()


# ---------------------------------------------------------------------------
# 8. summarize() returnerar konsekventa siffror
# ---------------------------------------------------------------------------

def test_summarize_matches_array():
    w = _uniform_weights(20, 20, seed=8)
    out = distribute_with_floor(w, 50.0, 0.5)
    n_active, total, min_act, max_act = summarize(out, 0.5)
    assert total == pytest.approx(out.sum(), rel=1e-9)
    flat = out[out > 0.0]
    assert n_active == flat.size
    if flat.size > 0:
        assert min_act == pytest.approx(flat.min(), rel=1e-9)
        assert max_act == pytest.approx(flat.max(), rel=1e-9)


def test_summarize_empty():
    out = np.zeros((5, 5))
    n_active, total, min_act, max_act = summarize(out, 0.5)
    assert n_active == 0
    assert total == 0.0
    assert min_act == 0.0
    assert max_act == 0.0
