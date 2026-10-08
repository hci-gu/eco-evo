"""The viewer hides only cells the model itself would sweep as extinct.

Section 54 zeroed every cell below 0.5 * min_split_biomass before the
viewer summed it. After porpoises went to min_split_biomass 250 kg with
extinction_threshold_factor 0.02, that hid live cells of 5-125 kg and
showed about a third less than biomass.jsonl (section 146). The clamp now
uses the FG's own extinction_threshold_factor.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = (4, 4)
FG = "porpoises"


@pytest.fixture
def viz(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    viewer = LiveVisualizer(fg_ids=[FG], grid_shape=GRID, mode="inference")
    if not viewer.enabled:
        pytest.skip("pygame display unavailable")
    yield viewer
    viewer.close()


def group(factor):
    """Porpoise-like FG in tonnes: split weight 0.25 t, cells 0.1 / 0.003 / 1.0 t."""
    arr = np.zeros(GRID, dtype=np.float32)
    arr[0, 0] = 0.1      # live: above 0.02 * 0.25 = 0.005, below 0.5 * 0.25
    arr[1, 1] = 0.003    # below the sweep threshold: residue
    arr[2, 2] = 1.0
    params = dict(biomass=arr, min_split_biomass=0.25)
    if factor is not None:
        params["extinction_threshold_factor"] = factor
    return SimpleNamespace(**params)


def test_live_cells_above_the_sweep_threshold_are_counted(viz):
    viz.update_biomass({FG: group(0.02)}, tick=0)
    assert viz._totals[FG] == pytest.approx(1.1, rel=1e-6)


def test_the_default_factor_still_hides_half_a_split_weight(viz):
    viz.update_biomass({FG: group(None)}, tick=0)
    assert viz._totals[FG] == pytest.approx(1.0, rel=1e-6)


def test_a_zero_factor_only_hides_numerical_residue(viz):
    viz.update_biomass({FG: group(0.0)}, tick=0)
    assert viz._totals[FG] == pytest.approx(1.103, rel=1e-6)
