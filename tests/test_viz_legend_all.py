"""Select all / deselect all in the plot legend (section 223.22).

A master checkbox above the legend entries of the active tab: ticked
when every entry is on, a dash when some are, empty when none. A click
turns every entry on, or off when all already are, in training and in
inference alike.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = (4, 4)
FGS = ["benthivorous_fish", "planktivorous_fish", "phytoplankton"]


@pytest.fixture(params=["train", "inference"])
def viz(request, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=FGS, grid_shape=GRID, mode=request.param, plot_fg_ids=FGS)
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    for step in range(5):
        for fid in FGS:
            v.update_series("biomass", fid, step, 100.0 + step)
    v.update_biomass({f: np.ones(GRID, dtype=np.float32) for f in FGS}, tick=1)
    v._active_tab = v._tabs.index("biomass")
    v._render_full()
    yield v
    v.close()


def centre(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


def test_the_master_checkbox_selects_and_deselects_every_entry(viz):
    ids = viz._active_plot_ids()
    assert viz._legend_all_rect is not None
    # Its row sits above the first entry.
    assert viz._legend_all_rect[1] < viz._legend_rects[0][0][1]
    viz._handle_click(centre(viz._legend_all_rect))  # all on -> all off
    assert not any(viz._plot_enabled[f] for f in ids)
    viz._render_full()
    viz._handle_click(centre(viz._legend_all_rect))  # none -> all on
    assert all(viz._plot_enabled[f] for f in ids)
    # Some on: a click selects all again (and clears a solo series).
    viz._render_full()
    viz._handle_click(centre(viz._legend_rects[0][0]))
    assert not viz._plot_enabled[ids[0]]
    viz._solo = ids[1]
    viz._render_full()
    viz._handle_click(centre(viz._legend_all_rect))
    assert all(viz._plot_enabled[f] for f in ids) and viz._solo is None
    viz._render_full()


def test_the_entries_keep_their_own_toggles_under_the_master_row(viz):
    first_rect, first = viz._legend_rects[0]
    viz._handle_click(centre(first_rect))
    assert not viz._plot_enabled[first]
    assert all(viz._plot_enabled[f] for _r, f in viz._legend_rects[1:])
    viz._lang = "sv"
    viz._render_full()  # "alla" in Swedish
