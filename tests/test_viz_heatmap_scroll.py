"""Heatmap scroll view, heatmap/plot splitter and window resize.

The window starts capped to the desktop (the dummy SDL driver reports
1024x768), so the status bar, the playback bar and the plot stay on
screen; the heatmaps live in a viewport between them, scrolled with the
mouse wheel or the scrollbar at its right edge. Controls scrolled out of
view must not claim clicks. A vertical splitter shares the width between
heatmaps and plot, and the window is resizable; both re-run the layout.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FGS = ["zooplankton", "pelagic_fish", "gadoids", "porpoises", "seals",
       "phytoplankton", "subsurface_phytoplankton", "benthic_community",
       "microzooplankton"]
GRID = (60, 60)


def make_viewer(monkeypatch, n_fgs, max_height=None):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    if max_height is None:
        monkeypatch.delenv("MARELD_VIZ_MAX_HEIGHT", raising=False)
    else:
        monkeypatch.setenv("MARELD_VIZ_MAX_HEIGHT", str(max_height))
    from lib.viz.pygame_viz import LiveVisualizer
    viewer = LiveVisualizer(fg_ids=FGS[:n_fgs], grid_shape=GRID, mode="training")
    if not viewer.enabled:
        pytest.skip("pygame display unavailable")
    viewer.set_b0_defaults({fid: 10.0 for fid in FGS[:n_fgs]})
    return viewer


def event(viewer, kind, **kw):
    pg = viewer._pg
    return viewer._handle_layout_event(
        SimpleNamespace(type=getattr(pg, kind), **kw))


def wheel(viewer, y, pos):
    viewer._pg.mouse.set_pos(pos)
    return event(viewer, "MOUSEWHEEL", x=0, y=y)


def slider_fids(viewer):
    return {fid for _rect, fid in viewer._slider_rects}


def drag_splitter(viewer, to_x):
    sx, sy, sw, sh = viewer._splitter_rect
    start = (sx + sw // 2, sy + sh // 2)
    assert event(viewer, "MOUSEBUTTONDOWN", button=1, pos=start)
    assert event(viewer, "MOUSEMOTION", pos=(to_x, start[1]))
    assert event(viewer, "MOUSEBUTTONUP", button=1, pos=(to_x, start[1]))


def assert_layout_consistent(viewer):
    W, H = viewer._win_w, viewer._win_h
    vx, vy, vw, vh = viewer._hm_view_rect
    sx, sy, sw, sh = viewer._splitter_rect
    px, py, pw, ph = viewer._plot_rect
    bx, by, bw, bh = viewer._playback_rect
    assert vx + vw <= sx and sx + sw == px
    assert px + pw <= W
    assert by == vy + vh and by + bh <= H and bw == W
    assert viewer._cols * viewer._panel_w <= vw


def test_tall_grid_fits_the_screen_and_scrolls(monkeypatch):
    viewer = make_viewer(monkeypatch, len(FGS))
    try:
        assert viewer._win_h <= 768 - 100
        assert viewer._win_w <= 1024
        assert_layout_consistent(viewer)
        assert viewer._hm_scrollbar_rect is not None
        vx, vy, vw, vh = viewer._hm_view_rect
        assert vh < viewer._hm_block_h

        viewer._render_full()
        last = FGS[-1]  # last row
        assert last not in slider_fids(viewer)
        for rect, _fid in viewer._slider_rects:
            assert vy <= rect[1] and rect[1] + rect[3] <= vy + vh

        # Wheel outside the viewport (over the plot) does not scroll.
        plot = viewer._plot_rect
        assert not wheel(viewer, -1, (plot[0] + 10, plot[1] + 10))
        assert viewer._hm_scroll_y == 0

        for _ in range(200):
            wheel(viewer, -1, (vx + 10, vy + 10))
        assert viewer._hm_scroll_y == viewer._hm_block_h - vh
        viewer._render_full()
        assert last in slider_fids(viewer)
        assert FGS[0] not in slider_fids(viewer)

        for _ in range(200):
            wheel(viewer, 1, (vx + 10, vy + 10))
        assert viewer._hm_scroll_y == 0
    finally:
        viewer.close()


def test_scrollbar_track_click_pages_and_thumb_drags(monkeypatch):
    viewer = make_viewer(monkeypatch, len(FGS))
    try:
        viewer._render_full()
        sx, sy, sw, sh = viewer._hm_scrollbar_rect
        assert event(viewer, "MOUSEBUTTONDOWN", button=1,
                     pos=(sx + 5, sy + sh - 2))
        assert viewer._hm_scroll_y == min(viewer._hm_max_scroll(),
                                          viewer._hm_view_rect[3])
        viewer._render_full()
        tx, ty, tw, th = viewer._hm_thumb_rect
        assert event(viewer, "MOUSEBUTTONDOWN", button=1, pos=(sx + 5, ty + 2))
        assert viewer._dragging_hm_scroll
        assert event(viewer, "MOUSEMOTION", pos=(sx + 5, sy))
        assert viewer._hm_scroll_y == 0
        assert event(viewer, "MOUSEBUTTONUP", button=1, pos=(sx + 5, sy))
        assert not viewer._dragging_hm_scroll
    finally:
        viewer.close()


def test_short_grid_keeps_the_natural_layout(monkeypatch):
    viewer = make_viewer(monkeypatch, 2)
    try:
        assert viewer._hm_scrollbar_rect is None
        assert viewer._hm_view_rect[3] == viewer._hm_block_h
        assert viewer._cols == 2
        assert viewer.cell_px == viewer._min_cell_px
        assert_layout_consistent(viewer)
    finally:
        viewer.close()


def test_height_override(monkeypatch):
    viewer = make_viewer(monkeypatch, len(FGS), max_height=500)
    try:
        assert viewer._win_h <= 500
        assert viewer._hm_scrollbar_rect is not None
        viewer._render_full()
    finally:
        viewer.close()


def test_splitter_shares_the_width(monkeypatch):
    viewer = make_viewer(monkeypatch, 2)
    try:
        cell0 = viewer.cell_px
        plot_w0 = viewer._plot_rect[2]
        # Wider heatmap area: bigger cells, narrower plot.
        drag_splitter(viewer, viewer._win_w - viewer._min_plot_w - 20)
        assert viewer.cell_px > cell0
        assert viewer._plot_rect[2] < plot_w0
        assert viewer._plot_rect[2] >= viewer._min_plot_w
        assert_layout_consistent(viewer)
        viewer._render_full()
        # Dragged far left: clamped to one reference panel, one column,
        # never below the reference cell size.
        drag_splitter(viewer, 0)
        assert viewer._splitter_rect[0] == viewer._min_hm_area_w
        assert viewer._cols == 1
        assert viewer.cell_px == viewer._min_cell_px
        assert_layout_consistent(viewer)
        viewer._render_full()
    finally:
        viewer.close()


def test_resize_relayouts_and_keeps_the_share(monkeypatch):
    viewer = make_viewer(monkeypatch, len(FGS))
    try:
        pg = viewer._pg
        frac = viewer._split_frac
        viewer._screen = pg.display.set_mode(
            (1600, 1100), pg.SWSURFACE | pg.RESIZABLE)
        assert event(viewer, "VIDEORESIZE", w=1600, h=1100, size=(1600, 1100))
        assert (viewer._win_w, viewer._win_h) == (1600, 1100)
        assert viewer._playback_rect[1] + viewer._playback_rect[3] <= 1100
        assert abs(viewer._splitter_rect[0] / 1600 - frac) < 0.01
        assert_layout_consistent(viewer)
        viewer._render_full()
    finally:
        viewer.close()
