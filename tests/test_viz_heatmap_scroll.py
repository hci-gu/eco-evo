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

FGS = ["zooplankton", "planktivorous_fish", "benthivorous_fish", "porpoises", "seals",
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
    viewer._fold_anim_s = 0.0          # the roll is tested on its own
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
    if viewer._plot_collapsed:
        assert pw == 0
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


# ---- folding the plot in under the heatmaps (sections 190-194) ------

def click_toggle(viewer):
    x, y, w, h = viewer._plot_toggle_rect
    assert event(viewer, "MOUSEBUTTONDOWN", button=1,
                 pos=(x + w // 2, y + h // 2))


def test_fold_button_collapses_and_restores_the_plot(monkeypatch):
    viewer = make_viewer(monkeypatch, 2)
    try:
        viewer._render_full()
        splitter0, plot0 = viewer._splitter_rect, viewer._plot_rect
        toggle0 = viewer._plot_toggle_rect
        view0, cell0, cols0 = viewer._hm_view_rect, viewer.cell_px, viewer._cols
        hm_sb0 = viewer._hm_scrollbar_rect
        frac0 = viewer._split_frac
        win0 = (viewer._win_w, viewer._win_h)
        old_plot_area = viewer._plot_area_rect
        assert viewer._tab_rects and old_plot_area is not None

        click_toggle(viewer)
        assert viewer._plot_collapsed
        # Only the plot folds: the heatmap area, and with it the fold
        # button under the pointer, stays exactly where it was; the
        # window ends at the strip (or the floor).
        assert viewer._plot_toggle_rect[:2] == toggle0[:2]
        assert viewer._splitter_rect[0] == splitter0[0]
        assert viewer._splitter_rect[2] == viewer._collapsed_strip_w
        assert viewer._win_w == max(viewer._min_collapsed_win_w,
                                    splitter0[0] + viewer._collapsed_strip_w)
        assert viewer._win_w < win0[0]
        assert viewer._screen.get_size() == (viewer._win_w, win0[1])
        assert viewer._hm_view_rect == view0
        assert viewer.cell_px == cell0 and viewer._cols == cols0
        assert viewer._hm_scrollbar_rect == hm_sb0
        assert_layout_consistent(viewer)
        viewer._render_full()
        # No stale plot click targets over the heatmaps.
        assert viewer._tab_rects == [] and viewer._legend_rects == []
        assert viewer._plot_area_rect is None
        assert viewer._save_plot_button_rect is None
        assert not viewer._hit_plot_area(
            (old_plot_area[0] + 5, old_plot_area[1] + 5))
        # Folded, the strip has no grip and does not drag (the window's
        # edges size the heatmaps); only its button unfolds.
        sx, sy, sw, sh = viewer._splitter_rect
        assert not viewer._hit_splitter((sx + 2, sy + sh // 2))
        assert not event(viewer, "MOUSEBUTTONDOWN", button=1,
                         pos=(sx + 2, sy + sh // 2))
        assert not viewer._dragging_splitter
        assert not viewer._hit_plot_toggle((sx + 2, sy + sh // 2))
        # The heatmap scrollbar right beside the strip keeps its clicks.
        assert not viewer._hit_splitter((sx - 1, sy + sh // 2))
        assert not viewer._hit_plot_toggle((sx - 1, sy + 5))

        # The button rolls the plot out again, unchanged, button in place.
        click_toggle(viewer)
        assert not viewer._plot_collapsed
        assert viewer._plot_toggle_rect == toggle0
        assert (viewer._win_w, viewer._win_h) == win0
        assert viewer._screen.get_size() == win0
        assert viewer._splitter_rect == splitter0
        assert viewer._plot_rect == plot0
        assert abs(viewer._split_frac - frac0) < 1e-3
        assert viewer._hm_area_fixed is None
        viewer._render_full()
        assert viewer._plot_area_rect == old_plot_area
    finally:
        viewer.close()


def test_fold_rolls_the_window_edge_without_re_creating_it(monkeypatch):
    """Section 194: the window edge glides in steps (ease-out), every
    step painted, the heatmaps and the button never move, and the
    window is resized in place, never through display.set_mode (which
    re-creates the X window here - the flicker of sections 190-193)."""
    viewer = make_viewer(monkeypatch, 2)
    try:
        viewer._render_full()
        viewer._fold_anim_s = 0.12
        toggle0, view0 = viewer._plot_toggle_rect, viewer._hm_view_rect
        widths, renders = [], []
        original = viewer._set_window_width

        def spy(w, render=True):
            original(w, render)
            widths.append(viewer._win_w)
            assert viewer._plot_toggle_rect[:2] == toggle0[:2]
            assert viewer._hm_view_rect == view0
        monkeypatch.setattr(viewer, "_set_window_width", spy)
        original_render = viewer._render_full

        def render_spy():
            original_render()
            renders.append(viewer._plot_anim_img is not None)
        monkeypatch.setattr(viewer, "_render_full", render_spy)
        set_modes = []
        pg = viewer._pg
        real_set_mode = pg.display.set_mode
        monkeypatch.setattr(pg.display, "set_mode",
                            lambda *a, **k: (set_modes.append(a),
                                             real_set_mode(*a, **k))[1])
        w0 = viewer._win_w
        click_toggle(viewer)
        folded = viewer._win_w
        assert len(widths) >= 3
        assert widths == sorted(widths, reverse=True)
        assert widths[0] < w0 and widths[-1] == folded
        assert any(renders)                    # the captured plot rolled
        widths.clear()
        click_toggle(viewer)
        assert len(widths) >= 3
        assert widths == sorted(widths)
        assert widths[-1] == w0
        assert viewer._plot_toggle_rect == toggle0
        assert set_modes == []
    finally:
        viewer.close()


def test_fold_button_is_not_the_splitter_grip(monkeypatch):
    viewer = make_viewer(monkeypatch, 2)
    try:
        x, y, w, h = viewer._plot_toggle_rect
        assert y == viewer._splitter_rect[1] and h < viewer._splitter_rect[3]
        assert not viewer._hit_splitter((x + w // 2, y + h // 2))
        # The splitter still drags from its middle.
        drag_splitter(viewer, viewer._win_w - viewer._min_plot_w - 20)
        assert not viewer._plot_collapsed
    finally:
        viewer.close()


def test_p_key_toggles_and_resize_keeps_the_state(monkeypatch):
    viewer = make_viewer(monkeypatch, len(FGS))
    try:
        pg = viewer._pg
        frac = viewer._split_frac
        win_w0 = viewer._win_w
        hm_w = viewer._splitter_rect[0]
        cols0, cell0 = viewer._cols, viewer.cell_px
        viewer._handle_key(pg.K_p)
        assert viewer._plot_collapsed
        assert viewer._win_w == max(viewer._min_collapsed_win_w,
                                    hm_w + viewer._collapsed_strip_w)
        assert (viewer._cols, viewer.cell_px) == (cols0, cell0)
        # A height-only resize while folded leaves the heatmap area.
        viewer._screen = pg.display.set_mode(
            (viewer._win_w, 1100), pg.SWSURFACE | pg.RESIZABLE)
        assert event(viewer, "VIDEORESIZE", w=viewer._win_w, h=1100,
                     size=(viewer._win_w, 1100))
        assert viewer._plot_collapsed and viewer._win_h == 1100
        assert viewer._splitter_rect[0] == hm_w
        assert not viewer._collapsed_resized
        assert_layout_consistent(viewer)
        viewer._render_full()
        # Unfolding gives the window its earlier width (the new height).
        viewer._handle_key(pg.K_p)
        assert not viewer._plot_collapsed
        assert (viewer._win_w, viewer._win_h) == (win_w0, 1100)
        assert viewer._splitter_rect[0] == hm_w
        assert abs(viewer._splitter_rect[0] / win_w0 - frac) < 0.01
        assert_layout_consistent(viewer)
        viewer._render_full()
    finally:
        viewer.close()


def test_status_sliders_stay_put_when_the_plot_folds(monkeypatch):
    viewer = make_viewer(monkeypatch, 2)
    try:
        viewer._render_full()
        track0 = viewer._ticks_track_rect
        viewer._handle_key(viewer._pg.K_p)
        viewer._render_full()
        assert viewer._ticks_track_rect == track0
    finally:
        viewer.close()


@pytest.mark.parametrize("mode", ["train", "inference"])
def test_scrub_slider_stays_draggable_with_the_plot_folded(monkeypatch, mode):
    """Folded, the window is too narrow for the full frame indicator
    beside the train buttons; the indicator goes compact so the scrub
    slider keeps its room (section 191)."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.delenv("MARELD_VIZ_MAX_HEIGHT", raising=False)
    from lib.viz.pygame_viz import LiveVisualizer
    viewer = LiveVisualizer(fg_ids=FGS, grid_shape=GRID, mode=mode)
    if not viewer.enabled:
        pytest.skip("pygame display unavailable")
    viewer._fold_anim_s = 0.0
    try:
        n = 7300
        viewer._current_rollout = [{"tick": i} for i in range(n)]
        viewer._playback_mode = "paused"
        viewer._playback_idx = 0
        viewer._render_full()
        assert not viewer._playback_compact
        viewer._handle_key(viewer._pg.K_p)
        viewer._render_full()
        assert viewer._plot_collapsed
        # The train buttons leave no room for the full texts.
        assert viewer._playback_compact == (mode == "train")
        track = viewer._playback_track_rect
        assert track is not None
        assert track[2] >= 60 and track[0] + track[2] <= viewer._win_w
        # Click near the right end, then drag to the left end.
        tx, ty, tw, th = track
        viewer._handle_click((tx + tw - 4, ty + th // 2))
        assert viewer._dragging_playback
        assert viewer._playback_idx > 0.9 * n
        viewer._playback_idx = viewer._playback_idx_from_x(tx)
        assert viewer._playback_idx == 0
        # Unfolded again: the full indicator comes back.
        viewer._handle_key(viewer._pg.K_p)
        viewer._render_full()
        assert not viewer._playback_compact
    finally:
        viewer.close()


def test_folded_window_edges_size_the_heatmaps(monkeypatch):
    """Section 195: folded, the strip has no grip; resizing the window
    by its own edges or corners sets the heatmap area (the window up to
    the strip). Unfolding puts the plot back beside the new area at the
    width it had, button where the strip's button was."""
    viewer = make_viewer(monkeypatch, 2)
    try:
        pg = viewer._pg
        viewer._split_frac = 0.0          # one column, the narrowest area
        viewer._compute_layout()
        viewer._render_full()
        plot_w0 = viewer._win_w - viewer._plot_rect[0]
        cell0 = viewer.cell_px
        click_toggle(viewer)
        assert viewer._plot_collapsed and viewer._cols == 1
        viewer._render_full()

        # The user drags the window's right edge out to 760 px.
        new_w = 760
        viewer._screen = pg.display.set_mode(
            (new_w, viewer._win_h), pg.SWSURFACE | pg.RESIZABLE)
        assert event(viewer, "VIDEORESIZE", w=new_w, h=viewer._win_h,
                     size=(new_w, viewer._win_h))
        hm_w = new_w - viewer._collapsed_strip_w
        assert viewer._splitter_rect[0] == hm_w
        assert viewer._splitter_rect[0] + viewer._splitter_rect[2] == new_w
        assert viewer._cols == 2 or viewer.cell_px > cell0
        assert_layout_consistent(viewer)
        viewer._render_full()
        cols, cell = viewer._cols, viewer.cell_px
        toggle = viewer._plot_toggle_rect

        # Unfold: the plot comes back beside the new heatmaps.
        click_toggle(viewer)
        assert not viewer._plot_collapsed
        assert (viewer._cols, viewer.cell_px) == (cols, cell)
        assert viewer._splitter_rect[0] == hm_w
        assert viewer._plot_toggle_rect[:2] == toggle[:2]
        assert abs(viewer._win_w - viewer._plot_rect[0] - plot_w0) <= 1 \
            or viewer._win_w == viewer._max_window_w()
        assert_layout_consistent(viewer)
        viewer._render_full()

        # Narrowed by its edge while folded: the heatmaps follow down.
        click_toggle(viewer)
        narrow = 500
        viewer._screen = pg.display.set_mode(
            (narrow, viewer._win_h), pg.SWSURFACE | pg.RESIZABLE)
        assert event(viewer, "VIDEORESIZE", w=narrow, h=viewer._win_h,
                     size=(narrow, viewer._win_h))
        assert viewer._splitter_rect[0] == max(
            viewer._min_hm_area_w, narrow - viewer._collapsed_strip_w)
        assert_layout_consistent(viewer)
        viewer._render_full()
    finally:
        viewer.close()


# ---- heatmaps grouped by decision makers / non-decision makers (198) -

NDMS = ["phytoplankton", "subsurface_phytoplankton", "benthic_community",
        "microzooplankton"]


def test_heatmaps_are_grouped_under_dm_and_ndm_headers(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.delenv("MARELD_VIZ_MAX_HEIGHT", raising=False)
    from lib.viz.pygame_viz import LiveVisualizer
    # NDMs interleaved in the FG order: the grouping must sort them out.
    order = ["phytoplankton", "zooplankton", "planktivorous_fish",
             "subsurface_phytoplankton", "benthivorous_fish", "porpoises",
             "benthic_community", "seals", "microzooplankton"]
    viewer = LiveVisualizer(fg_ids=order, grid_shape=GRID, mode="training",
                            ndm_ids=NDMS)
    if not viewer.enabled:
        pytest.skip("pygame display unavailable")
    viewer._fold_anim_s = 0.0
    viewer.set_b0_defaults({fid: 10.0 for fid in order})
    try:
        titles = [t for t, _y in viewer._hm_headers]
        assert titles == ["Decision makers", "Non-decision makers"]
        slots = viewer._hm_slots
        fids = [f for f, _x, _y in slots]
        dms = [f for f in order if f not in NDMS]
        assert fids == dms + [f for f in order if f in NDMS]
        cols, pw, ph = viewer._cols, viewer._panel_w, viewer._panel_h
        hh = viewer._hm_header_h
        dm_rows = (len(dms) + cols - 1) // cols
        # DMs right under their header, row by row.
        for i, (fid, x, y) in enumerate(slots[:len(dms)]):
            assert (x, y) == ((i % cols) * pw, hh + (i // cols) * ph)
        # The NDM header starts after the last DM row, NDMs on a new row.
        ndm_header_y = hh + dm_rows * ph
        assert viewer._hm_headers[1][1] == ndm_header_y
        first_ndm = slots[len(dms)]
        assert first_ndm[1] == 0 and first_ndm[2] == ndm_header_y + hh
        ndm_rows = (len(NDMS) + cols - 1) // cols
        assert viewer._hm_block_h == 2 * hh + (dm_rows + ndm_rows) * ph
        assert_layout_consistent(viewer)

        # Every panel is reachable by scrolling, last NDM included.
        vx, vy, vw, vh = viewer._hm_view_rect
        viewer._render_full()
        for _ in range(200):
            wheel(viewer, -1, (vx + 10, vy + 10))
        viewer._render_full()
        assert "microzooplankton" in slider_fids(viewer)
    finally:
        viewer.close()


def test_one_kind_only_keeps_the_untitled_grid(monkeypatch):
    viewer = make_viewer(monkeypatch, len(FGS))
    try:
        assert viewer._hm_headers == [] and viewer._hm_header_h == 0
        cols, pw, ph = viewer._cols, viewer._panel_w, viewer._panel_h
        for i, (fid, x, y) in enumerate(viewer._hm_slots):
            assert fid == FGS[i]
            assert (x, y) == ((i % cols) * pw, (i // cols) * ph)
    finally:
        viewer.close()
