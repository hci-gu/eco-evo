"""The heatmap normalisation toggle, asserted on rendered pixels.

Section 94 made the heatmap colour reference the rollout-start per-cell
maximum, so that declining biomass darkens and two ticks can be compared
by eye. That is the right default and the wrong answer when a field
collapses far enough to go black: the structure disappears with the
brightness. 'n' therefore switches the reference to the frame being
drawn -- the behaviour section 94 replaced -- and back.

The contract is about colours, so the tests read the pixels the viewer
actually blitted rather than the reference it computed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = (4, 4)
FG = "mover"


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


def field(peak):
    """A field whose single hot cell holds ``peak``."""
    arr = np.zeros(GRID, dtype=np.float32)
    arr[1, 2] = np.float32(peak)
    arr[0, 0] = np.float32(peak * 0.25)
    return arr


def hot_pixel(viewer, cell=(1, 2)):
    """The rendered colour of ``cell`` in the first FG's heatmap."""
    viewer._render_full()
    ox, oy = viewer._heatmap_origin
    # _draw_one_heatmap: hm_x = px + pad (4), hm_y = py + title_h (104).
    x = ox + 4 + int((cell[1] + 0.5) * viewer.cell_px)
    y = oy + 104 + int((cell[0] + 0.5) * viewer.cell_px)
    return tuple(viewer._screen.get_at((x, y)))[:3]


def lut(viewer, fraction):
    """The viewer's own colour for ``fraction`` of the reference."""
    idx = int(np.clip(fraction * 255.0, 0, 255))
    return tuple(int(c) for c in viewer._lut[idx])


def test_fixed_reference_is_the_default_and_darkens_on_decline(viz):
    """A tenfold collapse must read as a tenfold collapse."""
    assert viz._rolling_heatmap is False
    viz.update_biomass({FG: field(10.0)}, tick=0)
    assert hot_pixel(viz) == lut(viz, 1.0)
    viz.update_biomass({FG: field(1.0)}, tick=1)
    assert hot_pixel(viz) == lut(viz, 0.1)


def test_rolling_normalisation_uses_the_frame_maximum(viz):
    """With 'n' the same collapsed field fills the colour range again."""
    pg = viz._pg
    viz.update_biomass({FG: field(10.0)}, tick=0)
    viz.update_biomass({FG: field(1.0)}, tick=1)
    dim = hot_pixel(viz)

    viz._handle_key(pg.K_n)
    assert viz._rolling_heatmap is True
    assert hot_pixel(viz) == lut(viz, 1.0)
    # The structure the fixed reference flattened is back: the quarter
    # cell is a quarter of the way up the gradient, not a tenth of one.
    assert hot_pixel(viz, cell=(0, 0)) == lut(viz, 0.25)
    assert hot_pixel(viz) != dim

    viz._handle_key(pg.K_n)
    assert viz._rolling_heatmap is False
    assert hot_pixel(viz) == dim


def test_rolling_follows_growth_as_well_as_decline(viz):
    """Above the start reference the fixed scale saturates; rolling does not."""
    viz.update_biomass({FG: field(1.0)}, tick=0)
    viz.update_biomass({FG: field(4.0)}, tick=1)
    # Fixed: the hot cell and the quarter cell both sit at the top.
    assert hot_pixel(viz) == lut(viz, 1.0)
    assert hot_pixel(viz, cell=(0, 0)) == lut(viz, 1.0)

    viz._handle_key(viz._pg.K_n)
    assert hot_pixel(viz) == lut(viz, 1.0)
    assert hot_pixel(viz, cell=(0, 0)) == lut(viz, 0.25)


def test_the_display_scale_slider_still_multiplies_the_reference(viz):
    """The slider zooms whichever reference is in use, not only the fixed one."""
    # Two different references: start max 10, frame max 1. At tick 0 they
    # would coincide and the test could not tell them apart.
    viz.update_biomass({FG: field(10.0)}, tick=0)
    viz.update_biomass({FG: field(1.0)}, tick=1)
    viz._set_biomass_display_scale(2.0)
    assert hot_pixel(viz) == lut(viz, 1.0 / (10.0 * 2.0))

    viz._handle_key(viz._pg.K_n)
    assert hot_pixel(viz) == lut(viz, 1.0 / (1.0 * 2.0))
    viz._reset_biomass_display_scale()
    assert hot_pixel(viz) == lut(viz, 1.0)


def test_an_empty_frame_renders_black_in_both_modes(viz):
    """A zero reference must not divide by zero or paint noise."""
    viz.update_biomass({FG: np.zeros(GRID, dtype=np.float32)}, tick=0)
    assert hot_pixel(viz) == lut(viz, 0.0)
    viz._handle_key(viz._pg.K_n)
    assert hot_pixel(viz) == lut(viz, 0.0)


def test_log_mode_composes_with_both_references(viz):
    """'l' and 'n' are independent: log applies to the reference in use."""
    pg = viz._pg
    viz.update_biomass({FG: field(10.0)}, tick=0)
    viz.update_biomass({FG: field(1.0)}, tick=1)
    viz._handle_key(pg.K_l)
    assert viz._log_heatmap is True
    fixed_log = hot_pixel(viz)
    assert fixed_log == lut(viz, float(np.log1p(1.0) / np.log1p(10.0)))

    viz._handle_key(pg.K_n)
    assert hot_pixel(viz) == lut(viz, 1.0)
    assert hot_pixel(viz) != fixed_log


def test_rolling_follows_the_replayed_frame_not_the_live_one(viz):
    """During playback the reference must come from the frame on screen.

    ``_swap_in_frame`` replaces the live biomass while a recorded frame
    is drawn, and the rolling reference is read from that array, so a
    replayed tick is normalised by its own maximum. The toggle itself is
    deliberately NOT part of the snapshot: it is a view setting, so it
    can be flipped in the middle of a playback.
    """
    viz.update_biomass({FG: field(10.0)}, tick=0)
    frame = viz._capture_frame()
    viz.update_biomass({FG: field(1.0)}, tick=1)
    viz._handle_key(viz._pg.K_n)
    live = hot_pixel(viz)
    assert live == lut(viz, 1.0)

    saved = viz._swap_in_frame(frame)
    try:
        assert "_rolling_heatmap" not in saved
        assert viz._rolling_heatmap is True
        # The replayed frame's own max is its hot cell, so it saturates
        # too - but its quarter cell proves the array really is the old
        # one (0.25 of 10, not of 1).
        assert hot_pixel(viz) == lut(viz, 1.0)
        assert hot_pixel(viz, cell=(0, 0)) == lut(viz, 0.25)
        assert viz._totals[FG] == pytest.approx(12.5)
    finally:
        viz._swap_out_frame(saved)
    assert viz._totals[FG] == pytest.approx(1.25)


def centre(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


@pytest.mark.parametrize("mode", ["train", "inference"])
def test_the_toggle_has_a_visible_button_in_every_mode(monkeypatch, mode):
    """A key nobody can see is not a setting.

    The biomass-scale slider, the natural neighbour for a display
    control, is drawn in inference mode only, so during TRAINING there
    was nothing on screen for the normalisation at all. The button
    therefore lives in the status bar, which every mode draws.
    """
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    viewer = LiveVisualizer(fg_ids=[FG], grid_shape=GRID, mode=mode)
    if not viewer.enabled:
        pytest.skip("pygame display unavailable")
    try:
        viewer.update_biomass({FG: field(10.0)}, tick=0)
        viewer._render_full()
        rect = viewer._hm_norm_button_rect
        assert rect is not None, f"no normalisation button in {mode} mode"

        bx, by, bw, bh = rect
        sx, sy, sw, sh = viewer._status_rect
        assert sx <= bx and bx + bw <= sx + sw, "button hangs off the bar"
        assert sy <= by and by + bh <= sy + sh
        off = tuple(viewer._screen.get_at(centre(rect)))[:3]

        # The click toggles, and the button repaints to say so.
        viewer._handle_click(centre(rect))
        assert viewer._rolling_heatmap is True
        viewer._render_full()
        on = tuple(viewer._screen.get_at(centre(viewer._hm_norm_button_rect)))[:3]
        assert on != off, "the button looks the same in both states"

        viewer._handle_click(centre(viewer._hm_norm_button_rect))
        assert viewer._rolling_heatmap is False

        # A click on the button must never fall through to a heatmap.
        viewer._solo = None
        viewer._handle_click(centre(viewer._hm_norm_button_rect))
        assert viewer._solo is None
        assert viewer._rolling_heatmap is True
    finally:
        viewer.close()


class _SpyFont:
    """Delegating font that records every string it is asked to render."""

    def __init__(self, font, sink):
        self._font = font
        self._sink = sink

    def render(self, text, *a, **kw):
        self._sink.append(text)
        return self._font.render(text, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._font, name)


def test_the_button_keeps_its_slot_in_the_narrowest_window(monkeypatch):
    """The status TEXT yields to the button, not the other way round.

    One FG gives the narrowest window the viewer ever opens, and a
    long-running train run gives the longest status line (five-digit
    tick, simulated time, gen/iter/T). Placing the button after the text
    then pushed it off the right edge; right-aligning it without giving
    it a reserved slot would have put it under the text instead.
    """
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    viewer = LiveVisualizer(fg_ids=[FG], grid_shape=(60, 60), mode="train")
    if not viewer.enabled:
        pytest.skip("pygame display unavailable")
    try:
        viewer.update_biomass({FG: np.zeros((60, 60), dtype=np.float32)},
                              tick=73000,
                              extra={"gen": 12, "iter": 345, "T": 1.234})
        rendered = []
        original = viewer._font_big
        viewer._font_big = _SpyFont(original, rendered)
        try:
            viewer._render_full()
        finally:
            viewer._font_big = original

        x, y, w, h = viewer._status_rect
        bx, by, bw, bh = viewer._hm_norm_button_rect
        line = next(t for t in rendered if t.startswith("mode = "))
        text_end = x + 6 + original.size(line)[0]

        assert bx + bw <= x + w, "button runs off the right edge"
        assert text_end <= bx, "status text runs under the button"
        # The fields that were dropped to make room are the expendable
        # ones; what the row is for survives.
        for kept in ("mode = ", "tick = 73000", "time = 50 y 0 d"):
            assert kept in line, line
    finally:
        viewer.close()
