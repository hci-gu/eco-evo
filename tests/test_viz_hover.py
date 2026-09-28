"""The plot's hover tooltip during the perturbation rollouts.

``_draw_plot`` draws the tooltip from the live mouse position, so it
exists only while something repaints. A PROBE rollout repaints every
tick through ``update_biomass``; the perturbation rollouts do not call
it at all, and the only thing running is ``pump_events`` from the
trainer's pump callback - which repainted on clicks and keys but not on
plain pointer movement. The tooltip was therefore visible during the
probe and nowhere else.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = (4, 4)
FGS = ["gadoids", "pelagic_fish"]


@pytest.fixture
def viz(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=FGS, grid_shape=GRID, mode="train",
                       plot_fg_ids=FGS)
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    # Enough history for the plot to have an area and draw lines.
    for step in range(20):
        for fid in FGS:
            v.update_series("biomass", fid, 100.0 + step, step=step)
            v.update_reward(fid, float(step), step=step)
    v.update_biomass({f: np.ones(GRID, dtype=np.float32) for f in FGS}, tick=1)
    v._render_full()
    assert v._plot_area_rect is not None, "the plot never laid out an area"
    yield v
    v.close()


def motion(v, pos):
    """Post a plain pointer movement, as the window manager would."""
    pg = v._pg
    pg.event.post(pg.event.Event(pg.MOUSEMOTION, pos=pos, rel=(1, 1),
                                 buttons=(0, 0, 0)))


def count_renders(v, monkeypatch):
    calls = []
    original = v._render_full
    monkeypatch.setattr(v, "_render_full",
                        lambda *a, **kw: (calls.append(1), original(*a, **kw))[1])
    return calls


def inside(v):
    x, y, w, h = v._plot_area_rect
    return (x + w // 2, y + h // 2)


def outside(v):
    # The heatmap block sits above the plot; the top-left corner of the
    # window is never inside the plot area.
    return (2, 2)


def test_moving_over_the_plot_repaints(viz, monkeypatch):
    """This is the whole bug: no repaint, no tooltip."""
    calls = count_renders(viz, monkeypatch)
    viz._last_frame_ts = 0.0
    motion(viz, inside(viz))
    assert viz.pump_events() is True
    assert len(calls) == 1, "pointer movement over the plot did not repaint"
    assert viz._plot_hover is True


def test_moving_elsewhere_does_not_repaint(viz, monkeypatch):
    """Movement across the heatmaps must not cost a redraw per event."""
    calls = count_renders(viz, monkeypatch)
    viz._last_frame_ts = 0.0
    motion(viz, outside(viz))
    viz.pump_events()
    assert calls == []
    assert viz._plot_hover is False


def test_leaving_the_plot_repaints_once_to_erase_the_tooltip(viz, monkeypatch):
    """A tooltip left behind after the pointer has gone is worse than none."""
    motion(viz, inside(viz))
    viz._last_frame_ts = 0.0
    viz.pump_events()
    assert viz._plot_hover is True

    calls = count_renders(viz, monkeypatch)
    viz._last_frame_ts = 0.0
    motion(viz, outside(viz))
    viz.pump_events()
    assert len(calls) == 1, "the tooltip was not erased on the way out"
    assert viz._plot_hover is False

    # ...and the next movement outside costs nothing.
    viz._last_frame_ts = 0.0
    motion(viz, outside(viz))
    viz.pump_events()
    assert len(calls) == 1


def test_hover_repaints_are_frame_capped(viz, monkeypatch):
    """The pointer emits hundreds of events a second; a redraw is ~5 ms."""
    calls = count_renders(viz, monkeypatch)
    viz._last_frame_ts = 0.0
    for _ in range(20):
        motion(viz, inside(viz))
        viz.pump_events()
    assert len(calls) == 1, f"hover redrew {len(calls)} times in one frame budget"


def test_a_drag_still_takes_priority_over_hover(viz, monkeypatch):
    """Motion during a drag must keep forcing an immediate repaint.

    The new plain-motion branch is the last ``elif`` in the chain, so a
    drag claims the event first and keeps its uncapped redraw - dragging
    a slider through a 30 fps cap would feel like it was sticking.
    """
    assert viz._ticks_track_rect is not None, "no ticks slider laid out"
    tx, _ty, tw, _th = viz._ticks_track_rect
    viz._dragging_ticks = True
    calls = count_renders(viz, monkeypatch)
    try:
        for i in range(3):
            motion(viz, (tx + 4 + i * max(1, tw // 8), inside(viz)[1]))
            viz.pump_events()
    finally:
        viz._dragging_ticks = False
    assert len(calls) == 3, "a drag must not be frame-capped like a hover"


def screen(v):
    return v._pg.surfarray.array3d(v._screen).copy()


def test_the_tooltip_appears_with_only_pump_events_running(viz):
    """End to end, in the situation the report was about.

    No ``update_biomass`` anywhere: this is exactly the perturbation
    phase, where the trainer's pump callback is the only thing touching
    the viewer. The pixels have to change when the pointer enters the
    plot, and change back when it leaves.
    """
    pg = viz._pg
    out, over = outside(viz), inside(viz)

    pg.mouse.set_pos(out)
    viz._last_frame_ts = 0.0
    viz._render_full()
    bare = screen(viz)

    pg.mouse.set_pos(over)
    motion(viz, over)
    viz._last_frame_ts = 0.0
    viz.pump_events()
    hovered = screen(viz)
    assert not np.array_equal(bare, hovered), "no tooltip was drawn"

    x, y, w, h = viz._plot_area_rect
    # array3d is indexed [x, y]; the change must be inside the plot area.
    changed = np.argwhere(np.any(bare != hovered, axis=2))
    assert changed.size, "nothing changed at all"
    assert changed[:, 0].min() >= x and changed[:, 0].max() <= x + w
    assert changed[:, 1].min() >= y and changed[:, 1].max() <= y + h

    pg.mouse.set_pos(out)
    motion(viz, out)
    viz._last_frame_ts = 0.0
    viz.pump_events()
    assert np.array_equal(screen(viz), bare), "the tooltip was left behind"
