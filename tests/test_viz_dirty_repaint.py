"""Updates between probes reach the screen without a hover.

With --eval-every > 1 an iteration without a probe only calls
``update_reward`` and ``update_status`` and then ``pump_events``. Those
updates store data; nothing repainted until the pointer moved over the
plot. ``pump_events`` now repaints (frame-capped) whenever data is
waiting, and a repaint clears the flag.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = (4, 4)
FGS = ["benthivorous_fish", "planktivorous_fish"]


@pytest.fixture
def viz(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=FGS, grid_shape=GRID, mode="train",
                       plot_fg_ids=FGS)
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    for step in range(5):
        for fid in FGS:
            v.update_reward(fid, float(step), step=step)
    v.update_biomass({f: np.ones(GRID, dtype=np.float32) for f in FGS}, tick=1)
    v._render_full()
    yield v
    v.close()


def count_renders(v, monkeypatch):
    calls = []
    original = v._render_full
    monkeypatch.setattr(v, "_render_full",
                        lambda *a, **kw: (calls.append(1), original(*a, **kw))[1])
    return calls


def screen(v):
    return v._pg.surfarray.array3d(v._screen).copy()


def test_a_render_clears_the_flag(viz):
    assert viz._dirty is False


def test_an_idle_pump_does_not_repaint(viz, monkeypatch):
    calls = count_renders(viz, monkeypatch)
    viz._last_frame_ts = 0.0
    viz.pump_events()
    assert calls == []


@pytest.mark.parametrize("update", [
    lambda v: v.update_reward(FGS[0], 9.0, step=5),
    lambda v: v.update_series("energy", FGS[0], 50.0, step=5),
    lambda v: v.update_status(gen=3, iter=2, T=1.0),
    lambda v: v.update_loss_breakdown({FGS[0]: {"predation": 1.0}}),
    lambda v: v.update_diet_breakdown({FGS[0]: {FGS[1]: 1.0}}),
    lambda v: v.update_action_fracs(FGS[0], 10.0, 20.0, 70.0),
])
def test_an_update_is_drawn_by_the_next_pump(viz, monkeypatch, update):
    update(viz)
    calls = count_renders(viz, monkeypatch)
    viz._last_frame_ts = 0.0
    viz.pump_events()
    assert len(calls) == 1, "the update waited for a hover"
    assert viz._dirty is False
    # Drawn once, not on every pump after it.
    viz._last_frame_ts = 0.0
    viz.pump_events()
    assert len(calls) == 1


def test_an_update_held_back_by_the_frame_cap_is_kept(viz, monkeypatch):
    calls = count_renders(viz, monkeypatch)
    viz.update_status(gen=3, iter=2, T=1.0)
    viz._last_frame_ts = float("inf")       # inside the frame budget
    viz.pump_events()
    assert calls == [] and viz._dirty is True
    viz._last_frame_ts = 0.0
    viz.pump_events()
    assert len(calls) == 1


def test_the_new_reward_point_appears_on_screen(viz):
    """End to end: an iteration without a probe, then only the pump."""
    before = screen(viz)
    for fid in FGS:
        viz.update_reward(fid, 40.0, step=5)
    viz.update_status(gen=2, iter=1, T=1.0)
    viz._last_frame_ts = 0.0
    viz.pump_events()
    assert not np.array_equal(before, screen(viz))
