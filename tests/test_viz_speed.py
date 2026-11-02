"""Inference speed slider of the live viewer (section 236): a row under
the biomass scale, 0 holds the rollout, max (the default) runs it as fast
as it goes; ``pace_rollout`` holds inference.py's loop to it."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = (4, 4)
FGS = ["benthivorous_fish", "phytoplankton"]


@pytest.fixture
def viz(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz import pygame_viz
    v = pygame_viz.LiveVisualizer(fg_ids=FGS, grid_shape=GRID, mode="inference")
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    v.update_biomass({f: np.ones(GRID, dtype=np.float32) for f in FGS}, tick=0)
    v._render_full()
    yield v
    v.close()


def centre(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


def test_scale_runs_from_held_to_max(viz):
    assert viz.get_speed() is None                     # default: max
    viz._speed_frac = 0.0
    assert viz.get_speed() == 0.0
    viz._speed_frac = 0.5
    assert viz.get_speed() == pytest.approx(np.sqrt(1000.0))


def test_row_sits_under_the_biomass_scale_on_the_same_track(viz):
    bio, speed = viz._biomass_scale_track_rect, viz._speed_track_rect
    assert speed is not None and bio is not None
    assert speed[1] > bio[1]
    assert (speed[0], speed[2]) == (bio[0], bio[2])
    assert speed[1] + speed[3] <= viz._status_h + 6


def test_dragging_to_the_left_end_holds_and_the_label_resets(viz):
    x, y, w, h = viz._speed_track_rect
    viz._handle_click((x + 1, y + h // 2))
    assert viz._speed_frac == 0.0 and viz._dragging_speed
    viz._dragging_speed = False
    viz._render_full()
    viz._handle_click(centre(viz._speed_label_rect))
    assert viz._speed_frac == 1.0


def test_pace_holds_ticks_to_the_slider_rate(viz):
    viz._speed_frac = 1.0
    started = time.monotonic()
    for _ in range(5):
        assert viz.pace_rollout()
    assert time.monotonic() - started < 0.5            # max: no wait
    viz._speed_frac = (np.log(20.0) - np.log(1.0)) / (np.log(1000.0) - np.log(1.0))
    assert viz.get_speed() == pytest.approx(20.0)
    viz.pace_rollout()
    started = time.monotonic()
    viz.pace_rollout()
    viz.pace_rollout()
    assert time.monotonic() - started >= 2 * 0.05 * 0.9


def test_held_rollout_returns_on_a_restarting_control_or_close(viz):
    viz._speed_frac = 0.0
    viz._ticks_dirty = True
    assert viz.pace_rollout() is True
    viz._ticks_dirty = False
    viz._quit = True
    assert viz.pace_rollout() is False
