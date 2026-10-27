"""Playback bar of the training viewer (section 223.24).

Between two recorded rollouts nothing feeds the heatmaps, so the live
view equals the film's last frame: the live/replay button had nothing to
return to and is gone. The bar blinks when a new film arrives in any
playback mode (a viewer that has scrubbed is paused on the new film's
last frame), and playback starts at 60 fps.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = (4, 4)
FGS = ["benthivorous_fish", "phytoplankton"]
BLINK_BG = (90, 70, 30)


@pytest.fixture
def viz(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz import pygame_viz
    # A clock on an even half second keeps the 2 Hz blink in its "on" phase.
    monkeypatch.setattr(pygame_viz.time, "monotonic", lambda: 1000.0)
    v = pygame_viz.LiveVisualizer(fg_ids=FGS, grid_shape=GRID, mode="train", plot_fg_ids=FGS)
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    yield v
    v.close()


def record(v, ticks=4):
    v.begin_rollout_recording()
    for tick in range(1, ticks + 1):
        v.update_biomass({f: np.full(GRID, float(tick), dtype=np.float32) for f in FGS}, tick=tick)
    v.end_rollout_recording()


def button(v, action):
    return next(rect for rect, a in v._playback_rects if a == action)


def blinking(v):
    v._render_full()
    x, y, w, h = button(v, "step_back")
    return tuple(v._screen.get_at((x + 2, y + h // 2)))[:3] == BLINK_BG


def test_no_live_button_and_60_fps(viz):
    record(viz)
    viz._render_full()
    actions = [a for _, a in viz._playback_rects]
    assert "to_live" not in actions
    assert "toggle_train_pause" in actions
    assert viz._playback_fps == 60.0


def test_new_film_blinks_while_paused(viz):
    record(viz)
    viz._handle_playback_action("step_back")
    assert viz._playback_mode == "paused"
    assert not blinking(viz)
    record(viz, ticks=6)
    assert viz._playback_mode == "paused"
    assert viz._playback_idx == len(viz._current_rollout) - 1
    assert blinking(viz)
    viz._handle_playback_action("step_back")
    assert not blinking(viz)
