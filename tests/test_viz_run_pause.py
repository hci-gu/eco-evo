"""The run's pause button in the playback bar (section 226).

It pauses the run itself - the training loop, or the inference rollout -
and is distinct from the film's play/pause. It sits in its own segment at
the right end of the bar, set off from the film controls (which form one
unit: transport buttons, scrub slider, frame indicator), and is always
clickable, also with capture off (inference --no-capture).
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = (20, 20)
FGS = ["zooplankton", "planktivorous_fish"]


def make(monkeypatch, mode, capture=True):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.delenv("MARELD_VIZ_MAX_HEIGHT", raising=False)
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=FGS, grid_shape=GRID, mode=mode,
                       capture=capture)
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    v.update_biomass({f: np.ones(GRID, dtype=np.float32) for f in FGS}, tick=1)
    return v


def centre(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


def pause_rect(v):
    return next(r for r, a in v._playback_rects if a == "toggle_train_pause")


@pytest.mark.parametrize("mode", ["inference", "train"])
def test_run_pause_sits_right_of_the_film_controls(monkeypatch, mode):
    v = make(monkeypatch, mode)
    try:
        v._render_full()
        px = pause_rect(v)[0]
        others = [r for r, a in v._playback_rects if a != "toggle_train_pause"]
        assert others and all(r[0] + r[2] < px for r in others)
        # The scrub slider ends before the divider too.
        tx, _ty, tw, _th = v._playback_track_rect
        assert tx + tw < px - 10
        bx, _by, bw, _bh = v._playback_rect
        assert px + pause_rect(v)[2] <= bx + bw
    finally:
        v.close()


def test_inference_pause_is_clickable_without_capture(monkeypatch):
    v = make(monkeypatch, "inference", capture=False)
    try:
        v._render_full()
        play = next(r for r, a in v._playback_rects if a == "toggle_play")
        v._handle_click(centre(play))
        assert v._playback_mode == "live"          # no film: inert
        v._handle_click(centre(pause_rect(v)))
        assert v._paused is True
        v._render_full()
        v._handle_click(centre(pause_rect(v)))
        assert v._paused is False
    finally:
        v.close()


def test_the_button_does_not_move_with_its_label_or_language(monkeypatch):
    v = make(monkeypatch, "inference")
    try:
        v._render_full()
        before = pause_rect(v)
        v._paused = True
        for lang in ("en", "sv", "it"):
            v._lang = lang
            v._render_full()
            assert pause_rect(v) == before
    finally:
        v.close()


def test_pump_events_holds_the_rollout_until_resumed(monkeypatch):
    v = make(monkeypatch, "inference", capture=False)
    try:
        v._render_full()
        pos = centre(pause_rect(v))
        pg = v._pg
        v._handle_click(pos)
        assert v._paused

        def resume():
            time.sleep(0.4)
            pg.event.post(pg.event.Event(pg.MOUSEBUTTONDOWN, pos=pos, button=1))

        threading.Thread(target=resume, daemon=True).start()
        t0 = time.monotonic()
        assert v.pump_events() is True
        assert time.monotonic() - t0 >= 0.35
        assert v._paused is False
    finally:
        v.close()
