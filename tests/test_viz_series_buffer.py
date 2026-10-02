"""The plot series buffer must behave like the deque it replaced, and the
per-pixel decimation must keep each column's ink (first/min/max/last)."""
from collections import deque

import numpy as np

from lib.viz.pygame_viz import _SeriesBuffer, _decimate_polyline, _series_arrays


def test_buffer_matches_bounded_deque():
    buf, ref = _SeriesBuffer(maxlen=7), deque(maxlen=7)
    for t in range(1000):
        item = (t, float(np.sin(t)))
        buf.append(item)
        ref.append(item)
        assert len(buf) == len(ref)
    assert list(buf) == list(ref)
    assert buf[0] == ref[0] and buf[-1] == ref[-1]
    steps, vals, is_sorted = _series_arrays(buf)
    assert is_sorted and steps.tolist() == [s for s, _ in ref]
    buf.clear()
    assert not buf and len(buf) == 0


def test_unbounded_buffer_keeps_every_point():
    buf = _SeriesBuffer()
    for t in range(20000):
        buf.append((t, t * 0.5))
    assert len(buf) == 20000 and buf[0] == (0, 0.0) and buf[-1] == (19999, 9999.5)


def test_out_of_order_step_clears_sorted_flag():
    buf = _SeriesBuffer(maxlen=3)
    for s in (5, 6, 2):
        buf.append((s, 0.0))
    assert not buf.is_sorted
    for s in (3, 4):
        buf.append((s, 0.0))  # the 6 -> 2 drop has left the window
    assert buf.is_sorted


def test_decimation_keeps_column_extremes_and_order():
    rng = np.random.default_rng(0)
    xs = np.sort(rng.integers(0, 50, size=5000))
    ys = rng.integers(0, 300, size=5000)
    dx, dy = _decimate_polyline(xs, ys)
    assert len(dx) == 4 * len(np.unique(xs))
    assert np.all(np.diff(dx) >= 0)
    for c in np.unique(xs):
        col = ys[xs == c]
        got = dy[dx == c]
        assert got[0] == col[0] and got[-1] == col[-1]
        assert got.min() == col.min() and got.max() == col.max()


def _viewer(mode):
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    return LiveVisualizer(fg_ids=["gadoids"], grid_shape=(4, 4), mode=mode)


def test_inference_ticks_slider_reaches_50000():
    v = _viewer("inference")
    v.set_ticks_default(50000)
    assert v.get_ticks() == 50000
    assert v._ticks_value_from_pos(1.0) == 50000
    v.set_ticks_default(10 ** 6)
    assert v.get_ticks() == 50000
    assert _viewer("train")._ticks_max == 25000


def test_long_recording_is_evenly_strided_and_ends_on_last_tick():
    v = _viewer("inference")
    v._MAX_ROLLOUT_FRAMES = 100
    v._MIN_FRAME_INTERVAL = 1e9  # no rendering, recording only
    v.begin_rollout_recording()
    n = 1003
    for t in range(1, n + 1):
        v.update_biomass({"gadoids": np.full((4, 4), float(t))}, tick=t)
    v.end_rollout_recording()
    ticks = [f["tick"] for f in v._current_rollout]
    assert len(ticks) <= 101
    assert ticks[-1] == n
    gaps = set(np.diff(ticks[:-1]).tolist())
    assert gaps == {v._capture_stride} and v._capture_stride == 16
