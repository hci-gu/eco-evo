"""The warm-up keeps a live viewer answering.

The warm-up (section 274) runs on the viewer's thread in inference.py and
in the training viewer's probe; without its events drained the window
stopped answering until the warm-up was done. ``settle(..., pump=)`` calls
the pump after every CPU tick and while the tensor engine computes, the
viewer pump shows the progress in the status row, and an explicit False
from it stops the warm-up early (inference.py: window closed, or a control
that restarts the rollout).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.runners.warmup import clear_viewer_pump, settle, viewer_pump
from test_warmup_reanchor import _env, _policies, _total


class _Viewer:
    """The part of LiveVisualizer the pump uses."""

    def __init__(self):
        self.pumps = 0
        self.status = {}
        self.shown = []

    def pump_events(self):
        self.pumps += 1
        return True

    def update_status(self, **kw):
        for key, value in kw.items():
            if value is None:
                self.status.pop(key, None)
            else:
                self.status[key] = value
                self.shown.append(value)


def test_the_cpu_warmup_pumps_every_tick_and_shows_its_progress():
    env, viewer = _env(), _Viewer()
    assert settle(env, 6, seed=5, policies=_policies(env), pump=viewer_pump(viewer)) is True
    assert viewer.pumps == 6
    assert viewer.shown[-1] == 100 and viewer.shown == sorted(viewer.shown)
    clear_viewer_pump(viewer)
    assert "warmup" not in viewer.status


def test_pumping_leaves_the_settled_world_unchanged():
    a, b = _env(), _env()
    policies = _policies(a)
    settle(a, 6, seed=5, policies=policies)
    settle(b, 6, seed=5, policies=policies, pump=viewer_pump(_Viewer()))
    for fid in a.fgs:
        np.testing.assert_array_equal(a.fgs[fid].biomass, b.fgs[fid].biomass, err_msg=fid)


def test_a_stop_ends_the_warmup_early_and_still_finishes_the_world():
    env, viewer = _env(), _Viewer()
    day = int(env.light_start)
    stop_after = 2
    completed = settle(env, 10, seed=5, policies=_policies(env),
                       pump=viewer_pump(viewer, stop=lambda: viewer.pumps >= stop_after))
    assert completed is False
    assert viewer.pumps == stop_after
    # Finished as a warm-up of the ticks run: tick 0 on that day.
    assert env.tick_count == 0
    assert int(env.light_start) == (day + stop_after) % env.light_period


def test_no_viewer_gives_no_pump():
    assert viewer_pump(None) is None


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVIDIA GPU required"))])
def test_the_tensor_warmup_pumps_while_it_computes(device):
    env, viewer = _env(), _Viewer()
    start = {f: _total(env.fgs[f]) for f in env.fgs}
    assert settle(env, 8, seed=5, policies=_policies(env), engine="gpu", device=device,
                  pump=viewer_pump(viewer)) is True
    assert viewer.pumps >= 1
    assert env.tick_count == 0
    assert all(np.isfinite(_total(env.fgs[f])) for f in start)


def test_the_status_row_drops_an_entry_set_to_none():
    from lib.viz.pygame_viz import LiveVisualizer
    viz = LiveVisualizer.__new__(LiveVisualizer)
    viz.enabled, viz._status, viz._dirty = True, {"gen": 1}, False
    viz.update_status(warmup=40)
    assert viz._status == {"gen": 1, "warmup": 40}
    viz.update_status(warmup=None)
    assert viz._status == {"gen": 1}
