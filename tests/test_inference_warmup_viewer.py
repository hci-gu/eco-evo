"""inference.py --visual during the warm-up (section 297).

The viewer is pumped while the world warms up; a b0 slider released then
stops the stale warm-up, and the loop starts over once with the new value
(the change flags are cleared before the controls are read, so the change
is neither lost nor restarted forever). Closing the window during the
warm-up skips the baseline worlds' warm-ups.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRID = "5*6"


def _checkpoints(directory):
    """A policy_<fg>.pth per decision maker of the 5 x 6 world."""
    from inference import build_env
    from lib.runners.policy import PolicyNetwork
    env = build_env("mareld2.yaml", (5, 6), seed=7, verbose=False)
    env.build_static_caches()
    torch.manual_seed(0)
    for i, fid in enumerate(env.dm_ids):
        net = PolicyNetwork(int(env.per_dm_in_dim[i]), 5 + env.N_all)
        torch.save({"state_dict": net.state_dict()}, directory / f"policy_{fid}.pth")


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Run inference.main with a dummy-SDL viewer; ``on_pump(viz, n)`` is
    called at each pump during a warm-up. Returns the viewer and the
    warm-ups as (ticks, completed)."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import lib.viz
    import lib.runners.warmup as warmup
    from lib.viz.pygame_viz import LiveVisualizer
    _checkpoints(tmp_path)

    def go(on_pump, extra=()):
        created, warmups = [], []

        class Viewer(LiveVisualizer):
            warm_pumps = 0

            def pump_events(self):
                alive = super().pump_events()
                if "warmup" in self._status:
                    self.warm_pumps += 1
                    on_pump(self, self.warm_pumps)
                return alive and not self._quit

            def wait_for_close(self, banner=None):
                self._quit = True

        def create(**kwargs):
            viz = Viewer(**kwargs)
            created.append(viz)
            return viz

        original = warmup.settle

        def settle(*args, **kwargs):
            done = original(*args, **kwargs)
            warmups.append((int(args[1]), done))
            return done

        monkeypatch.setattr(lib.viz, "LiveVisualizer", create)
        monkeypatch.setattr(warmup, "settle", settle)
        monkeypatch.setattr(sys, "argv", [
            "inference.py", "--project", "mareld2.yaml", "--grid", GRID,
            "--checkpoints", str(tmp_path), "--visual", "--engine", "cpu",
            "--warmup-days", "2", "--ticks", "2", "--seed", "7", "--no-capture",
            *extra])
        import inference
        try:
            inference.main()
        finally:
            for viz in created:
                viz.close()
        return created[0], warmups

    return go


def test_a_slider_released_during_the_warmup_restarts_once(run):
    def release(viz, n):
        if n == 3 and not getattr(viz, "released", False):
            viz.released = True
            viz._b0_dirty = True

    viz, warmups = run(release)
    # 2 days = 8 ticks: the first warm-up stopped at the release, the
    # second ran whole, and nothing restarted after it.
    assert warmups == [(8, False), (8, True)]
    assert viz._b0_dirty is False
    assert "warmup" not in viz._status


def test_closing_the_window_skips_the_baseline_warmups(run):
    def close(viz, n):
        if n == 2:
            viz._quit = True

    viz, warmups = run(close, extra=("--rnd_baseline", "solo"))
    # The main world's warm-up stopped; every solo baseline world returned
    # before warming up at all.
    assert warmups == [(8, False)]
