"""Each impact map's "restart" box in the inference viewer (section 225).

Ticked (the default), releasing the map's intensity slider starts a new
rollout, as every other slider does. Unticked, the running world takes
the new intensity (k x the loaded field) and goes on from the tick it has
reached; after the end of the rollout the same world runs another rollout
length. b0, length and spawn changes always restart.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from inference import build_env, run_inference
from lib.runners.policy import PolicyNetwork

GRID = (5, 6)


def _world():
    # The tick shuffles the FG order with the global RNG.
    np.random.seed(0)
    env = build_env("mareld2.yaml", GRID, seed=7, verbose=False)
    env.build_static_caches()
    return env


def _trained(env):
    torch.manual_seed(0)
    policies = {fid: PolicyNetwork(int(env.per_dm_in_dim[i]), 5 + env.N_all).eval()
                for i, fid in enumerate(env.dm_ids)}
    mean = np.zeros((env.N_dm, int(env.max_in_dim)), dtype=np.float32)
    var = np.ones_like(mean)
    return policies, mean, var


class FakeViz:
    """The slice of LiveVisualizer that a live impact change touches."""

    enabled = True

    def __init__(self):
        self._quit = False
        self._spawn_dirty = False
        self._impact_dirty = False
        self._impact_live_dirty = False
        self._b0_dirty = False
        self._ticks_dirty = False
        self.scales = {}
        self.ticks = 3
        self.waits = 0
        self.on_wait = []

    def wait_for_close(self, banner=None):
        self.waits += 1
        if self.on_wait:
            self.on_wait.pop(0)(self)
        else:
            self._quit = True

    def consume_impact_live_change(self):
        d, self._impact_live_dirty = self._impact_live_dirty, False
        return d

    def get_impact_scales(self):
        return dict(self.scales)

    def get_ticks(self):
        return self.ticks

    def pump_events(self):
        return True

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None


def trawl_off(live):
    def act(v):
        v.scales = {"bottom_trawling": 0.0}
        if live:
            v._impact_live_dirty = True
        else:
            v._impact_dirty = True
    return act


def _base(env):
    return {"bottom_trawling": env.grid.get_map("bottom_trawling").copy()}


def test_unticked_impact_continues_the_world_after_the_end():
    env = _world()
    policies, mean, var = _trained(env)
    viz = FakeViz()
    viz.on_wait = [trawl_off(live=True)]
    history = run_inference(env, policies, mean, var, 2, verbose=False,
                            viz=viz, base_impact_maps=_base(env))
    # 2 ticks, then another rollout length (3) in the same world.
    assert len(history["phytoplankton"]) == 5
    assert viz.waits == 2
    assert float(env.grid.get_map("bottom_trawling").max()) == 0.0


def test_ticked_impact_returns_to_the_caller_to_restart():
    env = _world()
    policies, mean, var = _trained(env)
    viz = FakeViz()
    viz.on_wait = [trawl_off(live=False)]
    history = run_inference(env, policies, mean, var, 2, verbose=False,
                            viz=viz, base_impact_maps=_base(env))
    assert len(history["phytoplankton"]) == 2
    # The flag is left for main() to restart on; the world is untouched.
    assert viz._impact_dirty is True
    assert float(env.grid.get_map("bottom_trawling").max()) > 0.0


def test_b0_change_after_the_end_always_restarts():
    env = _world()
    policies, mean, var = _trained(env)
    viz = FakeViz()

    def b0(v):
        v._b0_dirty = True

    viz.on_wait = [b0]
    history = run_inference(env, policies, mean, var, 2, verbose=False,
                            viz=viz, base_impact_maps=_base(env))
    assert len(history["phytoplankton"]) == 2
    assert viz._b0_dirty is True


def test_unticked_impact_applies_mid_run_without_stopping():
    env = _world()
    policies, mean, var = _trained(env)
    viz = FakeViz()
    calls = {"n": 0}

    def pump():
        calls["n"] += 1
        if calls["n"] == 1:
            trawl_off(live=True)(viz)
        return True

    viz.pump_events = pump
    history = run_inference(env, policies, mean, var, 3, verbose=False,
                            viz=viz, base_impact_maps=_base(env))
    assert len(history["phytoplankton"]) == 3
    assert float(env.grid.get_map("bottom_trawling").max()) == 0.0
