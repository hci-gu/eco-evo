"""Training-only bookkeeping can be switched off (section 233).

inference.py runs frozen policies with frozen obs stats, so it skips the
running observation sums, and headless also the action diagnostics (the
viewer's move/rest/eat curves, and the inference plot HTML built from
them, need them). Neither may change the world.
"""
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.environments.ecosystem import EcosystemEnvironment  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402


def _env(collect):
    fg = FunctionalGroup("dm", {"is_decision_maker": True,
                                "movement_speed": 1.0, "growth_rate": 0.0})
    rng = np.random.default_rng(3)
    b = rng.uniform(0.5, 2.0, size=(4, 4)).astype(np.float32)
    fg.initialize_state(b.shape, initial_biomass=b, initial_energy_ratio=0.5)
    env = EcosystemEnvironment({"width": 4, "height": 4}, {"dm": fg})
    env.collect_observation_stats = collect
    env.collect_action_diagnostics = collect
    return env


def _run(env, ticks=5):
    for _ in range(ticks):
        env.step(env.policy_controller.forward(env.get_observation()))
    return env.fgs["dm"].biomass.copy(), env.fgs["dm"].energy_reserve.copy()


def test_defaults_collect_everything():
    env = _env(True)
    _run(env, ticks=2)
    assert env._obs_count > 0
    assert env._action_entropy_count == 2


def test_switched_off_collects_nothing_and_leaves_the_world_unchanged():
    on, off = _env(True), _env(False)
    b_on, e_on = _run(on)
    b_off, e_off = _run(off)
    np.testing.assert_array_equal(b_on, b_off)
    np.testing.assert_array_equal(e_on, e_off)
    assert getattr(off, "_obs_sum", None) is None
    assert getattr(off, "_action_entropy_sum", None) is None
