"""Warm-up before a training rollout (section 200).

A world starts from sampled totals that ignore the season: phytoplankton
far above the model's own level for the date, zooplankton 8-30x its
winter stock. The first month or two of every rollout was the world
settling, and the zooplankton policy partly learned in that transient.

With ``--warmup-days D`` each world first runs D days with the
UNPERTURBED policies (the current theta), shared by every candidate
evaluated on that world, and the reward starts from the state it
reaches. Shared, not per candidate, so the +/- candidates still start
from the same state (common random numbers, section 150) and nothing a
candidate does goes unscored. The calendar runs on through the warm-up:
the scored rollout starts D days after the sampled start day, which is
still uniform over the year with ``start_day_of_year: random``.

Both trainers use it: the GPU trainer once per world and update, the CPU
trainer inside every rollout task (the +/- tasks of a pair share seed
and base weights, and the warm-up runs on a stream seeded from that seed,
so they reach the same state).
"""
import math

import numpy as np

from lib.world.tick_time import ticks_per_day


def warmup_ticks(days, tick_hours=None):
    """Whole ticks in ``days`` at the run's tick length (0 = off)."""
    days = float(days or 0.0)
    if not math.isfinite(days) or days < 0.0:
        raise ValueError("--warmup-days must be a finite number >= 0")
    return int(round(days * ticks_per_day(tick_hours)))


def add_warmup_argument(parser):
    """Register ``--warmup-days`` (shared by train.py and train_gpu.py)."""
    parser.add_argument(
        "--warmup-days", "--warmup_days", dest="warmup_days", type=float,
        default=0.0,
        help=("Days each training world runs with the unperturbed policies "
              "before the reward starts (default 0 = off). Lets the "
              "plankton settle from the sampled start totals; shared by "
              "the +/- candidates of a world. Section 200."))


def run_warmup(env, ticks, seed=None):
    """Advance a CPU env ``ticks`` ticks with its current policies.

    The CPU tick draws its FG order and the seeding noise from the global
    ``np.random``, which is not re-seeded per rollout task. With ``seed``
    (the delta pair's world seed) the warm-up runs on its own stream, so
    the +/- tasks of a pair reach the same state on any worker; the
    previous global state is restored afterwards, so the scored rollout
    draws exactly as it did without a warm-up.

    The observation-normalisation and action-statistic accumulators are
    cleared afterwards, so the scored rollout reports only its own ticks
    (the GPU trainer never merges warm-up observations either).
    """
    saved = None
    if seed is not None:
        saved = np.random.get_state()
        np.random.seed((int(seed) ^ 0x5EED200) & 0x7FFFFFFF)
    try:
        for _ in range(int(ticks)):
            observation = env.get_observation()
            actions = env.policy_controller.forward(observation)
            env.step(actions)
    finally:
        if saved is not None:
            np.random.set_state(saved)
    env._obs_sum = None
    env._obs_sumsq = None
    env._obs_count = 0
    env._action_entropy_sum = None
