"""Minimal random-agent inference runner with visualization.

No CLI arguments, no checkpoint loading, no project files, no progress logs.
Run with:

    python simple_inference.py
"""

import numpy as np
import torch

from lib.config.config_loader import setup_full_mareld_mvp
from lib.environments.ecosystem import EcosystemEnvironment
from lib.viz.pygame_viz import LiveVisualizer


GRID_SIZE = (60, 60)
TICKS = 1000
SEED = 0


class RandomPolicy:
    """Uniform random legal actions via zero logits before env-side masking."""

    def __init__(self, output_dim):
        self.output_dim = int(output_dim)

    def get_action_logits_torch(self, flat_state):
        return torch.zeros((flat_state.shape[0], self.output_dim),
                           dtype=torch.float32)


def build_environment():
    height, width = GRID_SIZE
    fgs = setup_full_mareld_mvp(grid_size=GRID_SIZE, seed=SEED,
                                spawn_seed=SEED)
    env = EcosystemEnvironment(
        {
            "width": width,
            "height": height,
            "cell_size": 1000.0,
            "tick_duration": 6.0,
        },
        fgs,
        None,
        None,
        None,
        True
    )

    env._build_static_caches()
    output_dim = 5 + env.N_all
    env.policies = {fid: RandomPolicy(output_dim) for fid in env.dm_ids}
    env._batched_ready = False
    return env


def update_action_fractions(viz, env, tick, previous):
    move = getattr(env, "_action_move_frac", None)
    rest = getattr(env, "_action_rest_frac", None)
    eat = getattr(env, "_action_eat_frac", None)
    counts = getattr(env, "_action_active_ticks", None)
    if move is None or rest is None or eat is None or counts is None:
        return

    for i, fid in enumerate(env.dm_ids):
        count_now = float(counts[i])
        count_prev = float(previous.get((i, "count"), 0.0))
        delta_count = count_now - count_prev
        previous[(i, "count")] = count_now

        move_prev = float(previous.get((i, "move"), 0.0))
        rest_prev = float(previous.get((i, "rest"), 0.0))
        eat_prev = float(previous.get((i, "eat"), 0.0))
        previous[(i, "move")] = float(move[i])
        previous[(i, "rest")] = float(rest[i])
        previous[(i, "eat")] = float(eat[i])

        if delta_count <= 0.0:
            continue

        move_pct = 100.0 * (float(move[i]) - move_prev) / delta_count
        rest_pct = 100.0 * (float(rest[i]) - rest_prev) / delta_count
        eat_pct = 100.0 * (float(eat[i]) - eat_prev) / delta_count
        viz.update_series("move", fid, move_pct, step=tick)
        viz.update_series("rest", fid, rest_pct, step=tick)
        viz.update_series("eat", fid, eat_pct, step=tick)
        viz.update_action_fracs(fid, move_pct, rest_pct, eat_pct)


def main():
    env = build_environment()
    fg_ids = list(env.fgs.keys())
    ndm_ids = [fid for fid, fg in env.fgs.items()
               if not fg.is_decision_maker]
    viz = LiveVisualizer(
        fg_ids=fg_ids,
        grid_shape=GRID_SIZE,
        mode="inference",
        plot_fg_ids=fg_ids,
        ndm_ids=ndm_ids,
        title="Simple random inference",
    )

    start_biomass = {
        fid: max(float(fg.biomass.sum()), 1e-12)
        for fid, fg in env.fgs.items()
    }
    start_energy = {
        fid: max(float(getattr(fg, "energy_reserve", np.array([0.0])).sum()),
                 1e-12)
        for fid, fg in env.fgs.items()
    }
    previous_actions = {}

    def push_visual_frame(tick):
        viz.update_biomass(env.fgs, tick=tick)
        for fid, fg in env.fgs.items():
            biomass_pct = 100.0 * float(fg.biomass.sum()) / start_biomass[fid]
            energy = getattr(fg, "energy_reserve", None)
            energy_total = float(energy.sum()) if energy is not None else 0.0
            energy_pct = 100.0 * energy_total / start_energy[fid]
            viz.update_series("biomass", fid, biomass_pct, step=tick)
            viz.update_series("energy", fid, energy_pct, step=tick)

    try:
        viz.begin_rollout_recording()
        push_visual_frame(0)
        for tick in range(1, TICKS + 1):
            env.step()
            push_visual_frame(tick)
            update_action_fractions(viz, env, tick, previous_actions)
            if not viz.pump_events():
                break
        viz.end_rollout_recording()
        viz.wait_for_close()
    finally:
        viz.close()


if __name__ == "__main__":
    main()
