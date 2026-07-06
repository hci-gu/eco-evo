"""Keyboard-controlled gadoids play mode with live visualization.

Run with:

    python manual_play.py

Click/focus the pygame window, then use:

    Arrow keys  move all gadoids north/east/south/west for one tick
    E           make all gadoids eat for one tick
    H           make all gadoids hide/rest for one tick
    Q/Esc       quit

All other decision-making species act as random agents.
"""

import sys

import numpy as np
import torch

from lib.config.config_loader import setup_full_mareld_mvp
from lib.environments.ecosystem import EcosystemEnvironment
from lib.viz.pygame_viz import LiveVisualizer


GRID_SIZE = (10, 10)
CONTROLLED_FG = "gadoids"
SEED = 0
LOGIT_STRENGTH = 80.0
FPS_IDLE = 30


class RandomPolicy:
    """Uniform random legal actions via zero logits before env-side masking."""

    def __init__(self, output_dim):
        self.output_dim = int(output_dim)

    def get_action_logits_torch(self, flat_state):
        return torch.zeros((flat_state.shape[0], self.output_dim),
                           dtype=torch.float32)


class ManualPolicy:
    """Single preferred action for every controlled cell."""

    def __init__(self, output_dim):
        self.output_dim = int(output_dim)
        self.action_index = 4

    def set_action(self, action_index):
        self.action_index = int(action_index)

    def get_action_logits_torch(self, flat_state):
        logits = torch.zeros((flat_state.shape[0], self.output_dim),
                             dtype=torch.float32)
        logits[:, self.action_index] = LOGIT_STRENGTH
        return logits


def put_gadoids_in_center(env):
    fg = env.fgs[CONTROLLED_FG]
    total_biomass = float(fg.biomass.sum())
    total_energy = float(fg.energy_reserve.sum())

    fg.biomass[:] = 0.0
    fg.energy_reserve[:] = 0.0
    fg.temp_energy_gains[:] = 0.0

    y = env.grid.height // 2
    x = env.grid.width // 2
    fg.biomass[y, x] = total_biomass
    fg.energy_reserve[y, x] = total_energy


def build_environment():
    height, width = GRID_SIZE
    fgs = setup_full_mareld_mvp(grid_size=GRID_SIZE, seed=SEED,
                                spawn_seed=SEED)
    fgs[CONTROLLED_FG].speed = 1.0
    fgs[CONTROLLED_FG].params["movement_speed"] = 1.0
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
        True,
    )

    env.build_static_caches()
    put_gadoids_in_center(env)

    output_dim = 5 + env.N_all
    env.policies = {}
    for fid in env.dm_ids:
        if fid == CONTROLLED_FG:
            env.policies[fid] = ManualPolicy(output_dim)
        else:
            env.policies[fid] = RandomPolicy(output_dim)
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


def update_visualization(viz, env, tick, start_biomass, start_energy,
                         previous_actions, last_action):
    viz.update_biomass(env.fgs, tick=tick)
    for fid, fg in env.fgs.items():
        biomass_pct = 100.0 * float(fg.biomass.sum()) / start_biomass[fid]
        energy = getattr(fg, "energy_reserve", None)
        energy_total = float(energy.sum()) if energy is not None else 0.0
        energy_pct = 100.0 * energy_total / start_energy[fid]
        viz.update_series("biomass", fid, biomass_pct, step=tick)
        viz.update_series("energy", fid, energy_pct, step=tick)
    update_action_fractions(viz, env, tick, previous_actions)
    viz.update_status(tick=tick, player=CONTROLLED_FG, action=last_action)


def available_eat_targets(env, predator_id):
    i = env.dm_ids.index(predator_id)
    return [
        prey_id
        for j, prey_id in enumerate(env.global_fg_order)
        if float(env.eat_static_mask[i, j]) > 0.0
    ]


def choose_eat_target(env):
    targets = available_eat_targets(env, CONTROLLED_FG)
    if not targets:
        return None

    gadoids = env.fgs[CONTROLLED_FG].biomass
    occupied = gadoids > 0.0
    if np.any(occupied):
        local_scores = {
            prey_id: float(env.fgs[prey_id].biomass[occupied].sum())
            for prey_id in targets
        }
        best_local = max(local_scores, key=local_scores.get)
        if local_scores[best_local] > 0.0:
            return best_local
    return None


def current_gadoids_cell(env):
    biomass = env.fgs[CONTROLLED_FG].biomass
    if not np.any(biomass > 0.0):
        return None
    idx = int(np.argmax(biomass))
    return divmod(idx, env.grid.width)


def can_move(env, action_index):
    cell = current_gadoids_cell(env)
    if cell is None:
        return False
    y, x = cell
    if action_index == 0:
        return y > 0
    if action_index == 1:
        return x < env.grid.width - 1
    if action_index == 2:
        return y < env.grid.height - 1
    if action_index == 3:
        return x > 0
    return True


def action_from_key(pg, key, env):
    moves = {
        pg.K_UP: (0, "move north"),
        pg.K_RIGHT: (1, "move east"),
        pg.K_DOWN: (2, "move south"),
        pg.K_LEFT: (3, "move west"),
    }
    if key in moves:
        action_index, label = moves[key]
        if not can_move(env, action_index):
            return None, f"blocked: {label} would leave the map"
        return action_index, label
    if key == pg.K_h:
        return 4, "hide"
    if key == pg.K_e:
        prey_id = choose_eat_target(env)
        if prey_id is None:
            return None, "blocked: no edible prey in the gadoids cell"
        return 5 + env.global_fg_order.index(prey_id), f"eat {prey_id}"
    return None, None


def newly_pressed_action(pg, keys, previous_keys, env):
    for key in (pg.K_UP, pg.K_RIGHT, pg.K_DOWN, pg.K_LEFT, pg.K_e, pg.K_h):
        if keys[key] and not previous_keys.get(key, False):
            return action_from_key(pg, key, env)
    return None, None


def print_gadoids_summary(env, tick, action_label, previous_biomass):
    fg = env.fgs[CONTROLLED_FG]
    biomass = float(fg.biomass.sum())
    energy = float(fg.energy_reserve.sum())
    cell = current_gadoids_cell(env)
    delta = biomass - previous_biomass
    if np.any(fg.biomass > 0.0):
        level = float(fg.energy_level[fg.biomass > 0.0].mean())
    else:
        level = 0.0
    print(
        f"tick {tick:03d} | {action_label:<18} | "
        f"cell={cell} | B={biomass:.3f} ({delta:+.3f}) | "
        f"E={energy:.1f} | s={level:.3f}"
    )


def print_controls(env):
    prey = ", ".join(available_eat_targets(env, CONTROLLED_FG)) or "none"
    print("")
    print("Manual play is running in the pygame visualizer.")
    print("Focus the visualizer window, then press:")
    print("  arrow keys  move all gadoids")
    print("  E           eat best available local prey")
    print("  H           hide/rest")
    print("  Q/Esc       quit")
    print(f"Gadoids edible prey: {prey}")
    print("")


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
        title="Manual play: gadoids",
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

    print_controls(env)
    previous_actions = {}
    previous_keys = {}
    tick = 0
    last_action = "ready"
    last_reported_biomass = float(env.fgs[CONTROLLED_FG].biomass.sum())

    try:
        viz.begin_rollout_recording()
        update_visualization(viz, env, tick, start_biomass, start_energy,
                             previous_actions, last_action)
        print_gadoids_summary(env, tick, last_action, last_reported_biomass)
        clock = viz._pg.time.Clock()
        while True:
            if not viz.pump_events():
                break

            pg = viz._pg
            keys = pg.key.get_pressed()
            if keys[pg.K_q] or keys[pg.K_ESCAPE]:
                break

            action_index, action_label = newly_pressed_action(
                pg, keys, previous_keys, env)
            if action_index is not None:
                before_biomass = float(env.fgs[CONTROLLED_FG].biomass.sum())
                env.policies[CONTROLLED_FG].set_action(action_index)
                last_action = action_label
                observation = env.get_observation()
                actions = env.policy_controller.forward(observation)
                env.step(actions)
                tick += 1
                update_visualization(viz, env, tick, start_biomass,
                                     start_energy, previous_actions,
                                     last_action)
                print_gadoids_summary(env, tick, last_action, before_biomass)
                last_reported_biomass = float(
                    env.fgs[CONTROLLED_FG].biomass.sum())
            elif action_label:
                print(action_label)

            previous_keys = {
                pg.K_UP: bool(keys[pg.K_UP]),
                pg.K_RIGHT: bool(keys[pg.K_RIGHT]),
                pg.K_DOWN: bool(keys[pg.K_DOWN]),
                pg.K_LEFT: bool(keys[pg.K_LEFT]),
                pg.K_e: bool(keys[pg.K_e]),
                pg.K_h: bool(keys[pg.K_h]),
            }
            clock.tick(FPS_IDLE)
        viz.end_rollout_recording()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        viz.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
