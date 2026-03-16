#!/usr/bin/env python3
"""Interactive PettingZoo run where you control one training agent."""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import lib.config.const as const
import lib.model as model
from lib.config.settings import Settings, load_settings
from lib.runners.petting_zoo import PettingZooRunner, RandomBaselinePolicy


ACTION_ORDER = ("up", "down", "left", "right", "eat")
ACTION_INDEX = {name: idx for idx, name in enumerate(ACTION_ORDER)}
ACTION_ALIASES = {
    "u": "up",
    "d": "down",
    "l": "left",
    "r": "right",
    "e": "eat",
    "0": "up",
    "1": "down",
    "2": "left",
    "3": "right",
    "4": "eat",
}
ARROW_KEYS = {
    "\x1b[A": "up",     # up arrow
    "\x1b[B": "down",   # down arrow
    "\x1b[D": "left",   # left arrow
    "\x1b[C": "right",  # right arrow
}


def _one_hot(action_name: str) -> np.ndarray:
    vec = np.zeros(model.OUTPUT_SIZE, dtype=np.float32)
    vec[ACTION_INDEX[action_name]] = 1.0
    return vec


def _training_run_params(settings: Settings, explicit_max_steps: int | None) -> Tuple[int | None, float, float]:
    fitness_method = str(settings.fitness_method).strip().lower()
    uses_biomass_style_fitness = fitness_method in {"biomass_pct", "trajectory_shaped"}
    two_stage_enabled = bool(settings.two_stage_eval_enabled) and uses_biomass_style_fitness
    relative_baseline_enabled = bool(settings.relative_baseline_enabled) and fitness_method == "biomass_pct"

    initial_energy_scale = max(0.0, float(settings.training_initial_energy_scale))
    energy_decay_per_cycle = float(np.clip(settings.training_energy_decay_per_cycle, 0.0, 0.95))

    use_custom_pipeline = (
        two_stage_enabled
        or relative_baseline_enabled
        or abs(initial_energy_scale - 1.0) > 1e-9
        or energy_decay_per_cycle > 0.0
    )

    if use_custom_pipeline:
        if uses_biomass_style_fitness:
            short_steps = int(settings.fitness_eval_steps)
        else:
            short_steps = int(settings.max_steps)
        if two_stage_enabled:
            short_steps = int(settings.short_eval_steps)
        max_steps_override = max(1, short_steps)
    else:
        max_steps_override = None
        initial_energy_scale = 1.0
        energy_decay_per_cycle = 0.0

    if explicit_max_steps is not None:
        max_steps_override = max(1, int(explicit_max_steps))

    return max_steps_override, initial_energy_scale, energy_decay_per_cycle


def _resolve_controlled_species(
    species_arg: str,
    runner: PettingZooRunner,
    control_all_age_groups: bool,
) -> List[str]:
    available = list(runner.species_list)
    if species_arg in available:
        return [species_arg]

    matching_base = [
        sp for sp in available
        if runner.species_map[sp].base_species == species_arg
    ]
    if not matching_base:
        raise ValueError(
            f"Unknown species '{species_arg}'. Available acting species: "
            + ", ".join(available)
        )

    matching_base.sort(key=lambda sp: runner.species_map[sp].age_index)
    if control_all_age_groups:
        return matching_base
    return [matching_base[-1]]


def _base_metrics_for_species(runner: PettingZooRunner, species_name: str) -> Tuple[str, float, float]:
    base_name = runner.species_map[species_name].base_species
    total_biomass = 0.0
    total_energy_weighted = 0.0
    for sp, props in runner.species_map.items():
        if props.base_species != base_name:
            continue
        b_idx = model.MODEL_OFFSETS[sp]["biomass"]
        e_idx = model.MODEL_OFFSETS[sp]["energy"]
        biomass = runner.env.world[..., b_idx]
        energy = runner.env.world[..., e_idx]
        bio_sum = float(np.sum(biomass))
        total_biomass += bio_sum
        total_energy_weighted += float(np.sum(biomass * energy))
    mean_energy = total_energy_weighted / max(total_biomass, 1e-8)
    return base_name, total_biomass, mean_energy


def _collect_base_biomass_step_metrics(
    runner: PettingZooRunner,
    world: np.ndarray,
    cell_threshold: float,
) -> Dict[str, float]:
    if world.shape[0] >= 3 and world.shape[1] >= 3:
        world_core = world[1:-1, 1:-1, :]
    else:
        world_core = world

    metrics: Dict[str, float] = {}
    for base_species in const.BASE_SPECIES:
        total_biomass = 0.0
        nonzero_cells = np.zeros(world_core.shape[:2], dtype=bool)
        threshold_cells = np.zeros(world_core.shape[:2], dtype=bool)

        for species_name, props in runner.species_map.items():
            if props.base_species != base_species:
                continue
            b_idx = model.MODEL_OFFSETS[species_name]["biomass"]
            biomass = world_core[:, :, b_idx]
            total_biomass += float(np.sum(biomass))
            nonzero_cells |= biomass > 0.0
            threshold_cells |= biomass > cell_threshold

        metrics[f"{base_species}_biomass"] = float(total_biomass)
        metrics[f"{base_species}_cells_nonzero"] = float(np.count_nonzero(nonzero_cells))
        metrics[f"{base_species}_cells_above_threshold"] = float(np.count_nonzero(threshold_cells))

    return metrics


class HumanPolicy:
    """Interactive policy that applies one action distribution to all cells."""

    def __init__(self, species_name: str, runner: PettingZooRunner):
        self.species_name = species_name
        self.runner = runner
        self.last_action = _one_hot("eat")
        self._printed_key_help = False

    def _read_keypress(self) -> str | None:
        if not sys.stdin.isatty():
            return None

        try:
            import termios
            import tty
        except Exception:
            return None

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            first = sys.stdin.read(1)
            if first == "\x1b":
                second = sys.stdin.read(1)
                if second == "[":
                    third = sys.stdin.read(1)
                    return first + second + third
                return first + second
            return first
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _parse_key_action(self, key: str) -> Tuple[np.ndarray | None, str | None]:
        if key in ARROW_KEYS:
            return _one_hot(ARROW_KEYS[key]), None

        lowered = key.lower()
        if lowered in {"e", " "}:
            return _one_hot("eat"), None
        if lowered in {"q"}:
            self.runner.env.reason = "manual_quit"
            for agent in self.runner.env.agents:
                self.runner.env.terminations[agent] = True
            return self.last_action.copy(), None
        if lowered in {"\r", "\n"}:
            return self.last_action.copy(), None
        if lowered in {"h", "?"}:
            return None, (
                "Keys: arrows=move, e/space=eat, Enter=repeat last, q=quit."
            )
        return None, (
            f"Unmapped key '{repr(key)}'. Use arrows, e, Enter, h, or q."
        )

    def _parse_action(self, raw: str) -> Tuple[np.ndarray | None, str | None]:
        text = raw.strip().lower()
        if not text:
            return self.last_action.copy(), None

        if text in {"help", "h", "?"}:
            return None, (
                "Use one action (`up/down/left/right/eat`, `u/d/l/r/e`, or `0-4`) "
                "or a weighted mix like `up=0.2 right=0.3 eat=0.5`."
            )

        if text in {"none", "noop", "stay"}:
            return np.zeros(model.OUTPUT_SIZE, dtype=np.float32), None

        mapped = ACTION_ALIASES.get(text, text)
        if mapped in ACTION_INDEX:
            return _one_hot(mapped), None

        if "=" in text:
            values = np.zeros(model.OUTPUT_SIZE, dtype=np.float32)
            pieces = text.replace(",", " ").split()
            for piece in pieces:
                if "=" not in piece:
                    return None, f"Invalid token '{piece}'. Expected key=value."
                key, value_text = piece.split("=", 1)
                key = ACTION_ALIASES.get(key, key)
                if key not in ACTION_INDEX:
                    return None, f"Unknown action key '{key}'."
                try:
                    value = float(value_text)
                except ValueError:
                    return None, f"Invalid numeric value '{value_text}'."
                if value < 0:
                    return None, "Action weights must be non-negative."
                values[ACTION_INDEX[key]] = value

            total = float(values.sum())
            if total <= 0:
                return None, "At least one action weight must be > 0."
            return values / total, None

        return None, f"Could not parse '{raw}'. Type 'help' for formats."

    def _request_manual_action(self) -> np.ndarray:
        base_name, biomass, mean_energy = _base_metrics_for_species(self.runner, self.species_name)
        cycle = int(self.runner.env.cycle_count)
        print(
            f"\n[{self.species_name}] cycle={cycle} base={base_name} "
            f"biomass={biomass:.2f} mean_energy={mean_energy:.2f}"
        )
        if not self._printed_key_help:
            print(
                "Controls: arrows=move, e/space=eat, Enter=repeat, q=quit, h=help."
            )
            self._printed_key_help = True

        while True:
            key = self._read_keypress()
            if key is not None:
                action, error = self._parse_key_action(key)
            else:
                try:
                    command = input(
                        "Action (`up/down/left/right/eat`, `u/d/l/r/e`, `0-4`, mix `up=..`, empty=repeat, quit=stop):\n> "
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    command = "quit"
                lowered = command.lower()
                if lowered in {"quit", "q", "exit"}:
                    self.runner.env.reason = "manual_quit"
                    for agent in self.runner.env.agents:
                        self.runner.env.terminations[agent] = True
                    return self.last_action.copy()
                action, error = self._parse_action(command)

            if action is not None:
                self.last_action = action.astype(np.float32)
                return self.last_action.copy()
            print(error)

    def forward(self, x: np.ndarray) -> np.ndarray:
        chosen_action = self._request_manual_action()
        return np.tile(chosen_action, (x.shape[0], 1)).astype(np.float32)


def _build_candidates(
    runner: PettingZooRunner,
    controlled_species: List[str],
    python_seed: int | None,
) -> Dict[str, object]:
    seed = python_seed if python_seed is not None else random.randrange(2**32)
    rng = random.Random(seed)
    controlled_set = set(controlled_species)
    candidates: Dict[str, object] = {}
    for species_name in runner.species_list:
        if species_name in controlled_set:
            candidates[species_name] = HumanPolicy(species_name=species_name, runner=runner)
        else:
            candidates[species_name] = RandomBaselinePolicy(seed=rng.randrange(2**31))
    return candidates


def _cycle_summary_callback_factory(
    runner: PettingZooRunner,
    fitness_history: List[Tuple[int, float]],
    biomass_step_rows: List[Dict[str, float]],
    cell_threshold: float,
):
    last_cycle_reported = -1

    def _callback(world, fitness, done):
        nonlocal last_cycle_reported
        current_cycle = int(runner.env.cycle_count)
        if current_cycle != last_cycle_reported:
            fitness_history.append((current_cycle, float(fitness)))
            row: Dict[str, float] = {
                "cycle": float(current_cycle),
                "accumulated_fitness": float(fitness),
            }
            row.update(
                _collect_base_biomass_step_metrics(
                    runner=runner,
                    world=world,
                    cell_threshold=cell_threshold,
                )
            )
            biomass_step_rows.append(row)

            summary_parts = []
            for base in const.BASE_SPECIES:
                summary_parts.append(
                    f"{base}:B={row[f'{base}_biomass']:.1f},"
                    f"nz={int(row[f'{base}_cells_nonzero'])},"
                    f"sig={int(row[f'{base}_cells_above_threshold'])}"
                )
            totals_str = " | ".join(summary_parts)
            print(
                f"[cycle {current_cycle}] accumulated_fitness={float(fitness):.3f} | "
                f"{totals_str}"
            )
            last_cycle_reported = current_cycle
        if done:
            print(f"[done] reason={runner.env.reason or 'completed'}")

    return _callback


def _persist_and_show_fitness_history(
    history: List[Tuple[int, float]],
    output_dir: str,
) -> None:
    if not history:
        print("No fitness history captured.")
        return

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"manual_fitness_history_{timestamp}.csv")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("cycle,accumulated_fitness\n")
        for cycle, fit in history:
            f.write(f"{cycle},{fit:.8f}\n")

    print(f"Saved fitness history CSV: {csv_path}")

    try:
        import matplotlib.pyplot as plt

        cycles = [c for c, _ in history]
        fitness = [v for _, v in history]
        plt.figure(figsize=(8, 4))
        plt.plot(cycles, fitness, color="tab:blue", linewidth=2)
        plt.title("Manual Run: Accumulated Fitness Over Time")
        plt.xlabel("Cycle")
        plt.ylabel("Accumulated fitness")
        plt.grid(alpha=0.25)
        plt.tight_layout()

        png_path = os.path.join(output_dir, f"manual_fitness_history_{timestamp}.png")
        plt.savefig(png_path, dpi=160)
        print(f"Saved fitness plot: {png_path}")
        plt.close()
    except Exception as exc:
        print(f"Could not create fitness plot ({exc}).")


def _persist_biomass_step_log(
    rows: List[Dict[str, float]],
    output_dir: str,
) -> None:
    if not rows:
        print("No biomass step log captured.")
        return

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"manual_biomass_steps_{timestamp}.csv")

    columns = [
        "cycle",
        "accumulated_fitness",
        *[f"{base}_biomass" for base in const.BASE_SPECIES],
        *[f"{base}_cells_nonzero" for base in const.BASE_SPECIES],
        *[f"{base}_cells_above_threshold" for base in const.BASE_SPECIES],
    ]

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for key in columns:
                value = row.get(key, 0.0)
                if key in {"cycle"} or key.endswith("_cells_nonzero") or key.endswith("_cells_above_threshold"):
                    values.append(str(int(round(float(value)))))
                else:
                    values.append(f"{float(value):.8f}")
            f.write(",".join(values) + "\n")

    print(f"Saved biomass step log CSV: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a training-like PettingZoo episode where one acting species is "
            "controlled manually from the terminal."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional .txt settings file (same format as training configs).",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="cod",
        help=(
            "Species to control. Can be a concrete acting species (e.g. cod__a2) "
            "or a base species (e.g. cod)."
        ),
    )
    parser.add_argument(
        "--control-all-age-groups",
        action="store_true",
        help="If --species is a base species, control all its age groups.",
    )
    parser.add_argument(
        "--map-seed",
        type=int,
        default=None,
        help="Map seed for deterministic reset.",
    )
    parser.add_argument(
        "--python-seed",
        type=int,
        default=None,
        help="Python RNG seed used for reproducible opponent random policies.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["human", "none"],
        default="human",
        help="Set to 'none' to disable visualization window.",
    )
    parser.add_argument(
        "--map-folder",
        type=str,
        default="maps/baltic",
        help="Map folder path.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cycle override for this run.",
    )
    parser.add_argument(
        "--fitness-output-dir",
        type=str,
        default="results/manual_play",
        help="Directory for accumulated-fitness CSV/plot outputs.",
    )
    parser.add_argument(
        "--biomass-cell-threshold",
        type=float,
        default=1e-6,
        help="Cell biomass threshold for 'significant occupancy' logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = load_settings(args.config) if args.config else Settings()
    runner = PettingZooRunner(
        settings=settings,
        render_mode=args.render_mode,
        map_folder=args.map_folder,
        build_population=False,
    )

    try:
        controlled_species = _resolve_controlled_species(
            species_arg=args.species,
            runner=runner,
            control_all_age_groups=bool(args.control_all_age_groups),
        )
        print("Controlling species:", ", ".join(controlled_species))

        max_steps_override, initial_energy_scale, energy_decay_per_cycle = _training_run_params(
            settings=settings,
            explicit_max_steps=args.max_steps,
        )

        candidates = _build_candidates(
            runner=runner,
            controlled_species=controlled_species,
            python_seed=args.python_seed,
        )
        fitness_history: List[Tuple[int, float]] = []
        biomass_step_rows: List[Dict[str, float]] = []
        callback = _cycle_summary_callback_factory(
            runner=runner,
            fitness_history=fitness_history,
            biomass_step_rows=biomass_step_rows,
            cell_threshold=float(max(0.0, args.biomass_cell_threshold)),
        )

        fitness, episode_length, end_reason = runner.run(
            candidates=candidates,
            species_being_evaluated=controlled_species[-1],
            seed=args.map_seed,
            is_evaluation=False,
            callback=callback,
            collect_plot_data=bool(settings.enable_plotting),
            python_random_seed=args.python_seed,
            max_steps_override=max_steps_override,
            initial_energy_scale=initial_energy_scale,
            energy_decay_per_cycle=energy_decay_per_cycle,
        )

        print(
            "\nFinished interactive run: "
            f"fitness={float(fitness):.3f}, cycles={int(episode_length)}, reason={end_reason}"
        )
        _persist_and_show_fitness_history(
            history=fitness_history,
            output_dir=args.fitness_output_dir,
        )
        _persist_biomass_step_log(
            rows=biomass_step_rows,
            output_dir=args.fitness_output_dir,
        )
    finally:
        try:
            runner.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
