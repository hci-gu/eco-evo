from __future__ import annotations

"""Visualise biomass stability across multiple evaluation runs.

The script loads a trained single-agent model, replays it for several seeds in
the Petting Zoo environment, and records the total biomass across the selected
species over time.  A summary plot is emitted showing the mean biomass and its
variability across the runs, offering a quick view into ecosystem stability.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

import lib.constants as const
from lib.model import Model
from lib.runners.petting_zoo_single import PettingZooRunnerSingle


def _load_model(model_path: Path) -> Model:
    """Load model weights from ``model_path`` and return a :class:`Model`.

    Parameters
    ----------
    model_path:
        Path to the ``.npz`` file containing the agent weights produced during
        training.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with np.load(model_path) as data:
        chromosome: Dict[str, np.ndarray] = {key: data[key] for key in data.files}

    return Model(chromosome=chromosome)


def _discover_latest_model(model_dir: Path) -> Path:
    """Return the most recently modified model file within ``model_dir``."""
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    candidates = sorted(
        model_dir.glob("*.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No '.npz' model files found inside {model_dir.resolve()}"
        )

    return candidates[0]


def _total_biomass(world: np.ndarray, species: Iterable[str]) -> float:
    """Compute the total biomass across ``species`` for the given ``world``."""
    total = 0.0
    for sp in species:
        props = const.SPECIES_MAP.get(sp)
        if props is None:
            continue
        offset = props.get("biomass_offset")
        if offset is None:
            continue
        total += float(world[..., offset].sum())
    return total


def _run_single_simulation(
    runner: PettingZooRunnerSingle,
    model: Model,
    seed: int,
    tracked_species: Sequence[str],
) -> List[float]:
    """Execute one simulation run and return total biomass values per step."""
    biomass_history: List[float] = []

    def callback(world: np.ndarray, _fitness: float) -> None:
        biomass_history.append(_total_biomass(world, tracked_species))

    # Reset high-level constants to minimise cross-run side-effects.
    const.reset_constants()
    runner.run(model, seed=seed, is_evaluation=True, callback=callback)
    return biomass_history


def _prepare_time_axis(num_steps: int) -> np.ndarray:
    """Convert environment steps to years for plotting."""
    days = np.arange(num_steps) * const.DAYS_PER_STEP
    return days / 365.0


def _plot_stability(
    biomass_runs: Sequence[Sequence[float]],
    output_path: Path,
    title: str,
) -> None:
    if not biomass_runs:
        raise ValueError("No simulation data was collected; cannot plot stability.")

    max_length = max(len(run) for run in biomass_runs)
    aligned = np.full((len(biomass_runs), max_length), np.nan, dtype=float)

    for idx, run in enumerate(biomass_runs):
        aligned[idx, : len(run)] = run

    mean = np.nanmean(aligned, axis=0)
    std = np.nanstd(aligned, axis=0)
    time_axis = _prepare_time_axis(max_length)

    plt.figure(figsize=(10, 6))

    for idx, run in enumerate(biomass_runs, start=1):
        plt.plot(
            time_axis[: len(run)],
            run,
            alpha=0.3,
            linewidth=1,
            label=f"Run {idx}" if idx <= 5 else None,
        )

    plt.plot(time_axis, mean, color="black", linewidth=2.0, label="Mean biomass")
    plt.fill_between(
        time_axis,
        mean - std,
        mean + std,
        color="skyblue",
        alpha=0.4,
        label="±1 standard deviation",
    )

    plt.xlabel("Time (years)")
    plt.ylabel("Total biomass")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate biomass stability across multiple simulation runs."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to a specific trained model file ('.npz').",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing trained agent checkpoints. The most recently "
        "modified '.npz' file will be selected if --model-path is not provided.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of independent simulation runs to perform.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed offset for reproducibility. Individual runs will use sequential "
        "seeds starting at this value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("biomass_stability.png"),
        help="Path for the generated stability plot.",
    )
    parser.add_argument(
        "--include-plankton",
        action="store_true",
        help="Include plankton biomass when computing ecosystem totals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_path is None and args.model_dir is None:
        raise SystemExit("Either --model-path or --model-dir must be provided.")

    model_path = args.model_path
    if model_path is None:
        model_path = _discover_latest_model(args.model_dir)

    model = _load_model(model_path)
    runner = PettingZooRunnerSingle()

    tracked_species = [sp for sp in const.SPECIES_MAP.keys() if sp != "plankton"]
    if args.include_plankton:
        tracked_species = list(const.SPECIES_MAP.keys())

    biomass_runs: List[List[float]] = []
    for idx in range(args.runs):
        seed = args.seed + idx
        biomass_runs.append(
            _run_single_simulation(runner, model, seed, tracked_species)
        )

    title_species = ", ".join(tracked_species)
    title = f"Total biomass stability across runs ({title_species})"
    _plot_stability(biomass_runs, args.output, title)

    summary_lines = [
        f"Saved stability plot to {args.output.resolve()}",
        f"Model used: {model_path.resolve()}",
        f"Runs: {args.runs}",
        f"Tracked species: {title_species}",
    ]
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
