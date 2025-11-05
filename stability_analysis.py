"""Run trained agents multiple times and plot total biomass stability."""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List

try:
    import matplotlib
    matplotlib.use("Agg")  # Use a non-interactive backend suitable for scripts.
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - provides a friendlier error at runtime.
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - improves error clarity.
    raise ModuleNotFoundError("numpy is required to run stability_analysis.py.") from exc

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - improves error clarity.
    raise ModuleNotFoundError("pandas is required to run stability_analysis.py.") from exc

import lib.config.const as const
from lib.config.settings import Settings
from lib.model import MODEL_OFFSETS, Model
from lib.runners.petting_zoo import PettingZooRunner


@dataclass(frozen=True)
class SimulationResult:
    """Container for storing the biomass time-series of a single run."""

    run_index: int
    steps: List[int]
    total_biomass: List[float]


def _normalise_species_name(name: str) -> str:
    """Normalise the species identifier extracted from a filename.

    Historically, saved agent files may prefix species with symbols or contain
    typos (e.g. ``spat`` instead of ``sprat``). This helper performs a best-effort
    cleanup so that we can automatically map files back to the canonical species
    names expected by the environment.
    """

    cleaned = name.lower().replace("$", "")
    if "spat" in cleaned and "sprat" not in cleaned:
        cleaned = cleaned.replace("spat", "sprat")
    return cleaned


def discover_model_files(model_dir: str, required_species: Iterable[str]) -> Dict[str, str]:
    """Infer model file paths for each species from a directory.

    Parameters
    ----------
    model_dir:
        Directory containing ``.npz`` files with the trained agent weights.
    required_species:
        Iterable of species identifiers that must be present in the results.

    Returns
    -------
    Dict[str, str]
        Mapping from species name to the selected model file path.

    Raises
    ------
    FileNotFoundError
        If the directory does not contain a model for every required species.
    """

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    candidates: Dict[str, List[str]] = {}
    for file_name in os.listdir(model_dir):
        if not file_name.endswith((".npz", ".npy.npz")):
            continue
        normalised = _normalise_species_name(file_name)
        for species in required_species:
            if species in normalised:
                candidates.setdefault(species, []).append(file_name)
                break

    missing = [species for species in required_species if species not in candidates]
    if missing:
        raise FileNotFoundError(
            "Unable to find trained models for the following species: "
            + ", ".join(missing)
        )

    selected: Dict[str, str] = {}
    for species, files in candidates.items():
        files.sort()
        selected[species] = os.path.join(model_dir, files[-1])

    return selected


def load_models(model_paths: Dict[str, str]) -> Dict[str, Model]:
    """Load ``Model`` instances from ``.npz`` checkpoints."""

    loaded = {}
    for species, path in model_paths.items():
        weights = np.load(path)
        loaded[species] = Model(chromosome=weights)
    return loaded


def run_simulation(
    runner: PettingZooRunner,
    models: Dict[str, Model],
    run_index: int,
    seed: int | None,
    tracked_species: Iterable[str],
) -> SimulationResult:
    """Execute a single simulation and collect total biomass measurements."""

    steps: List[int] = []
    totals: List[float] = []

    def callback(world: np.ndarray, _fitness: float, done: bool) -> None:
        if done:
            return
        step_index = len(steps)
        total_biomass = 0.0
        for species in tracked_species:
            biomass_offset = MODEL_OFFSETS[species]["biomass"]
            total_biomass += float(world[..., biomass_offset].sum())
        steps.append(step_index)
        totals.append(total_biomass)

    runner.run(models, species_being_evaluated="cod", seed=seed, is_evaluation=True, callback=callback)
    return SimulationResult(run_index=run_index, steps=steps, total_biomass=totals)


def aggregate_results(results: List[SimulationResult]) -> pd.DataFrame:
    """Aggregate individual run data into a single tidy DataFrame."""

    records = []
    for result in results:
        for step, biomass in zip(result.steps, result.total_biomass):
            records.append({
                "run": result.run_index,
                "step": step,
                "total_biomass": biomass,
            })
    return pd.DataFrame.from_records(records)


def plot_stability(df: pd.DataFrame, output_path: str) -> None:
    """Create a stability plot with mean ± standard deviation shading."""

    if df.empty:
        raise ValueError("No simulation data collected; cannot create plot.")
    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is required to generate stability plots. Install it to continue."
        )

    grouped = df.groupby("step")["total_biomass"]
    mean_series = grouped.mean()
    std_series = grouped.std().fillna(0.0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_series.index, mean_series.values, label="Mean total biomass", color="#1f77b4")
    plt.fill_between(
        mean_series.index,
        mean_series.values - std_series.values,
        mean_series.values + std_series.values,
        color="#1f77b4",
        alpha=0.2,
        label="±1 standard deviation",
    )
    plt.xlabel("Simulation steps")
    plt.ylabel("Total biomass")
    plt.title("Ecosystem stability across simulation runs")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate biomass stability across runs.")
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing trained agent checkpoints (.npz).",
    )
    parser.add_argument(
        "--model",
        action="append",
        help=(
            "Explicit species-to-model mapping in the form 'species=path'. "
            "Can be provided multiple times. Overrides --model-dir when used."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of simulation runs to execute.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducible runs. Each run increments this value.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="stability_results.csv",
        help="Path to save the raw per-run biomass data.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="stability_plot.png",
        help="Path to save the generated stability plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    required_species = list(const.ACTING_SPECIES)

    explicit_models: Dict[str, str] = {}
    if args.model:
        for mapping in args.model:
            if "=" not in mapping:
                raise ValueError(f"Invalid --model entry '{mapping}'. Expected format 'species=path'.")
            species, path = mapping.split("=", 1)
            species = _normalise_species_name(species)
            if species not in required_species:
                raise ValueError(
                    f"Unknown species '{species}'. Expected one of: {', '.join(required_species)}."
                )
            explicit_models[species] = path

    if explicit_models:
        model_paths = explicit_models
    else:
        if not args.model_dir:
            raise ValueError("Either --model-dir or --model must be supplied to locate trained agents.")
        model_paths = discover_model_files(args.model_dir, required_species)

    models = load_models(model_paths)

    settings = Settings()
    runner = PettingZooRunner(settings=settings, render_mode="none")

    results: List[SimulationResult] = []
    base_seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)

    for run_idx in range(args.num_runs):
        run_seed = base_seed + run_idx if args.seed is not None else None
        result = run_simulation(
            runner=runner,
            models=models,
            run_index=run_idx,
            seed=run_seed,
            tracked_species=const.SPECIES,
        )
        results.append(result)

    df = aggregate_results(results)
    df.to_csv(args.output_csv, index=False)
    plot_stability(df, args.output_figure)
    print(f"Saved raw biomass data to {args.output_csv}")
    print(f"Saved stability plot to {args.output_figure}")


if __name__ == "__main__":
    main()
