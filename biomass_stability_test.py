"""Utilities for probing biomass stability against starting biomass perturbations.

This script runs an initial set of baseline simulations to estimate the average
biomass per active species (sprat, herring, cod) after discarding an initial
warm-up window. It then repeats the experiment while scaling the starting
biomass of each species by a set of percentages and reports how those averages
shift relative to the baseline.

Example usage
-------------

  python biomass_stability_test.py \
      --model-dir results/2025-10-23_8/agents \
      --baseline-runs 5 \
      --warmup-steps 50 \
      --scales 0.75 0.9 1.1 1.25
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

import lib.config.const as const
from lib.config.settings import Settings, load_settings
from lib.config.species import SpeciesMap, build_species_map
from lib.environments.petting_zoo import env as make_env
from lib.model import MODEL_OFFSETS, Model
from lib.runners.petting_zoo import PettingZooRunner


TrackedSpecies = Sequence[str]


@dataclass(frozen=True)
class SpeciesStats:
    mean: float
    std: float
    samples: int


@dataclass(frozen=True)
class ScenarioResult:
    label: str
    stats: Dict[str, SpeciesStats]
    run_reasons: List[str]


_FILENAME_RE = re.compile(
    r"""
    ^(?P<iter>\d+)_\$(?P<species>[A-Za-z0-9\-]+)_
    (?P<fitness>-?\d+(?:\.\d+)?)
    (?:\.npy)?\.npz$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalise_species_name(name: str) -> str:
    cleaned = name.lower().replace("$", "")
    if "spat" in cleaned and "sprat" not in cleaned:
        cleaned = cleaned.replace("spat", "sprat")
    return cleaned


def discover_best_model_files(model_dir: str, required_species: Iterable[str]) -> Dict[str, str]:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    best: Dict[str, tuple[float, int, float, str]] = {}
    required_lower = {s.lower() for s in required_species}

    for file_name in os.listdir(model_dir):
        match = _FILENAME_RE.match(file_name)
        if not match:
            continue
        species = _normalise_species_name(match.group("species"))
        if species not in required_lower:
            continue
        fitness = float(match.group("fitness"))
        iteration = int(match.group("iter"))
        full_path = os.path.join(model_dir, file_name)
        try:
            mtime = os.path.getmtime(full_path)
        except OSError:
            mtime = 0.0

        candidate = (fitness, iteration, mtime, full_path)
        current = best.get(species)
        if current is None or candidate[:3] > current[:3]:
            best[species] = candidate

    missing = [s for s in required_species if s.lower() not in best]
    if missing:
        raise FileNotFoundError(
            "Unable to find trained models for: " + ", ".join(sorted(missing))
        )

    return {species: best[species.lower()][3] for species in required_species}


def load_models(model_dir: str | None, tracked_species: Iterable[str]) -> Dict[str, Model]:
    if model_dir is None:
        return {species: Model() for species in tracked_species}

    model_paths = discover_best_model_files(model_dir, tracked_species)
    loaded: Dict[str, Model] = {}
    for species, path in model_paths.items():
        weights = np.load(path)
        loaded[species] = Model(chromosome=weights)
    return loaded


def make_runner(settings: Settings, species_map: SpeciesMap) -> PettingZooRunner:
    runner = PettingZooRunner(settings, render_mode="none")
    runner.species_map = species_map
    runner.env = make_env(settings=settings, species_map=species_map, render_mode="none")
    runner.empty_action = runner.env.action_space("plankton").sample()
    runner.env.reset()
    return runner


def generate_seeds(count: int, seed: int | None) -> List[int]:
    rng = random.Random(seed)
    return [rng.randrange(1, 10**6) for _ in range(count)]


def collect_biomass_timeseries(
    runner: PettingZooRunner,
    models: Mapping[str, Model],
    seeds: Sequence[int],
    warmup_steps: int,
    tracked_species: TrackedSpecies,
) -> tuple[List[Dict[str, List[float]]], List[str]]:
    records: List[Dict[str, List[float]]] = []
    reasons: List[str] = []

    for seed in seeds:
        per_species = {species: [] for species in tracked_species}
        step_index = -1

        def callback(world, _fitness, done):
            nonlocal step_index
            if done:
                return
            step_index += 1
            if step_index < warmup_steps:
                return
            for species in tracked_species:
                offset = MODEL_OFFSETS[species]["biomass"]
                biomass = float(world[..., offset].sum())
                per_species[species].append(biomass)

        fitness, episode_length, reason = runner.run(
            models, species_being_evaluated="cod", seed=seed, is_evaluation=True, callback=callback
        )

        # Propagate the run reason to help spot premature terminations.
        reasons.append(reason if reason else f"completed:{episode_length}")
        records.append(per_species)

    return records, reasons


def summarise(records: Sequence[Mapping[str, Sequence[float]]], tracked_species: TrackedSpecies) -> Dict[str, SpeciesStats]:
    summary: Dict[str, SpeciesStats] = {}
    for species in tracked_species:
        all_values: List[float] = []
        for entry in records:
            all_values.extend(entry.get(species, ()))
        if not all_values:
            summary[species] = SpeciesStats(mean=math.nan, std=math.nan, samples=0)
            continue
        arr = np.asarray(all_values, dtype=np.float32)
        summary[species] = SpeciesStats(
            mean=float(arr.mean()),
            std=float(arr.std(ddof=0)),
            samples=int(arr.size),
        )
    return summary


def scale_species_map(base_map: SpeciesMap, species: str, factor: float) -> SpeciesMap:
    updated: SpeciesMap = {}
    for key, params in base_map.items():
        if key == species:
            scaled = params.original_starting_biomass * factor
            updated[key] = replace(params, starting_biomass=scaled)
        else:
            updated[key] = params
    return updated


def run_scenario(
    settings: Settings,
    base_species_map: SpeciesMap,
    models: Mapping[str, Model],
    seeds: Sequence[int],
    warmup_steps: int,
    tracked_species: TrackedSpecies,
    label: str,
) -> ScenarioResult:
    runner = make_runner(settings, base_species_map)
    records, reasons = collect_biomass_timeseries(
        runner=runner,
        models=models,
        seeds=seeds,
        warmup_steps=warmup_steps,
        tracked_species=tracked_species,
    )
    stats = summarise(records, tracked_species)
    return ScenarioResult(label=label, stats=stats, run_reasons=list(reasons))


def format_stats(label: str, stats: Dict[str, SpeciesStats], baseline: Dict[str, SpeciesStats] | None) -> str:
    lines: List[str] = [label]
    for species, species_stats in stats.items():
        base_line = ""
        if baseline is not None and not math.isnan(species_stats.mean):
            base_stats = baseline.get(species)
            if base_stats and not math.isnan(base_stats.mean):
                delta = species_stats.mean - base_stats.mean
                pct = (delta / base_stats.mean) * 100 if base_stats.mean else math.nan
                base_line = f", Δ={delta:.2f}, Δ%={pct:.2f}"
        lines.append(
            f"  {species}: mean={species_stats.mean:.2f}, std={species_stats.std:.2f}, samples={species_stats.samples}{base_line}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate biomass stability under starting biomass perturbations.")
    parser.add_argument("--config", type=str, default=None, help="Optional settings override file.")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory containing trained agents (.npz).")
    parser.add_argument("--baseline-runs", type=int, default=5, help="Number of baseline runs to average.")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Initial callback steps to discard.")
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.75, 0.9, 1.1, 1.25],
        help="Multiplicative factors to apply to each species' starting biomass.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base seed for reproducible scenario seeds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config) if args.config else Settings()

    tracked_species: TrackedSpecies = tuple(const.ACTING_SPECIES)
    models = load_models(args.model_dir, tracked_species)

    base_species_map = build_species_map(settings)
    seeds = generate_seeds(args.baseline_runs, args.seed)

    baseline_result = run_scenario(
        settings=settings,
        base_species_map=base_species_map,
        models=models,
        seeds=seeds,
        warmup_steps=args.warmup_steps,
        tracked_species=tracked_species,
        label=f"Baseline (warmup={args.warmup_steps}, runs={args.baseline_runs})",
    )

    print(format_stats(baseline_result.label, baseline_result.stats, baseline=None))
    if baseline_result.run_reasons:
        print("  run reasons:", ", ".join(baseline_result.run_reasons))

    for species in tracked_species:
        for factor in args.scales:
            scenario_map = scale_species_map(base_species_map, species, factor)
            label = f"{species} × {factor:.2f}"
            scenario_result = run_scenario(
                settings=settings,
                base_species_map=scenario_map,
                models=models,
                seeds=seeds,
                warmup_steps=args.warmup_steps,
                tracked_species=tracked_species,
                label=label,
            )
            print(
                format_stats(
                    label=scenario_result.label,
                    stats=scenario_result.stats,
                    baseline=baseline_result.stats,
                )
            )
            if scenario_result.run_reasons:
                print("  run reasons:", ", ".join(scenario_result.run_reasons))


if __name__ == "__main__":
    main()
