"""Periodic CPU inference and persistent survival history for the live viewer."""

import copy
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from lib.config.config_loader import load_project_config, setup_full_mareld_mvp
from lib.environments.ecosystem import EcosystemEnvironment
from lib.gpu.config import DEFAULT_LIBRARY
from lib.environments.ecosystem_env.currents import CurrentConfig
from lib.environments.ecosystem_env.debug_food import FoodBlobConfig
from lib.environments.ecosystem_env.population_change import (
    DEFAULT_MORTALITY_MULTIPLIER)


def add_progress_arguments(parser):
    from lib.gpu.cli import positive_int
    parser.add_argument("--eval-every", type=positive_int, default=20,
                        help="Survival progress: evaluate every N training updates (default: 20)")
    parser.add_argument("--eval-ticks", type=positive_int, default=1000,
                        help="Survival progress: inference tick cap (default: 1000)")
    parser.add_argument("--biomass-bounds", type=float, nargs=2, default=(0.3, 3.0),
                        metavar=("LOWER", "UPPER"), help="Survival bounds as multiples of initial biomass")
    parser.add_argument("--eval-seed", type=int, default=20260530)
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--plot-dir", type=Path, help="Progress history directory (default: <run>/progress)")


def validate_progress_arguments(parser, args):
    lower, upper = args.biomass_bounds
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= 1 <= upper and lower < upper):
        parser.error("--biomass-bounds must be finite and satisfy 0 < LOWER <= 1 <= UPPER, LOWER < UPPER")
    if not 0 <= args.eval_seed < 2**32:
        parser.error("--eval-seed must be between 0 and 2**32 - 1")
    if not math.isfinite(args.eval_temperature) or args.eval_temperature <= 0:
        parser.error("--eval-temperature must be positive and finite")


@contextmanager
def evaluation_randomness(seed):
    """Replay environmental noise without consuming the trainer's random stream."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            np.random.seed(seed)
            random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def measure_survival(env, ticks, lower, upper, pump_events=None):
    """Count decision makers' post-step ticks inside [lower*B0, upper*B0].

    A breach on tick 1 scores 0; surviving the entire horizon scores `ticks`.
    Returning inside the band after a breach never restarts the counter.
    Groups with zero initial biomass have no defined ratio and score None.
    Stop once every decision maker has breached the bounds; non-acting
    groups still participate in the ecosystem but do not prolong evaluation.
    """
    ids = sorted(env.dm_ids)
    initial = np.array([env.fgs[f].biomass.sum() for f in ids], dtype=np.float64)
    if not np.isfinite(initial).all() or (initial < 0).any():
        raise ValueError("Inference initial biomass must be finite and nonnegative")
    present = initial > 0
    alive, survived = present.copy(), np.zeros(len(ids), dtype=np.int64)
    ticks_run = 0
    for _ in range(ticks):
        if not alive.any():
            break
        if pump_events is not None and not pump_events():
            raise InterruptedError("Survival evaluation cancelled because the viewer closed")
        env.tick()
        ticks_run += 1
        current = np.array([env.fgs[f].biomass.sum() for f in ids], dtype=np.float64)
        alive &= np.isfinite(current) & (current >= lower * initial) & (current <= upper * initial)
        survived += alive
    return {
        "survival_ticks": {f: int(survived[i]) if present[i] else None for i, f in enumerate(ids)},
        "initial_biomass": dict(zip(ids, initial.tolist())),
        "ticks_run": ticks_run,
    }


def inference_config(trainer, backend):
    """Use the training project's ecology, grid and inference biomass settings."""
    if backend == "gpu":
        builder = trainer.spec.builder
        grid, library = builder.grid, builder.library_path
        mortality, migration = builder.mortality, builder.migration
    else:
        builder = trainer.env_builder
        grid, library = (builder.grid_height, builder.grid_width), DEFAULT_LIBRARY
        mortality, migration = builder.apply_natural_mortality, builder.migration
    # ``--mortality_multiplier`` belongs to the ecology: dropping it here
    # evaluated every run with the unscaled library rates, so
    # ``--mortality on --mortality_multiplier 0`` measured extinction and
    # growth under FULL mortality while ``--mortality off`` measured none.
    return dict(project=str(Path(builder.project_path).resolve()) if builder.project_path else None,
                library=str(Path(library).resolve()), grid=list(grid),
                mortality=mortality, migration=migration,
                mortality_multiplier=float(
                    getattr(builder, "mortality_multiplier",
                            DEFAULT_MORTALITY_MULTIPLIER)),
                currents=builder.currents.metadata() if getattr(builder, "currents", None) else None,
                food_blobs=builder.food_blobs.metadata() if getattr(builder, "food_blobs", None) else None)


def comparable_config(config):
    """Fill in defaults so a pre-existing ``config.json`` stays resumable.

    Histories written before ``mortality_multiplier`` was recorded used
    the library rates unscaled, which is exactly the default factor.
    """
    normalised = dict(config)
    normalised.setdefault("mortality_multiplier", DEFAULT_MORTALITY_MULTIPLIER)
    normalised.setdefault("food_blobs", None)
    normalised["mortality_multiplier"] = float(normalised["mortality_multiplier"])
    return normalised


def build_inference_env(config, seed):
    kwargs = dict(library_path=config["library"], grid_size=tuple(config["grid"]),
                  seed=seed, spawn_seed=seed)
    if config["project"]:
        groups = load_project_config(config["project"], mode="inference", **kwargs)[0]
    else:
        groups = setup_full_mareld_mvp(**kwargs)
    height, width = config["grid"]
    return EcosystemEnvironment(dict(height=height, width=width), groups,
                                apply_natural_mortality=config["mortality"],
                                mortality_multiplier=config.get(
                                    "mortality_multiplier",
                                    DEFAULT_MORTALITY_MULTIPLIER),
                                migration=config["migration"],
                                currents=CurrentConfig(**config["currents"]) if config.get("currents") else None,
                                current_world_seed=seed,
                                food_blobs=FoodBlobConfig(**config["food_blobs"]) if config.get("food_blobs") else None)


@torch.no_grad()
def install_current_policies(env, trainer, backend):
    """Copy the current unperturbed weights and freeze normalization for inference."""
    if backend == "gpu":
        # bank.policies can still contain initialization/export-time weights;
        # theta is the authoritative result of the most recent ARS update.
        env.policies = {f: copy.deepcopy(p).cpu().eval() for f, p in trainer.bank.policies.items()}
        for fid, flat in zip(trainer.model.dm_ids, trainer.theta):
            torch.nn.utils.vector_to_parameters(flat.detach().cpu().clone(), env.policies[fid].parameters())
    else:
        env.policies = {f: copy.deepcopy(p).cpu().eval() for f, p in trainer.policies.items()}
    env.rebuild_batched_weights()
    if not trainer.obs_normalize:
        return
    if backend == "gpu":
        order = [trainer.model.dm_ids.index(f) for f in env.dm_ids]
        env.obs_mean = trainer.obs_mean[order].cpu().numpy().astype(np.float32, copy=True)
        env.obs_var = trainer.obs_var[order].cpu().numpy().astype(np.float32, copy=True)
    else:
        env.obs_mean = np.zeros((env.N_dm, env.max_in_dim), dtype=np.float32)
        env.obs_var = np.ones_like(env.obs_mean)
        for i, fid in enumerate(env.dm_ids):
            stats = trainer.obs_stats.get(fid)
            if stats is not None and stats["count"] > 0:
                env.obs_mean[i] = stats["mean"]
                env.obs_var[i] = stats["var"]


class TrainingProgress:
    def __init__(self, backend, every, ticks, lower, upper, seed, temperature, directory=None,
                 *, visualizer=None):
        self.backend, self.every, self.ticks = backend, every, ticks
        self.lower, self.upper, self.seed, self.temperature = lower, upper, seed, temperature
        self.directory = Path(directory) if directory is not None else None
        self.config = None
        self.records = []
        self.visualizer = visualizer

    def _start(self, trainer, step, run_dir, resume):
        ecology = inference_config(trainer, self.backend)
        self.config = dict(format_version=1, backend=self.backend, ticks=self.ticks,
                           lower=self.lower, upper=self.upper, seed=self.seed,
                           temperature=self.temperature, **ecology)
        self.directory = self.directory or Path(run_dir) / "progress"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.history = self.directory / "survival.jsonl"
        config_path = self.directory / "config.json"
        if config_path.exists():
            if not resume:
                raise ValueError(f"Progress output already exists in {self.directory}; use --resume or a new --plot-dir")
            if comparable_config(json.loads(config_path.read_text())) != comparable_config(self.config):
                raise ValueError("Evaluation settings changed; use a new --plot-dir to keep the graph comparable")
            if self.history.exists():
                self.records = [json.loads(line) for line in self.history.read_text().splitlines() if line.strip()]
                # A crash can leave evaluations ahead of the saved checkpoint.
                # Re-evaluate the resume point and discard the abandoned tail.
                self.records = [r for r in self.records if r["step"] < step]
                for number, record in enumerate(self.records, 1):
                    record["evaluation"] = number
        with evaluation_randomness(self.seed):
            self.template = build_inference_env(ecology, self.seed)
        config_path.write_text(json.dumps(self.config, indent=2) + "\n")

    @torch.no_grad()
    def evaluate(self, trainer):
        with evaluation_randomness(self.seed):
            env = copy.deepcopy(self.template)
            install_current_policies(env, trainer, self.backend)
            env.softmax_temperature = self.temperature
            pump = self.visualizer.pump_events if self.visualizer is not None else None
            return measure_survival(env, self.ticks, self.lower, self.upper, pump_events=pump)

    def __call__(self, trainer, step, run_dir, args):
        if self.visualizer is not None and not self.visualizer.enabled:
            self.visualizer = None
        first = self.config is None
        if first:
            self._start(trainer, step, run_dir, args.resume)
        if not first and step % self.every:
            return
        number = len(self.records) + 1
        print(f"[progress] evaluation={number}, step={step}: evaluating up to {self.ticks} inference ticks on CPU...", flush=True)
        started = time.perf_counter()
        try:
            result = self.evaluate(trainer)
            self.records.append(dict(evaluation=number, step=step,
                                     evaluation_seconds=time.perf_counter() - started, **result))
            # Replace complete files atomically, so image viewers/readers never
            # see a partially written plot or measurement history.
            temporary = self.history.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in self.records))
            temporary.replace(self.history)
            if self.visualizer is not None:
                self.visualizer.set_training_progress(self.records, self.config)
        except Exception as error:
            # Monitoring failures must not discard a completed training update.
            print(f"[progress] WARNING: evaluation/plot failed at step {step}: {error}", flush=True)
            return
        destination = self.history
        print(f"[progress] updated {destination} "
              f"({result['ticks_run']}/{self.ticks} ticks, {time.perf_counter() - started:.1f}s)", flush=True)


class LiveTrainingProgress(TrainingProgress):
    """Persist survival measurements and render them in the live viewer."""

    disabled = False

    def __call__(self, trainer, step, run_dir, args):
        if (self.disabled or self.visualizer is None or not self.visualizer.enabled
                or getattr(self.visualizer, "_quit", False)):
            return
        try:
            super().__call__(trainer, step, run_dir, args)
        except Exception as error:
            self.disabled = True
            self.visualizer.set_progress_message("Progress unavailable; see terminal for details")
            print(f"[progress] Live progress disabled: {error}", flush=True)


def make_visual_progress(backend, args, viz):
    """Create the evaluator used by either trainer's progress tab."""
    if viz is None or not viz.enabled:
        return None
    lower, upper = args.biomass_bounds
    return LiveTrainingProgress(backend, args.eval_every, args.eval_ticks, lower, upper,
                                args.eval_seed, args.eval_temperature, args.plot_dir,
                                visualizer=viz)
