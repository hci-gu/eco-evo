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
from lib.environments.ecosystem_env.population_change import (
    DEFAULT_MASS_BALANCE, DEFAULT_MORTALITY_MULTIPLIER,
    LEGACY_MASS_BALANCE)


def add_progress_arguments(parser):
    from lib.gpu.cli import positive_int
    parser.add_argument("--eval-every", type=positive_int, default=1,
                        help="Survival progress: evaluate every N training updates (default: 1)")
    parser.add_argument("--eval-ticks", type=positive_int, default=None,
                        help="Survival progress: inference tick cap (default: the "
                             "training rollout length, --n_eval_ticks)")
    parser.add_argument("--biomass-bounds", type=float, nargs=2, default=(0.3, 3.0),
                        metavar=("LOWER", "UPPER"), help="Survival bounds as multiples of initial biomass")
    parser.add_argument("--eval-seed", type=int, default=20260530)
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--plot-dir", type=Path, help="Progress history directory (default: <run>/progress)")


def validate_progress_arguments(parser, args):
    # Without --eval-ticks the cap follows the training rollout length
    # (train.py --n_eval_ticks, train_gpu.py's dest "ticks"), so the
    # progress evaluation covers the same span - a whole year with
    # 1460-tick rollouts - and not a fixed 1000 ticks. A resumed run keeps
    # the cap its history was written with (TrainingProgress._start).
    args.eval_ticks_default = args.eval_ticks is None
    if args.eval_ticks_default:
        args.eval_ticks = int(getattr(args, "n_eval_ticks", None)
                              or getattr(args, "ticks", None) or 1000)
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
                # Same argument as the multiplier above: the growth term
                # the run was trained under has to be the one the
                # progress rollout is evaluated under, or the curve
                # measures a different ecology (section 116.3).
                mass_balance=bool(
                    getattr(builder, "mass_balance", DEFAULT_MASS_BALANCE)),
                currents=builder.currents.metadata() if getattr(builder, "currents", None) else None)


def comparable_config(config):
    """Fill in defaults so a pre-existing ``config.json`` stays resumable.

    Histories written before ``mortality_multiplier`` was recorded used
    the library rates unscaled, which is exactly the default factor.
    """
    normalised = dict(config)
    normalised.setdefault("mortality_multiplier", DEFAULT_MORTALITY_MULTIPLIER)
    normalised["mortality_multiplier"] = float(normalised["mortality_multiplier"])
    # Histories written before --mass-balance existed used the legacy
    # growth term. That is LEGACY_MASS_BALANCE and not DEFAULT_MASS_BALANCE:
    # the default flipped in section 120 and flipping a default must not
    # retroactively relabel what an old run was doing.
    normalised.setdefault("mass_balance", LEGACY_MASS_BALANCE)
    normalised["mass_balance"] = bool(normalised["mass_balance"])
    return normalised


def build_inference_env(config, seed):
    kwargs = dict(library_path=config["library"], grid_size=tuple(config["grid"]),
                  seed=seed, spawn_seed=seed,
                  apply_natural_mortality=bool(config["mortality"]))
    impact_vars, observable = [], []
    if config["project"]:
        groups, impact_vars, _, observable = load_project_config(
            config["project"], mode="inference", **kwargs)
    else:
        groups = setup_full_mareld_mvp(**kwargs)
    height, width = config["grid"]
    env = EcosystemEnvironment(dict(height=height, width=width), groups,
                                observable_impact_vars=observable,
                                apply_natural_mortality=config["mortality"],
                                mortality_multiplier=config.get(
                                    "mortality_multiplier",
                                    DEFAULT_MORTALITY_MULTIPLIER),
                                mass_balance=config.get(
                                    "mass_balance", DEFAULT_MASS_BALANCE),
                                migration=config["migration"],
                                currents=CurrentConfig(**config["currents"]) if config.get("currents") else None,
                                current_world_seed=seed)
    if impact_vars:
        # The inference world's impact fields (section 161). Imported
        # here: inference.py is a script module that itself imports from
        # lib.runners.
        from inference import load_inference_impact_maps
        maps = load_inference_impact_maps(config["project"], impact_vars,
                                          height, width, seed=seed, verbose=False)
        for iv in impact_vars:
            env.grid.add_map(iv, maps[iv])
    return env


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
                 *, visualizer=None, ticks_default=False):
        self.backend, self.every, self.ticks = backend, every, ticks
        # True when ``ticks`` only followed the rollout length (no
        # --eval-ticks): a resumed history then keeps its own cap.
        self.ticks_default = ticks_default
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
            stored = json.loads(config_path.read_text())
            if self.ticks_default and stored.get("ticks") not in (None, self.ticks):
                print(f"[progress] keeping this run's evaluation cap of {stored['ticks']} ticks "
                      f"(pass --eval-ticks to change it with a new --plot-dir)", flush=True)
                self.ticks = int(stored["ticks"])
                self.config["ticks"] = self.ticks
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

    def interval(self):
        """Updates between evaluations: the viewer's slider, else --eval-every."""
        getter = getattr(self.visualizer, "get_eval_interval_override", None)
        override = getter() if callable(getter) else None
        return int(override) if override is not None else int(self.every)

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
        # Every multiple of the interval (a resume re-evaluates its first
        # step, then returns to the grid). The live viewer's "Evaluation
        # probe interval" slider can change the interval between updates.
        if not first and step % self.interval():
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


def probe_due(viz, step, first=False):
    """True when the viewer's biomass probe should run after update ``step``.

    The probe shares the viewer's "Evaluation probe interval" slider
    (default ``--eval-every``) with the survival evaluation, so both run
    on the same updates. The first update of a session always probes, so
    the viewer has a rollout to show; without a viewer the probe runs
    after every update, as before.
    """
    if first or viz is None:
        return True
    getter = getattr(viz, "get_eval_interval", None)
    if not callable(getter):
        return True
    return int(step) % max(1, int(getter())) == 0


def make_visual_progress(backend, args, viz):
    """Create the evaluator used by either trainer's progress tab."""
    if viz is None or not viz.enabled:
        return None
    lower, upper = args.biomass_bounds
    viz.set_eval_interval_default(args.eval_every)
    return LiveTrainingProgress(backend, args.eval_every, args.eval_ticks, lower, upper,
                                args.eval_seed, args.eval_temperature, args.plot_dir,
                                visualizer=viz,
                                ticks_default=getattr(args, "eval_ticks_default", False))
