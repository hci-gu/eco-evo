"""Periodic CPU inference for the live viewer: the three-year probe.

Since section 223.12 the viewer's progress tab shows how a frozen copy of
the current policies plays out over three years (``PROBE_YEARS``) of the
inference world: every group's biomass as a share of its start, sampled
once a day. The live viewer runs the probe in a separate process, so
training goes on meanwhile. Each probe also keeps the decision makers'
survival ticks (inside ``--biomass-bounds``) and every group's end ratio
per year in ``<run>/progress/probe.jsonl``; the latest trajectory is in
``probe_latest.json``.
"""

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
from lib.world.tick_time import LIBRARY_TICK_HOURS, ticks_per_day, ticks_per_year

# Length of the progress probe (section 223.12).
PROBE_YEARS = 3


def add_progress_arguments(parser):
    from lib.gpu.cli import positive_int
    parser.add_argument("--eval-every", type=positive_int, default=1,
                        help="Progress probe: run every N training updates (default: 1)")
    parser.add_argument("--eval-ticks", type=positive_int, default=None,
                        help="Progress probe length in ticks (default: three years "
                             "at the run's --tick-length)")
    parser.add_argument("--biomass-bounds", type=float, nargs=2, default=(0.3, 3.0),
                        metavar=("LOWER", "UPPER"), help="Survival bounds as multiples of initial biomass")
    parser.add_argument("--eval-seed", type=int, default=20260530)
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--plot-dir", type=Path, help="Progress history directory (default: <run>/progress)")


def validate_progress_arguments(parser, args):
    # Without --eval-ticks the probe runs three years (section 223.12; it
    # followed the training rollout length before). The probe world is
    # built at the library's tick length (build_inference_env passes no
    # tick length), so the years are counted in library ticks. A resumed
    # run keeps the length its history was written with
    # (TrainingProgress._start).
    args.eval_ticks_default = args.eval_ticks is None
    if args.eval_ticks_default:
        args.eval_ticks = PROBE_YEARS * ticks_per_year(LIBRARY_TICK_HOURS)
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


def run_probe(env, ticks, lower, upper, pump_events=None):
    """Run ``ticks`` inference ticks and record the probe (section 223.12).

    Returns the decision makers' survival ticks as ``measure_survival``
    counts them (but never stops early), every group's biomass as a share
    of its start once a day (``trajectory``; None for a group without
    starting biomass) and its end ratio per year (``year_end_ratio``).
    """
    dm_ids = sorted(env.dm_ids)
    ids = list(env.global_fg_order)
    initial = {f: float(np.asarray(env.fgs[f].biomass, dtype=np.float64).sum()) for f in ids}
    if any(not math.isfinite(v) or v < 0 for v in initial.values()):
        raise ValueError("Inference initial biomass must be finite and nonnegative")
    every = max(1, int(ticks_per_day(getattr(env, "tick_hours", None))))
    per_year = max(1, int(ticks_per_year(getattr(env, "tick_hours", None))))

    def ratios():
        out = {}
        for f in ids:
            b0 = initial[f]
            value = float(np.asarray(env.fgs[f].biomass, dtype=np.float64).sum())
            out[f] = (value / b0 if b0 > 0 and math.isfinite(value) else None)
        return out

    series = {f: [] for f in ids}
    for f, v in ratios().items():
        series[f].append(v)
    years = {f: [] for f in ids}
    present = np.array([initial[f] > 0 for f in dm_ids])
    alive = present.copy()
    survived = np.zeros(len(dm_ids), dtype=np.int64)
    b0_dm = np.array([initial[f] for f in dm_ids], dtype=np.float64)
    ticks_run = 0
    for t in range(1, ticks + 1):
        if pump_events is not None and not pump_events():
            raise InterruptedError("Probe cancelled because the viewer closed")
        env.tick()
        ticks_run += 1
        current = np.array([env.fgs[f].biomass.sum() for f in dm_ids], dtype=np.float64)
        alive &= np.isfinite(current) & (current >= lower * b0_dm) & (current <= upper * b0_dm)
        survived += alive
        if t % every == 0 or t % per_year == 0 or t == ticks:
            now = ratios()
            if t % every == 0:
                for f, v in now.items():
                    series[f].append(v)
            if t % per_year == 0:
                for f, v in now.items():
                    years[f].append(v)
    return {
        "survival_ticks": {f: int(survived[i]) if present[i] else None for i, f in enumerate(dm_ids)},
        "initial_biomass": {f: initial[f] for f in dm_ids},
        "ticks_run": ticks_run,
        "year_end_ratio": years,
        "trajectory": {"every_ticks": every, "ticks_per_year": per_year, "series": series},
    }


def _probe_in_process(config, seed, temperature, ticks, lower, upper,
                      policies, obs_mean, obs_var, threads):
    """Worker: the same probe as ``TrainingProgress.evaluate``, in a
    separate process (the live viewer, section 223.12)."""
    torch.set_num_threads(max(1, int(threads)))
    with evaluation_randomness(seed):
        env = build_inference_env(config, seed)
    with evaluation_randomness(seed):
        env.policies = policies
        env.rebuild_batched_weights()
        if obs_mean is not None:
            env.obs_mean, env.obs_var = obs_mean, obs_var
        env.softmax_temperature = temperature
        return run_probe(env, ticks, lower, upper)


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
        self.latest = None
        self.visualizer = visualizer

    def _start(self, trainer, step, run_dir, resume):
        ecology = inference_config(trainer, self.backend)
        self.config = dict(format_version=2, backend=self.backend, ticks=self.ticks,
                           lower=self.lower, upper=self.upper, seed=self.seed,
                           temperature=self.temperature, **ecology)
        self.directory = self.directory or Path(run_dir) / "progress"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.history = self.directory / "probe.jsonl"
        self.latest_path = self.directory / "probe_latest.json"
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
            if self.latest_path.exists():
                try:
                    latest = json.loads(self.latest_path.read_text())
                    self.latest = latest if latest.get("step", step) < step else None
                except ValueError:
                    self.latest = None
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
            return run_probe(env, self.ticks, self.lower, self.upper, pump_events=pump)

    def _record(self, number, step, seconds, result):
        """Append one probe to the history and publish it."""
        trajectory = result.pop("trajectory", None)
        self.records.append(dict(evaluation=number, step=step,
                                 evaluation_seconds=seconds, **result))
        # Replace complete files atomically, so readers never see a
        # partially written history.
        temporary = self.history.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in self.records))
        temporary.replace(self.history)
        if trajectory is not None:
            self.latest = dict(step=step, evaluation=number, seed=self.seed, **trajectory)
            temporary = self.latest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self.latest))
            temporary.replace(self.latest_path)
        if self.visualizer is not None:
            self.visualizer.set_training_progress(self.records, self.config, self.latest)

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
        print(f"[progress] probe={number}, step={step}: {self.ticks} inference ticks on CPU...", flush=True)
        started = time.perf_counter()
        try:
            result = self.evaluate(trainer)
            ticks_run = result["ticks_run"]
            self._record(number, step, time.perf_counter() - started, result)
        except Exception as error:
            # Monitoring failures must not discard a completed training update.
            print(f"[progress] WARNING: probe failed at step {step}: {error}", flush=True)
            return
        print(f"[progress] updated {self.history} "
              f"({ticks_run}/{self.ticks} ticks, {time.perf_counter() - started:.1f}s)", flush=True)


class LiveTrainingProgress(TrainingProgress):
    """The live viewer's probe, run in a separate process (section 223.12).

    A due probe is submitted with a frozen copy of the current policies
    and normalisation; training goes on, and the result is recorded at
    the first update after it finishes (or by ``wait``). A probe that is
    due while one is still running is skipped. ``asynchronous=False``
    runs it in-line, as the base class does.
    """

    disabled = False
    asynchronous = True
    worker_threads = 4

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._pool = None
        self._pending = None

    def _collect(self, block=False):
        if self._pending is None:
            return
        future, number, step, started = self._pending
        if not block and not future.done():
            return
        self._pending = None
        try:
            result = future.result()
            ticks_run = result["ticks_run"]
            self._record(number, step, time.perf_counter() - started, result)
            print(f"[progress] probe {number} (update {step}) done: "
                  f"{ticks_run}/{self.ticks} ticks, {time.perf_counter() - started:.0f}s", flush=True)
        except Exception as error:
            print(f"[progress] WARNING: probe failed at step {step}: {error}", flush=True)

    def _submit(self, trainer, step):
        import concurrent.futures
        import multiprocessing
        if self._pool is None:
            self._pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        env = copy.deepcopy(self.template)
        install_current_policies(env, trainer, self.backend)
        obs_mean = getattr(env, "obs_mean", None)
        obs_var = getattr(env, "obs_var", None)
        number = len(self.records) + 1
        future = self._pool.submit(_probe_in_process, self.config_ecology(), self.seed,
                                   self.temperature, self.ticks, self.lower, self.upper,
                                   env.policies, obs_mean, obs_var, self.worker_threads)
        self._pending = (future, number, step, time.perf_counter())
        print(f"[progress] probe={number}, step={step}: {self.ticks} ticks submitted "
              f"(separate process)", flush=True)

    def config_ecology(self):
        keys = ("project", "library", "grid", "mortality", "migration",
                "mortality_multiplier", "mass_balance", "currents")
        return {k: self.config[k] for k in keys if k in self.config}

    def __call__(self, trainer, step, run_dir, args):
        if (self.disabled or self.visualizer is None or not self.visualizer.enabled
                or getattr(self.visualizer, "_quit", False)):
            return
        try:
            if not self.asynchronous:
                super().__call__(trainer, step, run_dir, args)
                return
            first = self.config is None
            if first:
                self._start(trainer, step, run_dir, args.resume)
                if self.latest is not None:
                    self.visualizer.set_training_progress(self.records, self.config, self.latest)
            self._collect()
            if not first and step % self.interval():
                return
            if self._pending is not None:
                return
            self._submit(trainer, step)
        except Exception as error:
            self.disabled = True
            self.visualizer.set_progress_message("Progress unavailable; see terminal for details")
            print(f"[progress] Live progress disabled: {error}", flush=True)

    def wait(self):
        """Record a running probe and stop the worker (end of training)."""
        try:
            self._collect(block=True)
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None


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
