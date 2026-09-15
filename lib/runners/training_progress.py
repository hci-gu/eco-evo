"""Periodic inference on an isolated CPU world, with a single PNG progress plot."""

import copy
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from lib.config.config_loader import load_project_config, setup_full_mareld_mvp
from lib.environments.ecosystem import EcosystemEnvironment
from lib.gpu.config import DEFAULT_LIBRARY


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


def measure_survival(env, ticks, lower, upper):
    """Count consecutive post-step ticks inside [lower*B0, upper*B0].

    A breach on tick 1 scores 0; surviving the entire horizon scores `ticks`.
    Returning inside the band after a breach never restarts the counter.
    Groups with zero initial biomass have no defined ratio and score None.
    """
    ids = sorted(env.fgs)
    initial = np.array([env.fgs[f].biomass.sum() for f in ids], dtype=np.float64)
    if not np.isfinite(initial).all() or (initial < 0).any():
        raise ValueError("Inference initial biomass must be finite and nonnegative")
    present = initial > 0
    alive, survived = present.copy(), np.zeros(len(ids), dtype=np.int64)
    for _ in range(ticks):
        if not alive.any():
            break
        env.tick()
        current = np.array([env.fgs[f].biomass.sum() for f in ids], dtype=np.float64)
        alive &= np.isfinite(current) & (current >= lower * initial) & (current <= upper * initial)
        survived += alive
    return {
        "survival_ticks": {f: int(survived[i]) if present[i] else None for i, f in enumerate(ids)},
        "initial_biomass": dict(zip(ids, initial.tolist())),
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
    return dict(project=str(Path(builder.project_path).resolve()) if builder.project_path else None,
                library=str(Path(library).resolve()), grid=list(grid),
                mortality=mortality, migration=migration)


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
                                migration=config["migration"])


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
    def __init__(self, backend, every, ticks, lower, upper, seed, temperature, directory=None):
        self.backend, self.every, self.ticks = backend, every, ticks
        self.lower, self.upper, self.seed, self.temperature = lower, upper, seed, temperature
        self.directory = Path(directory) if directory is not None else None
        self.config = None
        self.records = []

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
            if json.loads(config_path.read_text()) != self.config:
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
            return measure_survival(env, self.ticks, self.lower, self.upper)

    def __call__(self, trainer, step, run_dir, args):
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
            self._plot()
        except Exception as error:
            # Monitoring failures must not discard a completed training update.
            print(f"[progress] WARNING: evaluation/plot failed at step {step}: {error}", flush=True)
            return
        print(f"[progress] updated {self.directory / 'latest.png'} ({time.perf_counter() - started:.1f}s)", flush=True)

    def _plot(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.ticker import MaxNLocator

        fig = Figure(figsize=(11, 6), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ids = sorted(self.records[-1]["survival_ticks"])
        missing = []
        for i, fid in enumerate(ids):
            values = [r["survival_ticks"][fid] for r in self.records]
            if all(v is None for v in values):
                missing.append(fid)
                continue
            ax.plot([r["evaluation"] for r in self.records],
                    [np.nan if v is None else v for v in values],
                    label=fid, marker="o", markersize=3, linewidth=1.6,
                    color=f"C{i % 10}", linestyle=("-", "--", ":")[i // 10 % 3])
        ax.set(title=f"Biomass survival within {self.lower:g}–{self.upper:g} × starting biomass",
               xlabel="Inference evaluation", ylabel="Consecutive ticks before first breach",
               ylim=(-0.02 * self.ticks, 1.08 * self.ticks))
        ax.axhline(self.ticks, color="0.5", linestyle=":", linewidth=1)
        ax.text(0.01, 0.97, f"Evaluation cap: {self.ticks} ticks · seed: {self.seed} · temperature: {self.temperature:g}",
                transform=ax.transAxes, va="top", color="0.35", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
        if len(missing) < len(ids):
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9, frameon=False)
        if missing:
            fig.supxlabel("No starting biomass (excluded): " + ", ".join(missing), fontsize=8)
        try:
            temporary = self.directory / "latest.png.tmp"
            fig.savefig(temporary, format="png", dpi=140)
            temporary.replace(self.directory / "latest.png")
        finally:
            fig.clear()
