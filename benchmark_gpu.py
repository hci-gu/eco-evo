"""Compare complete CPU and GPU ARS iterations, including world reset.

Example: uv run benchmark_gpu.py --project mareld2.yaml --ticks 150
         --n-deltas 16 --worlds 3 --repeats 5 --output benchmark.json
"""

import argparse
import contextlib
import gc
import io
import json
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from lib.gpu.cli import add_common_arguments, builder_from_args, positive_int, targets_from_args, trainer_options
from lib.gpu.config import ProjectSpec
from lib.gpu.trainer import TensorARSTrainer
from lib.runners.trainer import ARSTrainer


class SharedWorldBuilder:
    """Publish an M=1 world's seed to the existing CPU pool between iterations.

    The reference trainer only sends per-task builder overrides when M>1.
    This shared scalar gives M=1 the same fixed-world-per-iteration semantics
    as train.py and the tensor trainer without rebuilding the worker pool.
    """
    def __init__(self, builder, shared_seed):
        self.builder, self.shared_seed = builder, shared_seed

    def __call__(self, seed=None):
        return self.builder.with_world(self.shared_seed.value)(seed=seed)

    def with_world(self, spawn_seed):
        return self.builder.with_world(spawn_seed)


def cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def synchronized_time(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started


def benchmark_backend(name, args, spec, targets):
    device = torch.device(args.cuda_device if name == "gpu" else "cpu")
    np.random.seed(args.seed & 0xFFFFFFFF)
    torch.manual_seed(args.seed)
    options = trainer_options(args)
    options["top_deltas"] = args.top_deltas or max(1, args.n_deltas // 2)
    started = time.perf_counter()
    workers = None
    if name == "cpu":
        workers = args.workers or max(1, min(2 * args.n_deltas * args.worlds, cpu_count() - 1))
        for key in ("worlds", "seed", "execution", "graph_ticks", "pairs_per_batch"):
            options.pop(key)
        shared_seed = mp.get_context("spawn").Value("q", args.seed)
        builder = SharedWorldBuilder(spec.builder, shared_seed)
        trainer = ARSTrainer(builder, spec.policy_params, n_workers=workers, **options)
        trainer.softmax_temperature = args.temperature
        iteration = 0
        world_rng = np.random.default_rng(args.seed)
        def refresh_worlds():
            trainer.world_list = world_rng.integers(1, 2**31 - 1, size=args.worlds).tolist()
            shared_seed.value = trainer.world_list[0]
        def step():
            nonlocal iteration
            # Publish even one explicit spawn world; the baseline and tensor
            # implementation use the same K/M workload and refresh frequency.
            if args.coevolution:
                refresh_worlds()
                trainer.train_step_coevolution(list(targets), args.ticks)
            else:
                for fid in targets:
                    refresh_worlds()
                    trainer.train_step(fid, args.ticks)
            iteration += 1
    else:
        if name == "tensor-cpu":
            options["execution"] = "eager"
        trainer = TensorARSTrainer(spec, device=device, **options)
        trainer.set_temperature(args.temperature)
        def step():
            if args.coevolution:
                trainer.train_step(targets, args.ticks)
            else:
                for fid in targets:
                    trainer.train_step([fid], args.ticks)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    startup = time.perf_counter() - started
    print(f"[{name}] startup {startup:.3f}s; warming up (includes compile/capture)...", flush=True)
    # Suppress the reference trainer's per-species console lines while timing.
    # Its actual diagnostics computations remain part of the existing baseline.
    def quiet_step():
        with contextlib.redirect_stdout(io.StringIO()):
            step()
    try:
        cold = synchronized_time(quiet_step, device)
        print(f"[{name}] first iteration {cold:.3f}s", flush=True)
        for _ in range(args.warmup - 1):
            synchronized_time(quiet_step, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for i in range(args.repeats):
            elapsed = synchronized_time(quiet_step, device)
            timings.append(elapsed)
            print(f"[{name}] repeat {i + 1}/{args.repeats}: {elapsed:.4f}s", flush=True)
        rollouts = 2 * args.n_deltas * args.worlds * (1 if args.coevolution else len(targets))
        median = statistics.median(timings)
        result = dict(backend=name, device=str(device), workers=workers,
                      execution="reference" if name == "cpu" else options["execution"],
                      setup_seconds=startup, first_iteration_seconds=cold,
                      iteration_seconds=timings, median_seconds=median,
                      mean_seconds=statistics.mean(timings),
                      min_seconds=min(timings), max_seconds=max(timings),
                      rollouts_per_iteration=rollouts,
                      world_ticks_per_second=rollouts * args.ticks / median,
                      cell_ticks_per_second=rollouts * args.ticks * args.grid[0] * args.grid[1] / median)
        if name != "cpu":
            result["final_metrics"] = trainer.metrics()
            if not all(torch.isfinite(w).all().item() for w in trainer.theta):
                raise RuntimeError("Non-finite policy weights after benchmark")
        if device.type == "cuda":
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            result["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
            if args.trace:
                args.trace.parent.mkdir(parents=True, exist_ok=True)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True) as profile:
                    quiet_step()
                    torch.cuda.synchronize(device)
                profile.export_chrome_trace(str(args.trace))
                result["trace"] = str(args.trace)
        return result
    finally:
        trainer.close()
        del trainer
        gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--backends", nargs="+", choices=("cpu", "gpu", "tensor-cpu"), default=["cpu", "gpu"])
    parser.add_argument("--workers", type=int, default=0, help="CPU reference workers; 0 uses available cores")
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--warmup", type=positive_int, default=2, help="Warmup iterations, including the separately reported cold iteration")
    parser.add_argument("--temperature", type=float, default=1.0, help="Fixed policy temperature for both backends")
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    parser.add_argument("--trace", type=Path, help="Optional Chrome/Perfetto trace of one extra GPU iteration, outside timing")
    args = parser.parse_args(argv)
    if args.workers < 0 or len(set(args.backends)) != len(args.backends):
        parser.error("Workers must be nonnegative and backends must be unique")
    if "gpu" in args.backends and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. On this machine use --backends cpu tensor-cpu; run cpu gpu on NVIDIA.")
    if args.temperature <= 0 or not np.isfinite(args.temperature):
        parser.error("Temperature must be positive and finite")
    if args.top_deltas is not None and args.top_deltas > args.n_deltas:
        parser.error("top_deltas cannot exceed n_deltas")
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        trainer_options(args)  # validate architecture before spawning workers
        t0 = time.perf_counter()
        spec = ProjectSpec(builder_from_args(args), seed=args.seed)
        targets = targets_from_args(args, spec.env.dm_ids)
        schema_seconds = time.perf_counter() - t0
    except (ValueError, OSError) as error:
        parser.error(str(error))
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    report = dict(format_version=1, revision=revision, platform=platform.platform(),
                  python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                  cpu_count=cpu_count(), schema_setup_seconds=schema_seconds,
                  options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  groups=spec.env.global_fg_order, decision_makers=spec.env.dm_ids,
                  notes=["Complete ARS iterations include world refresh/reset, perturbations, rollouts, moments, and updates.",
                         "The existing CPU backend reloads configuration per rollout; that cost remains in its baseline.",
                         "CPU and GPU use different random number generators. Matching seeds specify repeatable runs, not identical worlds or training trajectories.",
                         "All backends use the same configured grid, K, M, horizon, policy architecture, and reward options. Numerical parity is covered separately by tests.",
                         "Warmup iterations update policies. Cold and warm timings are reported separately; output/checkpoint I/O is excluded."],
                  results=[])
    if "gpu" in args.backends:
        properties = torch.cuda.get_device_properties(args.cuda_device)
        report["gpu"] = dict(name=properties.name, memory_mib=properties.total_memory / 2**20,
                             compute_capability=f"{properties.major}.{properties.minor}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.backends:
        try:
            report["results"].append(benchmark_backend(name, args, spec, targets))
        except Exception as error:
            report["error"] = dict(backend=name, type=type(error).__name__, message=str(error))
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            raise
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    by_name = {r["backend"]: r for r in report["results"]}
    if "cpu" in by_name and "gpu" in by_name:
        report["cpu_to_gpu_speedup"] = by_name["cpu"]["median_seconds"] / by_name["gpu"]["median_seconds"]
        print(f"CPU/GPU speedup: {report['cpu_to_gpu_speedup']:.2f}x", flush=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Benchmark report: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
