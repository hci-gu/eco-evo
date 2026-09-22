"""GPU ecosystem training with an optional live viewer. See GPU_TRAINING.md."""

import argparse
import itertools
import json
import math
import signal
import time
from pathlib import Path

import torch

from lib.gpu.cli import (add_common_arguments, builder_from_args, positive_int,
                         targets_from_args, trainer_options)
from lib.gpu.config import ProjectSpec
from lib.gpu.trainer import TensorARSTrainer
from lib.training_profiles import add_profile_argument, parse_training_args
from lib.runners.training_progress import add_progress_arguments, validate_progress_arguments


def world_schedule(value):
    if not value:
        return []
    schedule = []
    for part in value.split(","):
        count, generation = map(int, part.split("@"))
        if count < 1 or generation < 0:
            raise ValueError("World schedule must use positive counts and nonnegative generations")
        schedule.append((generation, count))
    if len({g for g, _ in schedule}) != len(schedule):
        raise ValueError("World schedule contains duplicate generations")
    return sorted(schedule)


def save_checkpoint(trainer, directory, generation, iteration_in_generation, options):
    payload = dict(trainer=trainer.state_dict(), generation=generation,
                   iteration_in_generation=iteration_in_generation, options=options)
    temporary = directory / "trainer.pth.tmp"
    torch.save(payload, temporary)
    temporary.replace(directory / "trainer.pth")
    trainer.export_policies(directory)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:1, or cpu for validation")
    parser.add_argument("--generations", default="inf", help="Additional generations to run, or inf")
    parser.add_argument("--iter-per-gen", "--iter_per_gen", dest="iter_per_gen", type=positive_int, default=20)
    parser.add_argument("--run-name", "--run_name", dest="run_name", default="gpu")
    parser.add_argument("--output", type=Path, help="Override results/<run-name>")
    parser.add_argument("--resume", action="store_true", help="Resume the complete GPU trainer checkpoint")
    parser.add_argument("--init-from", type=Path, help="Warm-start from existing policy_<species>.pth files")
    parser.add_argument("--log-every", type=positive_int, default=1)
    parser.add_argument("--checkpoint-every", type=positive_int, default=1, help="Save every N generations")
    parser.add_argument("--snapshot-every", type=int, default=0, help="Save one ecosystem snapshot every N generations; 0 disables")
    parser.add_argument("--worlds-refresh", "--worlds_refresh", dest="worlds_refresh", choices=("iteration", "generation"), default="iteration")
    parser.add_argument("--worlds-schedule", "--rollouts_per_delta_schedule", dest="worlds_schedule")
    parser.add_argument("--temp-start", "--temp_start", dest="temp_start", type=float, default=3.0)
    parser.add_argument("--temp-end", "--temp_end", dest="temp_end", type=float, default=1.0)
    parser.add_argument("--temp-anneal-gens", "--temp_anneal_gens", dest="temp_anneal_gens", type=positive_int, default=10)
    parser.add_argument("--visual", action="store_true",
                        help="Open the live pygame viewer with CPU inference probes between GPU updates")
    add_profile_argument(parser)
    add_progress_arguments(parser)
    return parser


def main(argv=None, *, on_step=None):
    parser = build_parser()
    args = parse_training_args(parser, argv, destinations={
        "n_eval_ticks": "ticks", "rollouts_per_delta": "worlds",
    })
    validate_progress_arguments(parser, args)
    try:
        generations = None if args.generations.lower() == "inf" else positive_int(args.generations)
        schedule = world_schedule(args.worlds_schedule)
        options = trainer_options(args)
        if args.resume and args.init_from:
            raise ValueError("Use either --resume or --init-from")
        if args.snapshot_every < 0:
            raise ValueError("--snapshot-every cannot be negative")
        if args.device == "cpu" and "graph" in args.execution:
            raise ValueError("Use --execution eager or compile with --device cpu")
        torch.set_num_threads(1)
        # Full FP32 policy math is the reproducibility baseline. Allow later
        # precision experiments only after reference comparisons on the target.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
        spec = ProjectSpec(builder_from_args(args), seed=args.seed)
        targets = targets_from_args(args, spec.env.dm_ids)
        trainer = TensorARSTrainer(spec, device=args.device, **options)
        trainer.set_temperature(args.temp_start)
        if args.temp_end <= 0 or not math.isfinite(args.temp_end):
            raise ValueError("--temp-end must be positive")
        directory = args.output or Path("results") / args.run_name
        if (directory / "trainer.pth").exists() and not args.resume:
            raise ValueError("A trainer checkpoint already exists here; use --resume or a different output directory")
        if not args.resume and any(directory.glob("policy_*.pth")):
            raise ValueError("Policy files already exist here; use a new output directory and --init-from")
        generation, within = 0, 0
        if args.resume:
            payload = torch.load(directory / "trainer.pth", map_location="cpu", weights_only=False)
            trainer.load_state_dict(payload["trainer"])
            generation, within = int(payload["generation"]), int(payload["iteration_in_generation"])
            if within >= args.iter_per_gen:
                generation, within = generation + 1, 0
        elif args.init_from:
            print("Loaded policies:", ", ".join(trainer.import_policies(args.init_from)), flush=True)
        directory.mkdir(parents=True, exist_ok=True)
    except (ValueError, RuntimeError, OSError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(groups=trainer.model.ids, decision_makers=trainer.model.dm_ids,
                    torch_version=torch.__version__, cuda_version=torch.version.cuda)
    (directory / "gpu_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Device: {trainer.model.device}; {args.n_deltas} delta pairs, {args.worlds} worlds, "
          f"{args.ticks} ticks; execution={args.execution}", flush=True)
    print(f"Output: {directory}. First iteration includes warmup/compilation/capture.", flush=True)
    if trainer.model.food_blobs:
        config = trainer.model.food_blobs
        print(f"DEBUG FOOD BLOBS: constant food; speed={config.speed:g} cells/tick, "
              f"radius={config.radius:g} (0=automatic), seed={config.seed}", flush=True)
    if args.currents == "on":
        print(f"Currents: on; max drift={args.current_strength:g}/tick, "
              f"scroll={1 / args.current_period:g} cells/tick per axis, "
              f"scale={args.current_scale:g} cells, seed={args.current_seed}", flush=True)
    if args.mortality_multiplier != 1.0:
        print(f"Mortality multiplier: every natural_mortality scaled by "
              f"{args.mortality_multiplier:g} (mortality={args.mortality}).",
              flush=True)
    if args.population_stability:
        print(f"Population stability: biomass bounds [{args.population_min:g}, {args.population_max:g}) "
              "× start; warnings, capped energy reward, failure tail −5/tick.", flush=True)
    if args.local_reward:
        expression = "log(B/A)" if args.local_reward_metric == "log" else "B/A"
        print(f"Local reward: per-cell {expression} with B(c,t+1) source-tracked "
              f"through {{c,N,E,S,W}}; {args.local_reward_norm} over cells, "
              f"theta={args.local_reward_theta:g}, "
              f"clip=[{args.local_reward_clip[0]:g}, {args.local_reward_clip[1]:g}], "
              f"min energy factor={args.local_reward_min_energy_factor:g}.", flush=True)
    stop = None if generations is None else generation + generations
    next_generation, next_within = generation, within
    started = time.perf_counter()
    last_log_time, last_log_update = started, trainer.iterations_completed
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    save_final = False
    visual = None
    try:
        if args.visual:
            from lib.gpu.visual import GPUTrainingVisualizer
            visual = GPUTrainingVisualizer(trainer, args, directory)
            visual.update_progress(trainer, args)
        if on_step is not None:
            on_step(trainer, trainer.iterations_completed, directory, args)
        with (directory / "training.jsonl").open("a") as log:
            for gen in itertools.count(generation):
                if stop is not None and gen >= stop:
                    break
                count = args.worlds
                for boundary, value in schedule:
                    if gen >= boundary:
                        count = value
                trainer.set_world_count(count)
                fraction = min(1.0, gen / max(1, args.temp_anneal_gens - 1))
                trainer.set_temperature(args.temp_start + fraction * (args.temp_end - args.temp_start))
                for iteration in range(within if gen == generation else 0, args.iter_per_gen):
                    epoch = gen if args.worlds_refresh == "generation" else None
                    ticks = args.ticks if visual is None else visual.training_ticks(args.ticks)
                    batches = [targets] if args.coevolution else [[species] for species in targets]
                    for batch in batches:
                        if visual is None:
                            trainer.train_step(batch, ticks, epoch)
                        else:
                            visual.train_step(trainer, batch, ticks, epoch)
                            visual.update(trainer, batch, gen, iteration, args.ticks)
                            visual.update_progress(trainer, args)
                        if on_step is not None:
                            on_step(trainer, trainer.iterations_completed, directory, args)
                    next_generation, next_within = gen, iteration + 1
                    if trainer.iterations_completed % args.log_every == 0:
                        metrics = trainer.metrics()  # intentional compact readback
                        now = time.perf_counter()
                        record = dict(generation=gen, iteration=iteration,
                                      updates=trainer.iterations_completed,
                                      elapsed_seconds=now - started,
                                      seconds_per_update=(now - last_log_time) / (trainer.iterations_completed - last_log_update),
                                      worlds=count, **metrics)
                        last_log_time, last_log_update = now, trainer.iterations_completed
                        log.write(json.dumps(record, allow_nan=False) + "\n")
                        log.flush()
                        rewards = ", ".join(f"{f}={r:+.5f}" for f, r in zip(trainer.model.dm_ids, metrics["reward_mean"]))
                        print(f"gen={gen} iter={iteration + 1}/{args.iter_per_gen} "
                              f"seconds/update={record['seconds_per_update']:.3f} {rewards}", flush=True)
                    if stop_requested:
                        raise KeyboardInterrupt
                next_generation, next_within = gen + 1, 0
                if (gen + 1) % args.checkpoint_every == 0:
                    save_checkpoint(trainer, directory, next_generation, next_within, metadata)
                if args.snapshot_every and (gen + 1) % args.snapshot_every == 0:
                    import numpy as np
                    np.savez_compressed(directory / f"snapshot_{gen + 1:06d}.npz",
                                        group_ids=np.asarray(trainer.model.ids), **trainer.runner.snapshot())
            save_final = True
    except KeyboardInterrupt:
        print("Interrupted; saving the last completed update.", flush=True)
        save_final = True
    finally:
        if visual is not None:
            visual.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if save_final:
            save_checkpoint(trainer, directory, next_generation, next_within, metadata)
        trainer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
