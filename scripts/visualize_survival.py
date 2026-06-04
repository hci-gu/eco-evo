#!/usr/bin/env python3
"""Plot normalized biomass survival from trained inference checkpoints.

The script runs the same inference setup as ``inference.py`` and saves a line
plot of B(t) / B(0) for each functional group. The rollout stops early when
any normalizable species reaches the configured collapse/explosion thresholds,
but the x-axis always spans the requested max tick count.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import build_env, load_policies_and_stats, parse_grid_arg

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


def _resolve_checkpoint_dir(args: argparse.Namespace) -> str:
    checkpoint_dir = args.input_folder or args.checkpoints
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join("results", args.run_name)
    return checkpoint_dir


def _safe_name(path: str) -> str:
    name = os.path.basename(os.path.normpath(path)) or "checkpoints"
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def _run_until_threshold(env, policies, obs_mean, obs_var, max_ticks,
                         lower_threshold, upper_threshold, verbose=True):
    env.policies = dict(policies)
    env.obs_mean = obs_mean
    env.obs_var = obs_var
    env._rebuild_batched_weights()

    fg_ids = list(env.fgs.keys())
    initial = {
        fid: float(env.fgs[fid].biomass.sum())
        for fid in fg_ids
    }
    history = {fid: [initial[fid]] for fid in fg_ids}
    normalizable = [fid for fid in fg_ids if initial[fid] > 0.0]

    stop_reason = None
    stop_tick = None
    for tick in range(1, int(max_ticks) + 1):
        env.step()
        ratios = {}
        for fid in fg_ids:
            biomass = float(env.fgs[fid].biomass.sum())
            history[fid].append(biomass)
            if initial[fid] > 0.0:
                ratios[fid] = biomass / initial[fid]

        if verbose and (tick == 1 or tick % max(1, int(max_ticks) // 10) == 0):
            print(f"    tick {tick}/{max_ticks}")

        for fid, ratio in ratios.items():
            if ratio <= lower_threshold:
                stop_reason = f"{fid} reached {ratio:.3f}x initial biomass"
                stop_tick = tick
                break
            if ratio >= upper_threshold:
                stop_reason = f"{fid} reached {ratio:.3f}x initial biomass"
                stop_tick = tick
                break
        if stop_reason is not None:
            break

    normalized = {}
    for fid in fg_ids:
        values = np.asarray(history[fid], dtype=np.float64)
        if initial[fid] > 0.0:
            normalized[fid] = values / initial[fid]
        else:
            normalized[fid] = np.full(values.shape, np.nan, dtype=np.float64)

    return normalized, initial, normalizable, stop_tick, stop_reason


def _plot_survival(normalized, max_ticks, output_path, lower_threshold,
                   upper_threshold, stop_tick=None, stop_reason=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    for fid, ratios in normalized.items():
        if np.all(np.isnan(ratios)):
            continue
        x = np.arange(ratios.shape[0])
        ax.plot(x, ratios, linewidth=1.8, label=fid)

    ax.axhline(lower_threshold, color="#b3261e", linestyle="--", linewidth=1.2,
               label=f"{lower_threshold:g}x threshold")
    ax.axhline(upper_threshold, color="#b3261e", linestyle=":", linewidth=1.2,
               label=f"{upper_threshold:g}x threshold")
    if stop_tick is not None:
        ax.axvline(stop_tick, color="#4d4d4d", linestyle="-.", linewidth=1.0)

    ax.set_xlim(0, int(max_ticks))
    ymax = max(float(upper_threshold) * 1.08, 1.2)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Normalized biomass, B(t) / B(0)")
    ax.set_title("Population Survival During Inference")
    if stop_reason:
        ax.text(
            0.01, 0.98, f"Stopped at tick {stop_tick}: {stop_reason}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
        )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run inference and save a normalized biomass survival plot."
    )
    parser.add_argument("--input-folder", type=str, default=None,
                        help="Directory containing policy_<fg>.pth checkpoints.")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Alias for --input-folder; matches inference.py wording.")
    parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, default="default",
                        help="Fallback checkpoint run name when no input folder is given. "
                             "Uses results/<run-name>/, matching inference.py.")
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Directory where the survival plot image is saved.")
    parser.add_argument("--image-name", type=str, default=None,
                        help="Optional image filename. Default: survival_<checkpoint-folder>.png.")
    parser.add_argument("--project", type=str, default=None,
                        help="Path to project YAML file. If omitted, uses setup_full_mareld_mvp.")
    parser.add_argument("--grid", type=parse_grid_arg, default=(60, 60),
                        help="Grid dimensions as n*m (default: 60*60).")
    parser.add_argument("--ticks", type=int, default=100,
                        help="Maximum simulation ticks to run and the x-axis size. Default: 100.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for deterministic initial biomass distribution.")
    parser.add_argument("--mortality", choices=["on", "off"], default="off",
                        help="Toggle artificial natural mortality. Default: off.")
    parser.add_argument("--lower-threshold", type=float, default=0.3,
                        help="Stop when any species reaches this fraction of B(0). Default: 0.3.")
    parser.add_argument("--upper-threshold", type=float, default=3.0,
                        help="Stop when any species reaches this multiple of B(0). Default: 3.0.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output.")
    args = parser.parse_args(argv)

    verbose = not args.quiet
    checkpoint_dir = _resolve_checkpoint_dir(args)
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: checkpoint directory does not exist: {checkpoint_dir}", file=sys.stderr)
        return 1
    if args.ticks < 1:
        print("Error: --ticks must be >= 1.", file=sys.stderr)
        return 1
    if args.lower_threshold <= 0.0 or args.upper_threshold <= args.lower_threshold:
        print("Error: thresholds must satisfy 0 < lower < upper.", file=sys.stderr)
        return 1

    os.makedirs(args.output_folder, exist_ok=True)
    image_name = args.image_name or f"survival_{_safe_name(checkpoint_dir)}.png"
    output_path = os.path.join(args.output_folder, image_name)

    if verbose:
        print("==========================================")
        print("      MARELD SURVIVAL VISUALIZATION       ")
        print("==========================================")
        print(f"Project:       {args.project or '(built-in MVP)'}")
        print(f"Checkpoints:   {checkpoint_dir}")
        print(f"Grid:          {args.grid[0]}x{args.grid[1]}")
        print(f"Max ticks:     {args.ticks}")
        print(f"Thresholds:    {args.lower_threshold:g}x .. {args.upper_threshold:g}x")
        print(f"Mortality:     {args.mortality}")
        print(f"Output:        {output_path}")
        print("------------------------------------------")

    env = build_env(args.project, args.grid, seed=args.seed, verbose=verbose,
                    apply_natural_mortality=(args.mortality == "on"))
    policies, mean, var = load_policies_and_stats(env, checkpoint_dir, verbose=verbose)

    normalized, initial, normalizable, stop_tick, stop_reason = _run_until_threshold(
        env, policies, mean, var,
        max_ticks=args.ticks,
        lower_threshold=float(args.lower_threshold),
        upper_threshold=float(args.upper_threshold),
        verbose=verbose,
    )

    _plot_survival(
        normalized,
        max_ticks=args.ticks,
        output_path=output_path,
        lower_threshold=float(args.lower_threshold),
        upper_threshold=float(args.upper_threshold),
        stop_tick=stop_tick,
        stop_reason=stop_reason,
    )

    if verbose:
        zero_initial = [fid for fid, b0 in initial.items() if b0 <= 0.0]
        if stop_reason:
            print(f"Stopped at tick {stop_tick}: {stop_reason}")
        else:
            print(f"Completed max ticks without threshold breach: {args.ticks}")
        if zero_initial:
            print("Skipped threshold checks for zero-initial-biomass species: "
                  + ", ".join(zero_initial))
        print(f"Saved survival plot to {output_path}")
        print("==========================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
