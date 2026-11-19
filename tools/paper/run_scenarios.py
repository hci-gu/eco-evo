#!/usr/bin/env python3
"""Run the paper's scenarios with trained policies (inference, no training).

    # every scenario, its own seeds and length, on the GPU
    uv run tools/paper/run_scenarios.py scenarios/paper/*.yaml --device cuda

    # a local smoke test: 30 x 30, 20 days, one seed
    uv run tools/paper/run_scenarios.py scenarios/paper/current_fishing.yaml \\
        --grid 30*30 --days 20 --seeds 0 --out results/paper_smoke

Each (scenario, variant, seed) is written to
``<out>/<scenario>/<variant>/seed_<seed>/`` (``lib/scenarios/runner.py``);
a run whose ``meta.json`` exists is skipped unless ``--force``. See
``scenarios/paper/README.md`` and the notebooks in ``notebooks/paper/``.
"""

import argparse
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from lib.scenarios.runner import run_variant  # noqa: E402
from lib.scenarios.spec import load_scenario  # noqa: E402


def parse_grid(value):
    parts = value.lower().replace("x", "*").split("*")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("grid must be N*M")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenarios", nargs="+", help="scenario YAML files")
    parser.add_argument("--out", default="results/paper",
                        help="output root (default results/paper)")
    parser.add_argument("--device", default=None,
                        help="torch device for the tensor engine (default cuda if "
                             "available, else cpu)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="override every scenario's seeds")
    parser.add_argument("--run-name", default=None,
                        help="override every scenario's run_name (results/<run-name>/)")
    parser.add_argument("--grid", type=parse_grid, default=None,
                        help="override the grid, e.g. 30*30 (default: the scenario's)")
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--years", type=float, default=None,
                        help="override every scenario's length")
    length.add_argument("--days", type=float, default=None)
    parser.add_argument("--warmup-days", type=float, default=None,
                        help="override the warm-up (default: the scenario's, else the "
                             "run's training warm-up); for smoke tests")
    parser.add_argument("--only", nargs="+", default=None,
                        help="run only these variant names (e.g. k=0.5)")
    parser.add_argument("--force", action="store_true", help="re-run finished runs")
    parser.add_argument("--dry-run", action="store_true", help="list the runs and stop")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    jobs = []
    for path in args.scenarios:
        for variant in load_scenario(path):
            if args.only and variant.name not in args.only:
                continue
            for seed in (args.seeds if args.seeds is not None else variant.seeds):
                out = os.path.join(args.out, variant.scenario, variant.name, f"seed_{seed}")
                jobs.append((variant, seed, out))
    print(f"{len(jobs)} runs on {device}")
    if args.dry_run:
        for variant, seed, out in jobs:
            done = os.path.exists(os.path.join(out, "meta.json"))
            print(f"  {'done' if done else 'todo'}  {out}")
        return 0

    failures = 0
    for n, (variant, seed, out) in enumerate(jobs, 1):
        if not args.force and os.path.exists(os.path.join(out, "meta.json")):
            print(f"[{n}/{len(jobs)}] skip (done) {out}")
            continue
        print(f"[{n}/{len(jobs)}] {out}", flush=True)
        started = time.time()
        try:
            years = args.years
            if args.days is not None:
                years = args.days / 365.0
            meta = run_variant(variant, seed, out, device=device, grid=args.grid,
                               years=years, run_name=args.run_name,
                               warmup_days=args.warmup_days,
                               verbose=not args.quiet)
            print(f"    done in {time.time() - started:.0f} s "
                  f"({meta['years']:.2f} years)", flush=True)
        except Exception:
            failures += 1
            traceback.print_exc()
    if failures:
        print(f"{failures} run(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
