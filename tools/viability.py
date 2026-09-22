"""Long-term viability test for a project configuration - no ARS involved.

    python3 tools/viability.py --project mareld2.yaml --grid 20*20
    python3 tools/viability.py --years 1 --seeds 1 --behaviour eat   # quick
    python3 tools/viability.py --migration on --mortality on         # open box
    python3 tools/viability.py --spawn configured   # geometry as the library draws it

The rig is two-factorial: *behaviour* (``--behaviour``) and *spawn
geometry* (``--spawn``). The verdict is taken from the normative corner,
``--spawn colocated --behaviour greedy`` - the most favourable start and
the strongest frozen behaviour - and the exit code is 1 only when that
arm fails the criterion in ``VIABILITY.md``, so the rig can gate a
pipeline. Running only ``neutral``/``random`` yields no verdict at all:
a uniform or untrained policy dying says nothing about the world.

The verdict is a statement about the *world*: the behaviour is frozen and
the reward function is never evaluated, so a FAIL cannot be blamed on
training. ``--behaviour policy`` is available to compare a checkpoint
against the frozen arms, but it is not part of the default set.
"""
import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from lib.config.config_loader import project_tick_hours  # noqa: E402
from lib.diagnostics import viability  # noqa: E402


def parse_grid(text):
    for separator in ('*', 'x', 'X', ','):
        if separator in text:
            height, width = text.split(separator, 1)
            break
    else:
        height = width = text
    height, width = int(height), int(width)
    if height < 3 or width < 3:
        raise argparse.ArgumentTypeError("both grid dimensions must be >= 3")
    return height, width


def parse_behaviours(text):
    names = [name.strip() for name in text.split(',') if name.strip()]
    unknown = [name for name in names if name not in viability.BEHAVIOURS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown behaviour(s) {unknown}, "
            f"expected from {list(viability.BEHAVIOURS)}")
    return names or [viability.NEUTRAL]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Measure whether the ecosystem is viable without training.")
    parser.add_argument("--project", default="mareld2.yaml")
    parser.add_argument("--grid", type=parse_grid, default=(20, 20),
                        help="Grid as n*m (default: 20*20).")
    parser.add_argument("--years", type=float, default=5.0,
                        help="Horizon in simulated years (default: 5).")
    parser.add_argument("--ticks", type=int, default=None,
                        help="Horizon in ticks; overrides --years.")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of independent spawn layouts (default: 3).")
    parser.add_argument("--seed0", type=int, default=20260530,
                        help="First seed; the rest are consecutive.")
    parser.add_argument("--behaviour", type=parse_behaviours,
                        default=[viability.GREEDY, viability.GREEDY_HIDE,
                                 viability.EAT],
                        help="Comma-separated arms from greedy,greedy_hide,"
                             "eat,neutral,random,policy "
                             "(default: greedy,greedy_hide,eat).")
    parser.add_argument("--spawn", choices=(viability.SPAWN_COLOCATED,
                                            viability.SPAWN_CONFIGURED),
                        default=viability.SPAWN_COLOCATED,
                        help="'colocated' puts every predator on the richest "
                             "cells of its own prey, keeping its total "
                             "biomass and cell count, so spawn geometry stops "
                             "being a hidden term in the verdict. "
                             "'configured' uses the library's own layouts.")
    parser.add_argument("--colocated-spawn", "--colocated_spawn",
                        dest="spawn", action="store_const",
                        const=viability.SPAWN_COLOCATED,
                        help="Alias for --spawn colocated.")
    parser.add_argument("--run-name", "--run_name", dest="run_name",
                        default=None,
                        help="results/<run-name> for --behaviour policy.")
    parser.add_argument("--mortality", choices=("on", "off"), default="off",
                        help="Density-independent natural mortality (default: off, "
                             "matching train.py).")
    parser.add_argument("--mortality_multiplier", "--mortality-multiplier",
                        dest="mortality_multiplier", type=float, default=1.0,
                        help="Global scale on every FG's natural_mortality.")
    parser.add_argument("--migration", choices=("on", "off"), default="off",
                        help="'off' is a closed box, 'on' lets biomass cross the "
                             "grid edges. This is the open-vs-closed system "
                             "question; the verdict is reported per setting.")
    parser.add_argument("--floor", type=float, default=0.10,
                        help="Minimum final-window mean, as a fraction of spawn.")
    parser.add_argument("--ceiling", type=float, default=10.0,
                        help="Maximum final-window mean, as a multiple of spawn.")
    parser.add_argument("--max-drift", "--max_drift", dest="max_drift",
                        type=float, default=2.0,
                        help="Largest tolerated factor between the last two "
                             "windows (default: 2).")
    parser.add_argument("--window-frac", "--window_frac", dest="window_frac",
                        type=float, default=0.10,
                        help="Share of the horizon used as the final window.")
    parser.add_argument("--json", default=None,
                        help="Write the machine-readable summary here.")
    parser.add_argument("--csv", default=None,
                        help="Write the per-FG trajectories here.")
    parser.add_argument("--sample-every", "--sample_every",
                        dest="sample_every", type=int, default=10,
                        help="Tick stride for --csv (default: 10).")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _build_env(args, seed):
    from inference import build_env

    return build_env(
        args.project, args.grid, seed=seed, verbose=False,
        apply_natural_mortality=args.mortality == "on",
        mortality_multiplier=args.mortality_multiplier,
        migration=args.migration == "on",
    )


def _write_csv(path, rows, sample_every):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("behaviour,seed,fg,tick,biomass\n")
        for behaviour, seed, result in rows:
            for fid, values in result.series.items():
                for k in range(0, len(values), max(1, sample_every)):
                    handle.write(f"{behaviour},{seed},{fid},{k + 1},"
                                 f"{values[k]:.6g}\n")


def main(argv=None):
    args = build_parser().parse_args(argv)

    # The horizon is specified in simulated years, so the tick length
    # the project runs at decides how many ticks that is. Section 97.
    tick_hours = project_tick_hours(args.project)
    criterion = viability.ViabilityCriterion(
        years=(args.ticks / viability.ticks_per_year(tick_hours) if args.ticks
               else args.years),
        tick_hours=tick_hours,
        seeds=args.seeds,
        floor=args.floor,
        ceiling=args.ceiling,
        max_drift=args.max_drift,
        window_frac=args.window_frac,
    )
    checkpoint_dir = (os.path.join("results", args.run_name)
                      if args.run_name else None)

    height, width = args.grid
    colocated = args.spawn == viability.SPAWN_COLOCATED
    spawn_note = ("predators placed on their prey" if colocated
                  else "geometry as the library draws it")
    if not args.quiet:
        print(f"project        {args.project}")
        print(f"grid           {height}x{width}")
        print(f"tick           {tick_hours} h "
              f"({viability.ticks_per_year(tick_hours)} ticks/year)")
        print(f"spawn          {args.spawn} ({spawn_note})")
        print(f"mortality      {args.mortality} "
              f"(multiplier {args.mortality_multiplier:g})")
        print(f"migration      {args.migration} "
              f"({'open' if args.migration == 'on' else 'closed'} system)")
        print(f"criterion      {criterion.describe()}")
        print(f"arms           {', '.join(args.behaviour)}")
        print()

    arms, rows, overlap = [], [], []
    for behaviour in args.behaviour:
        arm = viability.ArmVerdict(behaviour=behaviour, criterion=criterion)
        for index in range(criterion.seeds):
            seed = args.seed0 + index
            started = time.time()
            env = _build_env(args, seed)
            if colocated:
                viability.colocate_spawn(env)
            if not overlap:
                # Tick-0 co-location, measured on the first rollout: a
                # FAIL cannot be read without knowing whether predator
                # and prey even started in the same cells.
                overlap = viability.spawn_overlap(env)
            provider = viability.install_behaviour(
                env, behaviour, seed=seed, checkpoint_dir=checkpoint_dir)
            result = viability.run_rollout(env, criterion.ticks, provider)
            result.seed = seed
            arm.per_seed[seed] = viability.evaluate_seed(result, criterion)
            rows.append((behaviour, seed, result))
            if not args.quiet:
                print(f"  [{behaviour}] seed {seed}: {criterion.ticks} ticks "
                      f"in {time.time() - started:.1f} s")
        arms.append(arm)
        if not args.quiet:
            print()
            print(viability.format_arm(arm))
            print()

    if overlap and not args.quiet:
        print(f"predator/prey overlap at tick 0 (seed {args.seed0}, "
              f"{args.spawn} spawn)")
        print(viability.format_overlap(overlap))
        print()

    summary = viability.summary_dict(arms, spawn=args.spawn, overlap=overlap)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        if not args.quiet:
            print(f"wrote {args.json}")
    if args.csv:
        _write_csv(args.csv, rows, args.sample_every)
        if not args.quiet:
            print(f"wrote {args.csv}")

    if summary["viable"] is None:
        print("overall: NO VERDICT - only diagnostic arms were run "
              f"({', '.join(args.behaviour)}); the verdict needs one of "
              f"{', '.join(viability.NORMATIVE_BEHAVIOURS)}")
        return 0
    verdict = "VIABLE" if summary["viable"] else "NOT VIABLE"
    scope = ("normative corner (colocated spawn, greedy)"
             if summary["normative_corner"]
             else f"{summary['verdict_arm']} arm, {args.spawn} spawn")
    print(f"overall: {verdict}  [{scope}]")
    return 0 if summary["viable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
