"""Score the plankton against SHARK station A17 (sections 201-204).

    python3 tools/probes/a17_score.py run  CKPT_DIR [--seed S] [--years N]
        [--burn-in N] [--project P] [--library L] [--grid H W]
        [--behaviour NAME] [--currents on|off] [--out FILE.json]
    python3 tools/probes/a17_score.py score FILE.json [FILE.json ...]

``run`` builds the probe world (as the 181-204 scratch probes: the
project on a 60x60 grid, M1 on, mass balance, migration, currents),
lets the policies in CKPT_DIR act (or a viability arm with
``--behaviour``), skips ``--burn-in`` years and averages the next
``--years`` by calendar month (the daylight calendar's month). It writes
JSON with the monthly means and prints the score; ``score`` re-prints
the score of saved runs side by side.

Targets (section 204; research_notes/"Primarproduktion och C:chl i
Skagerrak" and "Djupklorofyllmaximum", SHARK A17 2010-2023):

- column biomass 0-30 m (t WW/km2): A17 chl at the measured monthly
  column C:chl (21 34 29 27 30 34 34 35 36 25 20 29) and C:WW 0.1,
  against the model's mixed + subsurface producer per km2;
- surface carbon (ug C/l): A17 0-10 m chl at the surface C:chl
  (21 34 31 35 55 70 70 60 50 25 20 29), against the mixed-layer
  producer over the mixed-layer depth;
- nitrate (umol/l): A17 surface and 15-30 m, against the mixed and
  the subsurface pool;
- zooplankton (t WW/km2): A17 0-25 m, the resident stock in Dec-Mar
  (Calanus risen from the Norwegian Trench excluded, section 183);
- primary production 150 (120-200) g C/m2/yr and the f-ratio
  (new / gross production) 0.2-0.4.

Errors are the log-RMSE over the twelve months, log(x + 0.05) as in the
scratch scorer (resume 182); r is the correlation of the logs. Gross
production is the producers' pool-paid growth; new N is the net gain
of the pools from the reservoir (two-layer column) or from the pool
tables (one layer); both in producer-biomass equivalents, x C:WW 0.1.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
C_WW = 0.1
A17_COLUMN_CHL = np.array([23, 66, 73, 52, 34, 39, 26, 25, 25, 40, 34, 17.0])   # mg chl/m2, 0-30 m
A17_SURFACE_CHL = np.array([1.21, 3.88, 3.28, 1.48, 0.94, 0.80, 0.46, 0.65, 0.79,
                            1.81, 1.54, 0.77])                                    # ug chl/l, 0-10 m
COLUMN_C_CHL = np.array([21, 34, 29, 27, 30, 34, 34, 35, 36, 25, 20, 29.0])
SURFACE_C_CHL = np.array([21, 34, 31, 35, 55, 70, 70, 60, 50, 25, 20, 29.0])
A17_NITRATE = np.array([5.09, 4.84, 2.33, 0.32, 0.13, 0.10, 0.10, 0.10, 0.10, 0.32, 1.30, 3.61])
A17_NITRATE_DEEP = np.array([5.5, 6.4, 6.3, 5.6, 3.4, 2.4, 1.4, 0.8, 0.5, 0.6, 2.0, 3.75])
A17_ZOO_RESIDENT = np.array([0.21, 0.18, 1.56, 19.3, 26.0, 11.4, 6.9, 4.6, 3.7, 2.9, 3.5, 0.92])
PP_TARGET, PP_RANGE = 150.0, (120.0, 200.0)
F_RANGE = (0.2, 0.4)

# Derived targets: g C = mg chl x C:chl / 1000; t WW/km2 = g WW/m2 = g C / C:WW.
A17_COLUMN = A17_COLUMN_CHL * COLUMN_C_CHL / 1000.0 / C_WW
A17_SURFACE_C = A17_SURFACE_CHL * SURFACE_C_CHL


def log_rmse(model, obs):
    model, obs = np.asarray(model, float), np.asarray(obs, float)
    return float(np.sqrt(np.mean((np.log(model + 0.05) - np.log(obs + 0.05)) ** 2)))


def log_r(model, obs):
    model, obs = np.asarray(model, float), np.asarray(obs, float)
    if np.std(model) == 0.0:
        return float("nan")
    return float(np.corrcoef(np.log(model + 1e-3), np.log(obs + 1e-3))[0, 1])


def score(run):
    """Summary metrics of one run's monthly means (a dict from ``run``)."""
    column = np.array(run["mixed"]) + np.array(run["subsurface"])
    surface = np.array(run["surface_carbon"])
    out = {
        "column_err": log_rmse(column, A17_COLUMN),
        "column_r": log_r(column, A17_COLUMN),
        "column_ratio": float(column.mean() / A17_COLUMN.mean()),
        "surface_err": log_rmse(surface, A17_SURFACE_C),
        "surface_r": log_r(surface, A17_SURFACE_C),
        "nitrate_err": log_rmse(run["nitrate"], A17_NITRATE),
        "zoo_err": log_rmse(run["zooplankton"], A17_ZOO_RESIDENT),
        "zoo_r": log_r(run["zooplankton"], A17_ZOO_RESIDENT),
        "pp": float(run["pp"]),
        "f_ratio": float(run["f_ratio"]),
    }
    if any(v > 0 for v in run.get("nitrate_deep", [])):
        out["nitrate_deep_err"] = log_rmse(run["nitrate_deep"], A17_NITRATE_DEEP)
    return out


def _row(name, values, fmt="{:6.2f}"):
    return f"{name:24s}" + " ".join(fmt.format(v) for v in values)


def report(run, label=""):
    s = score(run)
    column = np.array(run["mixed"]) + np.array(run["subsurface"])
    lines = [f"== {label} (seed {run.get('seed')}, {run.get('years')} yr after "
             f"{run.get('burn_in')} burn-in)",
             " " * 24 + " ".join(f"{m:>6s}" for m in MONTHS),
             _row("column t/km2", column), _row("  A17", A17_COLUMN),
             _row("  ratio", column / A17_COLUMN),
             _row("surface ug C/l", run["surface_carbon"], "{:6.1f}"),
             _row("  A17", A17_SURFACE_C, "{:6.1f}"),
             _row("nitrate mixed", run["nitrate"]), _row("  A17 surface", A17_NITRATE)]
    if "nitrate_deep_err" in s:
        lines += [_row("nitrate subsurface", run["nitrate_deep"]),
                  _row("  A17 15-30 m", A17_NITRATE_DEEP)]
    lines += [_row("zooplankton t/km2", run["zooplankton"]),
              _row("  A17 resident", A17_ZOO_RESIDENT),
              _row("microzooplankton", run["microzooplankton"]),
              f"PP {s['pp']:.1f} g C/m2/yr (target {PP_TARGET:.0f}, "
              f"{PP_RANGE[0]:.0f}-{PP_RANGE[1]:.0f}); f-ratio {s['f_ratio']:.2f} "
              f"(target {F_RANGE[0]}-{F_RANGE[1]})",
              "log-RMSE: column {column_err:.2f} (r {column_r:+.2f}, mean ratio "
              "{column_ratio:.2f}) | surface C {surface_err:.2f} (r {surface_r:+.2f}) | "
              "nitrate {nitrate_err:.2f}".format(**s)
              + (f" | deep nitrate {s['nitrate_deep_err']:.2f}" if "nitrate_deep_err" in s else "")
              + " | zooplankton {zoo_err:.2f} (r {zoo_r:+.2f})".format(**s)]
    return "\n".join(lines)


def run_world(args):
    os.chdir(ROOT)
    from inference import build_env
    from lib.diagnostics import viability
    from lib.environments.ecosystem_env import interactions, population_change
    from lib.environments.ecosystem_env.currents import CurrentConfig
    from lib.world import daylight, nutrients
    from lib.world.tick_time import ticks_per_day

    currents = CurrentConfig(0.1, 20, 0, 12.0) if args.currents == "on" else None
    env = build_env(args.project, tuple(args.grid), seed=args.seed, verbose=False,
                    apply_natural_mortality=True, mass_balance=True, migration=True,
                    currents=currents, library_path=args.library)
    if not getattr(env, "_has_nutrients", False):
        raise SystemExit("the project has no nutrient pools; nothing to score")
    if args.behaviour:
        provider = viability.install_behaviour(env, args.behaviour)
    else:
        viability.install_behaviour(env, viability.POLICY, checkpoint_dir=args.checkpoints)
        provider = None
    ex = getattr(env, "layer_exchange", None)
    mixed = ex["mixed"] if ex else env.nutrient_ids[0]
    deep = ex["subsurface"] if ex else None
    cells = env.H * env.W
    hours = float(env.daylight["tick_hours"])
    tpd = ticks_per_day(hours)
    tpy = int(round(365 * tpd))
    flux = {"gross": 0.0, "new": 0.0}

    original_delta = population_change._nutrient_limited_delta

    def counted_delta(env_, fg_id, fg, rate, capacity):
        before = float(fg.energy_reserve.sum())
        delta = original_delta(env_, fg_id, fg, rate, capacity)
        paid = float(delta.sum())
        flux["gross"] += paid
        # the pool tables' supply (one-layer column; inert with the layer)
        flux["new"] += float(fg.energy_reserve.sum()) + paid - before
        return delta

    population_change._nutrient_limited_delta = counted_delta
    if ex:
        original_exchange = population_change.apply_layer_exchange

        def counted_exchange(env_):
            before = sum(float(env_.fgs[f].energy_reserve.sum()) for f in (mixed, deep))
            original_exchange(env_)
            after = sum(float(env_.fgs[f].energy_reserve.sum()) for f in (mixed, deep))
            flux["new"] += after - before

        population_change.apply_layer_exchange = counted_exchange

    try:
        return _collect(args, env, provider, mixed, deep, cells, hours, tpd, tpy, flux)
    finally:
        population_change._nutrient_limited_delta = original_delta
        if ex:
            population_change.apply_layer_exchange = original_exchange


def _collect(args, env, provider, mixed, deep, cells, hours, tpd, tpy, flux):
    from lib.environments.ecosystem_env import interactions
    from lib.world import daylight, nutrients
    keys = ("mixed", "subsurface", "surface_carbon", "nitrate", "nitrate_deep",
            "zooplankton", "microzooplankton")
    acc = {k: np.zeros(12) for k in keys}
    count = np.zeros(12)
    gross = new = 0.0
    for k in range((args.burn_in + args.years) * tpy):
        tick = interactions.light_index(env)
        month = daylight.month_of_day(int(tick / tpd))
        scoring = k >= args.burn_in * tpy
        if scoring:
            depth = float(nutrients._depth(env.nutrient_config, tick * hours + 0.5 * hours))
            top = float(env.fgs[mixed].biomass.sum()) / cells
            acc["mixed"][month] += top
            acc["surface_carbon"][month] += top / depth * C_WW * 1000.0
            acc["nitrate"][month] += float(nutrients.concentration(
                env.nutrient_config, env.nutrient_yield[mixed],
                float(env.fgs[mixed].energy_reserve.mean()), tick))
            if deep:
                acc["subsurface"][month] += float(env.fgs[deep].biomass.sum()) / cells
                acc["nitrate_deep"][month] += float(nutrients.concentration(
                    env.nutrient_config, env.nutrient_yield[deep],
                    float(env.fgs[deep].energy_reserve.mean()), tick, "subsurface"))
            for fid, key in (("zooplankton", "zooplankton"), ("microzooplankton", "microzooplankton")):
                if fid in env.fgs:
                    acc[key][month] += float(env.fgs[fid].biomass.sum()) / cells
            count[month] += 1
        flux["gross"] = flux["new"] = 0.0
        env.step(provider(env) if provider else env.calculate_decisions())
        if scoring:
            gross += flux["gross"]
            new += flux["new"]
    n = np.maximum(count, 1)
    years = max(args.years, 1)
    run = {k: (v / n).tolist() for k, v in acc.items()}
    run.update(seed=args.seed, years=args.years, burn_in=args.burn_in,
               checkpoints=args.checkpoints, behaviour=args.behaviour,
               project=args.project, library=args.library,
               pp=gross / cells / years * C_WW,
               f_ratio=(new / gross) if gross > 0 else float("nan"))
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run", help="run a world and score it")
    r.add_argument("checkpoints", nargs="?", default=None)
    r.add_argument("--behaviour", default=None, help="a viability arm instead of the policies")
    r.add_argument("--project", default="mareld2.yaml")
    r.add_argument("--library", default=None)
    r.add_argument("--seed", type=int, default=20260530)
    r.add_argument("--years", type=int, default=3)
    r.add_argument("--burn-in", dest="burn_in", type=int, default=1)
    r.add_argument("--grid", type=int, nargs=2, default=(60, 60))
    r.add_argument("--currents", choices=("on", "off"), default="on")
    r.add_argument("--out", default=None)
    s = sub.add_parser("score", help="score saved runs")
    s.add_argument("files", nargs="+")
    args = parser.parse_args(argv)
    if args.mode == "run":
        if not args.checkpoints and not args.behaviour:
            parser.error("give CKPT_DIR or --behaviour")
        run = run_world(args)
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(run, fh)
        print(report(run, args.checkpoints or args.behaviour))
    else:
        for path in args.files:
            with open(path) as fh:
                print(report(json.load(fh), os.path.basename(path)))


if __name__ == "__main__":
    main()
