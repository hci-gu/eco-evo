"""Can herring survive the winter by not feeding? Section 140.

Frozen-schedule test on top of trained policies: every decision maker
runs the checkpoint, only herring's actions are replaced by a rule.

Rules (herring only):
  trained   the checkpoint's own actions
  lowfoodX  rest fully in cells where zooplankton is below X t/cell,
            the checkpoint's actions elsewhere
  winter    rest 0.9 in December-February (calendar months), the
            checkpoint's actions otherwise
  hideSatT  rest a share S where gadoids exceed T t/cell (e.g.
            hide1at0.1, hide0.5at0.1), the checkpoint's actions elsewhere
            - hiding from the predator (gadoids -> herring floor 0.25)

Resting both saves energy (resting_cost instead of feeding_cost) and
hides from gadoids (visibility_floor 0.25). ``--gadoid-floor 1.0``
switches the hiding off for that pair, which separates the two.

    python3 tools/probes/herring_overwinter.py --run-name <run> --seeds 3
"""
import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

HERRING = "pelagic_fish"
ZOO = "zooplankton"
TICKS_PER_YEAR = 1460
WINTER_MONTHS = (11, 0, 1)        # Dec, Jan, Feb (0-based)
MONTH_START = np.cumsum((0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30))


def _month(env):
    from lib.environments.ecosystem_env import interactions
    hours = int(env.daylight["tick_hours"])
    day = (interactions.light_index(env) * hours) // 24
    return int(np.searchsorted(MONTH_START, day, side="right") - 1)


def _apply_rule(env, actions, rule):
    if rule == "trained":
        return actions
    i = env.dm_ids.index(HERRING)
    if rule.startswith("lowfood"):
        threshold = float(rule[len("lowfood"):])
        resting = env.fgs[ZOO].biomass < threshold
        share = np.where(resting, 1.0, 0.0).astype(actions.rest.dtype)
    elif rule.startswith("hide"):
        share_text, threshold_text = rule[len("hide"):].split("at")
        present = env.fgs["gadoids"].biomass > float(threshold_text)
        share = np.where(present, float(share_text), 0.0).astype(
            actions.rest.dtype)
    elif rule == "winter":
        share = np.full_like(actions.rest[i],
                             0.9 if _month(env) in WINTER_MONTHS else 0.0)
    else:
        raise ValueError(rule)
    keep = 1.0 - share
    actions.move[i] *= keep
    actions.eat[i] *= keep
    actions.rest[i] = actions.rest[i] * keep + share
    return actions


def run_one(job):
    from inference import build_env, load_policies_and_stats
    from lib.environments.ecosystem_env.currents import CurrentConfig

    np.random.seed(job["seed"])
    env = build_env(job["project"], (job["grid"], job["grid"]),
                    seed=job["seed"], verbose=False,
                    apply_natural_mortality=True, migration=True,
                    currents=CurrentConfig(0.1, 20, 0, 12.0),
                    mass_balance=True, library_path=job.get("library"))
    if job["gadoid_floor"] is not None:
        env.fgs["gadoids"].params["interaction"][
            "gadoids_preys_on_pelagic_fish"]["visibility_floor"] = float(
                job["gadoid_floor"])
    env.build_static_caches()
    policies, mean, var = load_policies_and_stats(env, job["ckpt"],
                                                  verbose=False)
    env.policies = dict(policies)
    env.obs_mean, env.obs_var = mean, var
    env.rebuild_batched_weights()

    i = env.dm_ids.index(HERRING)
    h0 = float(env.fgs[HERRING].biomass.sum())
    z0 = float(env.fgs[ZOO].biomass.sum())
    g0 = float(env.fgs["gadoids"].biomass.sum())
    series, zoo, rest, months, gad = [], [], [], [], []
    for _ in range(job["ticks"]):
        months.append(_month(env))
        actions = _apply_rule(env, env.calculate_decisions(), job["rule"])
        env.step(actions)
        b = env.fgs[HERRING].biomass
        series.append(float(b.sum()))
        zoo.append(float(env.fgs[ZOO].biomass.sum()))
        gad.append(float(env.fgs["gadoids"].biomass.sum()))
        rest.append(float((env.pi_rest[i] * b).sum() / max(b.sum(), 1e-12)))
    h = np.array(series)
    months = np.array(months)
    winter = np.isin(months, WINTER_MONTHS)
    # Winter loss: biomass at the end of the winter block over its start.
    # A run that starts in winter has two winter pieces; use the longest
    # contiguous one.
    idx = np.flatnonzero(winter)
    winter_ratio = float("nan")
    if idx.size:
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        block = max(runs, key=len)
        first = block[0] - 1 if block[0] > 0 else block[0]
        winter_ratio = h[block[-1]] / max(h[first], 1e-12)
    years = job["ticks"] / TICKS_PER_YEAR
    mean_h = max(float(h.mean()), 1e-12)
    eaten_by = sum(float(row.get(HERRING, 0.0))
                   for row in env.intake_by_pred_prey.values())
    gadoid_diet = env.intake_by_pred_prey.get("gadoids", {})
    gadoid_total = max(sum(float(v) for v in gadoid_diet.values()), 1e-12)
    gadoid_herring = float(gadoid_diet.get(HERRING, 0.0))
    return dict(job, end=h[-1] / h0, mean=float(h.mean()) / h0,
                min=float(h.min()) / h0, winter=winter_ratio,
                rest=float(np.mean(rest)),
                rest_winter=float(np.mean(np.array(rest)[winter]))
                if winter.any() else float("nan"),
                starve=env.loss_starvation.get(HERRING, 0.0) / mean_h / years,
                predation=eaten_by / mean_h / years,
                gadoid_m2=gadoid_herring / mean_h / years,
                gadoid_share=gadoid_herring / gadoid_total,
                gadoid_mean=float(np.mean(gad)) / g0,
                zoo_mean=float(np.mean(zoo)) / z0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", default="mareld2.yaml")
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=TICKS_PER_YEAR)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=20260530)
    parser.add_argument("--rules", nargs="+",
                        default=["trained", "lowfood2", "lowfood4", "winter"])
    parser.add_argument("--gadoid-floor", type=float, nargs="*",
                        default=[None, 1.0],
                        help="gadoids -> herring visibility_floor; the "
                             "library value is used for 'None'")
    parser.add_argument("--library", nargs="+", default=[None],
                        help="library variants to compare (default: the "
                             "project's fgconfig/fg_library.yaml)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    ckpt = (args.run_name if os.path.isdir(args.run_name)
            else os.path.join("results", args.run_name))
    floors = args.gadoid_floor or [None]
    jobs = [dict(project=args.project, grid=args.grid, ticks=args.ticks,
                 ckpt=ckpt, seed=args.seed0 + s, rule=rule, gadoid_floor=f,
                 library=lib)
            for lib, f, rule, s in itertools.product(
                args.library, floors, args.rules, range(args.seeds))]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run_one, jobs))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=1)

    print(f"{args.ticks} ticks, {args.grid}x{args.grid}, {args.seeds} seeds; "
          "herring biomass relative to spawn")
    print(f"{'library':>12} {'gad floor':>9} {'rule':>9} | {'rest':>5} "
          f"{'rest DJF':>8} | {'end':>6} {'mean':>5} {'min':>6} {'winter':>6} | "
          f"{'starve':>6} {'pred':>5} {'by gad':>6} | {'gad diet':>8} "
          f"{'gad mean':>8} | {'zoo mean':>8}")
    for lib, f, rule in itertools.product(args.library, floors, args.rules):
        rows = [r for r in results if r["library"] == lib
                and r["gadoid_floor"] == f and r["rule"] == rule]
        m = {k: float(np.nanmean([r[k] for r in rows])) for k in
             ("rest", "rest_winter", "end", "mean", "min", "winter",
              "starve", "predation", "gadoid_m2", "gadoid_share",
              "gadoid_mean", "zoo_mean")}
        label = "library" if f is None else f"{f:g}"
        name = "default" if lib is None else os.path.basename(lib)[:12]
        print(f"{name:>12} {label:>9} {rule:>9} | {m['rest']:5.2f} "
              f"{m['rest_winter']:8.2f} | {m['end']:6.3f} "
              f"{m['mean']:5.2f} {m['min']:6.3f} {m['winter']:6.2f} | "
              f"{m['starve']:6.2f} {m['predation']:5.2f} "
              f"{m['gadoid_m2']:6.2f} | {m['gadoid_share']:8.2f} "
              f"{m['gadoid_mean']:8.2f} | {m['zoo_mean']:8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
