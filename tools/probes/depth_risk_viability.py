"""Does exposure-weighted M1 make diel vertical migration pay? Section 139.

Frozen-schedule experiment for zooplankton on top of trained policies:
every other decision maker runs the checkpoint, zooplankton's hiding is
replaced by a fixed schedule, and the M1 parameters are swept.

Schedules (zooplankton only; the rest of the action mass eats
phytoplankton, nothing moves). "Day" is 06-18 local solar time, i.e.
two of four 6 h ticks, whatever the season:
  trained  the checkpoint's own actions (rests ~0.8 around the clock)
  flat80   rest 0.8 in every tick
  dvm80    rest 1.0 by day, 0.6 at night      (mean 0.8, as flat80)
  flat50   rest 0.5 in every tick
  dvm50    rest 1.0 by day, 0.0 at night      (mean 0.5, as flat50)

flat vs dvm at the same mean isolates WHEN hiding happens (the visual
part of M1 and herring follow the light); 80 vs 50 isolates HOW MUCH
(the tactile part is ``depth_risk_ratio`` times stronger on hidden
biomass and linear in the hidden fraction, so only the amount matters
for it). The hypothesis: with the split M1 and rho >= 2, dvm50 beats
flat80 and the trained 0.8, and herring's realised predation moves
towards the 5.24 /yr of the growth budget.

    python3 tools/probes/depth_risk_viability.py --run-name <run> \
        --rho 1 2 4 --seeds 2 --jobs 8

``--floors`` sweeps herring's visibility_floor on zooplankton instead
(constant M1, section 140):

    python3 tools/probes/depth_risk_viability.py --run-name <run> \
        --rho --schedules trained --floors 0.2 0.4 0.6 0.8 --seeds 3
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

SCHEDULES = ("trained", "flat80", "dvm80", "flat50", "dvm50")
#: (day, night) hidden fraction per schedule.
HIDE = {"flat80": (0.8, 0.8), "dvm80": (1.0, 0.6),
        "flat50": (0.5, 0.5), "dvm50": (1.0, 0.0)}
ZOO = "zooplankton"
TICKS_PER_YEAR = 1460


def _schedule(env, actions, schedule, phyto_col):
    """Overwrite zooplankton's action mass according to ``schedule``."""
    from lib.environments.ecosystem_env import interactions
    i = env.dm_ids.index(ZOO)
    move, rest, eat = actions.move, actions.rest, actions.eat
    if schedule == "trained":
        return actions
    move[i] = 0.0
    eat[i] = 0.0
    present = env.fgs["phytoplankton"].biomass > 0
    hours = int(env.daylight["tick_hours"])
    hour = (interactions.light_index(env) * hours) % 24
    day, night = HIDE[schedule]
    hide = np.full_like(rest[i], day if 6 <= hour < 18 else night)
    hide = np.where(present, hide, 1.0).astype(rest.dtype)
    rest[i] = hide
    eat[i, phyto_col] = 1.0 - hide
    return actions


def run_one(job):
    from inference import build_env, load_policies_and_stats
    from lib.environments.ecosystem_env.currents import CurrentConfig

    np.random.seed(job["seed"])
    env = build_env(job["project"], (job["grid"], job["grid"]),
                    seed=job["seed"], verbose=False,
                    apply_natural_mortality=True, migration=True,
                    currents=CurrentConfig(0.1, 20, 0, 12.0),
                    mass_balance=True)
    if job.get("floor") is not None:
        pair = env.fgs["pelagic_fish"].params["interaction"][
            "pelagic_fish_preys_on_zooplankton"]
        pair["visibility_floor"] = float(job["floor"])
    params = env.fgs[ZOO].params
    if job["rho"] is not None:
        params["m1_visual_share"] = job["visual"]
        params["m1_tactile_share"] = job["tactile"]
        params["depth_risk_ratio"] = job["rho"]
    env.build_static_caches()
    policies, mean, var = load_policies_and_stats(env, job["ckpt"],
                                                  verbose=False)
    env.policies = dict(policies)
    env.obs_mean, env.obs_var = mean, var
    env.rebuild_batched_weights()
    phyto_col = env.global_fg_order.index("phytoplankton")

    zoo0 = float(env.fgs[ZOO].biomass.sum())
    her0 = float(env.fgs["pelagic_fish"].biomass.sum())
    zoo_series, her_series, rest_series = [], [], []
    for _ in range(job["ticks"]):
        actions = env.calculate_decisions()
        actions = _schedule(env, actions, job["schedule"], phyto_col)
        env.step(actions)
        zoo_series.append(float(env.fgs[ZOO].biomass.sum()))
        her_series.append(float(env.fgs["pelagic_fish"].biomass.sum()))
        i = env.dm_ids.index(ZOO)
        b = env.fgs[ZOO].biomass
        rest_series.append(float((env.pi_rest[i] * b).sum()
                                 / max(b.sum(), 1e-12)))
    zoo = np.array(zoo_series)
    years = job["ticks"] / TICKS_PER_YEAR
    mean_zoo = float(zoo.mean())
    parts = env.loss_natural_parts.get(ZOO, {})
    eaten = float(env.intake_by_pred_prey.get("pelagic_fish", {}).get(ZOO,
                                                                      0.0))
    her = np.array(her_series)
    her_mean = float(her.mean())
    on_herring = sum(float(row.get("pelagic_fish", 0.0))
                     for row in env.intake_by_pred_prey.values())
    return dict(job, zoo_end=zoo[-1] / zoo0, zoo_mean=mean_zoo / zoo0,
                zoo_min=float(zoo.min()) / zoo0,
                herring_end=her_series[-1] / her0,
                rest=float(np.mean(rest_series)),
                m1=env.loss_natural.get(ZOO, 0.0) / mean_zoo / years,
                m1_visual=parts.get("visual", 0.0) / mean_zoo / years,
                m1_tactile=parts.get("tactile", 0.0) / mean_zoo / years,
                m2_herring=eaten / mean_zoo / years,
                herring_mean=her_mean / her0,
                herring_min=float(her.min()) / her0,
                herring_q=eaten / max(her_mean, 1e-9) / years,
                herring_m2=on_herring / max(her_mean, 1e-9) / years,
                herring_starve=env.loss_starvation.get("pelagic_fish", 0.0)
                / max(her_mean, 1e-9) / years)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", default="mareld2.yaml")
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=TICKS_PER_YEAR)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed0", type=int, default=20260530)
    parser.add_argument("--visual", type=float, default=2.0 / 19.7,
                        help="share of M1 that is visual (default 2/19.7)")
    parser.add_argument("--tactile", type=float, default=10.0 / 19.7,
                        help="share of M1 that is tactile (default 10/19.7)")
    parser.add_argument("--rho", type=float, nargs="*", default=[1.0, 2.0, 4.0])
    parser.add_argument("--floors", type=float, nargs="*", default=[],
                        help="herring -> zooplankton visibility_floor values "
                             "to sweep (constant M1); empty = library value")
    parser.add_argument("--schedules", nargs="+", default=list(SCHEDULES))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    ckpt = os.path.join("results", args.run_name)
    variants = [None] + list(args.rho)      # None = constant M1 (today)
    floors = list(args.floors) or [None]
    jobs = [dict(project=args.project, grid=args.grid, ticks=args.ticks,
                 ckpt=ckpt, seed=args.seed0 + s, rho=rho, floor=floor,
                 visual=args.visual, tactile=args.tactile, schedule=sched)
            for floor, rho, sched, s in itertools.product(
                floors, variants, args.schedules, range(args.seeds))]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run_one, jobs))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)

    print(f"{args.ticks} ticks, {args.grid}x{args.grid}, {args.seeds} seeds, "
          f"visual share {args.visual:.3f}, tactile share {args.tactile:.3f}")
    if not args.floors:
        print(f"{'M1':>10} {'schedule':>8} | {'rest':>5} {'zoo end':>7} "
              f"{'zoo mean':>8} {'zoo min':>7} | {'M1/yr':>6} {'vis':>5} "
              f"{'tact':>5} {'M2 her':>6} | {'her end':>7}")
    if args.floors:
        print(f"{'floor':>5} {'schedule':>8} | {'rest':>5} {'zoo mean':>8} "
              f"{'zoo min':>7} | {'M2 her':>6} | {'her end':>7} {'her mean':>8} "
              f"{'her min':>7} {'Q/B':>5} {'M2 on':>5} {'starve':>6}")
        for floor in floors:
            for sched in args.schedules:
                rows = [r for r in results if r["floor"] == floor
                        and r["schedule"] == sched and r["rho"] is None]
                m = {k: float(np.mean([r[k] for r in rows])) for k in
                     ("rest", "zoo_mean", "zoo_min", "m2_herring",
                      "herring_end", "herring_mean", "herring_min",
                      "herring_q", "herring_m2", "herring_starve")}
                print(f"{floor:5.2f} {sched:>8} | {m['rest']:5.2f} "
                      f"{m['zoo_mean']:8.2f} {m['zoo_min']:7.3f} | "
                      f"{m['m2_herring']:6.2f} | {m['herring_end']:7.3f} "
                      f"{m['herring_mean']:8.2f} {m['herring_min']:7.3f} "
                      f"{m['herring_q']:5.1f} {m['herring_m2']:5.2f} "
                      f"{m['herring_starve']:6.2f}")
        return 0
    for rho in variants:
        for sched in args.schedules:
            rows = [r for r in results
                    if r["rho"] == rho and r["schedule"] == sched]
            m = {k: float(np.mean([r[k] for r in rows])) for k in
                 ("rest", "zoo_end", "zoo_mean", "zoo_min", "m1",
                  "m1_visual", "m1_tactile", "m2_herring", "herring_end")}
            label = "constant" if rho is None else f"rho {rho:g}"
            print(f"{label:>10} {sched:>8} | {m['rest']:5.2f} "
                  f"{m['zoo_end']:7.2f} {m['zoo_mean']:8.2f} "
                  f"{m['zoo_min']:7.3f} | {m['m1']:6.1f} {m['m1_visual']:5.1f} "
                  f"{m['m1_tactile']:5.1f} {m['m2_herring']:6.2f} | "
                  f"{m['herring_end']:7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
