"""Wall probe: do frozen policies leave a grid corner they start in?

Section 308.5. A decision maker whose edge moves are masked (section 266.16:
every DM with its own outside exchange, today the five fish groups) meets
the grid edge as a wall. Training worlds start the groups away from it and
last one year, so a policy may never learn to turn there: fix110's
benthivores drifted west over years 2-4 of an unfished ten-year run and
stayed in the south-west corner (cell (58, 0) up to 1500 t).

The probe builds the inference world (``inference.build_env``, the run's
warm-up, the frozen checkpoints), moves ONE group's whole biomass into a
band around one corner (weight exp(-d / lambda), d the distance in cells
to the corner point; each cell keeps the group's mean reserve fill and
juvenile share), runs ``--days`` on the tensor engine and reports how the
group spreads: the share in the outer ring of cells against the ring's
area share, the share within ``--radius`` cells of the start corner, the
centre of mass's distance from the corner, the largest cell and Lloyd's
crowding. A row "none" is the same world without the move (the group's
settled layout), the reference. A diagnostic of a training artefact, not a
calibration target.

    python3 tools/probes/wall_probe.py results/<run> [--days 180] \\
        [--groups benthivorous_fish ...] [--corners sw nw ...] \\
        [--impact-scale bottom_trawling=0 --impact-scale pelagic_trawling=0]
"""
import argparse
import copy
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.environments.ecosystem_env.constants import EAST, NORTH, SOUTH, WEST  # noqa: E402

CORNERS = ("nw", "ne", "sw", "se")


def corner_point(corner, H, W):
    """(row, col) of a corner's cell centre; row 0 is north."""
    return ((0 if corner[0] == "n" else H - 1), (0 if corner[1] == "w" else W - 1))


def corner_weights(corner, H, W, length):
    """exp(-d / length) around the corner cell, summing to 1."""
    r0, c0 = corner_point(corner, H, W)
    rr, cc = np.mgrid[0:H, 0:W]
    d = np.hypot(rr - r0, cc - c0)
    w = np.exp(-d / float(length))
    return w / w.sum()


def closed_edge_groups(env):
    """Decision makers whose every edge move is masked."""
    mask = getattr(env, "dm_move_mask", None)
    if mask is None:
        return []
    out = []
    for i, fid in enumerate(env.dm_ids):
        m = mask[i]
        if (m[NORTH, 0, :].max() == 0 and m[SOUTH, -1, :].max() == 0
                and m[WEST, :, 0].max() == 0 and m[EAST, :, -1].max() == 0):
            out.append(fid)
    return out


def relocate(env, fid, weights):
    """Move ``fid``'s whole biomass onto ``weights`` (sum 1), keeping its
    total, its mean reserve fill (reserve per tonne) and its juvenile share."""
    fg = env.fgs[fid]
    total = float(fg.biomass.sum())
    if total <= 0.0:
        return
    allowed = env.grid.get_map("accessibility")
    w = weights if allowed is None else weights * (np.asarray(allowed) > 0)
    w = w / w.sum()
    per_tonne = float(fg.energy_reserve.sum()) / total
    new = (w * total).astype(fg.biomass.dtype)
    fg.energy_reserve = (new * per_tonne).astype(fg.energy_reserve.dtype)
    juv = getattr(fg, "juvenile", None)
    if juv is not None:
        fg.juvenile = (new * (float(juv.sum()) / total)).astype(juv.dtype)
    fg.biomass = new


def spread_metrics(b, corner, radius):
    """(ring share / ring area share, share within radius of the corner,
    centre-of-mass distance from the corner in cells, max cell, Lloyd)."""
    H, W = b.shape
    total = float(b.sum())
    if total <= 0.0:
        return (np.nan,) * 5
    ring = np.zeros((H, W), dtype=bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    rr, cc = np.mgrid[0:H, 0:W]
    r0, c0 = corner_point(corner, H, W)
    near = np.hypot(rr - r0, cc - c0) <= radius
    com = (float((rr * b).sum()) / total, float((cc * b).sum()) / total)
    return (float(b[ring].sum()) / total / ring.mean(),
            float(b[near].sum()) / total,
            float(np.hypot(com[0] - r0, com[1] - c0)),
            float(b.max()),
            float(b.size * np.square(b).sum() / total ** 2))


def run_case(env, fid, days, sample_days, policies, mean, var, device, tick_hours):
    """Run ``env`` on the tensor engine; {day: fid's biomass map}."""
    from lib.gpu.inference import build_runner
    per_day = int(round(24 / tick_hours))
    n_ticks = days * per_day
    model, runner = build_runner(env, policies, mean, var, n_ticks, device=device)
    g = model.ids.index(fid)
    shape = (runner.E, model.G, model.H, model.W)
    maps = {0: env.fgs[fid].biomass.astype(np.float64).copy()}
    done = 0
    try:
        for day in sorted(sample_days):
            if day <= 0:
                continue
            runner.run(day * per_day - done)
            done = day * per_day
            maps[day] = runner.biomass.reshape(shape)[0, g].double().cpu().numpy()
    finally:
        runner.close()
    return maps


def parse_scales(items):
    scales = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if not key or not value:
            raise SystemExit(f"--impact-scale wants ID=FACTOR, got {item!r}")
        scales[key.strip()] = float(value)
    return scales


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", help="results/<run> (policy_*.pth, gpu_run.json)")
    ap.add_argument("--project", default="mareld2.yaml")
    ap.add_argument("--grid", default="60*60")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--groups", nargs="*", default=None,
                    help="default: every decision maker with closed edges")
    ap.add_argument("--corners", nargs="*", default=list(CORNERS), choices=CORNERS)
    ap.add_argument("--length", type=float, default=3.0,
                    help="decay length of the corner band in cells (default 3)")
    ap.add_argument("--radius", type=float, default=10.0,
                    help="radius in cells of the 'near corner' share (default 10)")
    ap.add_argument("--impact-scale", action="append", default=[],
                    help="ID=FACTOR on an inference impact map (e.g. bottom_trawling=0)")
    ap.add_argument("--warmup-days", type=float, default=None,
                    help="default: the run's (gpu_run.json)")
    ap.add_argument("--device", default="auto")
    from lib.environments.ecosystem_env.currents import add_current_arguments, current_options
    add_current_arguments(ap)
    args = ap.parse_args(argv)

    import inference
    from lib.runners.warmup import settle, warmup_ticks
    from lib.world.tick_time import LIBRARY_TICK_HOURS

    H, W = (int(x) for x in args.grid.lower().replace("x", "*").split("*"))
    tick_hours = LIBRARY_TICK_HOURS
    env = inference.build_env(args.project, (H, W), seed=args.seed, verbose=False,
                              apply_natural_mortality=True, migration=True,
                              mass_balance=True, tick_hours=tick_hours,
                              currents=current_options(args))
    policies, mean, var = inference.load_policies_and_stats(env, args.checkpoints, verbose=False)
    base_maps = {iid: np.array(env.grid.get_map(iid), copy=True)
                 for iid in inference._load_inference_map_paths(args.project)
                 if env.grid.get_map(iid) is not None}
    scales = parse_scales(args.impact_scale)
    unknown = sorted(set(scales) - set(base_maps))
    if unknown:
        raise SystemExit(f"--impact-scale: no inference map for {unknown}; "
                         f"maps: {sorted(base_maps)}")
    inference.apply_impact_scales(env, base_maps, scales)
    days = (inference.run_warmup_days(args.checkpoints) if args.warmup_days is None
            else args.warmup_days)
    settle(env, warmup_ticks(days, tick_hours), seed=args.seed, policies=policies,
           obs=(mean, var), engine="gpu", device=args.device)

    groups = args.groups or closed_edge_groups(env)
    samples = sorted(d for d in {0, 30, 90, args.days} if d <= args.days)
    print(f"wall probe: {args.checkpoints}, {H}x{W}, seed {args.seed}, warm-up {days:g} d, "
          f"{args.days} d, band length {args.length:g} cells, impact scales {scales or 'none'}")
    print("ring = share in the outer ring / its area share (1 = even); near = share within "
          f"{args.radius:g} cells of the corner; dist = centre of mass from the corner (cells)")
    for fid in groups:
        print(f"\n{fid}")
        print(f"  {'start':6s} " + "  ".join(f"{'day ' + str(d):>34s}" for d in samples))
        print(f"  {'':6s} " + "  ".join(f"{'ring  near  dist   max t  Lloyd':>34s}" for _ in samples))
        for corner in ["none"] + list(args.corners):
            case = copy.deepcopy(env)
            if corner != "none":
                relocate(case, fid, corner_weights(corner, H, W, args.length))
            maps = run_case(case, fid, args.days, samples, policies, mean, var,
                            args.device, tick_hours)
            ref = corner if corner != "none" else "sw"
            cells = []
            for d in samples:
                ring, near, dist, mx, lloyd = spread_metrics(maps[d], ref, args.radius)
                cells.append(f"{ring:5.2f} {near:5.2f} {dist:5.1f} {mx:7.1f} {lloyd:6.1f}")
            print(f"  {corner:6s} " + "  ".join(f"{c:>34s}" for c in cells), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
