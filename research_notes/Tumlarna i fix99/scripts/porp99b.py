"""Porpoise budget under frozen policies (fix99): where the biomass goes.

Per 20-tick window: biomass, fill, occupied cells, share of B in cells with
fill below u, losses by cause (starvation split into catabolism and the
extinction sweep), edge out/in, intake per tonne by prey, action shares, and
the prey density (visible, accessible) in porpoise cells vs the grid mean.

    RUN=... LIB=... TICKS=1460 python3 porp99.py
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from inference import load_policies_and_stats  # noqa: E402
from lib.environments.ecosystem_env import movement, population_change  # noqa: E402
from lib.runners import training_progress as tp  # noqa: E402
from lib.world import daylight  # noqa: E402

RUN = os.environ['RUN']
LIB = os.environ['LIB']
TICKS = int(os.environ.get('TICKS', 1460))
WIN = int(os.environ.get('WIN', 20))
G = 'porpoises'
PREY = ['planktivorous_fish', 'benthivorous_fish', 'whiting', 'predatory_fish']

config = json.load(open(os.path.join(RUN, 'progress', 'config.json')))
config['library'] = LIB
seed = int(config['seed'])
with tp.evaluation_randomness(seed):
    env = tp.build_inference_env(config, seed)
with tp.evaluation_randomness(seed):
    policies, mean, var = load_policies_and_stats(env, RUN, verbose=False)
    env.policies = policies
    env.rebuild_batched_weights()
    env.obs_mean, env.obs_var = mean, var
    env.softmax_temperature = float(config.get('temperature', 1.0))

    i = env.dm_ids.index(G)
    fg = env.fgs[G]
    ARM = os.environ.get('ARM', 'base')
    if ARM == 'closed':
        from lib.environments.ecosystem_env import grid_masks
        env.dm_move_mask[i] = grid_masks.build_movement_mask(env.grid, False, env.dtype)
    if ARM == 'nounits':
        env.unit_splits = False
    print('ARM', ARM)
    flag = {'mv': False}
    _apply_mv = movement.apply_movement

    def mv_hook(e, settlement):
        before = np.asarray(e.fgs[G].biomass, dtype=np.float64).copy()
        flag['mv'] = True
        try:
            out = _apply_mv(e, settlement)
        finally:
            flag['mv'] = False
        after = np.asarray(e.fgs[G].biomass, dtype=np.float64)
        acc['moved'] += 0.5 * np.abs(after - before).sum()
        return out
    movement.apply_movement = mv_hook
    u = float(fg.maintenance_level)
    me = float(fg.max_energy_reserve)
    acc = {'moved': 0.0, 'sweep': 0.0, 'edge_out': 0.0, 'rest': 0.0, 'move': 0.0, 'eat': 0.0, 'n': 0}

    _edge = movement._edge_emigration

    def edge_hook(b_out, r_out):
        b, r = _edge(b_out, r_out)
        if flag['mv']:
            acc['edge_out'] += float(b[i])
        return b, r
    movement._edge_emigration = edge_hook

    _sweep = population_change.apply_extinction_threshold

    def sweep_hook(e):
        before = float(np.asarray(e.fgs[G].biomass, dtype=np.float64).sum())
        _sweep(e)
        acc['sweep'] += before - float(np.asarray(e.fgs[G].biomass, dtype=np.float64).sum())
    population_change.apply_extinction_threshold = sweep_hook

    _costs = movement.apply_energy_costs

    def cost_hook(e, actions):
        b = np.asarray(e.fgs[G].biomass, dtype=np.float64)
        rest = float((b * actions.rest[i]).sum())
        eat = float((b[None] * actions.eat[i]).sum())
        move = float((b * actions.move[i].sum(axis=0)).sum())
        tot = max(rest + eat + move, 1e-30)
        acc['rest'] += rest / tot
        acc['eat'] += eat / tot
        acc['move'] += move / tot
        acc['n'] += 1
        return _costs(e, actions)
    movement.apply_energy_costs = cost_hook

    def book(name):
        return float((getattr(env, name, None) or {}).get(G, 0.0))

    def intake():
        return dict(env.intake_by_pred_prey.get(G, {}))

    H, W = fg.biomass.shape
    prev = {k: book(k) for k in ('loss_starvation', 'loss_natural', 'loss_impact', 'loss_predation')}
    prev_int = intake()
    B0 = float(fg.biomass.sum())
    print(f"u {u} ME {me} unit {fg.min_split_biomass} t  B0 {B0:.3f} t  grid {H}x{W}")
    print(" tick mon      B  fill  cells lowF  starv  sweep  M1    edge   rest  eat  move  ration/yr | prey dens in porp cells / grid mean")
    totals = {'starv': 0.0, 'sweep': 0.0, 'm1': 0.0, 'edge': 0.0}
    for t in range(1, TICKS + 1):
        env.step()
        if t % WIN and t != TICKS:
            continue
        b = np.asarray(fg.biomass, dtype=np.float64)
        r = np.asarray(fg.energy_reserve, dtype=np.float64)
        B = b.sum()
        occ = b > 0
        fill_cell = np.where(occ, r / np.maximum(b * me, 1e-30), 0.0)
        low = float(b[occ & (fill_cell < u)].sum() / max(B, 1e-30))
        h = np.clip((1 - fill_cell) / (1 - u), 0, 1)
        hB = float((h * b).sum() / max(B, 1e-30))
        now = {k: book(k) for k in prev}
        d = {k: now[k] - prev[k] for k in now}
        prev = now
        sweep = acc['sweep']
        starv = d['loss_starvation'] - sweep
        it = intake()
        dint = sum(v - prev_int.get(p, 0.0) for p, v in it.items())
        by_prey = {p: v - prev_int.get(p, 0.0) for p, v in it.items()}
        prev_int = it
        ration = dint / max(B, 1e-30) * (1460 / WIN)
        dens = []
        for p in PREY:
            pb = np.asarray(env.fgs[p].biomass, dtype=np.float64)
            inside = (pb * b).sum() / max(B, 1e-30)
            dens.append(f"{p[:5]} {inside / max(pb.mean(), 1e-30):4.2f}")
        n = max(acc['n'], 1)
        m = daylight.calendar_at(env.daylight, t)['month_name']
        top = ' '.join(f"{p[:5]}:{v / max(dint, 1e-30):.2f}" for p, v in sorted(by_prey.items(), key=lambda x: -x[1])[:3] if v > 0)
        print(f"{t:5d} {m:4s} {B:6.3f} {r.sum() / max(B * me, 1e-30):5.3f} {int(occ.sum()):5d} {low:4.2f} "
              f"{starv:6.4f} {sweep:6.4f} {d['loss_natural']:6.4f} {acc['edge_out']:6.3f} "
              f"{acc['rest'] / n:5.2f} {acc['eat'] / n:4.2f} {acc['move'] / n:4.2f} {ration:6.1f} h {hB:4.2f} r/(51h) {ration / max(51.1 * hB, 1e-9):4.2f} mvd/B {acc['moved'] / max(B, 1e-30) / WIN * 1460:6.1f} | {' '.join(dens)} | {top}",
              flush=True)
        totals['starv'] += starv
        totals['sweep'] += sweep
        totals['m1'] += d['loss_natural']
        totals['edge'] += acc['edge_out']
        acc.update(moved=0.0, sweep=0.0, edge_out=0.0, rest=0.0, eat=0.0, move=0.0, n=0)
        if B < 1e-4:
            break
    print('totals (t):', {k: round(v, 3) for k, v in totals.items()}, 'B end', round(float(fg.biomass.sum()), 3))
