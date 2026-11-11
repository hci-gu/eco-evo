"""Where does the krill the fish eat come from? The open pool's flows in the probe world.
Frozen policies, the run's probe world (unfished as in training), the run's grid, seed and currents, CPU, nothing
altered. The pool (open_pool.grow, after M1 and predation): B <- T + (B (1 + g) - T) exp(-k), T = outside density x
depth weight x season. Per month and bed-depth class (< 100 / 100-150 / 150-250 / >= 250 m), t/km2 per cell-year:
  - B / T (the pool against its outside target);
  - local production g B, the import and the export (env.gain_exchange / loss_exchange books);
  - predation by predator (env.intake_by_pred_prey), and the realised M2 (predation / mean B).
usage: RUN=<run> POL=<dir> CLS=<class npy> krill_supply.py <years>"""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
years = int(sys.argv[1]); S = os.path.dirname(os.path.abspath(__file__))
import numpy as np, torch
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import interactions, open_pool
from lib.world import daylight
from lib.world.tick_time import ticks_per_day
torch.set_num_threads(3)
cfg = json.load(open('results/' + os.environ['RUN'] + '/progress/config.json')); seed = cfg['seed']
keys = ("project", "library", "grid", "mortality", "migration", "mortality_multiplier", "mass_balance", "currents", "tick_hours")
eco = {k: cfg[k] for k in keys if k in cfg}
with evaluation_randomness(seed): env = build_inference_env(eco, seed)
with evaluation_randomness(seed):
    pol, om, ov = load_policies_and_stats(env, os.environ['POL'], verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = om, ov
    env.softmax_temperature = cfg.get('temperature', 1.0)
K = 'krill'; fg = env.fgs[K]
cls = np.load(os.environ['CLS']); NC = 4; masks = [cls == c for c in range(NC)]
ncell = np.array([m.sum() for m in masks], dtype=float)
g = float(fg.growth_rate); k = float(fg.open_pool['rate'])
preds = [p for p in env.dm_ids if K in (env.fgs[p].params.get('interaction', {}) or {}) or
         f"{p}_preys_on_{K}" in (env.fgs[p].params.get('interaction', {}) or {})]
print('krill g per yr', g * 1460, '| k per yr', k * 1460, '| predators', preds, flush=True)
pc = lambda x: np.array([np.asarray(x, dtype=float)[m].sum() for m in masks])
# Per-cell exchange: wrap open_pool.grow to book the flow by class.
o_grow = open_pool.grow
tick = {}
def p_grow(env_, fid, fg_, rate):
    if fid == K:
        before = fg_.biomass.astype(np.float64).copy()
        grown = before * (1.0 + float(rate))
        o_grow(env_, fid, fg_, rate)
        flow = fg_.biomass.astype(np.float64) - grown
        tick['prod'] = pc(before * float(rate)); tick['imp'] = pc(np.maximum(flow, 0)); tick['exp'] = pc(-np.minimum(flow, 0))
        tick['T'] = pc(np.broadcast_to(open_pool.target(env_, fid), fg_.biomass.shape))
    else:
        o_grow(env_, fid, fg_, rate)
open_pool.grow = p_grow
# Per-cell predation on krill: wrap nothing; use the per-pred book (grid total) and the per-cell loss by difference.
tpd = ticks_per_day(float(env.daylight['tick_hours'])); tpy = int(round(365 * tpd))
out = []
with evaluation_randomness(seed):
    for y in range(years):
        A = {n: np.zeros((12, NC)) for n in ('B', 'T', 'prod', 'imp', 'exp', 'pred')}
        P = {p: np.zeros(12) for p in preds}; n = np.zeros(12)
        for t in range(tpy):
            month = daylight.month_of_day(int(interactions.light_index(env) / tpd))
            b0 = pc(fg.biomass)
            book0 = {p: env.intake_by_pred_prey.get(p, {}).get(K, 0.0) for p in preds}
            tick.clear(); env.tick()
            for p in preds:
                P[p][month] += env.intake_by_pred_prey.get(p, {}).get(K, 0.0) - book0[p]
            b1 = pc(fg.biomass)
            # Predation by class = start - end + production + import - export (M1 0 for krill).
            A['pred'][month] += b0 - b1 + tick['prod'] + tick['imp'] - tick['exp']
            for key in ('prod', 'imp', 'exp', 'T'):
                A[key][month] += tick[key]
            A['B'][month] += b0; n[month] += 1
        rec = dict(year=y + 1, classes=['<100', '100-150', '150-250', '>=250'], ncell=ncell.tolist())
        rec['monthly'] = {key: (A[key] / ncell).tolist() for key in A}  # per cell, summed over the month's ticks
        rec['n_ticks'] = n.tolist()
        rec['pred_by_predator'] = {p: (P[p] / ncell.sum()).tolist() for p in preds}
        out.append(rec)
        # Annual summary (t/km2/yr per class and box)
        tot = {key: A[key].sum(axis=0) for key in A}
        Bm = tot['B'] / tpy; Tm = tot['T'] / tpy
        print(f"=== year {y+1}: per class / box (t/km2 per cell; flows per yr) ===")
        for i, c in enumerate(rec['classes'] + ['box']):
            sel = slice(None) if c == 'box' else i
            nc = ncell.sum() if c == 'box' else ncell[i]
            f = lambda key: (tot[key][sel].sum() if c == 'box' else tot[key][sel]) / nc
            Bv = (Bm.sum() if c == 'box' else Bm[i]) / nc; Tv = (Tm.sum() if c == 'box' else Tm[i]) / nc
            print(f"  {c:8s} B {Bv:5.2f} T {Tv:5.2f} B/T {Bv / Tv:4.2f} | prod {f('prod'):6.2f} import {f('imp'):6.2f} "
                  f"export {f('exp'):6.2f} predation {f('pred'):6.2f} | M2 {f('pred') / Bv:5.2f} /yr  "
                  f"import share of eaten {f('imp') / max(f('pred'), 1e-9):4.2f}", flush=True)
        print("  monthly box B/T: " + ' '.join(f"{A['B'][m].sum() / A['T'][m].sum():.2f}" for m in range(12)))
        print("  monthly box M2 (/yr): " + ' '.join(f"{A['pred'][m].sum() / max(A['B'][m].sum(), 1e-9) * tpd * 365:.1f}" for m in range(12)))
        print("  eaten by predator (t/km2/yr):", {p: round(float(P[p].sum() / ncell.sum()), 2) for p in preds}, flush=True)
os.makedirs(f"{S}/../tables", exist_ok=True)
json.dump(out, open(f"{S}/../tables/krill_supply_{os.environ['RUN']}.json", 'w'))
