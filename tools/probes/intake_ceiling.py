"""What limits each DM's intake, and the production it could reach at all.

    python3 tools/probes/intake_ceiling.py CKPT_DIR [YEARS] [--scale FG=K ...]

Frozen policies, probe world (60x60, mortality, migration, currents
0.1/20/0/12, daylight + temperature). Per DM and calendar month,
biomass-weighted over the DM's cells:

  h      hunger gate (1-s)/(1-u); demand is B * eat * f(B_prey) * h
  sat    eat-weighted saturation f(B_vis) * handling of the pairs eaten
         from (1 = at the Holling ceiling 1/h_t)
  cap    realised / demanded intake (< 1: the prey harvest cap binds)
  gain   energy gained, MJ per t per day
  C1     gain at full appetite (gain / h), same eat mix and saturation
  cost   action costs (temperature-scaled), MJ per t per day
  Pmax   the most production the energy allows: growth g (s - u) is paid
         from the reserve (energy_content per t) and dilutes it (s ME
         per t); as g grows s falls towards u, h towards 1, so
         Pmax = (C1 - cost) / (energy_content + u ME), as /yr
  prod   realised growth term, /yr; budget = g (1 - u), /yr

--scale FG=K multiplies FG's growth_rate by K in memory (resume 160).
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import train as train_mod
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import (interactions, movement,
                                            population_change, predation)
from lib.environments.ecosystem_env.currents import CurrentConfig

ap = argparse.ArgumentParser()
ap.add_argument('ckpt')
ap.add_argument('years', type=int, nargs='?', default=2)
ap.add_argument('--scale', action='append', default=[])
args = ap.parse_args()

env = train_mod._ProbeEnvBuilder(
    project_path='mareld2.yaml', grid_size=(60, 60),
    apply_natural_mortality=True, migration=True,
    currents=CurrentConfig(0.1, 20, 0, 12.0))()
env.policies, env.obs_mean, env.obs_var = load_policies_and_stats(
    env, args.ckpt, verbose=False)
env._batched_ready = False
for attr in ('_setup_batched_policies', '_prepare_batched_policies',
             '_build_batched_policies'):
    fn = getattr(env, attr, None)
    if callable(fn):
        fn()
        break
for item in args.scale:
    fid, k = item.split('=')
    env.fgs[fid].growth_rate *= float(k)
    print(f"growth_rate of {fid} x {float(k)} -> {env.fgs[fid].growth_rate:.6f}/tick")
np.random.seed(20260530)
torch.manual_seed(20260530)

TPD, TPY = 4, 1460
DM = list(env.dm_ids)
ND = len(DM)
U = np.array([env.fgs[f].maintenance_level for f in DM])
G = np.array([env.fgs[f].growth_rate for f in DM])
ME = np.array([env.fgs[f].max_energy_reserve for f in DM])
EC = np.array([float(env.fgs[f].params.get('energy_content', 0.0) or 0.0) for f in DM])
KEYS = ('B', 'h', 'sat', 'sat_w', 'cap_real', 'cap_dem', 'gain', 'gainh', 'cost',
        'prod', 's')
acc = {}
cap = {}


def bucket(key):
    return acc.setdefault(key, {k: np.zeros(ND) for k in KEYS} | {'n': 0})


def aeff_hook(a, h, Bp, m3, interference=0.0):
    f = _orig_aeff(a, h, Bp, m3, interference)
    cap['f'] = f
    cap['Bp'] = np.broadcast_to(Bp, f.shape)
    return f


def predation_hook(env_, actions):
    b = cap['b']
    B = np.stack([env_.fgs[f].biomass for f in DM]).astype(np.float64)
    h = np.stack([env_.fgs[f].get_hunger() for f in DM]).astype(np.float64)
    before = {p: dict(env_.intake_by_pred_prey.get(p, {})) for p in DM}
    cap.pop('f', None)
    _orig_pred(env_, actions)
    gains = np.stack([env_.fgs[f].temp_energy_gains for f in DM]).astype(np.float64)
    tot = B.sum(axis=(1, 2))
    ok = tot > 0
    b['B'] += tot
    b['h'] += np.where(ok, (B * h).sum(axis=(1, 2)) / np.maximum(tot, 1e-30), 0)
    b['gain'] += np.where(ok, gains.sum(axis=(1, 2)) / np.maximum(tot, 1e-30), 0)
    # gain at full appetite: gains are linear in h unless the cap binds
    b['gainh'] += np.where(ok, (B * h).sum(axis=(1, 2)), 0)
    if 'f' in cap:
        f = cap['f'].astype(np.float64)                     # (ND, NA, H, W)
        ht = env_.handling_time_mat[:, :, None, None].astype(np.float64)
        eat = actions.eat.astype(np.float64)                # (ND, NA, H, W)
        sat = np.minimum(f * ht, 1.0)
        w = B[:, None] * eat * (ht > 0)
        b['sat'] += (w * sat).sum(axis=(1, 2, 3))
        b['sat_w'] += w.sum(axis=(1, 2, 3))
        demand = B[:, None] * eat * f * h[:, None]
        b['cap_dem'] += demand.sum(axis=(1, 2, 3))
        for i, p in enumerate(DM):
            after = env_.intake_by_pred_prey.get(p, {})
            b['cap_real'][i] += sum(v - before[p].get(k, 0.0) for k, v in after.items())


def costs_hook(env_, actions):
    b = cap['b']
    B = np.stack([env_.fgs[f].biomass for f in DM]).astype(np.float64)
    tot = B.sum(axis=(1, 2))
    T = interactions.metabolism_temperature(env_)
    m = env_.dm_resting_metabolism.astype(np.float64) * (T if T is not None else 1.0)
    c = (actions.rest * env_.dm_cost_rest[:, None, None]
         + actions.eat.sum(axis=1) * env_.dm_cost_eat[:, None, None]
         + actions.move.sum(axis=1) * env_.dm_cost_move[:, None, None])
    cost = (B * c).sum(axis=(1, 2)) * m
    b['cost'] += np.where(tot > 0, cost / np.maximum(tot, 1e-30), 0)
    s = np.stack([env_.fgs[f].energy_level for f in DM]).astype(np.float64)
    b['s'] += np.where(tot > 0, (B * s).sum(axis=(1, 2)) / np.maximum(tot, 1e-30), 0)
    return _orig_costs(env_, actions)


def growth_hook(env_, fg_id, fg_):
    if fg_id not in DM:
        return _orig_growth(env_, fg_id, fg_)
    i = DM.index(fg_id)
    B0 = float(fg_.biomass.sum())
    L0 = env_.loss_natural.get(fg_id, 0.0)
    _orig_growth(env_, fg_id, fg_)
    if B0 > 0:
        m1 = env_.loss_natural.get(fg_id, 0.0) - L0
        cap['b']['prod'][i] += (float(fg_.biomass.sum()) - B0 + m1) / B0


_orig_aeff = interactions.holling_a_eff
_orig_pred = predation.apply_predation
_orig_costs = movement.apply_energy_costs
_orig_growth = population_change._apply_decision_maker_population_change
interactions.holling_a_eff = aeff_hook
predation.apply_predation = predation_hook
movement.apply_energy_costs = costs_hook
population_change._apply_decision_maker_population_change = growth_hook

for t in range(args.years * TPY):
    month = min(11, int(interactions.light_index(env) / TPD / (365.0 / 12)))
    cap['b'] = bucket((t // TPY, month))
    cap['b']['n'] += 1
    env.step()

MON = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()
den = EC + U * ME
for i, fid in enumerate(DM):
    print(f"\n=== {fid}: u {U[i]}, ME {ME[i]:.0f}, energy_content {EC[i]:.0f} MJ/t "
          f"(Pmax denominator {den[i]:.0f}), budget g(1-u) {G[i]*(1-U[i])*TPY:.3f}/yr")
    for year in range(args.years):
        print(f"  year {year+1}: mon    B      s     h    sat   cap   gain    C1   cost  "
              "C1-cost   Pmax   prod")
        rows = []
        for mo in range(12):
            b = acc.get((year, mo))
            if not b or b['n'] == 0:
                continue
            n = b['n']
            hbar = b['h'][i] / n
            gain = b['gain'][i] / n * TPD
            c1 = gain / max(hbar, 1e-6)
            cost = b['cost'][i] / n * TPD
            pmax = (c1 - cost) / TPD * TPY / den[i] if den[i] > 0 else float('nan')
            prod = b['prod'][i] / n * TPY
            sat = b['sat'][i] / max(b['sat_w'][i], 1e-30)
            capr = b['cap_real'][i] / max(b['cap_dem'][i], 1e-30)
            rows.append((b['n'], pmax, prod, gain, c1, cost))
            print(f"           {MON[mo]}  {b['B'][i]/n:7.1f}  {b['s'][i]/n:.3f} {hbar:.2f}  {sat:.2f}  {capr:.2f}  "
                  f"{gain:6.1f} {c1:6.1f} {cost:6.1f}  {c1-cost:6.1f}  {pmax:6.2f}  {prod:6.2f}")
        n = sum(r[0] for r in rows)
        pm = sum(r[0] * r[1] for r in rows) / n
        pr = sum(r[0] * r[2] for r in rows) / n
        print(f"           year: Pmax {pm:.2f}/yr, realised {pr:.2f}/yr, budget "
              f"{G[i]*(1-U[i])*TPY:.2f}/yr, M1 {env.fgs[fid].natural_mortality*TPY:.2f}/yr")
