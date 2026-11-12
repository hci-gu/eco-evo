"""Monthly energy ledger of the five fish groups under frozen policies.

Builds the long-run probe world of a run (its progress/config.json, with
LIB replacing the library), loads the run's policies and books per fish
group and month: biomass, reserve fill, action shares, intake (tonnes and
assimilated MJ), action costs (rest / eat / move), the reserve exchanged
with the outside, the reserve carried out by losses, and the residual
(growth energy + catabolism), all per tonne of mean biomass per year.

    RUN=results/pzbpgp248nsrpfix97 LIB=.../lib269.yaml YEARS=3 python3 ledger.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from inference import load_policies_and_stats  # noqa: E402
from lib.environments.ecosystem_env import interactions, movement, predation  # noqa: E402
from lib.runners import training_progress as tp  # noqa: E402
from lib.world import daylight  # noqa: E402

RUN = os.environ.get('RUN', 'results/pzbpgp248nsrpfix97')
LIB = os.environ['LIB']
YEARS = int(os.environ.get('YEARS', 3))
OUT = os.environ.get('OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ledger.json'))
FISH = ['planktivorous_fish', 'benthivorous_fish', 'whiting', 'predatory_fish', 'mackerel']

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

    idx = {f: env.dm_ids.index(f) for f in FISH}
    tick = {}

    _costs = movement.apply_energy_costs

    def booking_costs(e, actions):
        biomass = np.stack([e.fgs[f].biomass for f in e.dm_ids], axis=0).astype(np.float64)
        rm = e.dm_resting_metabolism.astype(np.float64).copy()
        tm = interactions.metabolism_temperature(e)
        if tm is not None:
            rm = rm * np.asarray(tm, dtype=np.float64)
        for f, i in idx.items():
            b = biomass[i]
            rest = float((b * actions.rest[i]).sum())
            eat = float((b * actions.eat[i].sum(axis=0)).sum())
            move = float((b * actions.move[i].sum(axis=0)).sum())
            tot = max(rest + eat + move, 1e-30)
            tick[f]['c_rest'] = rm[i] * float(e.dm_cost_rest[i]) * rest
            tick[f]['c_eat'] = rm[i] * float(e.dm_cost_eat[i]) * eat
            tick[f]['c_move'] = rm[i] * float(e.dm_cost_move[i]) * move
            tick[f]['sh_eat'] = eat / tot
            tick[f]['sh_move'] = move / tot
            tick[f]['sh_rest'] = rest / tot
            tick[f]['gain'] = float(np.asarray(e.fgs[f].temp_energy_gains, dtype=np.float64).sum())
        return _costs(e, actions)

    movement.apply_energy_costs = booking_costs

    def snap():
        out = {}
        for f in FISH:
            fg = env.fgs[f]
            out[f] = dict(
                B=float(np.asarray(fg.biomass, dtype=np.float64).sum()),
                R=float(np.asarray(fg.energy_reserve, dtype=np.float64).sum()),
                intake=dict(env.intake_by_pred_prey.get(f, {})),
                loss={k: float((getattr(env, k, None) or {}).get(f, 0.0))
                      for k in ('loss_starvation', 'loss_predation', 'loss_impact', 'loss_natural')})
        return out

    ticks_year = 1460 // int(config.get('tick_hours', 6)) * 6
    ticks_year = int(round(365 * 24 / int(config.get('tick_hours', 6))))
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    prev = snap()
    for t in range(int(os.environ.get("TICKS", YEARS * ticks_year))):
        year = t // ticks_year
        month = daylight.calendar_at(env.daylight, t)['month_name']
        for f in FISH:
            tick[f] = {}
        env.step()
        now = snap()
        xb = getattr(env, 'tick_exchange_biomass', {}) or {}
        xr = getattr(env, 'tick_exchange_reserve', {}) or {}
        for f in FISH:
            p, n, r = prev[f], now[f], rows[(year, month)][f]
            fill = p['R'] / max(p['B'], 1e-30)
            losses = {k: n['loss'][k] - p['loss'][k] for k in n['loss']}
            lost_t = losses['loss_predation'] + losses['loss_impact'] + losses['loss_natural']
            r['ticks'] += 1
            r['B'] += p['B']
            r['R'] += p['R']
            for k in ('c_rest', 'c_eat', 'c_move', 'sh_eat', 'sh_move', 'sh_rest', 'gain'):
                r[k] += tick[f].get(k, 0.0) * (p['B'] if k.startswith('sh_') else 1.0)
            for prey, v in n['intake'].items():
                r['t:' + prey] += v - p['intake'].get(prey, 0.0)
            r['xB'] += float(xb.get(f, 0.0))
            r['xR'] += float(xr.get(f, 0.0))
            r['lost_t'] += lost_t
            r['starv_t'] += losses['loss_starvation']
            # Reserve carried out by the tick's losses (pro rata at the fill).
            r['lost_R'] += lost_t * fill
            r['dB'] += n['B'] - p['B']
            r['dR'] += n['R'] - p['R']
        prev = now
        if (t + 1) % ticks_year == 0:
            print(f"year {year + 1} done", flush=True)

meta = dict(run=RUN, lib=LIB, years=YEARS, ticks_year=ticks_year,
            rm={f: float(env.dm_resting_metabolism[i]) for f, i in idx.items()},
            ME={f: float(env.fgs[f].max_energy_reserve) for f in FISH})
dump = dict(meta=meta, rows=[dict(year=k[0], month=k[1], groups={f: dict(v) for f, v in g.items()})
                             for k, g in rows.items()])
json.dump(dump, open(OUT, 'w'))
print('wrote', OUT)
