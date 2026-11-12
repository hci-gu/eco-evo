"""What limits the fish's intake, month by month (frozen policies, probe world).

Wraps predation._reserve_room_cap: per fish group the demand's energy
before the reserve-room cap (pre) and after it (post), with the realised
gain after the prey-side harvest caps. pre/post > 1: the reserve room binds
(the fish could eat more); gain/post < 1: the prey side cuts it. The
ceiling share is pre / (B x ceiling energy), the ceiling taken as the
max over prey of E_gain_j / h_j at this tick's ingestion Q10.
Also the krill pool's annual budget: import, export, M2, local growth.

    RUN=... LIB=... YEARS=2 python3 limits.py
"""
import json, os, sys
from collections import defaultdict
import numpy as np
ROOT = '/home/simon/eco-evo'; sys.path.insert(0, ROOT); os.chdir(ROOT)
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import interactions, predation
from lib.runners import training_progress as tp
from lib.world import daylight

RUN = os.environ.get('RUN', 'results/pzbpgp248nsrpfix97'); LIB = os.environ['LIB']
YEARS = int(os.environ.get('YEARS', 2))
FISH = ['planktivorous_fish', 'benthivorous_fish', 'whiting', 'predatory_fish', 'mackerel']
config = json.load(open(os.path.join(RUN, 'progress', 'config.json'))); config['library'] = LIB
seed = int(config['seed'])
with tp.evaluation_randomness(seed):
    env = tp.build_inference_env(config, seed)
with tp.evaluation_randomness(seed):
    pol, mean, var = load_policies_and_stats(env, RUN, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = mean, var
    env.softmax_temperature = float(config.get('temperature', 1.0))
    idx = {f: env.dm_ids.index(f) for f in FISH}
    tick = {}
    _cap = predation._reserve_room_cap

    def cap(e, demand, predator_biomass, reserve_offset):
        def energy(d):
            p = (d * e.energy_gain_mat[:, :, None, None]).sum(axis=1)
            if reserve_offset is not None:
                p = p + (d * e.assimilation_mat[:, :, None, None] * reserve_offset[None]).sum(axis=1)
            return p.sum(axis=(1, 2))
        out = _cap(e, demand, predator_biomass, reserve_offset)
        pre, post = energy(demand), energy(out)
        for f, i in idx.items():
            tick[f] = (float(pre[i]), float(post[i]))
        return out
    predation._reserve_room_cap = cap

    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    kb = {k: 0.0 for k in ('gain_exchange', 'loss_exchange', 'loss_predation', 'loss_natural', 'B')}
    def book(name, f):
        return float((getattr(env, name, None) or {}).get(f, 0.0))
    k0 = {k: book(k, 'krill') for k in kb if k != 'B'}
    kB0 = float(np.asarray(env.fgs['krill'].biomass).sum())
    T = YEARS * 1460
    for t in range(T):
        m = daylight.calendar_at(env.daylight, t)['month_name']
        for f in FISH:
            tick[f] = (0.0, 0.0)
        env.step()
        for f in FISH:
            fg = env.fgs[f]; r = rows[m][f]
            r['B'] += float(np.asarray(fg.biomass).sum())
            r['pre'] += tick[f][0]; r['post'] += tick[f][1]
            r['gain'] += float(np.asarray(fg.temp_energy_gains).sum())
        if t >= T - 1460:
            kb['B'] += float(np.asarray(env.fgs['krill'].biomass).sum()) / 1460
        if t == T - 1461:
            k0 = {k: book(k, 'krill') for k in kb if k != 'B'}
            kB0 = float(np.asarray(env.fgs['krill'].biomass).sum())

print(f"{'group':>18} {'mon':>4} {'pre/post':>8} {'gain/post':>9} {'gain MJ/t/tick':>14} {'pre MJ/t/tick':>13}")
for f in FISH:
    for m in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']:
        r = rows[m][f]
        print(f"{f:>18} {m:>4} {r['pre'] / max(r['post'], 1e-9):>8.2f} {r['gain'] / max(r['post'], 1e-9):>9.2f} "
              f"{r['gain'] / r['B']:>14.1f} {r['pre'] / r['B']:>13.1f}")
kB1 = float(np.asarray(env.fgs['krill'].biomass).sum())
flows = {k: book(k, 'krill') - k0[k] for k in k0}
grow = kB1 - kB0 + flows['loss_exchange'] + flows['loss_predation'] + flows['loss_natural'] - flows['gain_exchange']
print(f"\nkrill, last year (t/yr; mean B {kb['B']:.0f} t, start-of-year {kB0:.0f}, end {kB1:.0f}):")
for k, v in flows.items():
    print(f"  {k:>15} {v:>9.0f}  ({v / kb['B']:.2f} /yr)")
print(f"  {'local growth':>15} {grow:>9.0f}  ({grow / kb['B']:.2f} /yr)")
