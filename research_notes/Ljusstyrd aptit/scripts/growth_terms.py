"""Which term of the material growth law binds for the planktivores, by month, under frozen policies.

Hooks movement.income_growth and books, per month, the planktivores' four terms
(the g ceiling B g (s - u) p, the material cap kappa_m I p / ec, the store above u,
the tick's surplus max(I - C, 0)) summed over cells, the paid G, and the
biomass-weighted share of cells where each term is the minimum.
    RUN=... LIB=... YEARS=2 python3 growth_terms.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from inference import load_policies_and_stats  # noqa: E402
from lib.environments.ecosystem_env import movement  # noqa: E402
from lib.runners import training_progress as tp  # noqa: E402
from lib.world import daylight, energy_balance  # noqa: E402

RUN, LIB = os.environ['RUN'], os.environ['LIB']
YEARS = int(os.environ.get('YEARS', 2))
P = 'planktivorous_fish'
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
    i = env.dm_ids.index(P)
    fg = env.fgs[P]
    g = float(fg.growth_rate)
    u = float(fg.maintenance_level)
    ec = float(fg.params['energy_content'])
    me = float(fg.max_energy_reserve)
    share = float(fg.params['material_growth_share'])
    book = defaultdict(lambda: defaultdict(float))
    state = {'month': None, 'year': 0}
    _growth = movement.income_growth

    def hook(e, biomass, reserve, gains, cost, sb, sr):
        b, r = biomass[i].astype(np.float64), reserve[i].astype(np.float64)
        cap = b * me
        fill = np.where(cap > 0, r / np.where(cap > 0, cap, 1.0), 0.0)
        price = energy_balance.tonne_price(ec, float(e.dm_growth_cost[i]), fill, me)
        ceiling = b * g * np.maximum(fill - u, 0.0) * price
        material = share * np.maximum(gains[i], 0.0) * price / ec
        above = np.maximum(sr[i] - u * me * sb[i], 0.0)
        surplus = np.maximum(gains[i] - cost[i], 0.0)
        terms = np.stack([ceiling, material, above, surplus])
        which = np.argmin(terms, axis=0)
        row = book[(state['year'], state['month'])]
        w = b
        for k, name in enumerate(('ceiling', 'material', 'above', 'surplus')):
            row[name] += float(terms[k].sum())
            row['bind_' + name] += float(w[which == k].sum())
        row['paid'] += float(np.min(terms, axis=0).sum())
        row['B'] += float(b.sum())
        row['ticks'] += 1
        return _growth(e, biomass, reserve, gains, cost, sb, sr)

    movement.income_growth = hook
    for t in range(YEARS * 1460):
        state['year'] = t // 1460
        state['month'] = daylight.calendar_at(env.daylight, t)['month_name']
        env.step()

M = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
y = YEARS - 1
print(f"{P}: g {g:.6g}/tick, u {u}, year {y + 1}; MJ/t/yr of mean biomass; binding share by biomass")
print(f"{'mon':>4} {'ceiling':>8} {'material':>8} {'above':>8} {'surplus':>8} {'paid':>8}   "
      f"{'b_ceil':>6} {'b_mat':>6} {'b_abv':>6} {'b_sur':>6}")
for m in M:
    r = book.get((y, m))
    if not r:
        continue
    per = 1460.0 / r['B']
    print(f"{m:>4} {r['ceiling'] * per:8.0f} {r['material'] * per:8.0f} {r['above'] * per:8.0f} "
          f"{r['surplus'] * per:8.0f} {r['paid'] * per:8.0f}   "
          + " ".join(f"{r['bind_' + k] / r['B']:6.2f}" for k in ('ceiling', 'material', 'above', 'surplus')))
