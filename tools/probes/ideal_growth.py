"""Policy-free growth_rate: the budget paid at the ideal-conditions reserve.

    python3 tools/probes/ideal_growth.py [--drop PAIR ...]

``growth_budget`` pays r_max + M1 + sum M2 at s = 1, but the hunger gate
h = (1 - s)/(1 - u) closes there, so no DM ever reaches it. Ideal
conditions here mean: unlimited prey of the best pair (intake at the
Holling ceiling, E_j / h_j MJ per t per tick), eating every tick, no
interference, the annual-mean temperature factor on the eat cost. The
reserve then settles where

    h(s) C = cost + g (s - u) (energy_content + s ME)

(growth is debited from the reserve and the new biomass dilutes it).
Asking that the production g (s - u) there equals the budget
P = (r_max + M1 + sum M2) / ticks_per_year gives, with no policy,

    s* = (C/(1-u) - cost - P ec) / (C/(1-u) + P ME),   g* = P / (s* - u),

feasible while P < Pmax = (C - cost) / (ec + u ME). Also printed: the
ideal-conditions reserve s0 and production with today's g, and the
mass-balance ceiling g <= ME / ec. --drop PAIR removes a predation pair
from the budget (e.g. gadoids_preys_on_gadoids). Resume section 160.
"""
import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import train as train_mod
from lib.world.energy_balance import intake_ceiling
from lib.world.growth_budget import TICKS_PER_YEAR

ap = argparse.ArgumentParser()
ap.add_argument('--drop', action='append', default=[])
args = ap.parse_args()

env = train_mod._ProbeEnvBuilder(project_path='mareld2.yaml', grid_size=(16, 16),
                                 apply_natural_mortality=True, migration=True)()
env._build_static_caches()
order = env.global_fg_order
temp = getattr(env, 'metabolism_temp_table', None)


def steady_s(g, C, cost, u, ec, me):
    """Reserve where h(s) C - cost = g (s - u)(ec + s me), s in [u, 1]."""
    f = lambda s: (1 - s) / (1 - u) * C - cost - g * (s - u) * (ec + s * me)
    if f(u) <= 0:
        return u
    lo, hi = u, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


print(f"{'DM':>13} {'u':>4} {'P/yr':>6} {'best prey':>13} {'C':>7} {'cost':>6} "
      f"{'Pmax/yr':>8} | {'today g':>9} {'s0':>5} {'P0/yr':>6} | "
      f"{'s*':>5} {'g*':>9} {'g*/g':>5} {'ME/ec':>7}")
print(f"{'':>13} {'':>4} {'':>6} {'':>13} {'MJ/t/d':>7} {'MJ/t/d':>6}")
for i, fid in enumerate(env.dm_ids):
    fg = env.fgs[fid]
    u = float(fg.maintenance_level)
    g = float(fg.growth_rate)
    me = float(fg.max_energy_reserve)
    ec = float(fg.params.get('energy_content', 0.0) or 0.0)
    P = g * (1 - u)                                     # budget per tick
    for pair in args.drop:
        pred, prey = pair.split('_preys_on_')
        if prey == fid and pred in env.dm_ids:
            m2 = float(env.fgs[pred].params.get('interaction', {}).get(pair, {})
                       .get('predation_mortality', 0.0) or 0.0)
            if not m2:
                import yaml
                lib = yaml.safe_load(open('fgconfig/fg_library.yaml'))
                m2 = float(lib['interaction_definitions'][pair]['predation_mortality'])
            P -= m2 / TICKS_PER_YEAR
    rates = {}
    for j, prey in enumerate(order):
        if env.eat_static_mask[i, j]:
            rates[prey] = (intake_ceiling(float(env.max_intake_mat[i, j]),
                                          float(env.handling_time_mat[i, j]))
                           * float(env.energy_gain_mat[i, j]))
    best = max(rates, key=rates.get)
    C = rates[best]
    tf = float(np.mean(temp[:, i])) if temp is not None else 1.0
    cost = float(env.dm_resting_metabolism[i]) * float(env.dm_cost_eat[i]) * tf
    pmax = (C - cost) / (ec + u * me)
    s0 = steady_s(g, C, cost, u, ec, me)
    p0 = g * (s0 - u)
    if P < pmax:
        s_star = (C / (1 - u) - cost - P * ec) / (C / (1 - u) + P * me)
        g_star = P / (s_star - u)
        tail = f"{s_star:5.3f} {g_star:9.6f} {g_star/g:5.2f}"
    else:
        tail = f"{'-':>5} {'INFEASIBLE':>9} {'-':>5}"
    print(f"{fid:>13} {u:4.2f} {P*TICKS_PER_YEAR:6.2f} {best:>13} {C*4:7.0f} {cost*4:6.1f} "
          f"{pmax*TICKS_PER_YEAR:8.2f} | {g:9.6f} {s0:5.3f} {p0*TICKS_PER_YEAR:6.2f} | "
          f"{tail} {me/ec if ec else float('nan'):7.4f}")
    others = ", ".join(f"{k} {v*4:.0f}" for k, v in sorted(rates.items(), key=lambda kv: -kv[1]) if k != best)
    print(f"{'':>13} other prey at the ceiling (MJ/t/d): {others or '-'}; temperature factor {tf:.2f}")
