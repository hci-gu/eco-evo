"""Month-by-month energy budget of one DM (default herring), frozen policies.

    python3 tools/probes/herring_monthly.py CKPT_DIR [YEARS] [FG]

Probe world (60x60, mortality, migration, currents 0.1/20/0/12, daylight
and temperature from mareld2.yaml). Per calendar month, biomass-weighted
over herring cells: reserve fill s, hunger h = (1-s)/(1-u), the action
mix, energy gains and action costs (MJ per t herring per day), realised
growth against the growth budget g*(1-u) (both as a yearly rate), the
food eaten per prey and the zooplankton density where the herring are.
Written for resume section 160 (why herring production stays at ~60 %
of its budget).
"""
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
from lib.environments.ecosystem_env import interactions, movement, population_change
from lib.environments.ecosystem_env.currents import CurrentConfig

ckpt = sys.argv[1]
years = int(sys.argv[2]) if len(sys.argv) > 2 else 3
env = train_mod._ProbeEnvBuilder(
    project_path='mareld2.yaml', grid_size=(60, 60),
    apply_natural_mortality=True, migration=True,
    currents=CurrentConfig(0.1, 20, 0, 12.0))()
env.policies, env.obs_mean, env.obs_var = load_policies_and_stats(
    env, ckpt, verbose=False)
env._batched_ready = False
for attr in ('_setup_batched_policies', '_prepare_batched_policies',
             '_build_batched_policies'):
    fn = getattr(env, attr, None)
    if callable(fn):
        fn()
        break
np.random.seed(20260530)
torch.manual_seed(20260530)

H = sys.argv[3] if len(sys.argv) > 3 else 'pelagic_fish'
fg = env.fgs[H]
IH = env.dm_ids.index(H)
u = float(fg.maintenance_level)
g = float(fg.growth_rate)
TPD = 4                                       # 6 h ticks
TPY = 1460
print(f"{H}: u {u}, g {g:.6f}/tick -> budget g(1-u) {g*(1-u)*TPY:.3f}/yr, "
      f"ME {fg.max_energy_reserve:.0f} MJ/t, energy_content "
      f"{fg.params.get('energy_content')} MJ/t, metabolism "
      f"{env.dm_resting_metabolism[IH]:.2f} MJ/t/tick, costs rest/eat/move "
      f"{env.dm_cost_rest[IH]:.2f}/{env.dm_cost_eat[IH]:.2f}/{env.dm_cost_move[IH]:.2f}")

KEYS = ('ticks', 'B', 's', 'h', 'full', 'eat', 'rest', 'move', 'gain', 'cost',
        'growth', 'wish', 'm1', 'zoo')
acc = {}
tick_state = {}


def bucket(month):
    return acc.setdefault(month, {k: 0.0 for k in KEYS} | {'prey': {}})


def costs_hook(env_, actions):
    B = fg.biomass.astype(np.float64)
    tot = B.sum()
    if tot > 0:
        T = interactions.metabolism_temperature(env_)
        mult = float(T[IH]) if T is not None else 1.0
        m = float(env_.dm_resting_metabolism[IH]) * mult
        rest = actions.rest[IH]
        eat = actions.eat[IH].sum(axis=0)
        move = actions.move[IH].sum(axis=0)
        cost = B * m * (rest * env_.dm_cost_rest[IH] + eat * env_.dm_cost_eat[IH]
                        + move * env_.dm_cost_move[IH])
        b = tick_state['b']
        b['gain'] += float(fg.temp_energy_gains.sum()) / tot
        b['cost'] += float(cost.sum()) / tot
        b['eat'] += float((B * eat).sum()) / tot
        b['rest'] += float((B * rest).sum()) / tot
        b['move'] += float((B * move).sum()) / tot
    return _orig_costs(env_, actions)


def growth_hook(env_, fg_id, fg_):
    if fg_id != H:
        return _orig_growth(env_, fg_id, fg_)
    b = tick_state['b']
    B0 = fg_.biomass.astype(np.float64)
    s = fg_.energy_level.astype(np.float64)
    keep = 1.0 - float(fg_.natural_mortality) * float(getattr(env_, 'mortality_multiplier', 1.0))
    L0 = env_.loss_natural.get(H, 0.0)
    tot = B0.sum()
    _orig_growth(env_, fg_id, fg_)
    if tot > 0:
        m1 = env_.loss_natural.get(H, 0.0) - L0
        b['m1'] += m1 / tot
        b['growth'] += (float(fg_.biomass.sum()) - tot + m1) / tot
        b['wish'] += float((B0 * keep * g * np.maximum(0.0, s - u)).sum()) / tot


_orig_costs = movement.apply_energy_costs
_orig_growth = population_change._apply_decision_maker_population_change
movement.apply_energy_costs = costs_hook
population_change._apply_decision_maker_population_change = growth_hook

zoo = env.fgs['zooplankton']
prev_intake = dict(env.intake_by_pred_prey.get(H, {}))
PREY = [p for p in env.global_fg_order
        if env.eat_static_mask[IH, env.global_fg_order.index(p)] > 0]
for t in range(years * TPY):
    day = interactions.light_index(env) / TPD
    month = min(11, int(day / (365.0 / 12)))
    year = t // TPY
    b = bucket((year, month))
    tick_state['b'] = b
    B = fg.biomass.astype(np.float64)
    tot = B.sum()
    if tot > 0:
        s = fg.energy_level.astype(np.float64)
        b['B'] += tot
        b['s'] += float((B * s).sum()) / tot
        b['h'] += float((B * np.clip((1 - s) / (1 - u), 0, 1)).sum()) / tot
        b['full'] += float(B[s >= 0.9].sum()) / tot
        b['zoo'] += float((B * zoo.biomass).sum()) / tot
    b['ticks'] += 1
    env.step()
    cur = env.intake_by_pred_prey.get(H, {})
    for k, v in cur.items():
        d = v - prev_intake.get(k, 0.0)
        if tot > 0:
            b['prey'][k] = b['prey'].get(k, 0.0) + d / tot
    prev_intake = dict(cur)

MON = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()
budget = g * (1 - u) * TPY
for year in range(years):
    print(f"\n== year {year + 1}: per month, biomass-weighted; gain/cost MJ/t/day; "
          f"growth, wish, M1 as /yr; budget {budget:.2f}/yr; food t/t/day")
    print("mon     B      s     h   s>=.9  eat  rest move   gain  cost  net  "
          "growth  wish   M1   zoo t/km2  food: " + " ".join(p[:5] for p in PREY))
    for mo in range(12):
        b = acc.get((year, mo))
        if not b or b['ticks'] == 0:
            continue
        n = b['ticks']
        p = b['prey']
        print(f"{MON[mo]}  {b['B']/n:6.0f}  {b['s']/n:.3f} {b['h']/n:.2f}  {b['full']/n:.2f}  "
              f"{b['eat']/n:.2f} {b['rest']/n:.2f} {b['move']/n:.2f}  "
              f"{b['gain']/n*TPD:5.1f} {b['cost']/n*TPD:5.1f} {(b['gain']-b['cost'])/n*TPD:5.1f}  "
              f"{b['growth']/n*TPY:5.2f}  {b['wish']/n*TPY:5.2f} {b['m1']/n*TPY:5.2f}   "
              f"{b['zoo']/n:6.2f}   "
              + " ".join(f"{p.get(k, 0)/n*TPD:.4f}" for k in PREY))
    tt = [acc[(year, mo)] for mo in range(12) if (year, mo) in acc]
    n = sum(b['ticks'] for b in tt)
    print(f"year  growth {sum(b['growth'] for b in tt)/n*TPY:.2f}/yr of budget {budget:.2f}"
          f" ({sum(b['growth'] for b in tt)/n*TPY/budget:.0%}); wish {sum(b['wish'] for b in tt)/n*TPY:.2f}; "
          f"mean s {sum(b['s'] for b in tt)/n:.3f}; M1 {sum(b['m1'] for b in tt)/n*TPY:.2f}/yr")
