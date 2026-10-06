"""One decision maker's energy budget over a year, with trained policies.

Runs the deterministic probe world (60x60, migration, mortality and
currents on, the project's daylight / temperature settings) with a
run's trained policies and prints monthly means, biomass-weighted over
the group's cells: reserve fill s, share of biomass below the
starvation threshold, hunger, action mix, realised gain and metabolic
cost per tonne (MJ/t/tick), the light multiplier on the group's
zooplankton attack rate (if it eats zooplankton), zooplankton at the
group's cells vs the grid mean, biomass lost to predation and the net
biomass change. Sections 143-144.

Environment variables:
    RUN        run name under results/ (default pzbpgp248nsrpfix47)
    FG         decision maker to book (default pelagic_fish)
    LIB        library variant instead of fgconfig/fg_library.yaml
    TICKS      ticks to run (default 1460 = one year at 6 h)
    RM_WINTER  in-memory factor on FG's resting_metabolism Nov-Feb
    U_X        in-memory maintenance_level override for FG

    python3 tools/probes/fg_year_budget.py
    FG=gadoids LIB=/tmp/variant.yaml python3 tools/probes/fg_year_budget.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import train as train_mod  # noqa: E402
from inference import load_policies_and_stats  # noqa: E402
from lib.environments.ecosystem_env import interactions, predation  # noqa: E402
from lib.environments.ecosystem_env.currents import CurrentConfig  # noqa: E402
from lib.world import daylight  # noqa: E402

RUN = os.environ.get('RUN', 'pzbpgp248nsrpfix47')
FG = os.environ.get('FG', 'pelagic_fish')
TICKS = int(os.environ.get('TICKS', 1460))
RM_WINTER = float(os.environ.get('RM_WINTER', 1.0))

builder = train_mod._ProbeEnvBuilder(
    project_path='mareld2.yaml', grid_size=(60, 60),
    library_path=os.environ.get('LIB') or None,
    apply_natural_mortality=True, migration=True, mass_balance=True,
    currents=CurrentConfig(0.1, 20, 0, 12.0))
env = builder()
policies, mean, var = load_policies_and_stats(
    env, os.path.join('results', RUN), verbose=False)
env.policies, env.obs_mean, env.obs_var = policies, mean, var
env.build_static_caches()
if env.daylight is None:
    sys.exit("the monthly table needs simulation_settings.daylight on")

i = env.dm_ids.index(FG)
fg = env.fgs[FG]
if os.environ.get('U_X'):
    fg.maintenance_level = float(os.environ['U_X'])
u = float(fg.maintenance_level)
jz = env.global_fg_order.index('zooplankton')
a_zoo = float(env.max_intake_mat[i, jz])
c_eat, c_move, c_rest = (float(env.dm_cost_eat[i]), float(env.dm_cost_move[i]),
                         float(env.dm_cost_rest[i]))
rm_base = env.dm_resting_metabolism.copy()
print(f"{FG}: run={RUN} resting_metabolism={float(rm_base[i])} "
      f"c_eat={c_eat} c_move={c_move} c_rest={c_rest} "
      f"ME={float(fg.max_energy_reserve)} u={u} "
      f"growth_rate={fg.growth_rate:.6g} starve_rate={fg.starve_rate}")

book = {}
_apply_predation = predation.apply_predation


def booking_predation(e, actions):
    """Run the real predation step, then book FG's state and flows."""
    before = float(fg.biomass.sum())
    _apply_predation(e, actions)
    total = max(float(fg.biomass.sum()), 1e-30)
    w = fg.biomass / total
    book['eaten'] = before - float(fg.biomass.sum())
    book['B'] = total
    book['gain'] = float(fg.temp_energy_gains.sum()) / total
    book['eat'] = float((actions.eat[i].sum(axis=0) * w).sum())
    book['move'] = float((actions.move[i].sum(axis=0) * w).sum())
    book['rest'] = float((actions.rest[i] * w).sum())
    book['s'] = float((fg.energy_level * w).sum())
    book['hung'] = float((fg.get_hunger() * w).sum())
    book['starving_share'] = float((w * (fg.energy_level < u)).sum())
    book['zoo_loc'] = float((e.fgs['zooplankton'].biomass * w).sum())
    book['mult'] = (float(interactions.attack_rate(e)[i, jz]) / a_zoo
                    if a_zoo > 0 else float('nan'))


predation.apply_predation = booking_predation
np.random.seed(20260530)
torch.manual_seed(20260530)

months = defaultdict(lambda: defaultdict(list))
order = []
for t in range(TICKS):
    cal = daylight.calendar_at(env.daylight, t)
    env.dm_resting_metabolism[:] = rm_base
    if cal['month_name'] in ('Nov', 'Dec', 'Jan', 'Feb'):
        env.dm_resting_metabolism[i] = rm_base[i] * RM_WINTER
    if not order or order[-1][1] != cal['month_name']:
        order.append((len(order), cal['month_name']))
    b_before = float(fg.biomass.sum())
    env.step()
    temp = interactions.metabolism_temperature(env)
    rm = float(env.dm_resting_metabolism[i]) * (1.0 if temp is None
                                                 else float(temp[i]))
    m = months[order[-1]]
    m['t'].append(t)
    for k in ('B', 's', 'hung', 'eat', 'move', 'rest', 'gain', 'zoo_loc',
              'mult', 'starving_share', 'eaten'):
        m[k].append(book[k])
    m['cost'].append(rm * (book['rest'] * c_rest + book['eat'] * c_eat
                           + book['move'] * c_move))
    m['dB'].append(float(fg.biomass.sum()) - b_before)
    m['zoo_mean'].append(float(env.fgs['zooplankton'].biomass.mean()))

print(f"\n{'month':>5} {'ticks':>9} {'B_end':>8} {'s':>5} {'starv%':>6} "
      f"{'hung':>5} {'eat':>5} {'mov':>5} {'rest':>5} {'gain/t':>7} "
      f"{'cost/t':>7} {'net':>6} {'a_mult':>6} {'zoo_loc':>7} {'zoo_mn':>7} "
      f"{'eaten':>7} {'dB':>7}")
for key in order:
    m = months[key]
    av = {k: float(np.mean(v)) for k, v in m.items()}
    print(f"{key[1]:>5} {m['t'][0]:>4}-{m['t'][-1]:<4} {m['B'][-1]:>8.1f} "
          f"{av['s']:>5.2f} {100 * av['starving_share']:>6.1f} "
          f"{av['hung']:>5.2f} {av['eat']:>5.2f} {av['move']:>5.2f} "
          f"{av['rest']:>5.2f} {av['gain']:>7.1f} {av['cost']:>7.1f} "
          f"{av['gain'] - av['cost']:>6.1f} {av['mult']:>6.2f} "
          f"{av['zoo_loc']:>7.2f} {av['zoo_mean']:>7.2f} "
          f"{sum(m['eaten']):>7.0f} {sum(m['dB']):>7.0f}")
print("\nend of run: " + "  ".join(
    f"{fid}={float(env.fgs[fid].biomass.sum()):.1f}" for fid in env.dm_ids))
