"""Porpoise winter energy budget: positional or budgetary? (section 143.1)

Same deterministic probe world as fg_year_budget.py; every other DM runs
its trained policy, only the porpoise row is replaced:

  trained    porpoises run their trained policy (baseline)
  eatonly    trained start cells, never move, eat 100 % on the best prey
  oracle K   every tick ALL porpoise biomass + reserve is put into the K
             cells with the highest gain potential, eating 100 % there
             (an omniscient pursuer; movement is free unless CHARGE_MOVE)
  greedy     move a share FRAC to the best neighbour when its potential
             beats the own cell by MARGIN, eat 100 % on the best prey

Environment variables: RUN (results/<run>, default pzbpgp248nsrpfix47),
TICKS (default 1460), MARGIN / FRAC (greedy), CHARGE_MOVE=1 (oracle pays
resting_metabolism * movement_cost every tick), CMOVE0=1 (free moves).

On fix47 (section 143.1): trained, eatonly and every greedy variant die
in Jan-Feb; only the free oracle survives, at zero margin.

    python3 tools/probes/porp_winter_oracle.py oracle 20
    MARGIN=0.1 FRAC=1.0 python3 tools/probes/porp_winter_oracle.py greedy
"""
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import train as train_mod  # noqa: E402
from inference import load_policies_and_stats  # noqa: E402
from lib.environments.ecosystem_env import impacts, predation  # noqa: E402
from lib.environments.ecosystem_env.currents import CurrentConfig  # noqa: E402

RUN = os.environ.get('RUN', 'pzbpgp248nsrpfix47')
MODE = sys.argv[1] if len(sys.argv) > 1 else 'trained'
K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
MARGIN = float(os.environ.get('MARGIN', 0.2))
FRAC = float(os.environ.get('FRAC', 0.5))
TICKS = int(os.environ.get('TICKS', 1460))
H = W = 60

builder = train_mod._ProbeEnvBuilder(
    project_path='mareld2.yaml', grid_size=(H, W),
    apply_natural_mortality=True, migration=True, mass_balance=True,
    currents=CurrentConfig(0.1, 20, 0, 12.0))
env = builder()
policies, mean, var = load_policies_and_stats(
    env, os.path.join('results', RUN), verbose=False)
env.policies = policies
env.obs_mean = mean
env.obs_var = var

ip = env.dm_ids.index('porpoises')
env.build_static_caches()
if os.environ.get('CMOVE0') == '1':
    env.dm_cost_move[ip] = 0.0
porp = env.fgs['porpoises']
prey = ['pelagic_fish', 'gadoids']
jj = [env.global_fg_order.index(p) for p in prey]
rm = float(env.dm_resting_metabolism[ip])
c_eat, c_move, c_rest = (float(env.dm_cost_eat[ip]), float(env.dm_cost_move[ip]),
                         float(env.dm_cost_rest[ip]))
u_x = float(porp.maintenance_level)
q = [float(env.energy_gain_mat[ip, j]) for j in jj]
a = [float(env.max_intake_mat[ip, j]) for j in jj]
h = [float(env.handling_time_mat[ip, j]) for j in jj]
print(f"mode={MODE} K={K} rm={rm} c_eat={c_eat} c_move={c_move} c_rest={c_rest} "
      f"u={u_x} q={q} a={a} h={h} interference="
      f"{float(np.ravel(env.dm_interference[ip])[0]) if env._has_interference else 0}")
print(f"pure-eat cost = {rm * c_eat:.1f} MJ/t/tick; need per-prey sat "
      f"{[round(rm * c_eat / (a[k] * q[k]), 3) for k in range(2)]}")


def gain_potential():
    """Per-tonne gain if a porpoise ate 100 % of the best prey in the cell
    (Holling II on total biomass, hunger 1, no interference)."""
    out = []
    for k, j in enumerate(jj):
        B = env.fgs[prey[k]].biomass
        out.append(q[k] * a[k] * B / (1.0 + a[k] * h[k] * B))
    return np.stack(out)          # (2, H, W)


def concentrate(cells):
    tb, tr = float(porp.biomass.sum()), float(porp.energy_reserve.sum())
    porp.biomass[:] = 0.0
    porp.energy_reserve[:] = 0.0
    n = len(cells[0])
    porp.biomass[cells] = tb / n
    porp.energy_reserve[cells] = tr / n


_orig = env.calculate_decisions


def decisions():
    acts = _orig()
    if MODE == 'greedy':
        gp = gain_potential()
        best = np.argmax(gp, axis=0)
        pot = gp.max(axis=0)
        nb = np.full((4,) + pot.shape, -np.inf)
        nb[0, 1:, :] = pot[:-1, :]     # N
        nb[1, :, :-1] = pot[:, 1:]     # E
        nb[2, :-1, :] = pot[1:, :]     # S
        nb[3, :, 1:] = pot[:, :-1]     # W
        d = np.argmax(nb, axis=0)
        go = nb.max(axis=0) > pot * (1.0 + MARGIN)
        acts.move[ip] = 0.0
        acts.rest[ip] = 0.0
        acts.eat[ip] = 0.0
        for k in range(4):
            acts.move[ip, k] = (go & (d == k)) * FRAC
        for k, j in enumerate(jj):
            acts.eat[ip, j] = (best == k) * (1.0 - go * FRAC)
        return acts
    if MODE in ('eatonly', 'oracle'):
        best = np.argmax(gain_potential(), axis=0)
        acts.move[ip] = 0.0
        acts.rest[ip] = 0.0
        acts.eat[ip] = 0.0
        for k, j in enumerate(jj):
            acts.eat[ip, j] = (best == k).astype(acts.eat.dtype)
    return acts


env.calculate_decisions = decisions

# Book the realised gain (MJ) right after predation.
_pred = predation.apply_predation
book = {}


def pred_hook(e, actions):
    _pred(e, actions)
    B = porp.biomass
    book['gain'] = float(porp.temp_energy_gains.sum())
    book['B'] = float(B.sum())
    w = B / max(book['B'], 1e-30)
    book['eat'] = float((actions.eat[ip].sum(axis=0) * w).sum())
    book['move'] = float((actions.move[ip].sum(axis=0) * w).sum())
    book['rest'] = float((actions.rest[ip] * w).sum())
    book['eat_split'] = [float((actions.eat[ip, j] * w).sum()) for j in jj]
    cf = impacts.impact_energy_cost_factor(e)
    cf = cf[ip] if np.ndim(cf) == 3 else cf
    book['cost_factor'] = float((np.broadcast_to(cf, B.shape) * w).sum())
    book['local'] = [float((e.fgs[p].biomass * w).sum()) for p in prey]
    book['s'] = float((porp.energy_level * w).sum())
    pfg = e.fgs['pelagic_fish']
    dens = np.divide(pfg.energy_reserve, pfg.biomass, out=np.zeros_like(pfg.biomass), where=pfg.biomass > 1e-9)
    book['pf_E'] = float((dens * w).sum())
    book['pf_rest'] = float((actions.rest[e.dm_ids.index('pelagic_fish')] * w).sum())
    ipf = e.dm_ids.index('pelagic_fish')
    book['pf_E_all'] = float(pfg.energy_reserve.sum() / max(pfg.biomass.sum(), 1e-9))
    book['hung'] = float((porp.get_hunger() * w).sum())


predation.apply_predation = pred_hook

np.random.seed(20260530)
torch.manual_seed(20260530)
print(f"{'tick':>5} {'B':>7} {'cells':>5} {'s':>5} {'hung':>5} {'eat':>5} "
      f"{'mov':>5} {'rest':>5} {'e_pf':>5} {'e_gd':>5} {'cf':>5} {'gain/t':>7} "
      f"{'cost/t':>7} {'pf_loc':>7} {'gd_loc':>7} {'pf_mean':>7} {'gd_mean':>7} "
      f"{'pot_loc':>7} {'pot_p95':>7}")
sum_gain = sum_cost = 0.0
died = None
for t in range(TICKS):
    if MODE == 'oracle' and porp.biomass.sum() > 0:
        pot = gain_potential().max(axis=0)
        flat = np.argsort(pot.ravel())[::-1][:K]
        concentrate(np.unravel_index(flat, pot.shape))
        if os.environ.get('CHARGE_MOVE') == '1':
            porp.energy_reserve[:] = np.maximum(
                0.0, porp.energy_reserve - porp.biomass * rm * c_move)
    pot = gain_potential().max(axis=0)
    w0 = porp.biomass / max(float(porp.biomass.sum()), 1e-30)
    pot_loc = float((pot * w0).sum())
    env.step()
    if not book or book['B'] <= 0:
        died = t
        print(f">>> porpoises extinct at tick {t}")
        break
    Bt = book['B']
    cost = rm * book['cost_factor'] * (book['rest'] * c_rest + book['eat'] * c_eat
                                       + book['move'] * c_move)
    g = book['gain'] / Bt
    sum_gain += g
    sum_cost += cost
    if t % 50 == 0 or t < 3:
        water = env.fgs['phytoplankton'].biomass > 0
        means = [float(env.fgs[p].biomass[water].mean()) for p in prey]
        print(f"{t:>5} {Bt:>7.3f} {int((porp.biomass > 0).sum()):>5} "
              f"{book['s']:>5.2f} {book['hung']:>5.2f} {book['eat']:>5.2f} "
              f"{book['move']:>5.2f} {book['rest']:>5.2f} {book['eat_split'][0]:>5.2f} "
              f"{book['eat_split'][1]:>5.2f} {book['cost_factor']:>5.2f} {g:>7.1f} "
              f"{cost:>7.1f} {book['local'][0]:>7.2f} {book['local'][1]:>7.2f} "
              f"{means[0]:>7.2f} {means[1]:>7.2f} {pot_loc:>7.1f} "
              f"{float(np.percentile(pot[water], 95)):>7.1f}  pfE={book['pf_E']:.0f} pfE_all={book['pf_E_all']:.0f} pfRest={book['pf_rest']:.2f}")
n = (died if died is not None else TICKS)
print(f"\nmode={MODE} K={K} died_at={died} final_B={float(porp.biomass.sum()):.4f} "
      f"(B0 12)  mean gain/t={sum_gain / max(n, 1):.1f}  mean cost/t={sum_cost / max(n, 1):.1f}")
print("intake by prey (t):", {p: round(v, 2) for p, v in
                              env.intake_by_pred_prey.get('porpoises', {}).items() if v > 0})
