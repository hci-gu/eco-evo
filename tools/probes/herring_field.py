"""Does the herring field EVER sustain the porpoise requirement?

The pinned/perfect experiment showed that even an oracle pursuer
re-concentrated onto the 5 best herring cells every tick starves. That
shifts the question from "can the porpoise find the school" to "does a
school dense enough to pay for a porpoise exist at all".

Requirement (derived in porp_budget.py):
    sat_min = rm*c_eat / (a*gain*h(u_x)) = 0.84
    -> visible herring >= 5.25 t/cell
    -> total herring >= 5.25 / (1 - pi_rest*(1 - 0.95)) ~ 5.3-5.5 t/cell

This probe runs the trained world with NO porpoises at all (so the
herring field is not being cropped by them) and reports, per tick, the
number of cells above the requirement plus the top densities.
"""
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import train as train_mod
from inference import load_policies_and_stats

RUN = 'pzbpgp248nsrpfix9'
TICKS = 1000
H = W = 16
KILL_PORPOISES = '--keep-porpoises' not in sys.argv
REQ = 5.25

builder = train_mod._ProbeEnvBuilder(
    project_path='mareld2.yaml', grid_size=(H, W),
    apply_natural_mortality=True, migration=True)
env = builder()
policies, mean, var = load_policies_and_stats(
    env, os.path.join('results', RUN), verbose=False)
env.policies = policies
env.obs_mean = mean
env.obs_var = var

pf = env.fgs['pelagic_fish']
porp = env.fgs['porpoises']
if KILL_PORPOISES:
    porp.biomass = np.zeros((H, W), dtype=np.float32)
    porp.energy_reserve = np.zeros((H, W), dtype=np.float32)

print(f"porpoises={'removed' if KILL_PORPOISES else 'kept'}  "
      f"requirement = {REQ} t visible/cell")
print(f"{'tick':>5} {'B_pf':>8} {'max':>8} {'top5':>8} {'top20':>8} "
      f"{'n>=req':>7} {'mean':>7} {'s_x_pf':>7} {'mov':>5} {'rst':>5}")

np.random.seed(20260530)
torch.manual_seed(20260530)
ipf = env.dm_ids.index('pelagic_fish')

for t in range(TICKS + 1):
    B = pf.biomass
    flat = np.sort(B.ravel())[::-1]
    occ = B > 0
    s_x = float(pf.energy_level[occ].mean()) if occ.any() else 0.0
    mov = float(env.pi_move[ipf][:, occ].sum(axis=0).mean()) \
        if (occ.any() and t > 0) else float('nan')
    rst = float(env.pi_rest[ipf][occ].mean()) \
        if (occ.any() and t > 0) else float('nan')
    if t % 50 == 0:
        print(f"{t:>5} {float(B.sum()):>8.1f} {float(flat[0]):>8.2f} "
              f"{float(flat[:5].mean()):>8.2f} "
              f"{float(flat[:20].mean()):>8.2f} "
              f"{int((B >= REQ).sum()):>7} {float(B.mean()):>7.3f} "
              f"{s_x:>7.3f} {mov:>5.3f} {rst:>5.3f}")
    if t < TICKS:
        env.step()

lp = {k: float(v) for k, v in env.loss_predation.items()}
ls = {k: float(v) for k, v in env.loss_starvation.items()}
li = {k: float(v) for k, v in getattr(env, 'loss_impact', {}).items()}
print()
print(f"{'fg':>18} {'predation':>12} {'starvation':>12} {'impact':>10} "
      f"{'nat.mort':>10} {'events':>7}")
for fid, fg in env.fgs.items():
    nm = float(getattr(fg, 'natural_mortality', 0.0) or 0.0)
    ev = int(env._extinction_events.get(fid, 0))
    print(f"{fid:>18} {lp.get(fid, 0.0):>12.1f} {ls.get(fid, 0.0):>12.1f} "
          f"{li.get(fid, 0.0):>10.1f} {nm:>10.5f} {ev:>7}")
print()
print("Natural mortality alone over 1000 ticks, as a surviving fraction:")
for fid, fg in env.fgs.items():
    nm = float(getattr(fg, 'natural_mortality', 0.0) or 0.0)
    if nm > 0:
        print(f"  {fid:>18} (1-{nm})^1000 = {(1.0 - nm) ** 1000:.4f}")
