"""Per-tick ENERGY BUDGET for the porpoises with trained weights.

The first version of this probe divided ``energy_reserve`` by
``max_energy_reserve`` only. That is wrong: ``energy_reserve`` is
R = B * E (total MJ in the cell), so the state variable the growth step
uses is ``fg.energy_level`` = (R / B) / ME. This version uses that, and
additionally books the two flows that decide q_x:

  gain  = temp_energy_gains / B          (MJ per tonne, from predation)
  cost  = rm * (pi_rest*c_rest + pi_eat*c_eat + pi_move*c_move)
                                          (MJ per tonne, metabolism)

so ds = (gain - cost) / ME per tick, and q_x = s_x - u_x drives growth.
Also reports the Holling SATURATION f/f_max = a*h*B_vis/(1 + a*h*B_vis),
which is the quantity Section 69's "0.84" requirement refers to - NOT
a_eff/a, which moves in the opposite direction.
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

RUN = sys.argv[1] if len(sys.argv) > 1 else 'pzbpgp248nsrpfix9'
TICKS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
H = W = 16

builder = train_mod._ProbeEnvBuilder(
    project_path='mareld2.yaml', grid_size=(H, W),
    apply_natural_mortality=True, migration=True)
env = builder()
policies, mean, var = load_policies_and_stats(
    env, os.path.join('results', RUN), verbose=False)
env.policies = policies
env.obs_mean = mean
env.obs_var = var

ip = env.dm_ids.index('porpoises')
jpf = env.global_fg_order.index('pelagic_fish')
porp = env.fgs['porpoises']
pf = env.fgs['pelagic_fish']
rm = float(porp.resting_metabolism)
c_eat = float(porp.feeding_cost)
c_move = float(porp.movement_cost)
c_rest = float(porp.resting_cost)
ME = float(porp.max_energy_reserve)
u_x = float(porp.maintenance_level)
a = float(env.max_intake_mat[ip, jpf])
h = float(env.handling_time_mat[ip, jpf])
vf = float(env.vis_floor_mat[ip, jpf])
gain_per_t = float(env.energy_gain_mat[ip, jpf])

print(f"run={RUN}  rm={rm} c_eat={c_eat} c_move={c_move} c_rest={c_rest}")
print(f"ME={ME} u_x={u_x} a={a} h={h} a*h={a*h:.4f} vis_floor={vf} "
      f"energy_gain_per_tonne_prey={gain_per_t}")
print(f"max gross intake  = a*gain = {a * gain_per_t:.1f} MJ/t/tick "
      f"(at sat=1, hunger=1)")
print(f"pure-eat cost     = rm*c_eat = {rm * c_eat:.1f} MJ/t/tick")
print(f"pure-move cost    = rm*c_move = {rm * c_move:.1f} MJ/t/tick")
print(f"break-even s_x    = u_x = {u_x}  ->  hunger at u_x = "
      f"{max(0.0, 1.0 - u_x / 0.9):.4f}")
print()
print(f"{'tick':>5} {'B':>8} {'cells':>5} {'s_x':>6} {'q_x':>7} "
      f"{'hung':>5} {'gain':>7} {'cost':>7} {'net':>7} "
      f"{'eat':>5} {'mov':>5} {'sat':>5} {'pf/cell':>8} {'B_pf':>7}")

np.random.seed(20260530)
torch.manual_seed(20260530)

for t in range(TICKS):
    env.step()
    B = porp.biomass
    occ = B > 0
    n = int(occ.sum())
    tot = float(B.sum())
    if tot <= 0.0:
        print(f"  >>> porpoises extinct at tick {t}")
        break
    w = B[occ] / tot
    s_x = float((porp.energy_level[occ] * w).sum())
    hung = float((porp.get_hunger()[occ] * w).sum())
    pi_e = float((env.pi_eat[ip][:, occ].sum(axis=0) * w).sum())
    pi_m = float((env.pi_move[ip][:, occ].sum(axis=0) * w).sum())
    pi_r = float((env.pi_rest[ip][occ] * w).sum())
    cost = rm * (pi_r * c_rest + pi_e * c_eat + pi_m * c_move)
    pf_cell = float((pf.biomass[occ] * w).sum())
    pf_rest = float((env.pi_rest[env.dm_ids.index('pelagic_fish')][occ]
                     * w).sum())
    pf_vis = pf_cell * (1.0 - pf_rest * (1.0 - vf))
    sat = a * h * pf_vis / (1.0 + a * h * pf_vis)
    gain = a * sat * hung * gain_per_t
    if t % 25 == 0 or t < 4:
        print(f"{t:>5} {tot:>8.4f} {n:>5} {s_x:>6.3f} {s_x - u_x:>7.3f} "
              f"{hung:>5.3f} {gain:>7.1f} {cost:>7.1f} {gain - cost:>7.1f} "
              f"{pi_e:>5.3f} {pi_m:>5.3f} {sat:>5.3f} {pf_cell:>8.2f} "
              f"{float(pf.biomass.sum()):>7.1f}")

print(f"\nfinal B_porp={float(porp.biomass.sum()):.5f}  "
      f"herring={float(pf.biomass.sum()):.1f}")
print("Holling saturation needed for gain >= cost at hunger=h(u_x):")
hu = max(0.0, 1.0 - u_x / 0.9)
need_sat = (rm * c_eat) / (a * gain_per_t * hu) if hu > 0 else float('inf')
print(f"  sat_min = rm*c_eat / (a*gain*h(u_x)) = {need_sat:.4f}")
if need_sat < 1.0:
    need_vis = need_sat / ((1.0 - need_sat) * a * h)
    print(f"  -> visible herring needed = {need_vis:.3f} t/cell")
else:
    print("  -> IMPOSSIBLE: eating at full satiation cannot pay the "
          "feeding cost at the break-even reserve level")
