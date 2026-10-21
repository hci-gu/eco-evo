"""Tick-by-tick trace of the porpoises with the pzbpgp248nsrpfix9 weights.

Reproduces the Section 71.13 ``porp_ckpt.py`` probe against the run the
user just finished, so the post-fix-C picture can be compared with the
pre-fix numbers (death at tick 158-205).
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
TICKS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
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
env._batched_ready = False
for attr in ('_setup_batched_policies', '_prepare_batched_policies',
             '_build_batched_policies'):
    fn = getattr(env, attr, None)
    if callable(fn):
        fn()
        break

ip = env.dm_ids.index('porpoises')
jpf = env.global_fg_order.index('pelagic_fish')
porp = env.fgs['porpoises']
pf = env.fgs['pelagic_fish']

print(f"run={RUN}  porpoise b0={float(porp.biomass.sum()):.3f} t  "
      f"herring b0={float(pf.biomass.sum()):.1f} t")
print(f"{'tick':>5} {'B_porp':>9} {'cells':>5} {'s_x':>6} "
      f"{'eat':>5} {'mov':>5} {'rst':>5} "
      f"{'pf_in_cell':>10} {'pf_vis':>8} {'a_eff/a':>7} {'B_pf':>8}")

np.random.seed(20260530)
torch.manual_seed(20260530)

a = float(env.attack_rate_mat[ip, jpf]) if hasattr(env, 'attack_rate_mat') \
    else float(env.max_intake_mat[ip, jpf])
h = float(env.handling_time_mat[ip, jpf])
vf = float(env.vis_floor_mat[ip, jpf])

died_at = None
for t in range(TICKS):
    env.step()
    B = porp.biomass
    occ = B > 0
    n = int(occ.sum())
    tot = float(B.sum())
    if tot <= 0.0 and died_at is None:
        died_at = t
        print(f"  >>> porpoises extinct at tick {t}")
        break
    pi_e = float(env.pi_eat[ip, :, occ].sum(axis=-1).sum()) / max(n, 1) \
        if n else 0.0
    pi_m = float(env.pi_move[ip][:, occ].sum()) / max(n, 1) if n else 0.0
    pi_r = float(env.pi_rest[ip][occ].mean()) if n else 0.0
    # Biomass-weighted herring density in the cells the porpoises occupy.
    if n:
        w = B[occ] / tot
        pf_cell = float((pf.biomass[occ] * w).sum())
    else:
        pf_cell = 0.0
    rest_pf = float(env.pi_rest[env.dm_ids.index('pelagic_fish')][occ].mean()) \
        if n else 0.0
    pf_vis = pf_cell * (1.0 - rest_pf * (1.0 - vf))
    a_eff_ratio = 1.0 / (1.0 + a * h * pf_vis)
    s_x = float(porp.energy_reserve[occ].mean() / porp.max_energy_reserve) \
        if n else 0.0
    if t % 10 == 0 or t < 5:
        print(f"{t:>5} {tot:>9.4f} {n:>5} {s_x:>6.3f} "
              f"{pi_e:>5.3f} {pi_m:>5.3f} {pi_r:>5.3f} "
              f"{pf_cell:>10.3f} {pf_vis:>8.3f} {a_eff_ratio:>7.3f} "
              f"{float(pf.biomass.sum()):>8.1f}")

print(f"\ndied_at={died_at}  a={a}  h={h}  vis_floor={vf}  a*h={a*h}")
print(f"required visible herring for a_eff/a >= 0.84: "
      f"{(1.0 / 0.84 - 1.0) / (a * h):.2f} t/cell")
