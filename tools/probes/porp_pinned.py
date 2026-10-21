"""Decisive experiment: is the porpoise death POSITIONAL or BUDGETARY?

Three variants, identical world (the deterministic probe world), all with
the porpoise policy overridden to "eat 100 %, never move":

  spread   the porpoise keeps whatever cells the trained policy gave it
           after its first few moving ticks (baseline reproduction)
  pinned   the porpoise biomass is CONCENTRATED in the k cells with the
           most herring at t=0 and stays there (zero dilution)
  perfect  same as pinned, but re-concentrated onto the current top-k
           herring cells every tick (an oracle pursuer: what fix A + B
           could achieve at best, with movement free)

If ``pinned`` survives, the killer is dilution and the fix belongs in the
movement representation (Section 71 fixes A/B). If even ``perfect`` dies,
the energy budget itself is mis-calibrated and no movement fix can help.
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
MODE = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5


def build():
    builder = train_mod._ProbeEnvBuilder(
        project_path='mareld2.yaml', grid_size=(H, W),
        apply_natural_mortality=(os.environ.get('MORT', '1') == '1'),
        migration=True)
    env = builder()
    policies, mean, var = load_policies_and_stats(
        env, os.path.join('results', RUN), verbose=False)
    env.policies = policies
    env.obs_mean = mean
    env.obs_var = var
    return env


def concentrate(porp, cells):
    """Move ALL porpoise biomass and reserve into ``cells`` (row, col)."""
    tot_b = float(porp.biomass.sum())
    tot_r = float(porp.energy_reserve.sum())
    porp.biomass = np.zeros((H, W), dtype=np.float32)
    porp.energy_reserve = np.zeros((H, W), dtype=np.float32)
    n = len(cells[0])
    for r, c in zip(*cells):
        porp.biomass[r, c] = tot_b / n
        porp.energy_reserve[r, c] = tot_r / n


def top_cells(field, k):
    flat = np.argsort(field.ravel())[::-1][:k]
    return np.unravel_index(flat, field.shape)


env = build()
porp = env.fgs['porpoises']
# Candidate recalibrations, applied in memory only (no YAML change yet).
# U_X    : maintenance_level, the reserve fill fraction at break-even.
# SAT_S  : satiation_scale, where the hunger gate closes.
# A_MULT : multiplier on max_intake_rate, with handling_time kept at
#          1/a so the Holling ceiling still equals the physiological max
#          (the fg_library convention).
import os as _os
_u = _os.environ.get('U_X')
if _u:
    porp.maintenance_level = float(_u)
_s = _os.environ.get('SAT_S')
if _s:
    porp.satiation_scale = float(_s)
_am = _os.environ.get('A_MULT')
if _am:
    m = float(_am)
    _i = env.dm_ids.index('porpoises')
    env.max_intake_mat[_i, :] *= m
    env.handling_time_mat[_i, :] /= m
pf = env.fgs['pelagic_fish']
ip = env.dm_ids.index('porpoises')
jpf = env.global_fg_order.index('pelagic_fish')
u_x = float(porp.maintenance_level)

np.random.seed(20260530)
torch.manual_seed(20260530)

if MODE in ('pinned', 'perfect'):
    concentrate(porp, top_cells(pf.biomass, K))

print(f"mode={MODE} K={K}  B0={float(porp.biomass.sum()):.4f} t  "
      f"cells={int((porp.biomass > 0).sum())}")
print(f"{'tick':>5} {'B':>9} {'cells':>5} {'s_x':>6} {'q_x':>7} "
      f"{'pf/cell':>8} {'B_pf':>7}")

_orig = env._calculate_decisions


def forced_decisions(*args, **kwargs):
    """Run the real decision step, then override the porpoise row with
    pure eating. Keeps every other DM (and therefore the herring field)
    exactly as the trained policies dictate."""
    out = _orig(*args, **kwargs)
    env.pi_move[ip] = 0.0
    env.pi_rest[ip] = 0.0
    env.pi_eat[ip] = 0.0
    env.pi_eat[ip, jpf] = 1.0
    return out


env._calculate_decisions = forced_decisions

died = None
for t in range(TICKS):
    if MODE == 'perfect' and float(porp.biomass.sum()) > 0.0:
        concentrate(porp, top_cells(pf.biomass, K))
    env.step()
    B = porp.biomass
    tot = float(B.sum())
    if tot <= 0.0:
        died = t
        print(f"  >>> extinct at tick {t}")
        break
    occ = B > 0
    w = B[occ] / tot
    s_x = float((porp.energy_level[occ] * w).sum())
    pf_cell = float((pf.biomass[occ] * w).sum())
    if t % 50 == 0:
        print(f"{t:>5} {tot:>9.4f} {int(occ.sum()):>5} {s_x:>6.3f} "
              f"{s_x - u_x:>7.3f} {pf_cell:>8.2f} "
              f"{float(pf.biomass.sum()):>7.1f}")

print(f"\nmode={MODE} K={K} died_at={died} "
      f"final_B={float(porp.biomass.sum()):.5f} "
      f"final_herring={float(pf.biomass.sum()):.1f}")
