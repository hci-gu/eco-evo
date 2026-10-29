"""Encounter-only diet of each water-column member under shared handling, at the probe world's start
state, for a range of member accessible shares f on the benthivores.

Usage (from the repository root): python3 <this> [f_list]

Same construction as the 223.27 shared-handling test (library h, pair f on the other prey, the
infauna refuge, no hiding, annual-mean light), but each member's own per-tonne encounter a per prey
from ../../Rovfiskens uppdelning/tables/split_encounter_223_stdout.txt ("Per-stage central a",
"layer" membership: a counts only where the capability score >= 0.1). Under a shared denominator
the diet share is a_j f_j B_j / sum_k a_k f_k B_k per cell, so h only enters through the common
denominator. Diet by mass.
"""
import os
import sys

import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
import train as train_mod  # noqa: E402
from lib.environments.ecosystem_env.currents import CurrentConfig  # noqa: E402
from lib.environments.ecosystem_env import refuge  # noqa: E402


class B(train_mod._ProbeEnvBuilder):
    PROBE_SEED = 20260530


env = B(project_path="mareld2.yaml", grid_size=(60, 60), apply_natural_mortality=True,
        migration=True, currents=CurrentConfig(0.1, 20, 0, 12.0))()
order = env.global_fg_order
PRED = "predatory_fish"
i = env.dm_ids.index(PRED)
bp = env.fgs[PRED].biomass.astype(np.float64)
Bp = np.stack([env.fgs[f].biomass for f in order]).astype(np.float64)
vis = refuge.accessible_rows(env, Bp.astype(np.float32)).astype(np.float64)
PREY = ["planktivorous_fish", "zooplankton", "benthivorous_fish", "whiting"]
js = [order.index(p) for p in PREY]
m3 = np.asarray(env._type3_pred_mask)
# per-stage central a (km2 per member tonne per tick), layer membership (score >= 0.1), and W
MEMBERS = {
    "saithe <60": (0.2588, [0.007607, 0.0003354, 0.08048, 0.0161]),
    "saithe >=60": (0.1506, [0.002889, 0.0001335, 0.03158, 0.006316]),
    "spurdog": (0.0500, [0.003369, 0.0001674, 0.06879, 0.01376]),
    "hake adult": (0.0245, [0.005702, 0.0002937, 0.07298, 0.0146]),
    "anglerfish": (0.0137, [0.0003812, 0.0, 0.02685, 0.00537]),
    "velvet belly": (0.0047, [0.01668, 0.000705, 0.3055, 0.06111]),
    "ling": (0.0070, [0.002139, 0.0, 0.0311, 0.006219]),
    "pollack": (0.0041, [0.003625, 0.0001647, 0.03916, 0.007832]),
    "mackerel": (0.0802, [0.04456, 0.002312, 0.0, 0.0]),
    "horse mackerel": (0.0345, [0.02648, 0.00204, 0.0, 0.0]),
}
fs = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.24, 0.043, 0.1, 0.2, 0.3, 0.5, 0.74]


def diet(avec, f_ben, a_scale=1.0):
    den = 1.0
    num = {}
    for k, j in enumerate(js):
        a = avec[k] * (a_scale if PREY[k] == "benthivorous_fish" else 1.0)
        h = float(env.handling_time_mat[i, j])
        f = f_ben if PREY[k] == "benthivorous_fish" else float(env.access_mat[i, j])
        v = vis[j] * f
        t3 = float(np.ravel(m3[i, j] if m3.ndim > 1 else m3[i])[0])
        x = v * v if t3 > 0.5 else v
        num[k] = a * x
        den = den + a * h * x
    intake = [float((bp * num[k] / den).sum()) for k in range(len(js))]
    tot = sum(intake)
    return [v / tot for v in intake] if tot > 0 else [0.0] * len(js)


print("check, library pair a and f:", ", ".join(
    f"{p} {100 * s:.1f} %" for p, s in zip(PREY, diet([float(env.max_intake_mat[i, j]) for j in js],
                                                       float(env.access_mat[i, js[2]])))))
print("\nbenthivore share of each member's encounter diet (%) by f on the benthivores:")
print("member          " + " ".join(f"f={f:<5}" for f in fs))
for m, (W, avec) in MEMBERS.items():
    print(f"{m:15s} " + " ".join(f"{100 * diet(avec, f)[2]:7.1f}" for f in fs))
