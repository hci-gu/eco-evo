"""Realised intake per (predator, prey) pair with frozen policies.

    python3 tools/probes/pair_intake.py CKPT_DIR SEED [YEARS] > out.json

Probe world as train.py builds it (mareld2.yaml, 60x60, mortality,
migration, currents 0.1/20/0/12), the spawn/start-day seed overridden.
Per year: intake t per pair, mean and end biomass per FG, and monthly
intake per pair; plus each pair's static assimilated quality (MJ/t at
the prey's reference fill). Written for resume section 168.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch

import train as train_mod
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import interactions
from lib.environments.ecosystem_env.currents import CurrentConfig

ckpt, seed = sys.argv[1], int(sys.argv[2])
years = int(sys.argv[3]) if len(sys.argv) > 3 else 1


class Builder(train_mod._ProbeEnvBuilder):
    PROBE_SEED = seed


env = Builder(project_path='mareld2.yaml', grid_size=(60, 60),
              apply_natural_mortality=True, migration=True,
              currents=CurrentConfig(0.1, 20, 0, 12.0))()
env.policies, env.obs_mean, env.obs_var = load_policies_and_stats(env, ckpt, verbose=False)
env._batched_ready = False
for attr in ('_setup_batched_policies', '_prepare_batched_policies', '_build_batched_policies'):
    fn = getattr(env, attr, None)
    if callable(fn):
        fn()
        break
np.random.seed(seed)
torch.manual_seed(seed)

TPD, TPY = 4, 1460
order = list(env.global_fg_order)
quality = {(p, q): float(env.energy_gain_mat[i, order.index(q)])
           for i, p in enumerate(env.dm_ids) for q in order}
start_day = interactions.light_index(env) / TPD
out = {"seed": seed, "start_day": start_day, "years": [], "quality": {f"{p}->{q}": v for (p, q), v in quality.items() if v}}
prev = {p: dict(r) for p, r in env.intake_by_pred_prey.items()}
for y in range(years):
    bsum = {f: 0.0 for f in env.fgs}
    intake, monthly = {}, {}
    for t in range(TPY):
        month = min(11, int(interactions.light_index(env) / TPD / (365.0 / 12)))
        for f, fg in env.fgs.items():
            bsum[f] += float(fg.biomass.sum())
        env.step()
        for p, row in env.intake_by_pred_prey.items():
            for q, v in row.items():
                d = v - prev.get(p, {}).get(q, 0.0)
                if d:
                    k = f"{p}->{q}"
                    intake[k] = intake.get(k, 0.0) + d
                    monthly.setdefault(k, [0.0] * 12)[month] += d
        prev = {p: dict(r) for p, r in env.intake_by_pred_prey.items()}
    out["years"].append({"mean_biomass": {f: v / TPY for f, v in bsum.items()},
                         "end_biomass": {f: float(fg.biomass.sum()) for f, fg in env.fgs.items()},
                         "intake": intake, "monthly": monthly})
print(json.dumps(out))
