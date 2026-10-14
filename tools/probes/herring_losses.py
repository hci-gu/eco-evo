"""Per-year herring loss budget and gadoid diet, frozen policies, probe world.

    python3 tools/probes/herring_losses.py CKPT_DIR YEARS [G_SCALE]

Per year: herring biomass, losses by channel (gadoids, porpoises, herring,
M1, starvation) as tonnes and as a rate on the mean biomass, GROSS
production (dB + every loss, so starvation - the negative branch of the
growth term - is added back; net = prod - starv), mean reserve fill s,
and the gadoid diet by mass. G_SCALE multiplies the herring growth_rate
in memory (resume section 160).
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
from lib.environments.ecosystem_env.currents import CurrentConfig

ckpt, years = sys.argv[1], int(sys.argv[2])
env = train_mod._ProbeEnvBuilder(
    project_path='mareld2.yaml', grid_size=(60, 60),
    apply_natural_mortality=True, migration=True,
    currents=CurrentConfig(0.1, 20, 0, 12.0))()
env.policies, env.obs_mean, env.obs_var = load_policies_and_stats(env, ckpt, verbose=False)
env._batched_ready = False
for attr in ('_setup_batched_policies', '_prepare_batched_policies', '_build_batched_policies'):
    fn = getattr(env, attr, None)
    if callable(fn):
        fn()
        break
K = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
env.fgs['pelagic_fish'].growth_rate *= K
print(f"herring growth_rate x {K}")
np.random.seed(20260530)
torch.manual_seed(20260530)

H, G = 'pelagic_fish', 'gadoids'
YEAR = 1460


def snap():
    return dict(
        B={k: float(f.biomass.sum()) for k, f in env.fgs.items()},
        intake={p: dict(r) for p, r in env.intake_by_pred_prey.items()},
        pred=dict(env.loss_predation), starv=dict(env.loss_starvation),
        nat=dict(env.loss_natural), imp=dict(env.loss_impact))


prev = snap()
print("year | herring B start->end, mean | herring losses t/yr (rate on mean B): "
      "by gadoids, by porpoises, by herring, M1, starv | production | "
      "gadoid diet mass shares")
for y in range(years):
    Bsum = {k: 0.0 for k in env.fgs}
    ssum = 0.0
    for _ in range(YEAR):
        env.step()
        for k, f in env.fgs.items():
            Bsum[k] += float(f.biomass.sum())
        fh = env.fgs[H]
        ssum += float((fh.biomass * fh.energy_level).sum()) / max(float(fh.biomass.sum()), 1e-30)
    cur = snap()
    d = lambda a, k1, k2=None: (cur[a][k1].get(k2, 0.0) - prev[a][k1].get(k2, 0.0)) if k2 else (cur[a][k1] - prev[a][k1])
    Bm = Bsum[H] / YEAR
    by_g = d('intake', G, H)
    by_p = d('intake', 'porpoises', H)
    by_h = d('intake', H, H)
    m1 = d('nat', H)
    st = d('starv', H)
    loss = d('pred', H) + m1 + st + d('imp', H)
    prod = cur['B'][H] - prev['B'][H] + loss
    gd = {k: d('intake', G, k) for k in cur['intake'].get(G, {})}
    tot = sum(gd.values()) or 1.0
    Gm = Bsum[G] / YEAR
    print(f"{y+1:3d} | {prev['B'][H]:6.0f}->{cur['B'][H]:6.0f} m{Bm:6.0f} | "
          f"G {by_g:6.0f} ({by_g/Bm:.2f})  P {by_p:4.0f} ({by_p/Bm:.3f})  "
          f"H {by_h:5.0f} ({by_h/Bm:.2f})  M1 {m1:5.0f} ({m1/Bm:.2f})  S {st:4.0f} | "
          f"prod {prod:6.0f} ({prod/Bm:.2f}) s {ssum/YEAR:.3f} | gadoids m{Gm:5.0f} eat {tot:6.0f} ({tot/Gm:.1f}/yr): "
          + " ".join(f"{k[:5]} {v/tot:.0%}" for k, v in sorted(gd.items(), key=lambda kv: -kv[1])),
          flush=True)
    prev = cur
