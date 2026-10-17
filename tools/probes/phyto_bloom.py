"""Why is there no phytoplankton spring bloom? Monthly phytoplankton under
variants of the grazing and the phytoplankton's own losses.

    python3 tools/probes/phyto_bloom.py PROJECT CKPT_DIR SEED VARIANT [YEARS] [BURN_IN]

VARIANT:
  base        trained policies everywhere
  nograze     zooplankton never eats (Rest), everyone else trained
  eat         zooplankton on the rig's eat rule (all mass on its prey)
  loss:X      base plus a phytoplankton loss of X per tick after growth
              (respiration / sinking / lysis, which the model lacks)

World as tools/probes/zoo_behaviour.py. Prints JSON with monthly means
(calendar months from the daylight calendar) over the years after
BURN_IN: phytoplankton and zooplankton t/km2, the light growth factor,
the phytoplankton's specific net growth per day from the growth step,
and grazing per day as a share of the phytoplankton stock. Written for
resume section 171.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

from inference import build_env
from lib.diagnostics import viability
from lib.environments.ecosystem_env import interactions, population_change
from lib.environments.ecosystem_env.currents import CurrentConfig
from lib.world import daylight

project, ckpt, seed, variant = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                sys.argv[4])
years = int(sys.argv[5]) if len(sys.argv) > 5 else 3
burn_in = int(sys.argv[6]) if len(sys.argv) > 6 else 1
ZOO, PHYTO = "zooplankton", "phytoplankton"
TPD, TPY = 4, 1460
loss = float(variant.split(":", 1)[1]) if variant.startswith("loss:") else 0.0

env = build_env(project, (60, 60), seed=seed, verbose=False,
                apply_natural_mortality=True, mass_balance=True,
                migration=True, currents=CurrentConfig(0.1, 20, 0, 12.0))
if variant == "eat":
    provider = viability.install_mixed(env, viability.EAT, ZOO, ckpt)
else:
    viability.install_behaviour(env, viability.POLICY, checkpoint_dir=ckpt)
    provider = None
iz = env.dm_ids.index(ZOO)
cells = env.H * env.W

growth = {"t": 0.0}
_ndm = population_change._apply_non_decision_maker_population_change


def _phyto_ndm(env_, fg_id, fg):
    before = float(fg.biomass.sum()) if fg_id == PHYTO else 0.0
    _ndm(env_, fg_id, fg)
    if fg_id == PHYTO:
        growth["t"] += float(fg.biomass.sum()) - before
        if loss > 0.0:
            fg.biomass = (fg.biomass * np.float32(1.0 - loss)).astype(
                env_.dtype, copy=False)


population_change._apply_non_decision_maker_population_change = _phyto_ndm

acc = {k: np.zeros(12) for k in ("n", "phyto", "zoo", "factor", "growth",
                                 "grazing")}
prev = float(env.intake_by_pred_prey.get(ZOO, {}).get(PHYTO, 0.0))
for k in range(years * TPY):
    year_tick = interactions.light_index(env)
    month = daylight.month_of_day((year_tick * 6) // 24)
    phyto = float(env.fgs[PHYTO].biomass.sum())
    zoo = float(env.fgs[ZOO].biomass.sum())
    factor = interactions.growth_light(env, PHYTO)
    actions = provider(env) if provider else env.calculate_decisions()
    if variant == "nograze":
        actions.rest[iz] = 1.0
        actions.eat[iz] = 0.0
        actions.move[iz] = 0.0
    growth["t"] = 0.0
    env.step(actions)
    now = float(env.intake_by_pred_prey.get(ZOO, {}).get(PHYTO, 0.0))
    grazed, prev = now - prev, now
    if k < burn_in * TPY:
        continue
    acc["n"][month] += 1
    acc["phyto"][month] += phyto / cells
    acc["zoo"][month] += zoo / cells
    acc["factor"][month] += factor
    acc["growth"][month] += growth["t"] / max(phyto, 1e-9) * TPD
    acc["grazing"][month] += grazed / max(phyto, 1e-9) * TPD

n = np.maximum(acc["n"], 1)
print(json.dumps({"seed": seed, "variant": variant,
                  **{k: (v / n).tolist() for k, v in acc.items() if k != "n"}}))
