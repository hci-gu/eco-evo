"""Is the trained zooplankton's resting individually rational?

    python3 tools/probes/zoo_tradeoff.py PROJECT CKPT_DIR SEED [YEARS] [BURN_IN]

Same world as tools/probes/zoo_behaviour.py. Per cell and tick, for one
tonne of zooplankton choosing eat over rest (Rest is the hide action):

  gain  = (intake at its appetite - extra feeding cost) / price of a
          tonne of new biomass at the maintenance reserve, i.e. the
          fraction of its biomass the extra energy buys this tick;
  risk  = mu_vis * (1 - floor), the extra fraction herring eat when it
          is fully visible instead of hidden.

A selfish individual eats where gain > risk. The trained policy
maximises the GROUP's energy, so where it rests although gain > risk
the restraint is not individually rational. mu_vis is the herring's
per-unit removal rate on visible zooplankton (Holling response with
this tick's light multiplier, the herring's own hunger and eat share),
checked against the realised herring intake. Written for resume
section 170.
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
from lib.environments.ecosystem_env import interactions
from lib.environments.ecosystem_env.currents import CurrentConfig
from lib.world import daylight

project, ckpt, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
years = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
burn_in = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
ZOO, PHYTO, HERRING = "zooplankton", "phytoplankton", "pelagic_fish"
TPY = 1460

env = build_env(project, (60, 60), seed=seed, verbose=False,
                apply_natural_mortality=True, mass_balance=True,
                migration=True, currents=CurrentConfig(0.1, 20, 0, 12.0))
viability.install_behaviour(env, viability.POLICY, checkpoint_dir=ckpt)
order = list(env.global_fg_order)
iz, ih = env.dm_ids.index(ZOO), env.dm_ids.index(HERRING)
jz = order.index(ZOO)
floor = (float(env.vis_floor_mat[ih, jz]) if env._has_pair_vis_floor
         else float(env._all_visibility_floor[jz]))
zoo_fg = env.fgs[ZOO]
price = (float(zoo_fg.params["energy_content"])
         + zoo_fg.maintenance_level * zoo_fg.max_energy_reserve)
rest_over_eat = float(env.dm_cost_rest[iz] / env.dm_cost_eat[iz])
latitude = float(env.daylight.get("latitude_deg", 58.15))
season_of_month = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]

acc = {k: np.zeros((4, 2)) for k in (
    "w", "gain", "risk", "better_eat_w", "rest_where_eat_better",
    "eat_where_rest_better", "rest", "est_herring_t", "real_herring_t")}
prev = float(env.intake_by_pred_prey.get(HERRING, {}).get(ZOO, 0.0))
for k in range(int(years * TPY)):
    year_tick = interactions.light_index(env)
    season = season_of_month[daylight.month_of_day((year_tick * 6) // 24)]
    day = int(daylight.solar_elevation_deg(latitude, year_tick * 6 + 3) > 0)
    measure = k >= burn_in * TPY

    actions = env.calculate_decisions()
    if measure:
        zoo = zoo_fg.biomass.astype(np.float64)
        herring = env.fgs[HERRING].biomass.astype(np.float64)
        hunger_z = zoo_fg.get_hunger().astype(np.float64)
        hunger_h = env.fgs[HERRING].get_hunger().astype(np.float64)
        payoff = viability._intake_payoff(env)[iz].astype(np.float64)
        extra_cost = (viability._feeding_cost(env)[iz].astype(np.float64)
                      * (1.0 - rest_over_eat))
        gain = (payoff * hunger_z - extra_cost) / price
        per_prey = viability._per_prey_intake(env)[ih, jz].astype(np.float64)
        share_h = actions.eat[ih, jz].astype(np.float64)
        removal_visible = herring * hunger_h * share_h * per_prey   # t/tick
        mu_vis = np.where(zoo > 0, removal_visible / np.maximum(zoo, 1e-12),
                          0.0)
        mu_vis = np.clip(mu_vis, 0.0, 1.0)
        risk = mu_vis * (1.0 - floor)
        rest = actions.rest[iz].astype(np.float64)
        eat = actions.eat[iz].sum(axis=0).astype(np.float64)
        better = gain > risk
        visible = 1.0 - rest * (1.0 - floor)
        a = acc
        a["w"][season, day] += zoo.sum()
        a["gain"][season, day] += (zoo * gain).sum()
        a["risk"][season, day] += (zoo * risk).sum()
        a["better_eat_w"][season, day] += zoo[better].sum()
        a["rest_where_eat_better"][season, day] += (zoo * rest)[better].sum()
        a["eat_where_rest_better"][season, day] += (zoo * eat)[~better].sum()
        a["rest"][season, day] += (zoo * rest).sum()
        a["est_herring_t"][season, day] += (removal_visible * visible).sum()
    env.step(actions)
    now = float(env.intake_by_pred_prey.get(HERRING, {}).get(ZOO, 0.0))
    if measure:
        acc["real_herring_t"][season, day] += now - prev
    prev = now

w = np.maximum(acc["w"], 1e-12)
bw = np.maximum(acc["better_eat_w"], 1e-12)
out = {
    "seed": seed, "floor": floor, "price_mj_per_t": price,
    "rows": "season DJF MAM JJA SON", "cols": "night, day",
    "gain_per_tick": (acc["gain"] / w).tolist(),
    "risk_per_tick": (acc["risk"] / w).tolist(),
    "share_where_eating_pays": (acc["better_eat_w"] / w).tolist(),
    "rest_share_where_eating_pays": (acc["rest_where_eat_better"] / bw).tolist(),
    "eat_share_where_resting_pays":
        (acc["eat_where_rest_better"] / np.maximum(acc["w"] - acc["better_eat_w"], 1e-12)).tolist(),
    "rest_share": (acc["rest"] / w).tolist(),
    "herring_intake_estimated_over_realised":
        (acc["est_herring_t"] / np.maximum(acc["real_herring_t"], 1e-12)).tolist(),
}
print(json.dumps(out))
