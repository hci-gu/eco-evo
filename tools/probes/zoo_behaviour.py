"""What a trained zooplankton policy does, in units the literature uses.

    python3 tools/probes/zoo_behaviour.py PROJECT CKPT_DIR SEED [YEARS] [BURN_IN] > out.json

World as tools/viability.py --spawn configured --mortality on
--migration on --currents on builds it (60x60, currents 0.1/20/0/12),
the trained policies from CKPT_DIR on every decision maker. After
BURN_IN years (default 1) it accumulates, weighted by zooplankton
biomass per cell:

  * monthly: action shares, stocks (t/km2), the share visible to
    herring (1 - rest * (1 - floor)), reserve fill, daily ration, and
    the grazing on phytoplankton against the phytoplankton's gross
    production that month;
  * diel: action shares per 6 h block and by sun above / below the
    horizon, per season;
  * per cell: rest share binned by reserve fill, by phytoplankton
    density and by the herring : zooplankton ratio in the cell.

Written for resume section 170 (does the zooplankton policy match
copepod behaviour?).
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

project, ckpt, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
years = int(sys.argv[4]) if len(sys.argv) > 4 else 3
burn_in = int(sys.argv[5]) if len(sys.argv) > 5 else 1

ZOO, PHYTO, HERRING = "zooplankton", "phytoplankton", "pelagic_fish"
TPD, TPY = 4, 1460

env = build_env(project, (60, 60), seed=seed, verbose=False,
                apply_natural_mortality=True, mass_balance=True,
                migration=True, currents=CurrentConfig(0.1, 20, 0, 12.0))
viability.install_behaviour(env, viability.POLICY, checkpoint_dir=ckpt)
i = env.dm_ids.index(ZOO)
order = list(env.global_fg_order)
env.build_static_caches()
# The floor the herring -> zooplankton pair uses (section 162: 0.5).
floor = (float(env.vis_floor_mat[env.dm_ids.index(HERRING), order.index(ZOO)])
         if env._has_pair_vis_floor
         else float(env._all_visibility_floor[order.index(ZOO)]))
me = float(env.dm_max_energy_reserve[i])
phyto_energy = float(env.fgs[PHYTO].params.get("energy_content", 0.0))
zoo_energy = float(env.fgs[ZOO].params.get("energy_content", 0.0))
cells = env.H * env.W
latitude = float(env.daylight.get("latitude_deg", 58.15))

# Phytoplankton gross production: the NDM growth step, recorded.
production = {"t": 0.0}
_ndm = population_change._apply_non_decision_maker_population_change


def _recording_ndm(env_, fg_id, fg):
    before = float(fg.biomass.sum()) if fg_id == PHYTO else 0.0
    _ndm(env_, fg_id, fg)
    if fg_id == PHYTO:
        production["t"] += float(fg.biomass.sum()) - before


population_change._apply_non_decision_maker_population_change = _recording_ndm


def month_of(year_tick):
    return daylight.month_of_day((year_tick * 6) // 24)


def zeros(*shape):
    return np.zeros(shape, dtype=np.float64)


monthly = {k: zeros(12) for k in (
    "w", "rest", "eat", "move", "vis_zoo", "vis_herr_w", "vis_herr",
    "fill", "zoo_t", "phyto_t", "herr_t", "intake_t", "prod_t", "ticks")}
season_of_month = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]   # DJF MAM JJA SON
diel = {k: zeros(4, 4) for k in ("w", "rest", "eat", "move")}   # season x block
sun = {k: zeros(4, 2) for k in ("w", "rest", "eat", "move")}    # season x night/day
FILL_BINS = np.linspace(0.0, 1.0, 21)
PHYTO_BINS = np.array([0, 1, 2, 5, 10, 20, 40, 80, 1e9], dtype=np.float64)
RATIO_BINS = np.array([0, 1e-6, 0.01, 0.03, 0.1, 0.3, 1, 1e9], dtype=np.float64)
binned = {name: {k: zeros(len(edges) - 1) for k in ("w", "rest", "eat")}
          for name, edges in (("fill", FILL_BINS), ("phyto", PHYTO_BINS),
                              ("herring_zoo", RATIO_BINS))}


def add_binned(name, edges, values, weight, rest, eat):
    idx = np.clip(np.digitize(values.ravel(), edges) - 1, 0, len(edges) - 2)
    w = weight.ravel()
    acc = binned[name]
    acc["w"] += np.bincount(idx, w, len(edges) - 1)
    acc["rest"] += np.bincount(idx, w * rest.ravel(), len(edges) - 1)
    acc["eat"] += np.bincount(idx, w * eat.ravel(), len(edges) - 1)


prev_intake = float(env.intake_by_pred_prey.get(ZOO, {}).get(PHYTO, 0.0)) \
    if getattr(env, "intake_by_pred_prey", None) else 0.0
for k in range(years * TPY):
    year_tick = interactions.light_index(env)
    month = month_of(year_tick)
    season = season_of_month[month]
    block = year_tick % TPD
    elevation = daylight.solar_elevation_deg(latitude, year_tick * 6 + 3)
    measure = k >= burn_in * TPY

    zoo = env.fgs[ZOO].biomass.astype(np.float64)
    herring = env.fgs[HERRING].biomass.astype(np.float64)
    phyto = env.fgs[PHYTO].biomass.astype(np.float64)
    reserve = env.fgs[ZOO].energy_reserve.astype(np.float64)

    actions = env.calculate_decisions()
    production["t"] = 0.0
    env.step(actions)
    intake_now = float(env.intake_by_pred_prey.get(ZOO, {}).get(PHYTO, 0.0))
    intake = intake_now - prev_intake
    prev_intake = intake_now
    if not measure:
        continue

    rest = actions.rest[i].astype(np.float64)
    eat = actions.eat[i].sum(axis=0).astype(np.float64)
    move = actions.move[i].sum(axis=0).astype(np.float64)
    visible = 1.0 - rest * (1.0 - floor)
    w = zoo
    W = w.sum()
    if W <= 0:
        continue
    fill = np.where(zoo > 0, reserve / np.maximum(zoo * me, 1e-12), 0.0)

    m = monthly
    m["w"][month] += W
    m["rest"][month] += (w * rest).sum()
    m["eat"][month] += (w * eat).sum()
    m["move"][month] += (w * move).sum()
    m["vis_zoo"][month] += (w * visible).sum()
    hz = herring * zoo
    m["vis_herr_w"][month] += hz.sum()
    m["vis_herr"][month] += (hz * visible).sum()
    m["fill"][month] += (w * fill).sum()
    m["zoo_t"][month] += W
    m["phyto_t"][month] += phyto.sum()
    m["herr_t"][month] += herring.sum()
    m["intake_t"][month] += intake
    m["prod_t"][month] += production["t"]
    m["ticks"][month] += 1

    for acc, col in ((diel, block), (sun, int(elevation > 0.0))):
        acc["w"][season, col] += W
        acc["rest"][season, col] += (w * rest).sum()
        acc["eat"][season, col] += (w * eat).sum()
        acc["move"][season, col] += (w * move).sum()

    add_binned("fill", FILL_BINS, fill, w, rest, eat)
    add_binned("phyto", PHYTO_BINS, phyto, w, rest, eat)
    add_binned("herring_zoo", RATIO_BINS,
               np.where(zoo > 0, herring / np.maximum(zoo, 1e-12), 0.0),
               w, rest, eat)


def share(acc, key):
    return (acc[key] / np.maximum(acc["w"], 1e-12)).tolist()


m = monthly
ticks = np.maximum(m["ticks"], 1)
out = {
    "seed": seed, "years": years, "burn_in": burn_in, "floor": floor,
    "zoo_energy_mj_t": zoo_energy, "phyto_energy_mj_t": phyto_energy,
    "monthly": {
        "rest": share(m, "rest"), "eat": share(m, "eat"),
        "move": share(m, "move"),
        "visible_mean": share(m, "vis_zoo"),
        "visible_seen_by_herring":
            (m["vis_herr"] / np.maximum(m["vis_herr_w"], 1e-12)).tolist(),
        "reserve_fill": share(m, "fill"),
        "zoo_t_km2": (m["zoo_t"] / ticks / cells).tolist(),
        "phyto_t_km2": (m["phyto_t"] / ticks / cells).tolist(),
        "herring_t_km2": (m["herr_t"] / ticks / cells).tolist(),
        # t phyto eaten per t zooplankton per day
        "ration_mass_per_day":
            (m["intake_t"] / np.maximum(m["zoo_t"], 1e-12) * TPD).tolist(),
        "grazing_over_gross_production":
            (m["intake_t"] / np.maximum(m["prod_t"], 1e-12)).tolist(),
        "grazing_per_day_over_stock":
            (m["intake_t"] / np.maximum(m["phyto_t"], 1e-12) * TPD).tolist(),
    },
    "diel_block": {k: share(diel, k) for k in ("rest", "eat", "move")},
    "sun_night_day": {k: share(sun, k) for k in ("rest", "eat", "move")},
    "binned": {name: {"edges": edges.tolist(),
                      "weight": binned[name]["w"].tolist(),
                      "rest": share(binned[name], "rest"),
                      "eat": share(binned[name], "eat")}
               for name, edges in (("fill", FILL_BINS),
                                   ("phyto", PHYTO_BINS),
                                   ("herring_zoo", RATIO_BINS))},
}
print(json.dumps(out))
