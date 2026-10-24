"""predator_share phi per prey pool for the predatory_fish group (stage x species scores).

Usage: python3 -I pred_prey_shares.py <predators_dir> [n_draws]

phi_j = sum_i w_i sum_k s_ik score_ikj, with
  w_i   long-run (2001-25) depth-stratified GOV biomass share of species i in the group
        (tables/pred_season.csv: pooled, Q1, Q3),
  s_ik  stage share of species i's biomass from tables/pred_length_biomass.csv
        (whiting < 20 cm juvenile; saithe >= 60 cm large; hake < 20 cm juvenile),
  score inputs/pred_scores.csv (min / central / max of access x capability; reused for saithe,
        whiting, hake from ../../Torskfiskarnas bytesandelar, judged for the others).
The zooplankton pool is f x krill-sized + (1 - f) x copepod-sized, f = krill share of the model's
zooplankton biomass (0.5, triangular 0.25-0.75; the pool is A17-calibrated, mostly copepods).
Joint caps: per stage max over pools, i.e. the share that can feed at the full rate on at least
one of the pools in the set. Monte Carlo: triangular scores and f, n_draws (default 4000).
"""
import os
import sys

import numpy as np
import pandas as pd

d = sys.argv[1]
ND = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
rng = np.random.default_rng(2026)
SC = pd.read_csv(os.path.join(d, "inputs", "pred_scores.csv"))
SE = pd.read_csv(os.path.join(d, "tables", "pred_season.csv"))
LB = pd.read_csv(os.path.join(d, "tables", "pred_length_biomass.csv"))
CUT = {"whiting": ("juvenile", "adult", 20), "saithe": ("adult", "large", 60),
       "hake": ("juvenile", "adult", 20)}
POOLS = ["benthos", "plank_fish", "zoo", "benth_fish"]
JOINT = {"plank_fish|zoo": ["plank_fish", "zoo"], "plank_fish|benth_fish": ["plank_fish", "benth_fish"],
         "any fish": ["plank_fish", "benth_fish"], "all four": POOLS}
sp_all = sorted(SC.species.unique())


def stage_rows(col):
    """(species, stage, weight) for the quarter column col (q1, q3, pooled)."""
    tot = SE.set_index("common").loc[sp_all, col]
    w = tot / tot.sum()
    rows = []
    for sp in sp_all:
        if sp in CUT:
            lo_st, hi_st, cut = CUT[sp]
            x = LB[LB.common == sp]
            s_hi = x[x.L5 >= cut][col].sum() / x[col].sum()
            rows += [(sp, lo_st, w[sp] * (1 - s_hi)), (sp, hi_st, w[sp] * s_hi)]
        else:
            rows.append((sp, "all", w[sp]))
    return rows


def tri(lo, c, hi, n):
    lo, c, hi = np.minimum(lo, c), c, np.maximum(hi, c)
    out = np.full(n, c, float)
    m = hi > lo
    if m:
        out = rng.triangular(lo, c, hi, n)
    return out


def phi(col, central=True):
    rows = stage_rows(col)
    n = 1 if central else ND
    f = np.full(n, 0.5) if central else rng.triangular(0.25, 0.5, 0.75, n)
    acc = {p: np.zeros(n) for p in POOLS}
    jt = {k: np.zeros(n) for k in JOINT}
    for sp, st, w in rows:
        s = {}
        for p in ["benthos", "plank_fish", "zoo_krill", "zoo_cop", "benth_fish"]:
            r = SC[(SC.species == sp) & (SC.stage == st) & (SC.pool == p)].iloc[0]
            s[p] = np.full(n, r.s_c) if central else tri(r.s_min, r.s_c, r.s_max, n)
        s["zoo"] = f * s["zoo_krill"] + (1 - f) * s["zoo_cop"]
        for p in POOLS:
            acc[p] += w * s[p]
        for k, ps in JOINT.items():
            jt[k] += w * np.max([s[p] for p in ps], axis=0)
    return acc, jt


res = []
for col in ["pooled", "q1", "q3"]:
    a, j = phi(col, True)
    am, jm = phi(col, False)
    for k in list(a) + list(j):
        c = (a.get(k) if k in a else j[k])[0]
        mc = am.get(k) if k in am else jm[k]
        res.append(dict(weights=col, pool=k, central=c, p05=np.percentile(mc, 5), p95=np.percentile(mc, 95)))
R = pd.DataFrame(res)
R.to_csv(os.path.join(d, "tables", "pred_phi.csv"), index=False)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
a, _ = phi("pooled", True)
print("\nsum of the four caps (pooled, central):", round(sum(v[0] for v in a.values()), 3))
# contributions by species, pooled central
rows = stage_rows("pooled")
con = {}
for sp, st, w in rows:
    for p in ["benthos", "plank_fish", "zoo_krill", "zoo_cop", "benth_fish"]:
        r = SC[(SC.species == sp) & (SC.stage == st) & (SC.pool == p)].iloc[0]
        con.setdefault(sp, {}).setdefault(p, 0.0)
        con[sp][p] += w * r.s_c
C = pd.DataFrame(con).T
C["zoo(f=0.5)"] = 0.5 * C.zoo_krill + 0.5 * C.zoo_cop
print("\ncontributions to phi by species (pooled, central):")
print(C.round(3).to_string())
