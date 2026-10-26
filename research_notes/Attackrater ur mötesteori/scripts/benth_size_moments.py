"""Individuals per tonne and length moments of the benthivorous_fish members (long-run 2001-25).

Usage: python3 -I benth_size_moments.py <length_biomass_longrun.csv> <stage_shares_longrun.csv>

Input: ../../Fiskgrupper i rutan/benthivores/tables/length_biomass_longrun.csv (NS-IBTS GOV Q1+Q3,
depth-stratified, 1-cm classes, long-run biomass share per class and the class weight w_g).

Per species (and per stage) the per-tonne sums that an encounter model needs:
  N/t      = sum_L s_L * 1e6 / W_L                (individuals per tonne)
  M1       = sum_L s_L * 1e6 / W_L * L            (m of body length per tonne)
  M2       = sum_L s_L * 1e6 / W_L * L^2          (m2 of L^2 per tonne)
  M3       = sum_L s_L * 1e6 / W_L * L^3
with L in metres. If speed ~ L (BL/s) and path width ~ L, the area searched per tonne is
c * M2; if width is a fixed distance, c' * M1; with a 3-D cylinder pi R^2 v and R ~ L, c'' * M3.
Effective lengths: L_eff1 = M1/N, L_eff2 = sqrt(M2/N), L_eff3 = (M3/N)^(1/3).
"""
import sys

import numpy as np
import pandas as pd

lb = pd.read_csv(sys.argv[1])
st = pd.read_csv(sys.argv[2])
CUTS = {"cod": [30, 50], "haddock": [40], "long rough dab": [20], "dab": [20], "plaice": [25],
        "witch": [30], "lemon sole": [25], "grey gurnard": [25], "starry ray": [30]}


def moments(d):
    s = d.share_2001_25 / d.share_2001_25.sum()
    n = s * 1e6 / d.w_g
    L = (d.L_cm + 0.5) / 100.0  # class mid-point, m
    return dict(N_per_t=n.sum(), M1=(n * L).sum(), M2=(n * L ** 2).sum(), M3=(n * L ** 3).sum(),
                stage_share=d.share_2001_25.sum())


rows = []
for sp, d in lb.groupby("species"):
    d = d[d.share_2001_25 > 0]
    tot = d.share_2001_25.sum()
    m = moments(d)
    rows.append(dict(species=sp, stage="all", **m))
    edges = [0] + CUTS.get(sp, []) + [1000]
    for lo, hi in zip(edges[:-1], edges[1:]):
        dd = d[(d.L_cm >= lo) & (d.L_cm < hi)]
        if dd.empty:
            continue
        m = moments(dd)
        m["stage_share"] = dd.share_2001_25.sum() / tot
        rows.append(dict(species=sp, stage=f"{lo}-{hi if hi < 1000 else ''}", **m))
out = pd.DataFrame(rows)
out["L_eff1_cm"] = 100 * out.M1 / out.N_per_t
out["L_eff2_cm"] = 100 * np.sqrt(out.M2 / out.N_per_t)
out["L_eff3_cm"] = 100 * np.cbrt(out.M3 / out.N_per_t)
out["mean_W_g_numbers"] = 1e6 / out.N_per_t
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False))
print("\nCheck: stage shares vs stage_shares_longrun.csv")
print(st.round(3).to_string(index=False))
