"""Individuals per tonne (N/t), biomass-median length and stage shares of the
predatory_fish members from the long-run NS-IBTS GOV length biomass table.

Usage: python3 -I pred_numbers_per_tonne.py <pred_length_biomass.csv> <length_weight_all.csv> [out.csv]
The number-weighted mean length is the length at which a clearance rate linear in L (v = BL/s x L)
gives the stage's clearance per tonne exactly: sum_c (b_c/W_c) L_c / sum_c (b_c/W_c).
N/t = sum_c B_c / sum_c (B_c / W(L_mid)) inverted: N per tonne = sum_c (b_c / W_c) * 1e6,
b_c = biomass share in 5-cm class c, W_c = a L_mid^b (g), L_mid = L5 + 2.5 cm.
"""
import sys
import pandas as pd

lb = pd.read_csv(sys.argv[1])
lw = pd.read_csv(sys.argv[2], index_col=0)
members = ["saithe", "whiting", "mackerel", "spurdog", "horse mackerel", "hake",
           "anglerfish", "ling", "velvet belly", "pollack"]
stage_cut = {"whiting": 20, "saithe": 60, "hake": 20}  # cm, as in params_predatory_fish.md
print("species, stage, biomass share of species, N per tonne, mean W (g), biomass-median L (cm), number-weighted mean L (cm)")
rows = []
for sp in members:
    d = lb[lb.common == sp].copy()
    if d.empty:
        print(sp, "no rows"); continue
    ap = int(d.aphia.iloc[0])
    a, b = lw.loc[ap, "a"], lw.loc[ap, "b"]
    d["Lmid"] = d.L5 + 2.5
    d["W"] = a * d.Lmid ** b
    d = d[d.pooled > 0]
    tot = d.pooled.sum()
    stages = [("all", d)]
    if sp in stage_cut:
        c = stage_cut[sp]
        lab = ("juvenile", "adult") if sp != "saithe" else ("adult", "large")
        stages += [(lab[0], d[d.L5 < c]), (lab[1], d[d.L5 >= c])]
    for st, s in stages:
        share = s.pooled.sum() / tot
        n_per_t = (s.pooled / s.W).sum() / s.pooled.sum() * 1e6
        cum = s.sort_values("L5").pooled.cumsum() / s.pooled.sum()
        med = s.sort_values("L5").Lmid[cum.values >= 0.5].iloc[0]
        nw = s.pooled / s.W
        lnum = (nw * s.Lmid).sum() / nw.sum()
        print(f"{sp}, {st}, {share:.3f}, {n_per_t:.0f}, {1e6 / n_per_t:.0f}, {med:.1f}, {lnum:.1f}")
        rows.append((sp, st, share, n_per_t, 1e6 / n_per_t, med, lnum))
pd.DataFrame(rows, columns=["species", "stage", "share", "n_per_t", "mean_w_g", "median_L_cm", "num_mean_L_cm"]).to_csv(
    sys.argv[3] if len(sys.argv) > 3 else "/dev/null", index=False)
