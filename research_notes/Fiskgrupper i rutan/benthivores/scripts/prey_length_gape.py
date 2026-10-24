"""Prey length structure near the bottom in the grid region (NS-IBTS GOV, Q1 and Q3, 2001-2025) and a
"can this predator stage swallow this prey" table from the gape rule prey_length <= k x predator_length.

Usage: python3 -I prey_length_gape.py <cpuel_2000_2020.csv.gz> <cpuel_2021_2025.csv.gz> <length_weight_all.csv>
                                      <gape_k.csv> <out_dir>
Hauls: 57.0-59.2 N, 7.5-11.5 E, depth 50-300 m (the grid and its surroundings). Simple mean over hauls
(no depth stratification; a judgement adequate for length structure). The GOV catches the part of the
pelagic stock that is near the bottom by day. Length class = LngtClas/10 + half a class (cm).
Weight W = a L^b from length_weight_all.csv (CA fits where available).
gape_k.csv columns: predator, k_low, k_c, k_high, stages (space-separated predator lengths in cm), basis.
"""
import sys, os, csv
import pandas as pd, numpy as np
c1, c2, lwf, kf, out = sys.argv[1:6]
PREY = {126444: "norway_pout", 126417: "herring", 126425: "sprat", 126439: "blue_whiting",
        126438: "whiting", 126441: "saithe", 126437: "haddock", 126436: "cod", 127137: "long_rough_dab",
        126445: "poor_cod", 125516: "sandeel_Ammodytes_marinus", 126751: "sandeel_Ammodytes_tobianus",
        150637: "grey_gurnard", 105865: "starry_ray", 127139: "dab", 127143: "plaice"}
cp = pd.concat([pd.read_csv(c1), pd.read_csv(c2)], ignore_index=True)
cp = cp[(cp.Year >= 2001) & cp.Quarter.isin([1, 3]) & cp.ShootLat.between(57.0, 59.2) &
        cp.ShootLon.between(7.5, 11.5) & cp.Depth.between(50, 300) & cp.AphiaID.isin(list(PREY))].copy()
cp = cp.drop_duplicates(["Year", "Quarter", "Ship", "HaulNo", "AphiaID", "Sex", "LngtClas"])
lw = pd.read_csv(lwf).rename(columns={"Unnamed: 0": "AphiaID"})
lw = lw.set_index(lw.columns[0])
# half-class: herring/sprat 0.5 cm classes, others 1 cm
half = np.where(cp.AphiaID.isin([126417, 126425]), 0.25, 0.5)
cp["L"] = cp.LngtClas / 10.0 + half
cp["W"] = [lw.loc[a, "a"] * L ** lw.loc[a, "b"] if a in lw.index else np.nan for a, L in zip(cp.AphiaID, cp.L)]
cp["kg_h"] = cp.CPUE_number_per_hour * cp.W / 1000.0
cp["sp"] = cp.AphiaID.map(PREY)
nh = cp.groupby("Quarter").apply(lambda d: d[["Year", "Ship", "HaulNo"]].drop_duplicates().shape[0])
rows = []
for (q, sp), d in list(cp.groupby(["Quarter", "sp"])) + [((("pooled"), sp), d) for sp, d in cp.groupby("sp")]:
    d = d.sort_values("L")
    for wcol, lab in (("kg_h", "biomass"), ("CPUE_number_per_hour", "numbers")):
        g = d.groupby("L")[wcol].sum()
        cum = g.cumsum() / g.sum()
        pct = lambda p: float(g.index[np.searchsorted(cum.values, p)])
        rows.append([q, sp, lab, round(g.sum(), 2), pct(0.1), pct(0.5), pct(0.9),
                     round(100 * g[g.index < 10].sum() / g.sum(), 1), round(100 * g[g.index < 15].sum() / g.sum(), 1),
                     round(100 * g[g.index < 20].sum() / g.sum(), 1)])
os.makedirs(out, exist_ok=True)
pd.DataFrame(rows, columns=["quarter", "species", "weight", "total_per_haul_sum", "L_p10", "L_p50", "L_p90",
                            "pct_lt10cm", "pct_lt15cm", "pct_lt20cm"]).to_csv(os.path.join(out, "prey_length_region.csv"), index=False)
# gape table: share of each prey species' near-bottom biomass (pooled Q1+Q3) that a stage can swallow
K = list(csv.DictReader(open(kf)))
prey_list = ["norway_pout", "herring", "sprat", "blue_whiting", "whiting", "poor_cod", "haddock", "cod",
             "long_rough_dab", "sandeel_Ammodytes_marinus"]
gt = []
for r in K:
    for Ls in r["stages"].split():
        L = float(Ls)
        row = [r["predator"], L, round(float(r["k_c"]) * L, 1), round(float(r["k_low"]) * L, 1), round(float(r["k_high"]) * L, 1)]
        for sp in prey_list:
            d = cp[cp.sp == sp]
            if d.kg_h.sum() <= 0:
                row += ["", ""]
                continue
            f = lambda k: round(100 * d.kg_h[d.L <= k * L].sum() / d.kg_h.sum(), 0)
            row += [f(float(r["k_c"])), f"{f(float(r['k_low'])):.0f}-{f(float(r['k_high'])):.0f}"]
        gt.append(row)
cols = ["predator", "pred_cm", "max_prey_cm_c", "max_prey_cm_low", "max_prey_cm_high"]
for sp in prey_list:
    cols += [f"{sp}_pct_biomass", f"{sp}_range"]
pd.DataFrame(gt, columns=cols).to_csv(os.path.join(out, "gape_table.csv"), index=False)
print("hauls per quarter:", dict(nh))
print(pd.read_csv(os.path.join(out, "prey_length_region.csv")).to_string())
# predator-biomass-weighted swallowable share: integrate the gape result over the predator's own
# near-bottom biomass length distribution in the region (pooled Q1+Q3)
pw = []
for r in K:
    pred = r["predator"]
    dp = cp[cp.sp == pred]
    if dp.kg_h.sum() <= 0:
        continue
    gp = dp.groupby("L").kg_h.sum(); gp = gp / gp.sum()
    for kk in ("k_low", "k_c", "k_high"):
        k = float(r[kk]); row = [pred, kk, k]
        for sp in prey_list:
            d = cp[cp.sp == sp]
            g = d.groupby("L").kg_h.sum(); g = g / g.sum()
            cum = g.cumsum()
            share = sum(w * float(g[g.index <= k * L].sum()) for L, w in gp.items())
            row.append(round(100 * share, 1))
        pw.append(row)
pd.DataFrame(pw, columns=["predator", "k_variant", "k"] + [f"{sp}_pct" for sp in prey_list]).to_csv(
    os.path.join(out, "gape_weighted_by_predator_length.csv"), index=False)
print(pd.read_csv(os.path.join(out, "gape_weighted_by_predator_length.csv")).to_string())
