"""Length structure (by biomass and numbers) of the model's pelagic_fish pool in the box region from
NS-IBTS GOV Q1/Q3 2021-25: herring, sprat, mackerel, horse mackerel. Depth-stratified like
length_biomass.py. Length-weight a, b below are APPROXIMATE generic values (not taken from a cited
source); they scale biomass between species only, and herring dominates the near-bottom catch.

Usage: python3 -I pelagic_prey_sizes.py <species_mix_dir> <out_dir>
The GOV is a demersal trawl: these are the pelagic fish that are near the bottom by day, which is the
part a demersal predator meets; it is not the stock's size structure.
"""
import os
import sys

import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
SP = {126417: ("herring", 0.0060, 3.10), 126425: ("sprat", 0.0055, 3.13),
      127023: ("mackerel", 0.0050, 3.15), 126822: ("horse mackerel", 0.0090, 2.95)}
BINS = ["<100", "100-200", "200-300", ">=300"]
hh = pd.read_csv(os.path.join(src, "tables", "hauls_region.csv"))
hh["Ship"] = hh.Ship.astype(str).str.strip()
hh = hh[hh.region]
area = pd.read_csv(os.path.join(src, "tables", "box_area_by_depth.csv"), index_col=0).iloc[:, 0]
cp = pd.read_csv(os.path.join(src, "data", "extract", "CPUEL.csv.gz"))
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp = cp[cp.AphiaID.isin(SP)].copy()
cp["species"] = cp.AphiaID.map(lambda a: SP[a][0])
cp["L"] = cp.LngtClas / 10.0 + 0.5
cp["kg_h"] = cp.CPUE_number_per_hour * cp.AphiaID.map(lambda a: SP[a][1]) * cp.L ** cp.AphiaID.map(
    lambda a: SP[a][2]) / 1000
cp = cp.merge(hh[["Year", "Quarter", "Ship", "HaulNo", "dbin"]], on=["Year", "Quarter", "Ship", "HaulNo"])
nh = hh.groupby(["Quarter", "dbin"]).size()
cp["cls"] = pd.cut(cp.L, [0, 8, 10, 12, 15, 20, 25, 30, 60], right=False).astype(str)
rows = []
for q in [1, 3]:
    for val in ["kg_h", "CPUE_number_per_hour"]:
        g = cp[cp.Quarter == q].groupby(["dbin", "species", "cls"])[val].sum().reset_index()
        g["mean"] = g[val] / [nh[(q, d)] for d in g.dbin]
        p = g.pivot_table(index=["species", "cls"], columns="dbin", values="mean", aggfunc="sum",
                          fill_value=0.0).reindex(columns=BINS, fill_value=0.0)
        p[">=300"] = p["200-300"]
        st = (p * area.reindex(BINS).values).sum(axis=1)
        for (s, c), v in st.items():
            rows.append(dict(quarter=q, measure="biomass" if val == "kg_h" else "numbers", species=s, cls=c, value=v))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(out, "pelagic_prey_sizes.csv"), index=False)
# 1-cm biomass of the whole pool (all four species), per quarter, for the gape calculation
cp["Lc"] = np.floor(cp.L).astype(int)
lrows = []
for q in [1, 3]:
    res = {}
    for val in ["kg_h", "CPUE_number_per_hour"]:
        g = cp[cp.Quarter == q].groupby(["dbin", "Lc"])[val].sum().reset_index()
        g["mean"] = g[val] / [nh[(q, d)] for d in g.dbin]
        p = g.pivot_table(index="Lc", columns="dbin", values="mean", aggfunc="sum", fill_value=0.0).reindex(
            columns=BINS, fill_value=0.0)
        p[">=300"] = p["200-300"]
        res[val] = (p * area.reindex(BINS).values).sum(axis=1)
    for lc in res["kg_h"].index:
        lrows.append(dict(quarter=q, L_cm=lc, kg_h=res["kg_h"][lc], n_h=res["CPUE_number_per_hour"][lc]))
pd.DataFrame(lrows).to_csv(os.path.join(out, "pelagic_prey_length_biomass.csv"), index=False)
pd.set_option("display.width", 200)
for (q, m), g in R.groupby(["quarter", "measure"]):
    t = g.pivot_table(index="species", columns="cls", values="value", aggfunc="sum").fillna(0)
    tot = t.sum(axis=1)
    print(f"Q{q} {m}: totals", tot.round(1).to_dict())
    print((100 * t.div(tot, axis=0)).round(1).to_string())
    print("pool total by class (%):", (100 * t.sum() / t.values.sum()).round(1).to_dict())
