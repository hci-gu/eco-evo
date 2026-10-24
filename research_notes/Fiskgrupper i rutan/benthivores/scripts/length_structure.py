"""Long-run (2001-25) length structure of the main benthivorous_fish species in the Mareld grid.

Usage: python3 -I length_structure.py <extract_old> <extract_new> <lw_csv> <depth_npz> <out_dir>

Same hauls, keys, depth strata and length-weight as ../../scripts/long_species.py (NS-IBTS GOV
Q1+Q3, all-species hauls, Skagerrak + 44G0/44G1/45G0/45G1), but kept per 1-cm length class.
Per year: depth-stratified kg/h per species and length class, pooled = mean of Q1 and Q3;
long-run = mean over the usable years. Output:
- length_biomass_longrun.csv: species, length (cm), long-run kg/h, share of species biomass.
- stage_shares_longrun.csv: biomass share per stage (boundaries below), long-run and 2021-25.
- size_factors_longrun.csv: biomass-weighted mean individual weight and the biomass-weighted mean
  of W^CB for CB = -0.15 ... -0.30 (the mass exponent of a Cmax or SMR per unit mass), so a
  per-gram rate f(W) = CA W^CB can be averaged over the stock's biomass.
"""
import os
import sys

import numpy as np
import pandas as pd

old, new, lw_csv, depth_npz, out = sys.argv[1:6]
LW = pd.read_csv(lw_csv, index_col=0)
SPECIES = {126437: "haddock", 126436: "cod", 127137: "long rough dab", 127139: "dab",
           127143: "plaice", 127136: "witch", 127140: "lemon sole", 150637: "grey gurnard",
           105865: "starry ray"}
HALF_CM = {126417, 126425, 126426, 126421, 125516, 126751, 126756, 126754, 125909}
STAGES = {"cod": [30, 50], "haddock": [40], "long rough dab": [20], "dab": [20],
          "plaice": [25], "witch": [30], "lemon sole": [25], "grey gurnard": [25],
          "starry ray": [30]}

hh = pd.concat([pd.read_csv(os.path.join(d, "HH.csv.gz")) for d in (old, new)])
hh = hh[(hh.HaulVal.astype(str).str.strip() == "V") & hh.Quarter.isin([1, 3])
        & (hh.Gear.astype(str).str.strip() == "GOV") & (hh.BySpecRecCode == 1)]
for c in ["Ship", "StatRec"]:
    hh[c] = hh[c].astype(str).str.strip()
hh["la"], hh["lo"] = hh.ShootLat.round(3), hh.ShootLong.round(3)
hh = hh.drop_duplicates(["Year", "Quarter", "Ship", "HaulNo", "la", "lo"])
lat, lon = hh.ShootLat, hh.ShootLong
x1, y1, x2, y2 = 8.60, 57.12, 7.05, 57.98
east = ((x2 - x1) * (lat - y1) - (y2 - y1) * (lon - x1)) < 0
skag = east & (lat < 59.2) & ~((lon > 10.6) & (lat < 57.75)) & (lon < 12.5) & (lat > 57.0)
hh = hh[skag | hh.StatRec.isin(["44G0", "44G1", "45G0", "45G1"])].copy()
hh = hh[(hh.Year >= 2001) & (hh.Year <= 2025)]
BINS = ["<100", "100-200", "200-300", ">=300"]
hh["dbin"] = pd.cut(hh.Depth, [0, 100, 200, 300, 600], labels=BINS, right=False)
hh["key"] = list(zip(hh.Year, hh.Quarter, hh.Ship, hh.HaulNo, hh.la, hh.lo))

cp = pd.concat([pd.read_csv(os.path.join(d, "CPUEL.csv.gz")) for d in (old, new)])
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp["key"] = list(zip(cp.Year, cp.Quarter, cp.Ship, cp.HaulNo, cp.ShootLat.round(3),
                     cp.ShootLon.round(3)))
cp = cp[cp.key.isin(set(hh.key)) & cp.AphiaID.isin(SPECIES)].copy()
cp["L"] = cp.LngtClas / 10.0 + np.where(cp.AphiaID.isin(HALF_CM), 0.25, 0.5)
cp["Lcm"] = np.floor(cp.L).astype(int)
cp["w_g"] = cp.AphiaID.map(LW.a) * cp.L ** cp.AphiaID.map(LW.b)
cp["kg_h"] = cp.CPUE_number_per_hour * cp.w_g / 1000
cp["col"] = cp.AphiaID.astype(str) + "_" + cp.Lcm.astype(str)
m = cp.groupby(["key", "col"]).kg_h.sum().unstack(fill_value=0.0)
cols = list(m.columns)
X = hh.set_index("key")[["Year", "Quarter", "dbin"]].join(m, how="left").fillna(0.0)

z = np.load(depth_npz)["depth"].astype(float)
z = np.abs(z[np.isfinite(z)])
area = pd.cut(pd.Series(z), [0, 100, 200, 300, 600], labels=BINS, right=False).value_counts(
    normalize=True).reindex(BINS)


def strat(df):
    mu = df.groupby("dbin", observed=False)[cols].mean().reindex(BINS)
    if mu.loc["200-300"].isna().any():
        mu.loc["200-300"] = mu.loc["100-200"]
    mu.loc[">=300"] = mu.loc["200-300"]
    if mu.loc[BINS[:3]].isna().any().any():
        return None
    return mu.mul(area, axis=0).sum()


years = {}
for y, d in X.groupby("Year"):
    q = [strat(d[d.Quarter == k]) for k in (1, 3)]
    if q[0] is not None and q[1] is not None:
        years[y] = (q[0] + q[1]) / 2
Y = pd.DataFrame(years).T
print("years used:", len(Y), min(Y.index), "-", max(Y.index))

rows = []
for c in cols:
    aph, L = (int(v) for v in c.split("_"))
    rows.append(dict(species=SPECIES[aph], aphia=aph, L_cm=L,
                     kgh_2001_25=Y[c].mean(), kgh_2021_25=Y.loc[2021:2025, c].mean()))
D = pd.DataFrame(rows).sort_values(["species", "L_cm"])
for c in ["kgh_2001_25", "kgh_2021_25"]:
    D["share_" + c[4:]] = D[c] / D.groupby("species")[c].transform("sum")
D["w_g"] = D.aphia.map(LW.a) * (D.L_cm + 0.5) ** D.aphia.map(LW.b)
D.to_csv(os.path.join(out, "length_biomass_longrun.csv"), index=False)

st, sf = [], []
for sp, d in D.groupby("species"):
    edges = [0] + STAGES[sp] + [999]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (d.L_cm >= lo) & (d.L_cm < hi)
        st.append(dict(species=sp, stage=f"{lo}-{hi if hi < 999 else ''}",
                       share_2001_25=d.loc[sel, "share_2001_25"].sum(),
                       share_2021_25=d.loc[sel, "share_2021_25"].sum()))
    r = dict(species=sp, kgh_2001_25=d.kgh_2001_25.sum(),
             mean_w_g_biomass_weighted=(d.share_2001_25 * d.w_g).sum(),
             median_L_biomass=float(d.L_cm[(d.share_2001_25.cumsum() >= 0.5).values].iloc[0]))
    for cb in (-0.15, -0.195, -0.2, -0.25, -0.3):
        r[f"E_W^{cb}"] = (d.share_2001_25 * d.w_g ** cb).sum()
        r[f"W_eff_{cb}"] = r[f"E_W^{cb}"] ** (1 / cb)
    sf.append(r)
ST = pd.DataFrame(st)
SF = pd.DataFrame(sf)
ST.to_csv(os.path.join(out, "stage_shares_longrun.csv"), index=False)
SF.to_csv(os.path.join(out, "size_factors_longrun.csv"), index=False)
pd.set_option("display.width", 220, "display.max_columns", 30)
print(ST.round(3).to_string(index=False))
print(SF.round(3).to_string(index=False))
