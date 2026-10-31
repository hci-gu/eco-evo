"""Box length structure of blue whiting (NS-IBTS GOV, Q1 and Q3, 2001-25), for the planktivore A denominator.

Usage: python3 -I blue_whiting_lengths.py <extract_old> <extract_new> <lw_csv> <depth_npz> <out_dir>

Same hauls, box polygon, depth strata and length-weight as
research_notes/"Fiskgrupper i rutan"/benthivores/scripts/length_structure.py (copied logic, species
replaced by blue whiting, AphiaID 126439), kept per quarter. Raw GOV (q = 1; no published GOV q for
blue whiting). Output: blue_whiting_lengths.csv (L_cm, kg/h Q1, Q3, pooled, shares).
Blue whiting has no length at age 1 in the stage table, so the box J/B counts it as adult
("Arsungarnas andel" juvenile_share_by_quarter.py); the engine applies f_A to its tonnes.
"""
import os
import sys

import numpy as np
import pandas as pd

old, new, lw_csv, depth_npz, out = sys.argv[1:6]
LW = pd.read_csv(lw_csv, index_col=0)
APH = 126439

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
cp["key"] = list(zip(cp.Year, cp.Quarter, cp.Ship, cp.HaulNo, cp.ShootLat.round(3), cp.ShootLon.round(3)))
cp = cp[cp.key.isin(set(hh.key)) & (cp.AphiaID == APH)].copy()
cp["L"] = cp.LngtClas / 10.0 + 0.5
cp["Lcm"] = np.floor(cp.L).astype(int)
cp["w_g"] = LW.loc[APH, "a"] * cp.L ** LW.loc[APH, "b"]
cp["kg_h"] = cp.CPUE_number_per_hour * cp.w_g / 1000
m = cp.groupby(["key", "Lcm"]).kg_h.sum().unstack(fill_value=0.0)
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


yq = {1: {}, 3: {}}
for y, d in X.groupby("Year"):
    for k in (1, 3):
        s = strat(d[d.Quarter == k])
        if s is not None:
            yq[k][y] = s
Q = {k: pd.DataFrame(v).T.mean() for k, v in yq.items()}
print(f"years used: Q1 {len(yq[1])}, Q3 {len(yq[3])}; hauls with blue whiting {cp.key.nunique()} of {len(hh)}")
D = pd.DataFrame({"L_cm": cols, "kgh_q1": [Q[1][c] for c in cols], "kgh_q3": [Q[3][c] for c in cols]})
D["kgh_pooled"] = (D.kgh_q1 + D.kgh_q3) / 2
for c in ("q1", "q3", "pooled"):
    D["share_" + c] = D["kgh_" + c] / D["kgh_" + c].sum()
D.to_csv(os.path.join(out, "blue_whiting_lengths.csv"), index=False)
for c in ("q1", "q3", "pooled"):
    s = D["share_" + c].values
    cs = np.cumsum(s)
    pct = [D.L_cm.values[np.searchsorted(cs, p)] + 0.5 for p in (0.05, 0.1, 0.5, 0.9)]
    print(f"  {c:6s} kg/h {D['kgh_' + c].sum():.2f}; biomass p5/p10/p50/p90 "
          + " / ".join(f"{x:.1f}" for x in pct) + " cm; share < 15 cm "
          f"{s[D.L_cm.values + 0.5 < 15].sum():.3f}, < 20 cm {s[D.L_cm.values + 0.5 < 20].sum():.3f}")
