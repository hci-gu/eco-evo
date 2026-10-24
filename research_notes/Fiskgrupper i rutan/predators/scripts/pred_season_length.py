"""Predatory fish in the Mareld grid: Q1 vs Q3 presence and length-biomass structure, 2001-25.

Usage: python3 -I pred_season_length.py <extract_old> <extract_new> <guilds_csv> <lw_csv> <depth_npz> <out_dir> [guild letter, default R; B gives the prey-size table of the benthivores]

Same hauls, keys, depth strata and length-weight as ../../scripts/long_species.py (NS-IBTS GOV,
BySpecRecCode 1, Skagerrak + box rectangles, depth-stratified by the grid's area per depth bin).
Outputs, per predatory_fish species:
- pred_season.csv: long-run (mean over years) stratified kg/h in Q1, Q3 and pooled; Q3 share.
- pred_length_biomass.csv: long-run stratified kg/h per 5-cm class (pooled Q1+Q3), and per quarter.
"""
import os
import sys

import numpy as np
import pandas as pd

old, new, guilds_csv, lw_csv, depth_npz, out = sys.argv[1:7]
GUILD = sys.argv[7] if len(sys.argv) > 7 else "R"  # B = benthivores (prey sizes)
PFX = "pred" if GUILD == "R" else "guild" + GUILD
G = pd.read_csv(guilds_csv).set_index("aphia")
LW = pd.read_csv(lw_csv, index_col=0)
PRED = [a for a in G.index if G.loc[a, "guild"] == GUILD]
HALF_CM = {126417, 126425, 126426, 126421, 125516, 126751, 126756, 126754, 125909}

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
cp = cp[cp.key.isin(set(hh.key)) & cp.AphiaID.isin(PRED)].copy()
cp["L"] = cp.LngtClas / 10.0 + np.where(cp.AphiaID.isin(HALF_CM), 0.25, 0.5)
cp["kg_h"] = cp.CPUE_number_per_hour * cp.AphiaID.map(LW.a) * cp.L ** cp.AphiaID.map(LW.b) / 1000
cp["L5"] = (np.floor(cp.L / 5) * 5).astype(int)
cp["col"] = list(zip(cp.AphiaID, cp.L5))
m = cp.groupby(["key", "col"]).kg_h.sum().unstack(fill_value=0.0)
X = hh.set_index("key")[["Year", "Quarter", "dbin"]].join(m, how="left").fillna(0.0)
cols = list(m.columns)

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


q1, q3 = {}, {}
for y, d in X.groupby("Year"):
    a, b = strat(d[d.Quarter == 1]), strat(d[d.Quarter == 3])
    if a is not None and b is not None:
        q1[y], q3[y] = a, b
Q1 = pd.DataFrame(q1).T.mean()
Q3 = pd.DataFrame(q3).T.mean()
P = (Q1 + Q3) / 2

L = pd.DataFrame({"q1": Q1, "q3": Q3, "pooled": P})
L.index = pd.MultiIndex.from_tuples(L.index, names=["aphia", "L5"])
L = L.reset_index()
L["common"] = L.aphia.map(G.common)
L.to_csv(os.path.join(out, PFX + "_length_biomass.csv"), index=False)

S = L.groupby(["aphia", "common"])[["q1", "q3", "pooled"]].sum().reset_index()
S["q3_share_of_q1_plus_q3"] = S.q3 / (S.q1 + S.q3)
S["share_pooled_pct"] = 100 * S.pooled / S.pooled.sum()
S["share_q1_pct"] = 100 * S.q1 / S.q1.sum()
S["share_q3_pct"] = 100 * S.q3 / S.q3.sum()
S = S.sort_values("pooled", ascending=False)
S.to_csv(os.path.join(out, PFX + "_season.csv"), index=False)
pd.set_option("display.width", 200)
print("years used:", len(q1))
print(S[S.share_pooled_pct > 0.3].round(3).to_string(index=False))

if GUILD != "R":
    T = L.groupby("L5")[["q1", "q3", "pooled"]].sum()
    T = T / T.sum()
    print("\nprey-size biomass share of the guild below L (cm), pooled | Q1 | Q3")
    for lim in (10, 15, 20, 25, 30):
        x = T[T.index < lim].sum()
        print(f"< {lim}: {x.pooled:.3f} | {x.q1:.3f} | {x.q3:.3f}")
    sys.exit(0)
# stage shares used by the stage scores (../../../Torskfiskarnas bytesandelar)
cuts = {"whiting": [20], "saithe": [30, 60], "hake": [20], "mackerel": [25, 30],
        "spurdog": [60], "horse mackerel": [20], "anglerfish": [40]}
print("\nlength-class biomass shares (pooled; Q1 | Q3), lower edge of the 5-cm class")
for sp, cc in cuts.items():
    d = L[L.common == sp]
    edges = [0] + cc + [999]
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (d.L5 >= lo) & (d.L5 < hi)
        parts.append(f"{lo}-{hi}: {d[sel].pooled.sum() / d.pooled.sum():.2f} "
                     f"({d[sel].q1.sum() / max(d.q1.sum(), 1e-12):.2f} | {d[sel].q3.sum() / d.q3.sum():.2f})")
    med = d.sort_values("L5")
    cum = med.pooled.cumsum() / med.pooled.sum()
    print(f"{sp}: " + "; ".join(parts) + f"; biomass-median class {int(med.L5[cum >= 0.5].iloc[0])} cm")
