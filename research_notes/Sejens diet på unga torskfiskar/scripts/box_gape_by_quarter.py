"""Benthivore biomass inside a saithe gape, Q1 vs Q3, in the Mareld box (NS-IBTS GOV).

Usage: python3 -I box_gape_by_quarter.py <extract_old> <extract_new> <lw_csv> <depth_npz> <out_dir>

Same hauls, box, depth strata (area-weighted on maps/depth.npz) and length-weight as
../../Fiskgrupper i rutan/benthivores/scripts/length_structure.py, but kept per quarter.
Per year and quarter: depth-stratified kg/h per species and 1-cm class; long run = mean of the
years with both quarters (2001-25). Prints, per quarter, the guild biomass and the share of it
below 15 / 19 / 25 cm by species, and haddock's length-frequency (numbers) below 30 cm.
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
months = hh.groupby("Quarter").Month.describe()[["min", "25%", "50%", "75%", "max"]]

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


def table(value):
    m = cp.groupby(["key", "col"])[value].sum().unstack(fill_value=0.0)
    cols = list(m.columns)
    X = hh.set_index("key")[["Year", "Quarter", "dbin"]].join(m, how="left").fillna(0.0)
    return X, cols


z = np.load(depth_npz)["depth"].astype(float)
z = np.abs(z[np.isfinite(z)])
area = pd.cut(pd.Series(z), [0, 100, 200, 300, 600], labels=BINS, right=False).value_counts(
    normalize=True).reindex(BINS)


def strat(df, cols):
    mu = df.groupby("dbin", observed=False)[cols].mean().reindex(BINS)
    if mu.loc["200-300"].isna().any():
        mu.loc["200-300"] = mu.loc["100-200"]
    mu.loc[">=300"] = mu.loc["200-300"]
    if mu.loc[BINS[:3]].isna().any().any():
        return None
    return mu.mul(area, axis=0).sum()


def longrun(value):
    X, cols = table(value)
    per_q = {1: {}, 3: {}}
    for y, d in X.groupby("Year"):
        q = {k: strat(d[d.Quarter == k], cols) for k in (1, 3)}
        if q[1] is not None and q[3] is not None:
            for k in (1, 3):
                per_q[k][y] = q[k]
    out_q = {}
    for k in (1, 3):
        Y = pd.DataFrame(per_q[k]).T
        s = Y.mean()
        r = [dict(species=SPECIES[int(c.split("_")[0])], L_cm=int(c.split("_")[1]), v=s[c])
             for c in s.index]
        out_q[k] = pd.DataFrame(r)
    return out_q, len(per_q[1])


B, ny = longrun("kg_h")
N, _ = longrun("CPUE_number_per_hour")
print("years with both quarters:", ny)
print("haul months by quarter:\n", months.to_string())
rows = []
for k in (1, 3):
    d = B[k]
    tot = d.v.sum()
    print(f"\nQ{k}: guild kg/h {tot:.2f}")
    for g in (15, 19, 25):
        sub = d[d.L_cm < g]
        bys = sub.groupby("species").v.sum() / tot
        print(f"  < {g} cm: share of guild {sub.v.sum() / tot:.3f}; by species "
              + ", ".join(f"{s} {v:.3f}" for s, v in bys.sort_values(ascending=False).items()
                          if v >= 0.001))
        for s, v in bys.items():
            rows.append(dict(quarter=k, gape_cm=g, species=s, share_of_guild=v))
    had = d[d.species == "haddock"]
    print(f"  haddock share of guild {had.v.sum() / tot:.3f}")
pd.DataFrame(rows).to_csv(os.path.join(out, "box_gape_share_by_quarter.csv"), index=False)
print("\nhaddock numbers per hour by 1-cm class (< 30 cm), Q1 | Q3")
n1 = N[1][N[1].species == "haddock"].set_index("L_cm").v
n3 = N[3][N[3].species == "haddock"].set_index("L_cm").v
for L in range(3, 30):
    print(f"  {L:2d} cm  {n1.get(L, 0.0):8.2f} | {n3.get(L, 0.0):8.2f}")


def q_haddock(L):
    """Walker et al. 2017 Q3 GOV catchability at length for haddock (door-spread basis),
    as printed in ../../Torskfiskarnas artsammansättning/evidence_B_catchability.md."""
    L = np.asarray(L, dtype=float)
    poly = -0.370453 + 0.04227 * L - 0.000685 * L ** 2
    return np.where(L <= 11, 0.0116121, np.where(L >= 46, 0.123268, poly))


print("\nhaddock, catchability-corrected (Walker 2017 q-at-length, Q3 values used in both quarters)")
for k in (1, 3):
    d = B[k][B[k].species == "haddock"].copy()
    d["vq"] = d.v / q_haddock(d.L_cm + 0.5)
    for g in (15, 19, 25):
        raw = d[d.L_cm < g].v.sum() / d.v.sum()
        cor = d[d.L_cm < g].vq.sum() / d.vq.sum()
        print(f"  Q{k} < {g} cm: share of haddock biomass raw {raw:.3f}, q-corrected {cor:.3f}")


print("\nguild share below L (raw survey; haddock also q-corrected), mean of Q1 and Q3; split gadoids / flatfish+rays+gurnard")
GAD = {"haddock", "cod"}
for L in (10, 12, 15, 19, 25, 27, 28, 30, 35, 40, 45):
    s_raw, s_gad, s_flat = [], [], []
    for k in (1, 3):
        d = B[k]
        tot = d.v.sum()
        sub = d[d.L_cm < L]
        s_raw.append(sub.v.sum() / tot)
        s_gad.append(sub[sub.species.isin(GAD)].v.sum() / tot)
        s_flat.append(sub[~sub.species.isin(GAD)].v.sum() / tot)
    print(f"  < {L:2d} cm: all {np.mean(s_raw):.3f} (Q1 {s_raw[0]:.3f}, Q3 {s_raw[1]:.3f}); "
          f"haddock+cod {np.mean(s_gad):.3f}; flatfish/rays/gurnard {np.mean(s_flat):.3f}")
