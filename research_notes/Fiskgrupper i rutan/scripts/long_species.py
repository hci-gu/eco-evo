"""Long-run (2001-25) species composition of each fish group in the Mareld grid.

Usage: python3 -I long_species.py <extract_old> <extract_new> <guilds_csv> <lw_csv> <depth_npz> <out_dir>

Same hauls, keys, strata and length-weight as long_series.py (all-species hauls, BySpecRecCode 1).
Per year: depth-stratified kg/h per species, pooled = mean of Q1 and Q3 (a year with a missing
shallow stratum is skipped; a missing 200-300 m stratum borrows 100-200 m). Long-run = mean over
years of kg/h; shares within each group; t/km2 at wing spread from the 2021-25 kg/h -> t/km2
ratio per group.
"""
import os
import sys

import numpy as np
import pandas as pd

old, new, guilds_csv, lw_csv, depth_npz, out = sys.argv[1:7]
G = pd.read_csv(guilds_csv).set_index("aphia")
LW = pd.read_csv(lw_csv, index_col=0)
NAMES = {"P": "planktivorous_fish", "B": "benthivorous_fish", "R": "predatory_fish", "X": "other"}
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
cp = cp[cp.key.isin(set(hh.key)) & cp.AphiaID.isin(G.index)].copy()
cp["L"] = cp.LngtClas / 10.0 + np.where(cp.AphiaID.isin(HALF_CM), 0.25, 0.5)
cp["kg_h"] = cp.CPUE_number_per_hour * cp.AphiaID.map(LW.a) * cp.L ** cp.AphiaID.map(LW.b) / 1000
m = cp.groupby(["key", "AphiaID"]).kg_h.sum().unstack(fill_value=0.0)
X = hh.set_index("key")[["Year", "Quarter", "dbin"]].join(m, how="left").fillna(0.0)
taxa = list(m.columns)

z = np.load(depth_npz)["depth"].astype(float)
z = np.abs(z[np.isfinite(z)])
area = pd.cut(pd.Series(z), [0, 100, 200, 300, 600], labels=BINS, right=False).value_counts(
    normalize=True).reindex(BINS)


def strat(df):
    mu = df.groupby("dbin", observed=False)[taxa].mean().reindex(BINS)
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
lr = Y.mean()
gr = pd.read_csv(os.path.join(out, "guild_box_depth_strat.csv"), index_col=0)
ratio = gr.t_km2_pooled / gr.kgh_pooled
rows = []
for aph, v in lr.items():
    g = NAMES[G.loc[aph, "guild"]]
    rows.append(dict(aphia=aph, common=G.loc[aph, "common"], guild=g, kgh_2001_25=v,
                     kgh_2021_25=Y.loc[2021:2025, aph].mean(), t_km2=v * ratio.get(g, np.nan)))
S = pd.DataFrame(rows)
S["share_in_group_pct"] = 100 * S.kgh_2001_25 / S.groupby("guild").kgh_2001_25.transform("sum")
S["share_2021_25_pct"] = 100 * S.kgh_2021_25 / S.groupby("guild").kgh_2021_25.transform("sum")
S = S.sort_values(["guild", "kgh_2001_25"], ascending=[True, False])
S.to_csv(os.path.join(out, "long_run_species_by_group.csv"), index=False)
pd.set_option("display.width", 200, "display.max_rows", 200)
print("years used:", sorted(Y.index))
for g, d in S.groupby("guild"):
    print(f"\n{g}: {d.t_km2.sum():.2f} t/km2 (wing, q = 1, mean 2001-25)")
    print(d[d.share_in_group_pct >= 0.3][["common", "t_km2", "share_in_group_pct", "share_2021_25_pct"]].round(2).to_string(index=False))
