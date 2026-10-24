"""Fish groups and the gadoid mix in the Mareld box per year, NS-IBTS 2000-2025.

Usage: python3 -I long_series.py <extract_old> <extract_new> <guilds_csv> <lw_csv> <phi_contrib_csv>
                                 <depth_npz> <out_dir>

<extract_old>: HH/CPUEL extracts 2000-2020 (this folder's data/extract), <extract_new>: 2021-25
(research_notes/"Torskfiskarnas artsammansattning"/data/extract, Swedish letters). Untrusted data,
only parsed. Length-weight from tables/length_weight_all.csv (2021-25 CA fits and body-form
defaults; applied to every year). Species-level prey scores from the stage-resolved
phi_contributions.csv (raw variant, Q1/Q3 mean): score = sum(contribution) / sum(biomass share).

Same conventions as fish_guilds.py (valid GOV hauls, Q1 + Q3, Skagerrak + box rectangles, depth
strata weighted by the grid's area, >= 300 m takes 200-300 m, pooled = mean of Q1 and Q3), plus:
only hauls that recorded all species (BySpecRecCode 1) enter the all-fish groups.
"""
import os
import sys

import numpy as np
import pandas as pd

old, new, guilds_csv, lw_csv, contrib_csv, depth_npz, out = sys.argv[1:8]
os.makedirs(out, exist_ok=True)
G = pd.read_csv(guilds_csv).set_index("aphia")
LW = pd.read_csv(lw_csv, index_col=0)
NAMES = {"P": "planktivorous_fish", "B": "benthivorous_fish", "R": "predatory_fish", "X": "other"}
HALF_CM = {126417, 126425, 126426, 126421, 125516, 126751, 126756, 126754, 125909}
GADOIDS = {126437: "haddock", 126438: "whiting", 126441: "saithe", 126436: "cod", 126484: "hake"}

hh = pd.concat([pd.read_csv(os.path.join(d, "HH.csv.gz")) for d in (old, new)])
hh = hh[(hh.HaulVal.astype(str).str.strip() == "V") & hh.Quarter.isin([1, 3])
        & (hh.Gear.astype(str).str.strip() == "GOV")]
for c in ["Country", "Ship", "StatRec"]:
    hh[c] = hh[c].astype(str).str.strip()
# Danish and Swedish cruises share the ship code 26D4 and haul numbers: key on position too.
hh["la"], hh["lo"] = hh.ShootLat.round(3), hh.ShootLong.round(3)
hh = hh.drop_duplicates(["Year", "Quarter", "Ship", "HaulNo", "la", "lo"])
lat, lon = hh.ShootLat, hh.ShootLong
x1, y1, x2, y2 = 8.60, 57.12, 7.05, 57.98
east = ((x2 - x1) * (lat - y1) - (y2 - y1) * (lon - x1)) < 0
skag = east & (lat < 59.2) & ~((lon > 10.6) & (lat < 57.75)) & (lon < 12.5) & (lat > 57.0)
box = hh.StatRec.isin(["44G0", "44G1", "45G0", "45G1"])
hh = hh[skag | box].copy()
hh["box"] = box[skag | box]
BINS = ["<100", "100-200", "200-300", ">=300"]
hh["dbin"] = pd.cut(hh.Depth, [0, 100, 200, 300, 600], labels=BINS, right=False)
hh["allsp"] = hh.BySpecRecCode == 1
hh["key"] = list(zip(hh.Year, hh.Quarter, hh.Ship, hh.HaulNo, hh.la, hh.lo))

cp = pd.concat([pd.read_csv(os.path.join(d, "CPUEL.csv.gz")) for d in (old, new)])
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp["key"] = list(zip(cp.Year, cp.Quarter, cp.Ship, cp.HaulNo, cp.ShootLat.round(3), cp.ShootLon.round(3)))
cp = cp[cp.key.isin(set(hh.key)) & cp.AphiaID.isin(G.index)].copy()
cp["L"] = cp.LngtClas / 10.0 + np.where(cp.AphiaID.isin(HALF_CM), 0.25, 0.5)
cp["kg_h"] = cp.CPUE_number_per_hour * cp.AphiaID.map(LW.a) * cp.L ** cp.AphiaID.map(LW.b) / 1000
m = cp.groupby(["key", "AphiaID"]).kg_h.sum().unstack(fill_value=0.0)
X = hh.set_index("key")[["Year", "Quarter", "dbin", "box", "allsp", "Country"]].join(m, how="left").fillna(0.0)
taxa = [c for c in m.columns]

z = np.load(depth_npz)["depth"].astype(float)
z = np.abs(z[np.isfinite(z)])
area = pd.cut(pd.Series(z), [0, 100, 200, 300, 600], labels=BINS, right=False).value_counts(
    normalize=True).reindex(BINS)


FILLED, CUR = set(), [None]


def strat(df, cols):
    mu = df.groupby("dbin", observed=False)[cols].mean().reindex(BINS)
    if mu.loc["200-300"].isna().any():  # no deep haul this year/quarter: borrow 100-200 m
        mu.loc["200-300"] = mu.loc["100-200"]
        FILLED.add(CUR[0])
    mu.loc[">=300"] = mu.loc["200-300"]
    if mu.loc[BINS[:3]].isna().any().any():
        return None  # a shallower stratum has no haul
    return mu.mul(area, axis=0).sum()


gg = G.guild.map(NAMES).reindex(taxa)
guild_cols = ["planktivorous_fish", "benthivorous_fish", "predatory_fish"]
Xg = X[["Year", "Quarter", "dbin", "box", "allsp"]].join(X[taxa].T.groupby(gg).sum().T.reindex(
    columns=guild_cols, fill_value=0.0))
Xs = X[["Year", "Quarter", "dbin", "allsp"]].join(X[[a for a in GADOIDS if a in X]].rename(columns=GADOIDS))
Xs["blue whiting"] = X.get(126439, 0.0)
Xs["herring"] = X.get(126417, 0.0)
Xs["Norway pout"] = X.get(126444, 0.0)
gad = list(GADOIDS.values())

# species-level prey scores from the stage-resolved contributions
c = pd.read_csv(contrib_csv)
c = c[(c.variant == "raw") & c.quarter.astype(str).isin(["1", "3"]) & c.pool.isin(["benthos", "fish", "zoo"])]
sc = c.groupby(["species", "pool"])[["contribution", "biomass_share"]].sum()
sc = (sc.contribution / sc.biomass_share).unstack()
sc.to_csv(os.path.join(out, "species_prey_scores.csv"))

rows = []
for y in sorted(X.Year.unique()):
    r = dict(year=int(y))
    CUR[0] = int(y)
    for q in (1, 3):
        d = Xg[(Xg.Year == y) & (Xg.Quarter == q)]
        r[f"hauls_Q{q}"] = len(d)
        r[f"box_hauls_Q{q}"] = int(d.box.sum())
        r[f"allsp_Q{q}"] = int(d.allsp.sum())
    qs = {}
    for q in (1, 3):
        d = Xg[(Xg.Year == y) & (Xg.Quarter == q) & Xg.allsp]
        qs[q] = strat(d, guild_cols) if len(d) else None
    if qs[1] is not None and qs[3] is not None:
        p = (qs[1] + qs[3]) / 2
        for k in guild_cols:
            r[k + "_pct"] = 100 * p[k] / p.sum()
            r[k + "_kgh"] = p[k]
    # gadoid mix and phi (standard species: every valid haul)
    gq = {}
    for q in (1, 3):
        d = Xs[(Xs.Year == y) & (Xs.Quarter == q)]
        gq[q] = strat(d, gad + ["blue whiting", "herring", "Norway pout"]) if len(d) else None
    if gq[1] is not None and gq[3] is not None:
        p = (gq[1] + gq[3]) / 2
        for k in gad + ["blue whiting", "herring", "Norway pout"]:
            r[k + "_kgh"] = p[k]
        mix = p[gad] / p[gad].sum()
        for k in gad:
            r[k + "_mix"] = 100 * mix[k]
        for pool in ["benthos", "fish", "zoo"]:
            r["phi_" + pool] = float((mix * sc.reindex(gad)[pool].fillna(0.0)).sum())
    rows.append(r)
T = pd.DataFrame(rows).set_index("year")
T.to_csv(os.path.join(out, "long_series_by_year.csv"))

# periods
per = {"2000-04": (2000, 2004), "2005-09": (2005, 2009), "2010-14": (2010, 2014),
       "2015-19": (2015, 2019), "2020-25": (2020, 2025), "2000-25": (2000, 2025)}
P = pd.DataFrame({k: T.loc[a:b].mean(numeric_only=True) for k, (a, b) in per.items()}).T
P.to_csv(os.path.join(out, "long_series_by_period.csv"))

print("years with the 200-300 m stratum borrowed from 100-200 m:", sorted(FILLED))
T["deep_borrowed"] = [y in FILLED for y in T.index]
T.to_csv(os.path.join(out, "long_series_by_year.csv"))
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 100)
print("species-level prey scores (gadoids)\n", sc.round(3))
print("\nhauls per year/quarter (region, box, all-species recorded)\n",
      T[["hauls_Q1", "box_hauls_Q1", "allsp_Q1", "hauls_Q3", "box_hauls_Q3", "allsp_Q3"]].to_string())
print("\nfish groups, % of all fish (depth-stratified, pooled Q1/Q3, all-species hauls)\n",
      T[[k + "_pct" for k in guild_cols]].round(1).to_string())
print("\ngadoid mix % and phi\n", T[[k + "_mix" for k in gad] + ["phi_benthos", "phi_fish", "phi_zoo"]].round(2).to_string())
print("\nkg/h\n", T[[k + "_kgh" for k in gad + ["blue whiting", "herring", "Norway pout"]]].round(1).to_string())
print("\nperiod means\n", P[[k + "_pct" for k in guild_cols] + [k + "_mix" for k in gad]
                           + ["phi_benthos", "phi_fish", "phi_zoo"]].round(2).to_string())
