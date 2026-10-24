"""Gadoid biomass composition in the Mareld box and the Skagerrak from DATRAS NS-IBTS.

Usage: python3 -I gadoid_mix.py <extract_dir> <depth_npz> <out_dir>

Inputs (untrusted data, only parsed): HH.csv.gz, CPUEL.csv.gz, CA.csv.gz written by
extract_datras.py; the repo's maps/depth.npz (read only).

Method (same as research_notes/Fiskeridodlighet i Skagerrak idag/ices_gadoid_f.md, extended):
- valid GOV hauls (HaulVal V) in Q1 and Q3 2021-2025; a species absent from a haul is a zero;
- CPUE per 1-cm length class (DATRAS lower bound, mm) -> weight at the class midpoint;
- length-weight W = a L^b (g, cm) fitted to NS-IBTS CA individual weights 2021-2025 (n >= 50,
  3-SD log-residual trim), else FishBase Bayesian LWR (summary pages, fetched 2026-10-24);
- kg/h averaged over hauls (arithmetic) and as rectangle means then mean;
- swept-area density at q = 1 with wing spread x distance per haul.
"""
import os
import sys

import numpy as np
import pandas as pd

ext, depth_npz, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)

SPECIES = {  # AphiaID: (short name, FishBase a, b)
    126437: ("haddock", None, None),
    126438: ("whiting", None, None),
    126441: ("saithe", None, None),
    126436: ("cod", None, None),
    126484: ("hake", None, None),
    126461: ("ling", 0.00380, 3.08),
    126459: ("blue ling", 0.00282, 3.05),
    126447: ("tusk", 0.00525, 3.10),
    126440: ("pollack", 0.00692, 3.08),
    126501: ("greater forkbeard", 0.00457, 3.14),
    126446: ("poor cod", 0.00741, 3.11),
    126445: ("bib", 0.00741, 3.10),
    126435: ("silvery pout", 0.00661, 3.08),
    126450: ("fourbeard rockling", 0.00407, 3.07),
    126458: ("threebeard rockling", 0.00513, 2.99),
    126444: ("Norway pout", 0.00603, 3.06),
    126439: ("blue whiting", None, None),
}
ZOOPL = ["Norway pout", "blue whiting"]
BOX_RECTS = ["44G0", "44G1", "45G0", "45G1"]
BOX_LAT, BOX_LON = (57.81, 58.31), (10.07, 11.15)

# ---------------------------------------------------------------- length-weight
ca = pd.read_csv(os.path.join(ext, "CA.csv.gz"), low_memory=False)
ca["LngtCode"] = ca["LngtCode"].astype(str).str.strip()
ca = ca[(ca.Year >= 2021) & (ca.Year <= 2025) & ca.Quarter.isin([1, 3])]
scale = {"1": 1.0, ".": 0.1, "0": 0.1, "5": 1.0}
ca["L"] = ca.LngtClass * ca.LngtCode.map(scale)
# midpoint of the recorded class: 1 cm classes +0.5, mm +0.05, 0.5 cm classes +0.25
ca["L"] += ca.LngtCode.map({"1": 0.5, ".": 0.05, "0": 0.25, "5": 2.5})
lw = []
for aph, (name, fa, fb) in SPECIES.items():
    k = ca[(ca.Valid_Aphia == aph) & (ca.L > 0) & (ca.IndWgt > 0)]
    row = dict(species=name, n=len(k), a_fb=fa, b_fb=fb)
    if len(k) >= 50:
        x, y = np.log(k.L.values), np.log(k.IndWgt.values)
        for _ in range(2):
            b, la = np.polyfit(x, y, 1)
            r = y - (la + b * x)
            keep = np.abs(r) < 3 * r.std()
            x, y = x[keep], y[keep]
        b, la = np.polyfit(x, y, 1)
        row.update(a=np.exp(la), b=b, n_used=len(x), source="DATRAS CA fit")
    else:
        row.update(a=fa, b=fb, n_used=0, source="FishBase")
    lw.append(row)
lw = pd.DataFrame(lw).set_index("species")
lw.to_csv(os.path.join(out, "length_weight.csv"))

# ---------------------------------------------------------------- hauls
hh = pd.read_csv(os.path.join(ext, "HH.csv.gz"))
hh = hh[(hh.HaulVal.str.strip() == "V") & hh.Quarter.isin([1, 3]) & (hh.Gear.str.strip() == "GOV")]
for c in ["Country", "Ship", "StatRec"]:
    hh[c] = hh[c].astype(str).str.strip()
hh = hh.drop_duplicates(["Year", "Quarter", "Ship", "HaulNo"])
lat, lon = hh.ShootLat, hh.ShootLong
# Skagerrak: east of the Hanstholm (8.60 E, 57.12 N) - Lindesnes (7.05 E, 57.98 N) line,
# excluding the Kattegat (south of 57.75 N east of 10.6 E); north limit 59.2 N.
x1, y1, x2, y2 = 8.60, 57.12, 7.05, 57.98
side = (x2 - x1) * (lat - y1) - (y2 - y1) * (lon - x1)
east = side < 0
hh["skag"] = east & (lat < 59.2) & ~((lon > 10.6) & (lat < 57.75)) & (lon < 12.5) & (lat > 57.0)
hh["box"] = hh.StatRec.isin(BOX_RECTS)
hh["box_strict"] = lat.between(*BOX_LAT) & lon.between(*BOX_LON)
hh["dbin"] = pd.cut(hh.Depth, [0, 100, 200, 300, 600], labels=["<100", "100-200", "200-300", ">=300"],
                    right=False)
hh["region"] = hh.skag | hh.box | hh.box_strict  # Skagerrak plus the box rectangles
hh = hh[hh.region].copy()
# swept area per hour at wing spread (km2/h); fall back to medians where -9
ws = hh.WingSpread.where(hh.WingSpread > 0, np.nan)
dist = hh.Distance.where(hh.Distance > 0, np.nan)
dist = dist.fillna(hh.GroundSpeed.where(hh.GroundSpeed > 0) * 1852 * hh.HaulDur / 60)
hh["km2_per_h"] = (ws.fillna(ws.median()) / 1000) * (dist.fillna(dist.median()) / 1000) / (hh.HaulDur / 60)
hh.to_csv(os.path.join(out, "hauls_region.csv"), index=False)

# ---------------------------------------------------------------- CPUE -> kg/h
cp = pd.read_csv(os.path.join(ext, "CPUEL.csv.gz"))
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp = cp[cp.AphiaID.isin(SPECIES)]
cp["species"] = cp.AphiaID.map(lambda a: SPECIES[a][0])
cp["L"] = cp.LngtClas / 10.0 + 0.5
cp["a"] = cp.species.map(lw.a)
cp["b"] = cp.species.map(lw.b)
cp["kg_h"] = cp.CPUE_number_per_hour * cp.a * cp.L ** cp.b / 1000
cp["key"] = list(zip(cp.Year, cp.Quarter, cp.Ship, cp.HaulNo))
hh["key"] = list(zip(hh.Year, hh.Quarter, hh.Ship, hh.HaulNo))
m = cp.groupby(["key", "species"]).kg_h.sum().unstack(fill_value=0.0)
names = [v[0] for v in SPECIES.values()]
H = hh.set_index("key").join(m, how="left")
for s in names:
    if s not in H:
        H[s] = 0.0
H[names] = H[names].fillna(0.0)
H.reset_index().to_csv(os.path.join(out, "haul_species_kgh.csv"), index=False)


def mean_kgh(df, how="haul"):
    if how == "haul":
        return df[names].mean()
    return df.groupby("StatRec")[names].mean().mean()


def share_table(sel, label):
    rows = []
    for q, qname in [((1, 3), "Q1+Q3"), ((1,), "Q1"), ((3,), "Q3")]:
        d = H[sel & H.Quarter.isin(q)]
        for how in ["haul", "rect"]:
            mu = mean_kgh(d, how)
            rows.append(dict(area=label, quarter=qname, avg=how, n=len(d), **mu.to_dict()))
    return pd.DataFrame(rows)


tabs = pd.concat([share_table(H.box, "box rects"), share_table(H.box_strict, "box strict"),
                  share_table(H.skag, "Skagerrak")])
tabs.to_csv(os.path.join(out, "mean_kgh.csv"), index=False)

# per haul depth bins, within Skagerrak (for the depth-stratified box estimate)
# depth bins from all region hauls (Skagerrak + box rectangles)
db = H.groupby("dbin", observed=False)[names].mean()
db.insert(0, "n", H.groupby("dbin", observed=False).size())
db.to_csv(os.path.join(out, "region_kgh_by_depth.csv"))
dbq = H.groupby(["Quarter", "dbin"], observed=False)[names].mean()
dbq.insert(0, "n", H.groupby(["Quarter", "dbin"], observed=False).size())
dbq.to_csv(os.path.join(out, "region_kgh_by_quarter_depth.csv"))

# box area share by depth bin
z = np.load(depth_npz)["depth"].astype(float)
z = np.abs(z[np.isfinite(z)])
area = pd.cut(pd.Series(z), [0, 100, 200, 300, 600], labels=db.index, right=False).value_counts(
    normalize=True).reindex(db.index)
area.to_csv(os.path.join(out, "box_area_by_depth.csv"))
strat = (db[names].mul(area, axis=0)).sum()  # depth-stratified box kg/h, Skagerrak hauls

# coverage: hauls per rectangle x depth bin (box rects) and per quarter
cov = pd.crosstab([H[H.box].StatRec], H[H.box].dbin)
cov.to_csv(os.path.join(out, "box_hauls_rect_depth.csv"))
covq = pd.crosstab([H[H.box].Year, H[H.box].Quarter], H[H.box].StatRec)
covq.to_csv(os.path.join(out, "box_hauls_year_quarter.csv"))

# recording check: minor species CPUE by country in Skagerrak hauls
minor = ["ling", "pollack", "greater forkbeard", "poor cod", "bib", "silvery pout", "fourbeard rockling",
         "threebeard rockling", "blue ling", "tusk"]
rec = H.groupby("Country")[minor].agg(lambda v: (v > 0).sum())
rec.insert(0, "hauls", H.groupby("Country").size())
rec.to_csv(os.path.join(out, "minor_species_positive_hauls_by_country.csv"))

# swept-area density at q = 1 (t/km2), haul mean of kg/km2
dens = H[names].div(H.km2_per_h, axis=0) / 1000
dens_box = dens[H.box & H.Quarter.isin([1, 3])].mean()
dens_box_rect = dens[H.box].groupby(H[H.box].StatRec).mean().mean()
dens_skag = dens[H.skag].mean()
pd.DataFrame(dict(box_haul=dens_box, box_rect=dens_box_rect, skag=dens_skag,
                  box_depth_strat_kgh_no300=strat)).to_csv(os.path.join(out, "density_q1.csv"))

# ---------------------------------------------------------------- print summary
pd.set_option("display.width", 250, "display.max_columns", 30)
print(lw[["n", "n_used", "a", "b", "a_fb", "b_fb", "source"]].round(5))
print("\nhauls: Skagerrak", int(H.skag.sum()), "box rects", int(H.box.sum()), "box strict",
      int(H.box_strict.sum()))
print("CPUE hauls not in HH list:", len(set(m.index) - set(hh.key)))
print("\ncoverage box rects x depth\n", cov)
print("\nbox area share by depth\n", area.round(3))
print("\nregion hauls by depth bin\n", db["n"])
print("\nyear x quarter x rect\n", covq)
t = tabs.copy()
t["total"] = t[names].sum(axis=1)
print("\nmean kg/h\n", t[["area", "quarter", "avg", "n", "total"] + names].round(2).to_string())
print("\npositive hauls by country\n", rec)
print("\ndensity t/km2 at q=1\n", pd.DataFrame(dict(box=dens_box, boxrect=dens_box_rect, skag=dens_skag)).round(4))
print("totals", dens_box.sum(), dens_box_rect.sum(), dens_skag.sum())
print("\ndepth-stratified box kg/h (region hauls x box area; >=300 m bin missing, see phi_mix.py)\n", strat.round(2), strat.sum())
print("\nkg/h by depth bin\n", db.round(2).T)

# ---------------------------------------------------------------- bootstrap over hauls (box rects, Q1+Q3)
rng = np.random.default_rng(1)
core = [s for s in names if s not in ZOOPL]
B = H[H.box]
boot = {"with": [], "without": []}
for _ in range(2000):
    d = pd.concat([g.sample(len(g), replace=True, random_state=rng.integers(1 << 31)) for _, g in B.groupby("Quarter")])
    mu = d[names].mean()
    boot["with"].append(mu / mu.sum())
    boot["without"].append(mu[core] / mu[core].sum())
for k, v in boot.items():
    v = pd.DataFrame(v)
    ci = pd.DataFrame(dict(p05=v.quantile(0.05), p50=v.quantile(0.5), p95=v.quantile(0.95)))
    ci.to_csv(os.path.join(out, f"box_share_bootstrap_{k}_zoopl.csv"))
    print(f"\nbootstrap share CI, box rects Q1+Q3, {k} Norway pout/blue whiting\n", (100 * ci).round(2))
# single-haul leverage
top = B[names].div(B[names].sum(axis=0), axis=1).max()
print("\nlargest single-haul share of the box sum by species\n", top.round(2))
