"""NS-IBTS GOV herring (and benthivore) density by bed-depth class inside the model grid (reconcile step B).

Usage: python3 -I reconcile_gov_nearbed.py <extract_2000_20> <extract_2021_25> <guilds_csv> <lw_csv> <out_dir>
  extracts: compact DATRAS CSVs (HH.csv.gz, CPUEL.csv.gz) - research_notes/Fiskgrupper i rutan/data/extract and
  research_notes/Torskfiskarnas artsammansättning/data/extract (untrusted downloads, parsed only).
  Writes <out_dir>/reconcile_gov_by_class.csv.

Footing [D]: hauls whose shoot position falls inside the model grid (SWEREF 99 TM E 211-271 km,
N 6414-6474 km), GOV, valid, day hauls, 2001-25, Q1 (Jan-Feb) and Q3 (Aug-Sep); the bbox
57.7-58.4 N / 9.9-11.3 E as a sensitivity. Bed-depth classes as the HERAS step (haul Depth).
Density per haul = catch numbers (CPUE/h x duration) x W(L) / (wing spread x distance); q = 1 ("raw").
Length correction (variant "Lq"): Walker et al. 2017 Table 3 herring multipliers, interpolated on
length, used RELATIVE to the adult value (biomass-weighted q of fish >= 20 cm in the same class) - i.e.
only the size selectivity is corrected, not the level. The level of Walker's q (0.22-0.30 for adults)
is anchored to assessment numbers and "ignore[s] the vertical component" (Walker 2017 text, quoted in
../../Vitlingens fiskbyten/scripts/step2_gov_catchability.py), so it CONTAINS the vertical
availability and cannot be used to lift a near-bed share.
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/simon/eco-evo/tools")
from make_fishing_maps import CELL, E0, GRID, N0, wgs84_to_sweref99tm  # noqa: E402  (repo code)

old, new, guilds_csv, lw_csv, out = sys.argv[1:6]
G = pd.read_csv(guilds_csv).set_index("aphia")
LW = pd.read_csv(lw_csv, index_col=0)
HALF_CM = {126417, 126425, 126426, 126421, 125516, 126751, 126756, 126754, 125909}
HER = 126417
BEN = set(G.index[G.guild == "B"])
CLASSES = [(0, 80), (80, 110), (110, 130), (130, 150), (150, 200), (200, 300), (300, 1e4)]
LAB = ["<80", "80-110", "110-130", "130-150", "150-200", "200-300", ">=300"]
# Walker 2017 Table 3 (herring, length-at-age, multiplier); flat beyond the end points
T3 = [(9.1, 0.11), (12.6, 0.41), (19.5, 0.30), (23.1, 0.26), (24.6, 0.22), (25.2, 0.28), (25.7, 0.22),
      (26.2, 0.22), (26.8, 0.22)]


def q_t3(L):
    x = np.array([p[0] for p in T3]); y = np.array([p[1] for p in T3])
    return np.interp(L, x, y)


hh = pd.concat([pd.read_csv(os.path.join(d, "HH.csv.gz")) for d in (old, new)])
hh = hh[(hh.HaulVal.astype(str).str.strip() == "V") & hh.Quarter.isin([1, 3])
        & (hh.Gear.astype(str).str.strip() == "GOV")]
for c in ["Ship", "StatRec", "DayNight"]:
    hh[c] = hh[c].astype(str).str.strip()
hh["la"], hh["lo"] = hh.ShootLat.round(3), hh.ShootLong.round(3)
hh = hh.drop_duplicates(["Year", "Quarter", "Ship", "HaulNo", "la", "lo"])
hh = hh[(hh.Year >= 2001) & (hh.Year <= 2025)]
e, n = wgs84_to_sweref99tm(hh.ShootLong.values, hh.ShootLat.values)
col = np.floor((e - E0) / CELL); row = GRID - 1 - np.floor((n - N0) / CELL)
hh["grid"] = (col >= 0) & (col < GRID) & (row >= 0) & (row < GRID)
hh["bbox"] = hh.ShootLat.between(57.7, 58.4) & hh.ShootLong.between(9.9, 11.3)
hh = hh[hh.bbox | hh.grid].copy()
# swept area per haul (km2): wing spread x distance, with fallbacks
ws = hh.WingSpread.where(hh.WingSpread.between(8, 35))
ws = ws.fillna(hh.groupby("Ship").WingSpread.transform(lambda s: s[s.between(8, 35)].median())).fillna(18.0)
dist = hh.Distance.where(hh.Distance.between(800, 9000))
dist = dist.fillna(hh.GroundSpeed.where(hh.GroundSpeed.between(2, 6)) * 1852 * hh.HaulDur / 60).fillna(
    4.0 * 1852 * hh.HaulDur / 60)
hh["swept_km2"] = ws * dist / 1e6
hh["opening"] = hh.Netopening.where(hh.Netopening.between(2, 10))
hh["k"] = pd.cut(hh.Depth, [c[0] for c in CLASSES] + [1e5], labels=LAB, right=False)
hh["key"] = list(zip(hh.Year, hh.Quarter, hh.Ship, hh.HaulNo, hh.la, hh.lo))
print(f"hauls in bbox or grid (Q1+Q3, 2001-25): {len(hh)}; in grid {int(hh.grid.sum())}; "
      f"night hauls {int((hh.DayNight == 'N').sum())} (dropped)")
print(f"wing spread median {ws.median():.1f} m, distance median {dist.median():.0f} m, swept {hh.swept_km2.median():.4f} km2, "
      f"net opening median {hh.opening.median():.1f} m (p10 {hh.opening.quantile(.1):.1f}, p90 {hh.opening.quantile(.9):.1f})")
hh = hh[hh.DayNight == "D"]

cp = pd.concat([pd.read_csv(os.path.join(d, "CPUEL.csv.gz")) for d in (old, new)])
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp["key"] = list(zip(cp.Year, cp.Quarter, cp.Ship, cp.HaulNo, cp.ShootLat.round(3), cp.ShootLon.round(3)))
cp = cp[cp.key.isin(set(hh.key))].copy()
cp["L"] = cp.LngtClas / 10.0 + np.where(cp.AphiaID.isin(HALF_CM), 0.25, 0.5)
cp = cp[cp.AphiaID.isin(LW.index)]
cp["w_kg"] = cp.AphiaID.map(LW.a) * cp.L ** cp.AphiaID.map(LW.b) / 1000
dur = hh.set_index("key").HaulDur
cp["n"] = cp.CPUE_number_per_hour * cp.key.map(dur) / 60
cp["kg"] = cp.n * cp.w_kg
her = cp[cp.AphiaID == HER].copy()
her["q"] = q_t3(her.L.values)

hh = hh.set_index("key")
sw = hh.swept_km2
hh["her_raw"] = her.groupby("key").kg.sum().reindex(hh.index).fillna(0) / 1000 / sw
hh["her_lt13"] = her[her.L < 13].groupby("key").kg.sum().reindex(hh.index).fillna(0) / 1000 / sw
hh["ben"] = cp[cp.AphiaID.isin(BEN)].groupby("key").kg.sum().reindex(hh.index).fillna(0) / 1000 / sw
hh["wht"] = cp[cp.AphiaID == 126438].groupby("key").kg.sum().reindex(hh.index).fillna(0) / 1000 / sw
hh["kgh_her"] = her.groupby("key").kg.sum().reindex(hh.index).fillna(0) / (hh.HaulDur / 60)


def lq_density(sel_keys):
    """Size-selectivity-corrected herring density: divide by q(L)/q_adult (adults >= 20 cm, same hauls)."""
    h = her[her.key.isin(sel_keys)]
    if h.kg.sum() <= 0:
        return 1.0
    ad = h[h.L >= 20]
    qa = (ad.kg * ad.q).sum() / ad.kg.sum() if ad.kg.sum() > 0 else 0.26
    corr = (h.kg / (h.q / qa)).sum() / h.kg.sum()
    return corr


rng = np.random.default_rng(1)
rows = []
for foot in ("grid", "bbox"):
    for q in (3, 1):
        s = hh[hh[foot] & (hh.Quarter == q)]
        print(f"\n--- {foot}, Q{q} (months {sorted(s.Month.unique())}), day hauls {len(s)} ---")
        print("class    hauls  years | herring t/km2 raw (boot 10-90 %)  share of hauls with herring | "
              "<13 cm share | Lq factor | herring kg/h | mean L (cm) | benthivores t/km2 | net opening m")
        for k in LAB:
            c = s[s.k == k]
            if len(c) == 0:
                continue
            boots = [c.her_raw.values[rng.integers(0, len(c), len(c))].mean() for _ in range(2000)]
            hk = her[her.key.isin(set(c.index))]
            mL = (hk.L * hk.kg).sum() / hk.kg.sum() if hk.kg.sum() > 0 else float("nan")
            f_lq = lq_density(set(c.index))
            lt13 = c.her_lt13.sum() / c.her_raw.sum() if c.her_raw.sum() > 0 else float("nan")
            print(f"{k:8s} {len(c):5d} {c.Year.nunique():5d} | {c.her_raw.mean():6.3f} ({np.percentile(boots, 10):.3f}-"
                  f"{np.percentile(boots, 90):.3f})  {np.mean(c.her_raw > 0):.2f} | {lt13:.2f} | x{f_lq:.2f} | "
                  f"{c.kgh_her.mean():7.1f} | {mL:5.1f} | {c.ben.mean():6.3f} | {c.opening.mean():.1f}")
            rows.append(dict(footing=foot, quarter=q, cls=k, hauls=len(c), years=c.Year.nunique(),
                             her_raw=c.her_raw.mean(), her_p10=np.percentile(boots, 10),
                             her_p90=np.percentile(boots, 90), her_lq_factor=f_lq, her_lt13_share=lt13,
                             her_kgh=c.kgh_her.mean(), her_meanL=mL, ben=c.ben.mean(), opening=c.opening.mean()))
        # depth-stratified grid mean with the grid's own cell shares (maps/depth.npz)
pd.DataFrame(rows).to_csv(os.path.join(out, "reconcile_gov_by_class.csv"), index=False)
# per-haul values for the bootstrap, and herring biomass per unit backscatter (kg per m2 sigma_bs, TS 20logL-71.2)
her["sig"] = 10 ** ((20 * np.log10(her.L) - 71.2) / 10) * her.n
hh["her_kg"] = her.groupby("key").kg.sum().reindex(hh.index).fillna(0)
hh["her_sig"] = her.groupby("key").sig.sum().reindex(hh.index).fillna(0)
hh["her_lqf"] = [1.0] * len(hh)
out_h = hh[["Year", "Quarter", "Month", "Depth", "k", "grid", "bbox", "her_raw", "her_lt13", "ben", "wht", "her_kg", "her_sig"]]
out_h.to_csv(os.path.join(out, "reconcile_gov_hauls.csv"), index=False)
print(f"wrote {os.path.join(out, 'reconcile_gov_hauls.csv')} ({len(out_h)} day hauls)")
print(f"\nwrote {os.path.join(out, 'reconcile_gov_by_class.csv')}")
print("t/km2 per kg/h (herring, grid Q3 hauls): "
      f"{(hh[hh.grid & (hh.Quarter == 3)].her_raw.sum() / hh[hh.grid & (hh.Quarter == 3)].kgh_her.sum()):.4f}")
