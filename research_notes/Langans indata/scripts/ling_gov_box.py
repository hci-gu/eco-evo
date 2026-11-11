"""Ling (Molva molva, AphiaID 126461) in NS-IBTS GOV, Q1 + Q3 2000-25: haul counts, numbers, length composition
and bed-depth class shape in the box and the wider Skagerrak / Norwegian Trench window; the adult benthivores
(cod >= 27.1 cm, haddock >= 27 cm, raw catch rates) in the same hauls for comparison.

Usage: python3 -I ling_gov_box.py <extract_2000_20> <extract_2021_25> <lw_csv> <depth_npz>
  extracts as research_notes/"Rekryternas bottendjup"/scripts/gov_juv_bed.py (HH.csv.gz, CPUEL.csv.gz; ICES DATRAS,
  CC BY 4.0); the haul selection, regions, bed classes and the equal-cell class normalisation are copied from it.
Biomass per hour: n/h x W(L), W from the CA fit in length_weight_all.csv (ling: as used for N/t in
research_notes/"Attackrater ur motesteori" pred_numbers_per_tonne.py), L = class + 0.5 cm (DATRAS LngtClas mm).
No catchability correction (no GOV q for ling). Nothing is fitted.
"""
import os
import sys

import numpy as np
import pandas as pd

E1, E2, LWF, DEP = sys.argv[1:5]
EDGES = [100.0, 150.0, 250.0]
CLS = ["<100", "100-150", "150-250", ">=250"]
REG = {"box": (57.7, 58.4, 9.9, 11.3), "wide": (57.3, 59.2, 8.0, 12.0)}
z = np.abs(np.load(DEP)["depth"].astype(float))
z = z[np.isfinite(z)]
CELL = np.array([(np.digitize(z, EDGES) == k).mean() for k in range(4)])

hh = pd.concat([pd.read_csv(os.path.join(d, "HH.csv.gz")) for d in (E1, E2)])
hh = hh[(hh.HaulVal.astype(str).str.strip() == "V") & (hh.Gear.astype(str).str.strip() == "GOV")
        & hh.Quarter.isin([1, 3]) & (hh.BySpecRecCode == 1) & (hh.Depth > 0)].copy()
hh["Ship"] = hh.Ship.astype(str).str.strip()
hh["la"], hh["lo"] = hh.ShootLat.round(3), hh.ShootLong.round(3)
a0, a1, o0, o1 = REG["wide"]
hh = hh[(hh.ShootLat >= a0) & (hh.ShootLat <= a1) & (hh.ShootLong >= o0) & (hh.ShootLong <= o1)]
hh = hh.drop_duplicates(["Year", "Quarter", "Ship", "HaulNo", "la", "lo"]).copy()
hh["key"] = list(zip(hh.Year, hh.Quarter, hh.Ship, hh.HaulNo, hh.la, hh.lo))
hh["cl"] = np.digitize(hh.Depth, EDGES)
a0, a1, o0, o1 = REG["box"]
hh["box"] = (hh.ShootLat >= a0) & (hh.ShootLat <= a1) & (hh.ShootLong >= o0) & (hh.ShootLong <= o1)
hh = hh.set_index("key")

lw = pd.read_csv(LWF, index_col=0)
print("ling length-weight in length_weight_all.csv (g, cm):", lw.loc[126461, ["a", "b"]].to_dict())
cp = pd.concat([pd.read_csv(os.path.join(d, "CPUEL.csv.gz")) for d in (E1, E2)])
cp = cp[cp.AphiaID.isin([126461, 126436, 126437])].copy()
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp["key"] = list(zip(cp.Year, cp.Quarter, cp.Ship, cp.HaulNo, cp.ShootLat.round(3), cp.ShootLon.round(3)))
cp = cp[cp.key.isin(set(hh.index))].copy()
cp["L"] = cp.LngtClas / 10.0 + 0.5
cp["kg"] = cp.CPUE_number_per_hour * cp.AphiaID.map(lw.a) * cp.L ** cp.AphiaID.map(lw.b) / 1000.0
ling = cp[cp.AphiaID == 126461]
adult = cp[((cp.AphiaID == 126436) & (cp.L >= 27.1)) | ((cp.AphiaID == 126437) & (cp.L >= 27.0))]

for reg in ("box", "wide"):
    h = hh[hh.box] if reg == "box" else hh
    lr = ling[ling.key.isin(set(h.index))]
    nh = lr.groupby("key").CPUE_number_per_hour.sum()
    print(f"\n== {reg}: hauls {len(h)} (Q1 {int((h.Quarter == 1).sum())}, Q3 {int((h.Quarter == 3).sum())}); hauls with ling"
          f" {int((nh > 0).sum())}; ling n/h summed over hauls {nh.sum():.1f} (~individuals at 1 h tows)")
    print(f"   hauls with ling by year: " + ", ".join(f"{y}:{c}" for y, c in
                                                     h.loc[nh[nh > 0].index].groupby("Year").size().items()))
    # length composition (numbers and biomass), 10-cm classes
    bins = np.arange(0, 210, 10)
    n = np.histogram(lr.L, bins=bins, weights=lr.CPUE_number_per_hour)[0]
    b = np.histogram(lr.L, bins=bins, weights=lr.kg)[0]
    print("   length class (cm): numbers share / biomass share")
    for lo_, nn, bb in zip(bins[:-1], n / n.sum(), b / b.sum()):
        if nn > 0 or bb > 0:
            print(f"     {lo_:3.0f}-{lo_ + 10:3.0f}: {nn:.3f} / {bb:.3f}")
    Lnum = (lr.L * lr.CPUE_number_per_hour).sum() / lr.CPUE_number_per_hour.sum()
    srt = lr.sort_values("L")
    cb = srt.kg.cumsum() / srt.kg.sum()
    print(f"   number-mean L {Lnum:.1f} cm; biomass-median L {srt.L[cb.values >= 0.5].iloc[0]:.1f} cm; "
          f"L range {lr.L.min():.0f}-{lr.L.max():.0f} cm; share of numbers >= 60 cm "
          f"{lr.CPUE_number_per_hour[lr.L >= 60].sum() / lr.CPUE_number_per_hour.sum():.2f}")
    # bed-class shape: pooled haul mean kg/h per class, equal-cell normalised (as gov_juv_bed.py)
    for name, src in (("ling", lr), ("adult cod+haddock", adult[adult.key.isin(set(h.index))])):
        per = src.groupby("key").kg.sum().reindex(h.index).fillna(0.0)
        d = np.array([per[h.cl == k].mean() if (h.cl == k).any() else np.nan for k in range(4)])
        nk = [int((h.cl == k).sum()) for k in range(4)]
        ok = np.isfinite(d)
        f = d / np.nansum(CELL[ok] * d[ok])
        print(f"   {name:18s} kg/h per class {np.round(d, 3).tolist()} (hauls {nk}); shape f (equal-cell mean 1)"
              f" {np.round(f, 2).tolist()}")
print(f"\nmodel map cell shares per class {np.round(CELL, 3).tolist()} ({CLS})")

# co-occurrence with the benthivore prey shapes the gate uses, relative to the water-column group's shape
# (lib/diagnostics/field_diet.FIELD_BED_SHARES predatory_fish = saithe, GOV in the box: class shares)
WC = np.array([0.043, 0.127, 0.782, 0.048])
SHAPES = {"J recruit_depth_table": np.array([2.71, 1.00, 0.21, 0.05]), "A GOV annual": np.array([1.33, 1.48, 0.59, 0.41])}
print("\nco-occurrence F = sum_k s_pred,k f_prey,k (class shares of the predator x prey shape, equal-cell mean 1):")
for reg in ("box", "wide"):
    h = hh[hh.box] if reg == "box" else hh
    lr = ling[ling.key.isin(set(h.index))]
    per = lr.groupby("key").kg.sum().reindex(h.index).fillna(0.0)
    d = np.array([per[h.cl == k].mean() if (h.cl == k).any() else 0.0 for k in range(4)])
    s = CELL * d / (CELL * d).sum()
    print(f"  ling {reg}: class shares {np.round(s, 3).tolist()} (WC group {WC.tolist()})")
    for nm, f in SHAPES.items():
        print(f"    {nm:22s} F ling {float(s @ f):.3f}, F WC group {float(WC @ f):.3f}, ratio {float(s @ f) / float(WC @ f):.2f}")
