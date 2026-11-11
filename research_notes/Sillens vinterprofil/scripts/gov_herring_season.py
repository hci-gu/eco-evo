"""Herring and sprat near the bed in winter vs summer: NS-IBTS GOV catch rates by quarter, bed depth, length and sun
elevation (box and Skagerrak), and the stock-wide GOV availability ratio Q1 / Q3 from the NSAS SAM catchabilities.

Usage: python3 -I gov_herring_season.py <gov dir> <gov_q2q4 dir> > ../tables/gov_herring_season_stdout.txt
  <gov dir>      research_notes/"Sejens djuplage" (a-umlaut)/data/gov  (NS-IBTS Q1 + Q3 2000-25, valid GOV hauls >= 56 N,
                 CPUE per length, parsed from DATRAS by that note's extract_gov.py; read in place)
  <gov_q2q4 dir> same note, data/gov_q2q4_1991_97 (Q2 May-Jun, Q4 Nov, 1991-97)

GOV = bottom trawl, headline 4-5 m (Netopening median 4.6 m): a catch rate is a near-bed availability index (the
fish within ~5 m of the bed and the sweeps' herding), not a column density. Nothing is fitted; no model output read.
Length-weight herring / sprat a 0.0060, b 3.1 (the gov_prey_overlap.py convention; ratios within a species do not
depend on it).
"""
import math
import sys

import numpy as np
import pandas as pd

gov, q24 = sys.argv[1:3]
KEY = ["Year", "Quarter", "Ship", "HaulNo", "ShootLat"]
HER, SPR = 126417, 126425


def load(d):
    H = pd.read_csv(f"{d}/hauls.csv", dtype={"Ship": str, "TimeShot": str})
    C = pd.read_csv(f"{d}/catch.csv.gz", dtype={"Ship": str})
    H["dep"] = pd.to_numeric(H["Depth"], errors="coerce")
    H = H[H.dep > 0].copy()
    C = C[C.aphia.isin([HER, SPR])].copy()
    C["kg"] = C.n_per_h * 0.006 * (C.length_cm + 0.5) ** 3.1 / 1000.0
    cols = {}
    for name, sel in (("her", (C.aphia == HER)), ("her_lt20", (C.aphia == HER) & (C.length_cm < 20)),
                      ("her_20_25", (C.aphia == HER) & (C.length_cm >= 20) & (C.length_cm < 25)),
                      ("her_ge25", (C.aphia == HER) & (C.length_cm >= 25)), ("spr", (C.aphia == SPR))):
        cols[name] = C[sel].groupby(KEY).kg.sum().rename(name)
    nher = C[C.aphia == HER].groupby(KEY).n_per_h.sum().rename("n_her")
    for s in list(cols.values()) + [nher]:
        H = H.merge(s.reset_index(), on=KEY, how="left")
    for c in list(cols) + ["n_her"]:
        H[c] = H[c].fillna(0.0)
    # sun elevation at mid-haul (DATRAS TimeShot is UTC)
    ts = H.TimeShot.astype(str).str.zfill(4)
    hh = pd.to_numeric(ts.str[:2], errors="coerce") + pd.to_numeric(ts.str[2:], errors="coerce") / 60.0
    hh = hh + pd.to_numeric(H.HaulDur, errors="coerce").fillna(30) / 120.0
    doy = pd.to_datetime(dict(year=H.Year, month=H.Month, day=H.Day), errors="coerce").dt.dayofyear
    dec = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    ha = np.radians(15 * (hh + H.ShootLong / 15.0 - 12))
    la = np.radians(H.ShootLat)
    H["elev"] = np.degrees(np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)))
    return H


H = pd.concat([load(gov), load(q24)], ignore_index=True)


def box(df):
    return df[(df.ShootLat >= 57.7) & (df.ShootLat <= 58.4) & (df.ShootLong >= 9.9) & (df.ShootLong <= 11.3)]


def ska(df):
    return df[(df.ShootLat >= 57.5) & (df.ShootLat < 58.9) & (df.ShootLong >= 8.5) & (df.ShootLong < 11.8)]


CELLS = np.array([0.233, 0.302, 0.317, 0.148])   # box cell shares by bed class (<100, 100-150, 150-250, >= 250 m)
CL = [0, 100, 150, 250, 1e4]
CLN = ["<100", "100-150", "150-250", ">=250"]

print("=" * 110)
print("1. Box GOV catch rate (kg/h, haul mean) by quarter and bed class; Q1/Q3 2000-25, Q2/Q4 1991-97 (other years)")
print("=" * 110)
B = box(H)
res = {}
for q in (1, 2, 3, 4):
    d = B[B.Quarter == q]
    if d.empty:
        continue
    line = f"  Q{q} months {sorted(d.Month.unique())} years {d.Year.min()}-{d.Year.max()}"
    print(line)
    for col in ("her", "her_lt20", "her_20_25", "her_ge25", "spr"):
        r, n = [], []
        for lo, hi in zip(CL[:-1], CL[1:]):
            m = (d.dep >= lo) & (d.dep < hi)
            r.append(d.loc[m, col].mean() if m.any() else float("nan"))
            n.append(int(m.sum()))
        res[(q, col)] = np.array(r)
        print(f"    {col:10s} " + "  ".join(f"{c:>7s} {v:7.1f} (n{k:3d})" for c, v, k in zip(CLN, r, n)))

print("\n  Ratio Q1 / Q3 by bed class (box):")
for col in ("her", "her_lt20", "her_20_25", "her_ge25", "spr"):
    r = res[(1, col)] / res[(3, col)]
    print(f"    {col:10s} " + "  ".join(f"{c:>7s} {v:6.2f}" for c, v in zip(CLN, r)))
print("  Ratio Q4 (Nov 1991-97) / Q3 (2000-25), different years [indicative]:")
for col in ("her", "her_lt20"):
    if (4, col) in res:
        r = res[(4, col)] / res[(3, col)]
        print(f"    {col:10s} " + "  ".join(f"{c:>7s} {v:6.2f}" for c, v in zip(CLN, r)))

print("\n" + "=" * 110)
print("2. Skagerrak 57.5-58.9 N, 8.5-11.8 E: herring and sprat kg/h by finer bed bins, Q1 vs Q3 (2000-25)")
print("=" * 110)
S = ska(H)
BINS = [0, 75, 100, 125, 150, 175, 200, 250, 300, 600]
for col in ("her", "her_lt20", "her_20_25", "her_ge25", "spr"):
    for q in (1, 3):
        d = S[S.Quarter == q]
        out = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (d.dep >= lo) & (d.dep < hi)
            out.append((d.loc[m, col].mean() if m.sum() >= 3 else float("nan"), int(m.sum())))
        print(f"  {col:10s} Q{q} " + " ".join(f"{lo}-{hi}:{v:6.1f}(n{k})" for (lo, hi), (v, k) in
                                               zip(zip(BINS[:-1], BINS[1:]), out)))

print("\n  Herring length composition (share of kg/h in < 20 / 20-25 / >= 25 cm), Skagerrak, by bed bin and quarter")
for q in (1, 3):
    d = S[S.Quarter == q]
    for lo, hi in ((0, 100), (100, 150), (150, 250), (250, 600)):
        m = (d.dep >= lo) & (d.dep < hi)
        t = d.loc[m, "her"].sum()
        if t > 0:
            print(f"    Q{q} {lo:3d}-{hi:3d} m: " + " / ".join(f"{d.loc[m, c].sum() / t:.2f}" for c in
                                                         ("her_lt20", "her_20_25", "her_ge25")) + f"  (n {int(m.sum())})")

print("\n" + "=" * 110)
print("3. Time of day in Q1 and Q3: herring kg/h by sun elevation at mid-haul (all hauls >= 56 N, 3-12 E, so the")
print("   eastern North Sea + Skagerrak; bed >= 100 m and < 100 m). Year-demeaned log(1 + kg/h) means in brackets.")
print("=" * 110)
E = H[(H.ShootLong >= 3) & (H.ShootLong <= 12) & (H.ShootLat >= 56)].copy()
ECL = [(-90, -6, "< -6 (dark)"), (-6, 0, "-6..0 (twilight)"), (0, 10, "0..10"), (10, 90, "> 10")]
for q in (1, 3):
    for lo_d, hi_d, lab in ((0, 100, "bed < 100 m"), (100, 1000, "bed >= 100 m")):
        d = E[(E.Quarter == q) & (E.dep >= lo_d) & (E.dep < hi_d)].copy()
        for col in ("her_lt20", "her"):
            d["lg"] = np.log1p(d[col])
            d["lgd"] = d.lg - d.groupby("Year").lg.transform("mean")
            row = []
            for a, b, nm in ECL:
                m = (d.elev >= a) & (d.elev < b)
                row.append(f"{nm}: {d.loc[m, col].mean() if m.sum() else float('nan'):6.1f} kg/h "
                           f"[{d.loc[m, 'lgd'].mean() if m.sum() else float('nan'):+.2f}] (n{int(m.sum())})")
            print(f"  Q{q} {lab:13s} {col:9s} " + "; ".join(row))

print("\n" + "=" * 110)
print("4. Stock-wide GOV availability, NSAS herring, same cohort Feb vs Aug (HAWG 2023 Table 2.6.2.2, SAM")
print("   catchabilities; IBTS-Q1 and IBTS-Q3 indices both delta-GAM standardised from DATRAS, NSAS stock annex B.3.2)")
print("=" * 110)
q1, q1cv = 0.1059, 0.06679
q3, q3cv = 0.04618, 0.1219
r = q1 / q3
cv = math.sqrt(q1cv ** 2 + q3cv ** 2)
print(f"  q IBTS-Q1 age 1 (1-ringers in Feb) {q1} (CV {q1cv}); q IBTS-Q3 age 1 (the same 1-ringers in Aug) {q3} (CV {q3cv})")
print(f"  ratio Q1 / Q3 = {r:.2f} (approx. 90 % range {r * math.exp(-1.645 * cv):.2f}-{r * math.exp(1.645 * cv):.2f})")
# size: Feb 1-ringers ~ 12-15 cm, Aug 1-ringers ~ 17-21 cm. Walker et al. 2017 GOV herring efficiency at length:
# Absolute 5-cm classes 10: 0.3748, 15: 0.3707, 20: 0.2643; final Efficiency flat 0.2437 (Vitlingens fiskbyten
# step2_gov_catchability.py inputs).
for lab, s in (("Walker Efficiency (flat)", 1.0), ("Walker Absolute 15 vs 20 cm class", 0.3707 / 0.2643),
               ("Walker Absolute 10-15 vs 15-20 mid", 0.3727 / 0.3175)):
    print(f"  size correction ({lab}): x1/{s:.2f} -> availability ratio {r / s:.2f}")
print("  The index is Q1 Jan-Feb vs Q3 Jul-Aug; the ratio carries any seasonal change in the fraction of herring within")
print("  the GOV's ~5 m above the bed AND any change in horizontal coverage (inshore < 10 m water is unsampled in both).")
