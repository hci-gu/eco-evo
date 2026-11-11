"""Seasonal check of the haddock + cod juvenile bed shape: NS-IBTS GOV Q2 (May) and Q4 (Nov) 1991-97, the only
years the quarterly survey ran. Haddock and cod only (the extraction has no flatfish); few hauls, no bootstrap.

Usage: python3 -I gov_q2q4_check.py <gov_q2q4_dir> <depth_npz>
  <gov_q2q4_dir> = research_notes/"Sejens djuplage" (a-umlaut)/data/gov_q2q4_1991_97 (hauls.csv, catch.csv.gz;
  ICES DATRAS, CC BY 4.0). Parsed only.
Same definitions as gov_juv_bed.py: J below 27.0 (haddock) / 27.1 (cod) cm, Walker q, n/h x W(L) with the
gov_juv_overlap.py length-weight constants (only ratios within a species matter), member weights haddock 0.386,
cod 0.314, bed classes < 100 / 100-150 / 150-250 / >= 250 m, shape normalised on the model map's cell shares; an
empty class takes its neighbour's value [J].
"""
import os
import sys

import numpy as np
import pandas as pd

G, DEP = sys.argv[1:3]
z = np.abs(np.load(DEP)["depth"].astype(float))
z = z[np.isfinite(z)]
CELL = np.array([(np.digitize(z, [100, 150, 250]) == k).mean() for k in range(4)])
H = pd.read_csv(os.path.join(G, "hauls.csv"))
C = pd.read_csv(os.path.join(G, "catch.csv.gz"))
HAD, COD = 126437, 126436
AB = {HAD: (0.0059, 3.13), COD: (0.0068, 3.08)}
CUT = {HAD: 27.0, COD: 27.1}
W = {HAD: 0.386, COD: 0.314}


def q_walker(a, L):
    L = np.asarray(L, dtype=float)
    if a == HAD:
        return np.where(L <= 11, 0.0116121, np.where(L >= 46, 0.123268, -0.370453 + 0.04227 * L - 0.000685 * L ** 2))
    return np.where(L <= 25, 0.15963, np.where(L >= 97, 0.66822, 0.20935 - 0.00434 * L + 0.00009 * L ** 2))


C = C[C.aphia.isin([HAD, COD])].copy()
C["L"] = C.length_cm + 0.5
C["kg"] = [n * AB[a][0] * l ** AB[a][1] / 1000.0 / q_walker(a, l) for n, a, l in zip(C.n_per_h, C.aphia, C.L)]
C["J"] = [l < CUT[a] for a, l in zip(C.aphia, C.L)]
KEY = ["Year", "Quarter", "Ship", "HaulNo", "ShootLat"]
H = H[H.Depth > 0].copy()
for a in (HAD, COD):
    for st, m in (("J", C.J), ("B", C.J | ~C.J)):
        s = C[(C.aphia == a) & m].groupby(KEY).kg.sum().rename(f"{a}{st}")
        H = H.merge(s, on=KEY, how="left")
H = H.fillna(0.0)
H["cl"] = np.digitize(H.Depth, [100, 150, 250])
REG = {"box": (57.7, 58.4, 9.9, 11.3), "wide": (57.3, 59.2, 8.0, 12.0)}
print(f"model map cell shares {np.round(CELL, 3).tolist()}")
for rn, (a0, a1, o0, o1) in REG.items():
    h0 = H[(H.ShootLat >= a0) & (H.ShootLat <= a1) & (H.ShootLong >= o0) & (H.ShootLong <= o1)]
    for q in (2, 4):
        h = h0[h0.Quarter == q]
        n = [int((h.cl == k).sum()) for k in range(4)]
        gJ, gB = np.zeros(4), np.zeros(4)
        for a in (HAD, COD):
            dJ = np.array([h[f"{a}J"][h.cl == k].mean() if n[k] else np.nan for k in range(4)])
            dB = np.array([h[f"{a}B"][h.cl == k].mean() if n[k] else np.nan for k in range(4)])
            for arr in (dJ, dB):
                for k in range(4):
                    if np.isnan(arr[k]):
                        arr[k] = arr[k - 1] if k > 0 else arr[k + 1]
            DB = (CELL * dB).sum()
            gJ += W[a] * dJ / DB
            gB += W[a] * dB / DB
        gA = gB - gJ
        f = lambda g: g / (CELL * g).sum()
        print(f"{rn:4s} Q{q}: hauls by class {n}; J/B {(CELL * gJ).sum() / (CELL * gB).sum():.3f}; "
              f"J f {'/'.join(f'{x:.2f}' for x in f(gJ))}  A f {'/'.join(f'{x:.2f}' for x in f(gA))}  "
              f"J shares {'/'.join(f'{x:.3f}' for x in CELL * f(gJ))}")
