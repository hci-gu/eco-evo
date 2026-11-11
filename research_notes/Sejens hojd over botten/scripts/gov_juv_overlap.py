"""Where adult saithe and the juvenile haddock inside their gape sit by bed depth, in NS-IBTS GOV hauls of the box
(and a wider Skagerrak / Norwegian Trench window): the horizontal co-occurrence factor that the diet gate's
equal-cell and field-spread forms (benthivores uniform, juveniles a fixed share of every cell) do not carry.

Usage: python3 -I gov_juv_overlap.py <gov dir>
  <gov dir> = research_notes/"Sejens djuplage" (a-umlaut)/data/gov (hauls.csv, catch.csv.gz: NS-IBTS Q1 + Q3
  2000-25, CPUE per length; ICES DATRAS, CC BY 4.0).

Quantities (biomass rates from n/h x W(L), W = a L^b with species a, b from FishBase-type values; only ratios
within a species matter):
  saithe adults (>= 35 cm); haddock in-gape (< 19 cm: the 0-group from settlement in Q3, the 1-group in Q1;
  research_notes/"Sejens diet pa unga torskfiskar" section 5); the gadoid benthivores haddock + cod (all
  lengths) as the group proxy (flatfish not in the extraction).
Factors, per quarter and box-presence weighted (Q3 0.9 / Q1 0.1, wc_depth evidence 1.2):
  F_gate   = sum_c s_c J_c / sum_c w_c J_c: in-gape haddock density met by the mean adult saithe (bed classes, s_c the
             saithe share by class, w_c the box cell shares 0.21 / 0.31 / 0.33 / 0.15) relative to the box-mean density
             (the gate credits 1: benthivores uniform, J a fixed share);
  F_engine = (sum_c s_c J_c / sum_c s_c B_c) / (sum_c w_c J_c / sum_c w_c B_c): the J:B ratio saithe meet relative to
             the box mean (what an engine with J a fixed share of B, but B placed as in the field, would over-credit);
  F_haul   = sum_h S_h J_h / (sum_h S_h x mean_h J_h): the same at haul level (adds within-class co-occurrence; haul-
             weighted, not cell-weighted).
"""
import os
import sys

import numpy as np
import pandas as pd

G = os.path.abspath(sys.argv[1])
H = pd.read_csv(os.path.join(G, "hauls.csv"))
C = pd.read_csv(os.path.join(G, "catch.csv.gz"))
SAI, HAD, COD = 126441, 126437, 126436
AB = {SAI: (0.0085, 3.0), HAD: (0.0059, 3.13), COD: (0.0068, 3.08)}
C = C[C.aphia.isin([SAI, HAD, COD])].copy()
C["kg"] = [n * AB[a][0] * l ** AB[a][1] / 1000.0 for n, a, l in zip(C.n_per_h, C.aphia, C.length_cm)]
KEY = ["Year", "Quarter", "Ship", "HaulNo", "ShootLat"]
H = H[H.Quarter.isin([1, 3]) & (H.Depth > 0)].copy()


def agg(mask, name):
    return C[mask].groupby(KEY).kg.sum().rename(name)


H = H.merge(agg((C.aphia == SAI) & (C.length_cm >= 35), "S"), on=KEY, how="left")
H = H.merge(agg((C.aphia == HAD) & (C.length_cm < 19), "J"), on=KEY, how="left")
H = H.merge(agg(C.aphia.isin([HAD, COD]), "B"), on=KEY, how="left")
H = H.merge(agg((C.aphia == HAD) & (C.length_cm < 19), "Jn"), on=KEY, how="left")
for c in ("S", "J", "B"):
    H[c] = H[c].fillna(0.0)
CELLS = np.array([0.21, 0.31, 0.33, 0.15])
H["cl"] = np.digitize(H.Depth, [100, 150, 250])
REG = {"box (57.7-58.4 N, 9.9-11.3 E)": (57.7, 58.4, 9.9, 11.3),
       "Skagerrak / Trench (57.3-59.2 N, 8-12 E)": (57.3, 59.2, 8.0, 12.0)}
RES = {}
for rn, (a0, a1, o0, o1) in REG.items():
    h0 = H[(H.ShootLat >= a0) & (H.ShootLat <= a1) & (H.ShootLong >= o0) & (H.ShootLong <= o1)]
    print("=" * 110)
    print(f"{rn}: hauls Q1 {int((h0.Quarter == 1).sum())}, Q3 {int((h0.Quarter == 3).sum())}")
    for q in (1, 3):
        h = h0[h0.Quarter == q]
        n = np.array([int((h.cl == k).sum()) for k in range(4)])
        S = np.array([h.S[h.cl == k].mean() if n[k] else np.nan for k in range(4)])
        J = np.array([h.J[h.cl == k].mean() if n[k] else np.nan for k in range(4)])
        B = np.array([h.B[h.cl == k].mean() if n[k] else np.nan for k in range(4)])
        # classes without hauls: take the neighbouring class (beds > 250 m often unsampled) [J]
        for arr in (S, J, B):
            for k in range(4):
                if np.isnan(arr[k]):
                    arr[k] = arr[k - 1] if k > 0 else arr[k + 1]
        s = S * CELLS / (S * CELLS).sum()
        Fg = float((s * J).sum() / (CELLS * J).sum())
        Fe = float(((s * J).sum() / (s * B).sum()) / ((CELLS * J).sum() / (CELLS * B).sum()))
        Fh = float((h.S * h.J).sum() / (h.S.sum() * h.J.mean())) if h.S.sum() > 0 else np.nan
        RES[(rn, q)] = (Fg, Fe, Fh)
        print(f"  Q{q}: hauls by class {n.tolist()}")
        print(f"      kg/h  saithe >= 35 cm {np.round(S, 2).tolist()}; haddock < 19 cm {np.round(J, 3).tolist()};"
              f" haddock + cod {np.round(B, 2).tolist()}")
        print(f"      saithe class shares {np.round(s, 3).tolist()}; in-gape haddock class shares "
              f"{np.round(CELLS * J / (CELLS * J).sum(), 3).tolist()}; J:B by class {np.round(J / B, 4).tolist()}")
        print(f"      F_gate {Fg:.3f}   F_engine {Fe:.3f}   F_haul {Fh:.3f}")
    fq = {k: 0.1 * RES[(rn, 1)][i] + 0.9 * RES[(rn, 3)][i] for i, k in enumerate(("F_gate", "F_engine", "F_haul"))}
    fe = {k: 0.5 * RES[(rn, 1)][i] + 0.5 * RES[(rn, 3)][i] for i, k in enumerate(("F_gate", "F_engine", "F_haul"))}
    print(f"  presence-weighted (Q3 0.9): " + ", ".join(f"{k} {v:.3f}" for k, v in fq.items()))
    print(f"  equal quarters:             " + ", ".join(f"{k} {v:.3f}" for k, v in fe.items()))
print("\nReading: F < 1 = adult saithe stand where the in-gape juveniles are scarcer than the box mean. The J part of the"
      " gate's WC -> benthivores flow (0.5-0.7 of it) scales with F_gate; the A part (bed members on adult tonnes) does not.")
