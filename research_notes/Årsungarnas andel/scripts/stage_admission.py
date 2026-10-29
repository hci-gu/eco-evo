"""Which juvenile stages fit each predator's prey-size window, over the box predator length mix.

Usage: python3 -I stage_admission.py <pred_length_biomass.csv> <benthivore length_biomass_longrun.csv>

Window = ratio x predator length, ratio by predator-length band (p95 and p99 of the realised
prey/predator length in the stomachs; values from ../tables/sms_prey_size_stdout.txt and
ices_prey_size_stdout.txt, chosen in ../evidence_prey_size.md). Predator lengths: NS-IBTS GOV box
biomass by 5-cm class (fish predators, ../../Fiskgrupper i rutan/predators/tables/pred_length_biomass.csv,
pooled Q1+Q3 unless stated) and 1-cm class for cod (benthivores/tables/length_biomass_longrun.csv,
2001-25). Prints the window at the predator-biomass p10/p50/p90 length and the share of predator
biomass whose window reaches 13 cm (age 0 in August), 16 cm (age 0 in November) and 25 cm (top of age 1).
"""
import sys

import numpy as np
import pandas as pd

plb, blb = sys.argv[1:3]
P = pd.read_csv(plb)
B = pd.read_csv(blb)

# (upper band edge cm, p95, p99): first band whose edge exceeds L applies
RATIO = {
    "saithe": [(999, 0.30, 0.41)],                                             # SMS pooled = 45-65 band after de-duplication
    "cod": [(30, 0.40, 0.56), (45, 0.34, 0.36), (65, 0.30, 0.39), (999, 0.27, 0.35)],  # SMS + ICES Skag/NNS, P prey
    "whiting": [(20, 0.50, 0.71), (30, 0.40, 0.49), (999, 0.33, 0.38)],         # SMS all fish prey (ICES 0.46-0.48)
    "hake": [(40, 0.50, 0.52), (60, 0.48, 0.52), (999, 0.33, 0.39)],            # ICES 2024 Skagerrak, all fish
    "spurdog": [(999, 0.35, 0.38)],                                             # ICES 1991 CSNS, n 18
    "anglerfish": [(999, 0.57, 0.71)],                                          # ICES 2022, n 16
    "mackerel": [(999, 0.33, 0.56)], "horse mackerel": [(999, 0.31, 0.34)],     # Makrillens bytesstorlek
    "velvet belly": [(999, 0.25, 0.30)], "ling": [(999, 0.28, 0.35)],           # judgement
}


def ratio(sp, L, k):
    for edge, r95, r99 in RATIO[sp]:
        if L < edge:
            return r95 if k == 95 else r99


def dist(sp, col="pooled"):
    if sp == "cod":
        d = B[B.species == "cod"]
        return d.L_cm.values + 0.5, d.share_2001_25.values
    d = P[P.common == sp]
    return d.L5.values + 2.5, d[col].values


def pct(L, w, q):
    o = np.argsort(L)
    cw = np.cumsum(w[o]) / w.sum()
    return L[o][np.searchsorted(cw, q)]


print("predator (season) | pred L p10/p50/p90 | window p95 at p10/p50/p90 | p99 at p50 | "
      "share of pred biomass with p95 window >= 13 / 16 / 25 cm | same with p99")
for sp, col in [("saithe", "pooled"), ("cod", None), ("whiting", "q1"), ("whiting", "q3"), ("hake", "q1"),
                ("hake", "q3"), ("spurdog", "pooled"), ("anglerfish", "pooled"), ("mackerel", "q3"),
                ("horse mackerel", "q3"), ("velvet belly", "pooled"), ("ling", "pooled")]:
    L, w = dist(sp, col) if col else dist(sp)
    ok = w > 0
    L, w = L[ok], w[ok] / w[ok].sum()
    q = [pct(L, w, x) for x in (0.1, 0.5, 0.9)]
    g95 = np.array([ratio(sp, l, 95) * l for l in L])
    g99 = np.array([ratio(sp, l, 99) * l for l in L])
    s95 = [w[g95 >= c].sum() for c in (13, 16, 25)]
    s99 = [w[g99 >= c].sum() for c in (13, 16, 25)]
    print(f"{sp:14s} ({col or '2001-25'}) | {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} cm | "
          f"{ratio(sp, q[0], 95) * q[0]:.1f}/{ratio(sp, q[1], 95) * q[1]:.1f}/{ratio(sp, q[2], 95) * q[2]:.1f} cm | "
          f"{ratio(sp, q[1], 99) * q[1]:.1f} cm | {s95[0]:.2f} / {s95[1]:.2f} / {s95[2]:.2f} | "
          f"{s99[0]:.2f} / {s99[1]:.2f} / {s99[2]:.2f}")
