"""Two proxies for ling's consumption / biomass Q/B (no ling Q/B, ration or evacuation measurement was found).

Usage: python3 -I qb_ling.py <pred_length_biomass.csv> <NS2023_consum_ab.in>

(1) Palomares & Pauly (1998) empirical equation (the Ecopath default for fish without a measured Q/B):
      log10 Q/B = 7.964 - 0.204 log10 Winf - 1.965 T' + 0.083 A + 0.532 h + 0.398 d,  T' = 1000 / (T + 273.15)
    h = d = 0 (carnivore). Winf (g) from Linf with the stock-annex length-weight W = 0.0055 L^3.012 (Norwegian
    fleet, 4.a / 6.a; ICES stock annex lin.27.346-91214, 2026) and, as a check, the FishBase Bayesian 0.0038 L^3.08.
    Linf / K pairs (stock annex Table 2): northern North Sea 189 / 0.080, 4.a + 6.ab 183 / 0.118, BHAC 160 / 0.09.
    T: the box bottom annual mean 7.4 C (library temperature table, AGENTS.md) and 6-9 C. A (caudal aspect ratio):
    no ling value in FishBase (swimming table: subcarangiform only); a rounded gadoid tail [J] 0.8-1.6.
(2) The SMS North Sea key-run 2023 cod ration (consum_ab.in: kg per quarter = a_q W^b, W in kg; the only
    size-resolved gadoid field ration of the area), applied to ling's box length mix (pred_length_biomass.csv
    pooled, biomass-weighted) - a PROXY from a more active gadoid (cod) at the same weights; an upper side.
Nothing is fitted.
"""
import math
import sys

import numpy as np
import pandas as pd

PLB, CAB = sys.argv[1:3]


def pp(winf_g, T, A):
    return 10 ** (7.964 - 0.204 * math.log10(winf_g) - 1.965 * 1000.0 / (T + 273.15) + 0.083 * A)


print("(1) Palomares & Pauly 1998, carnivore (h = d = 0)")
print(f"  {'Linf':>5s} {'K':>6s} {'LW':>12s} {'Winf kg':>8s} | Q/B at T 7.4 C, A 0.8 / 1.2 / 1.6 | T 6 / 9 C at A 1.2")
for Linf, K, lab in ((189, 0.080, "N North Sea"), (183, 0.118, "4.a+6.ab"), (160, 0.09, "BHAC")):
    for a, b, lwlab in ((0.0055, 3.012, "annex"), (0.0038, 3.08, "FishBase")):
        W = a * Linf ** b
        r = [pp(W, 7.4, A) for A in (0.8, 1.2, 1.6)]
        t = [pp(W, T, 1.2) for T in (6.0, 9.0)]
        print(f"  {Linf:5d} {K:6.3f} {lwlab:>12s} {W / 1000:8.1f} | {r[0]:.2f} / {r[1]:.2f} / {r[2]:.2f}"
              f" | {t[0]:.2f} / {t[1]:.2f}   ({lab})")

# (2) SMS cod ration at ling weights
lines = open(CAB).read().splitlines()
i = [k for k, l in enumerate(lines) if l.strip().startswith("#") and "Cod" in l][0]
ab = [tuple(float(x) for x in lines[i + 1 + q].split("#")[0].split()) for q in range(4)]
print("\n(2) SMS 2023 cod ration a_q W^b (kg/quarter):", ab)
d = pd.read_csv(PLB)
d = d[(d.common == "ling") & (d.pooled > 0)]
L = d.L5.values + 2.5
for a_, b_, lab in ((0.0037105, 3.0882, "CA fit used for N/t"), (0.0055, 3.012, "annex")):
    W = a_ * L ** b_ / 1000.0                                   # kg
    qb_ind = sum(aq * W ** bq for aq, bq in ab) / W             # per year per kg
    wbio = d.pooled.values / d.pooled.sum()
    print(f"  LW {lab}: biomass-weighted Q/B over the box length mix {float((wbio * qb_ind).sum()):.2f} /yr; "
          f"at 2 kg {sum(aq * 2.0 ** bq for aq, bq in ab) / 2.0:.2f}, at 5 kg {sum(aq * 5.0 ** bq for aq, bq in ab) / 5.0:.2f},"
          f" at 15 kg {sum(aq * 15.0 ** bq for aq, bq in ab) / 15.0:.2f}")
print("\nReference Q/B of the other water-column members (research_notes/'Fiskgrupper i rutan' evidence_C):"
      " saithe 3.0, hake 3.5, anglerfish 1.8, spurdog 2.0")
