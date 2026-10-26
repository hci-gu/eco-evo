"""Benthivore attack rates on the two new bottom groups (copy and extension of section 1 of
research_notes/"Attackrater ur motesteori"/scripts/benth_encounter.py; the original is not edited).

Usage: python3 -I benth_encounter_split.py <length_biomass_longrun.csv> [n_mc]

  <length_biomass_longrun.csv>: research_notes/"Fiskgrupper i rutan"/benthivores/tables/ (NS-IBTS GOV
      Q1+Q3 2001-25, 1-cm classes, share_2001_25 and class weight w_g per species).

Engine definition (interactions.holling_a_eff, predation.apply_predation): intake per tonne of predator
per 6-h tick = eat x hunger x a B / (1 + a h B + w B_pred), B = visible, accessible prey (t/km2).
predator_share is being removed, so a here is per tonne of the WHOLE benthivore group (members weighted
by their biomass share x the length structure; NO stage capability scores).

2-D swept area per fish per tick = v x w x p x 21 600 s, summed over 1-cm classes per tonne.
  infauna pair : w = path width of contact search (barbel and fin rays for gadoids, the original's
                 0.15 L), p = capture on infauna (the original's p_i).
  epifauna pair: w_epi = 2 R_epi, R_epi = detection distance of MOVING epifauna (lateral line, smell, close
                 sight at the dim bottom); p = capture on epifauna (the original's p_e).
Factors are (low, central, high); MC = each factor log-uniform between low and high, independent.
"""
import math
import sys

import numpy as np
import pandas as pd

LB = pd.read_csv(sys.argv[1])
NMC = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
RNG = np.random.default_rng(20261026)
TICK = 21600.0
GROUP = {"haddock": 38.62, "cod": 31.38, "long rough dab": 12.63, "dab": 4.49, "plaice": 3.80,
         "witch": 1.67, "starry ray": 1.47, "grey gurnard": 0.89, "lemon sole": 0.87}
SP_MODE = {"haddock": "gadoid", "cod": "gadoid", "dab": "flat", "plaice": "flat", "witch": "flat",
           "lemon sole": "flat", "long rough dab": "lrd", "grey gurnard": "gurnard", "starry ray": "ray"}
# v (BL/s), w_inf (x L), p_inf, w_epi (x L), p_epi.  v, w_inf, p_inf, p_epi as in benth_encounter.py.
MODE = {
    "gadoid": dict(v=(0.2, 0.4, 0.8), wi=(0.07, 0.15, 0.4), pi=(0.25, 0.5, 0.9),
                   we=(0.3, 0.6, 1.2), pe=(0.1, 0.3, 0.7)),
    "flat": dict(v=(0.04, 0.12, 0.3), wi=(0.25, 0.6, 1.2), pi=(0.4, 0.7, 1.0),
                 we=(0.3, 0.8, 1.5), pe=(0.15, 0.4, 0.8)),
    "lrd": dict(v=(0.04, 0.12, 0.3), wi=(0.25, 0.6, 1.2), pi=(0.25, 0.5, 0.9),
                we=(0.3, 0.8, 1.5), pe=(0.15, 0.4, 0.8)),
    "gurnard": dict(v=(0.1, 0.2, 0.4), wi=(0.15, 0.3, 0.6), pi=(0.25, 0.5, 0.9),
                    we=(0.3, 0.6, 1.2), pe=(0.1, 0.3, 0.7)),
    "ray": dict(v=(0.05, 0.15, 0.3), wi=(0.2, 0.4, 0.8), pi=(0.25, 0.5, 0.9),
                we=(0.3, 0.6, 1.2), pe=(0.1, 0.3, 0.7)),
}
PAIRS = {  # name: (width key, capture key, h, accessible prey t/km2 today, total stock t/km2)
    "infauna": ("wi", "pi", float(sys.argv[3]) if len(sys.argv) > 3 else 90.0, 0.2 * 92.0, 92.0),
    "mobile_epifauna": ("we", "pe", float(sys.argv[4]) if len(sys.argv) > 4 else 138.7, 0.31 * 7.63, 7.63),
}


def classes(sp):
    d = LB[(LB.species == sp) & (LB.share_2001_25 > 0)].copy()
    d["s"] = d.share_2001_25 / d.share_2001_25.sum()
    d["L"] = (d.L_cm + 0.5) / 100.0
    return d


CL = {sp: classes(sp) for sp in GROUP}
WT = {sp: GROUP[sp] / sum(GROUP.values()) for sp in GROUP}   # biomass weights, renormalised (hagfish,
# weever and 'others' have no length data and are left out as in the original)
M2L = {sp: float((CL[sp].s * 1e6 / CL[sp].w_g * CL[sp].L ** 2).sum()) for sp in GROUP}  # sum n L^2 per t


def logu(lo, hi, n):
    return np.exp(RNG.uniform(math.log(lo), math.log(hi), n))


draws = {m: {k: logu(v[0], v[2], NMC) for k, v in spec.items()} for m, spec in MODE.items()}
out = {}
for pair, (wk, pk, h, Bacc, Btot) in PAIRS.items():
    c = 0.0
    mc = np.zeros(NMC)
    per_sp = {}
    for sp, wt in WT.items():
        md, dd = MODE[SP_MODE[sp]], draws[SP_MODE[sp]]
        a_c = md["v"][1] * md[wk][1] * md[pk][1] * TICK * M2L[sp] / 1e6
        a_m = dd["v"] * dd[wk] * dd[pk] * TICK * M2L[sp] / 1e6
        per_sp[sp] = a_c
        c += wt * a_c
        mc += wt * a_m
    p5, p50, p95 = np.percentile(mc, [5, 50, 95])
    out[pair] = c
    print("=" * 96)
    print(f"benthivorous_fish -> {pair}: h {h:.1f}, accessible prey today {Bacc:.2f} t/km2 (total {Btot})")
    print("  per member (central, km2 per t of that member per tick): " +
          ", ".join(f"{sp} {a:.3f}" for sp, a in sorted(per_sp.items(), key=lambda x: -WT[x[0]])))
    print(f"  GROUP a central {c:.4f}  MC 5/50/95 % {p5:.4f} {p50:.4f} {p95:.4f}")
    print(f"  1/(a h) = {1 / (c * h):.4f} t/km2 accessible (MC 5-95 %: {1 / (p95 * h):.4f} - {1 / (p5 * h):.4f})"
          f"; in total-stock units 1/(a h f) = {Btot / Bacc / (c * h):.3f} t/km2")
    for frac in (1.0, 0.1, 0.01):
        x = c * h * Bacc * frac
        print(f"  at {frac:5.0%} of today's accessible stock: x = a h B = {x:8.2f} -> {x / (1 + x):.3f} of ceiling")
print("=" * 96)
print(f"ratio a_epi / a_inf (central) = {out['mobile_epifauna'] / out['infauna']:.2f}")
# Diet check (not used to set a): at saturation the engine pays the ceiling on either pair, so the
# diet split is the policy's eat shares; below saturation the ratio of intakes per eat share is
# (a_e B_e) / (a_i B_i) for the accessible stocks.
Be, Bi = PAIRS["mobile_epifauna"][3], PAIRS["infauna"][3]
print(f"unsaturated intake ratio epi/inf per unit eat share = {out['mobile_epifauna'] * Be / (out['infauna'] * Bi):.3f}"
      f" (mass); measured diet ratio 0.29/0.71 = {0.29 / 0.71:.3f}")
