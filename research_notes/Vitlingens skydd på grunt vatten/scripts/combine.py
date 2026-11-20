"""Combine the per-mechanism factors into the shallow (< 100 m) : 150-250 m hazard ratio on whiting, per predator pair
and for the two fish pairs together (tables/factors.csv is written by hand from this and the other outputs).

Usage: python3 -I combine.py > ../tables/combine_stdout.txt
Inputs are typed in from this note's tables (each line names its source table); nothing is fitted to model output.
Pair class a's (live library 280, a x class multiplier): research_notes/"Vitlingens placering"
tables/pair_depth_mult_live_vs_pre279.txt. Group densities t/km2: water column 0.617, benthivores 1.53 (AGENTS.md
section 4); the pair a's are per tonne of the whole group.
"""
import csv
import os

import numpy as np

A = {"WC": np.array([0.1713, 0.1104, 0.0326, 0.0281]), "benth": np.array([0.0718, 0.0466, 0.0162, 0.0148])}
B = {"WC": 0.617, "benth": 1.53}
# field class shapes (density / box equal-cell mean), gov_predators_by_class_stdout.txt, Q1 + Q3:
S = {"WC": np.array([0.23, 0.43, 2.30, 0.36]), "benth": np.array([0.77, 1.86, 0.69, 0.26])}
S_LO = {"WC": np.array([0.14, 0.25, 2.51, 0.36]), "benth": np.array([0.58, 1.60, 0.85, 0.26])}  # shallow low / deep high
S_HI = {"WC": np.array([0.41, 0.66, 2.03, 0.36]), "benth": np.array([0.98, 2.11, 0.53, 0.26])}
# measured near-bed light (mechanism 2) on the class a. < 100 m: Swedish side 0.55 (0.42-0.82; shallow_light_factor
# _stdout.txt), Danish side estimated with the Swedish column under the Danish surface layer 0.78 (0.59-0.97;
# _level4/5/6_stdout.txt), weighted 0.61 / 0.39 by shallow whiting biomass [D/J]. 100-150 m: 0.48 (0.26-0.80).
F2 = {"central": np.array([0.64, 0.48, 1.0, 1.0]), "low": np.array([0.49, 0.26, 1.0, 1.0]),
      "high": np.array([0.88, 0.80, 1.0, 1.0])}
# mechanism 1 (light-dependent escape) [J] on the lit classes; mechanism 3 (diel position) [D/J]
F1 = {"central": np.array([0.85, 0.9, 1.0, 1.0]), "low": np.array([0.6, 0.75, 1.0, 1.0]),
      "high": np.array([1.0, 1.0, 1.0, 1.0])}
F3 = {"central": np.ones(4), "low": np.array([0.8, 0.9, 1, 1]), "high": np.array([1.2, 1.1, 1, 1])}


def ratio(h):
    return h[0] / h[2]


def hz(pairs, fac, shape):
    return sum(A[p] * B[p] * shape[p] * fac for p in pairs)


rows = []
print("shallow (< 100 m) : 150-250 m hazard ratio on whiting (per unit whiting biomass)")
for pairs, lab in ((("WC",), "WC -> whiting"), (("benth",), "benthivores -> whiting"), (("WC", "benth"), "both fish pairs")):
    one = {p: np.ones(4) for p in pairs}
    base = ratio(hz(pairs, 1.0, one))
    print(f"\n  {lab}: model, uniform predators {base:.2f}")
    for key in ("central", "low", "high"):
        f2 = ratio(hz(pairs, F2[key], one))
        f12 = ratio(hz(pairs, F2[key] * F1[key] * F3[key], one))
        print(f"    {key:7s}: x measured bed light {f2:.2f} (factor {f2 / base:.2f}); x escape and diel {f12:.2f} "
              f"(factor {f12 / base:.2f})")
    for key, sh in (("central", S), ("shallow-low", S_LO), ("shallow-high", S_HI)):
        fp = ratio(hz(pairs, 1.0, {p: sh[p] for p in pairs}))
        fall = ratio(hz(pairs, F2["central"] * F1["central"], {p: sh[p] for p in pairs}))
        print(f"    field predator placement ({key}): {fp:.2f} (factor {fp / base:.2f}); with light + escape (central) "
              f"{fall:.2f}")
    lo = ratio(hz(pairs, F2["low"] * F1["low"] * F3["low"], {p: S_LO[p] for p in pairs}))
    hi = ratio(hz(pairs, F2["high"] * F1["high"] * F3["high"], {p: S_HI[p] for p in pairs}))
    print(f"    all corners together (placement + light + escape + diel): {lo:.2f} - {hi:.2f}")
print("\n  class profile, both pairs, relative to 150-250 m (< 100 / 100-150 / 150-250 / >= 250):")
one = {"WC": np.ones(4), "benth": np.ones(4)}
for lab, fac, sh in (("model, uniform predators", 1.0, one), ("measured light + escape (central)",
                                                                 F2["central"] * F1["central"], one),
                     ("field placement", 1.0, S), ("field placement + light + escape", F2["central"] * F1["central"], S)):
    h = hz(("WC", "benth"), fac, sh)
    print(f"    {lab:36s} " + " / ".join(f"{x / h[2]:.2f}" for x in h))
