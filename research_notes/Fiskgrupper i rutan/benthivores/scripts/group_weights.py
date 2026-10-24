"""Biomass-weighted group numbers for benthivorous_fish from the per-species table in this script.

Values are the choices documented in evidence_benthivores_B_population_fishing.md (central, low, high).
Weights: long-run 2001-25 GOV shares (tables/long_run_species_by_group.csv of the parent folder),
'others' (2.5 %) carried at a generic value. Usage: python3 -I group_weights.py
"""
import math
# species: weight, r_prior, r_now (c, lo, hi), M_total, M0, nonmodel (c, lo, hi), F (c, lo, hi), pelagic share of F
S = {
 "haddock":      (0.386, 0.50, (0.24, 0.15, 0.33), 0.59, 0.20, (0.11, 0.05, 0.16), (0.06, 0.045, 0.075), 0.017),
 "cod":          (0.314, 0.51, (0.35, 0.26, 0.40), 0.70, 0.20, (0.11, 0.05, 0.25), (0.39, 0.35, 0.50), 0.0),
 "long rough dab":(0.126, 0.19, (0.30, 0.19, 0.45), 0.25, 0.20, (0.03, 0.01, 0.06), (0.10, 0.05, 0.20), 0.0),
 "dab":          (0.045, 0.68, (0.55, 0.45, 0.68), 0.35, 0.20, (0.05, 0.02, 0.08), (0.18, 0.10, 0.26), 0.0),
 "plaice":       (0.038, 0.50, (0.40, 0.28, 0.50), 0.35, 0.20, (0.05, 0.02, 0.08), (0.04, 0.03, 0.08), 0.0),
 "witch":        (0.017, 0.32, (0.28, 0.25, 0.42), 0.20, 0.17, (0.02, 0.01, 0.04), (0.12, 0.08, 0.18), 0.0),
 "starry ray":   (0.015, 0.09, (0.15, 0.09, 0.25), 0.23, 0.15, (0.03, 0.01, 0.06), (0.03, 0.02, 0.05), 0.0),
 "hagfish":      (0.013, 0.10, (0.10, 0.07, 0.15), 0.15, 0.13, (0.02, 0.00, 0.04), (0.01, 0.00, 0.03), 0.0),
 "grey gurnard": (0.009, 0.74, (0.65, 0.50, 0.74), 0.30, 0.20, (0.05, 0.02, 0.08), (0.03, 0.02, 0.05), 0.0),
 "lemon sole":   (0.009, 0.71, (0.50, 0.42, 0.72), 0.28, 0.20, (0.03, 0.01, 0.05), (0.09, 0.06, 0.13), 0.0),
 "greater weever":(0.008, None, (0.50, 0.30, 0.80), 0.40, 0.20, (0.05, 0.02, 0.08), (0.02, 0.00, 0.05), 0.0),
 "others":       (0.025, None, (0.40, 0.20, 0.60), 0.30, 0.20, (0.05, 0.02, 0.08), (0.10, 0.05, 0.15), 0.0),
}
wt = sum(v[0] for v in S.values())
print(f"sum of weights {wt:.3f} (normalised)")
def wmean(fn):
    return sum(v[0] * fn(v) for v in S.values()) / wt
for lab, i in (("central", 0), ("low", 1), ("high", 2)):
    print(f"{lab:8s} r_now {wmean(lambda v: v[2][i]):.3f}  nonmodel {wmean(lambda v: v[5][i]):.3f}  F {wmean(lambda v: v[6][i]):.3f}")
print(f"r_prior (FishBase; weever/others at 0.5) {wmean(lambda v: v[1] if v[1] else 0.5):.3f}")
print(f"M0 {wmean(lambda v: v[4]):.3f}   M_total {wmean(lambda v: v[3]):.3f}")
Fp = wmean(lambda v: v[6][0] * v[7]); F = wmean(lambda v: v[6][0])
print(f"F on pelagic map {Fp:.4f} = {Fp/F*100:.1f} % of group F; bottom map {F-Fp:.4f}")
# F contribution by species
print("F contribution (w*F/wt):", ", ".join(f"{k} {v[0]*v[6][0]/wt:.3f}" for k, v in S.items()))
# Sensitivity: cod/haddock only
ch = {k: S[k] for k in ("cod", "haddock")}
w2 = sum(v[0] for v in ch.values())
print(f"cod+haddock only: F {sum(v[0]*v[6][0] for v in ch.values())/w2:.3f}, r_now {sum(v[0]*v[2][0] for v in ch.values())/w2:.3f}")
# per tick
for nm, x in (("M0", wmean(lambda v: v[4])), ("nonmodel", wmean(lambda v: v[5][0])), ("F", F)):
    print(f"{nm} per tick (6 h) 1-exp(-x/1460) = {1-math.exp(-x/1460):.3e}")
