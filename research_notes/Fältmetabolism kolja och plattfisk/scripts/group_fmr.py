"""Biomass-weighted group FMR (MJ/t/yr at 7-8 C) from the otolith evidence, and the group multiplier.

Route B (cod-anchored, the main route): cod stays at the prior notes' 11 338 MJ/t/yr (Chung et al. 2019
2.2 x SMR at 1.2 kg, `Tillvaxt mot ranson` W.1); haddock and long rough dab are scaled by the
within-site otolith ratio to cod (Barents site, one DIC baseline and one calibration; ratios printed by
otolith_fmr.py, robust to the diet baseline); plaice and witch (North Sea, measured near 8 C, no
extrapolation) at face value as central, with the cod anchoring factor (prior cod / otolith cod,
0.70-0.88) as the LOW case and an SMR-like mass scaling to the model's 150-190 g as the HIGH case.
Route A (absolute): every value at the Chung calibration's face value, Barents fish Q10-extrapolated.
Inputs are numbers printed in tables/otolith_fmr_stdout.txt and prior-note values. No model output.
Usage: python3 -I group_fmr.py
"""
W = {"cod": 31.4, "haddock": 38.6, "LRD": 12.6, "dab": 4.5, "plaice": 3.8, "witch": 1.7,
     "starry ray": 1.5, "hagfish": 1.3, "grey gurnard": 0.9, "lemon sole": 0.9, "greater weever": 0.8}
# prior per-species FMR (Tillvaxt mot ranson tables/weighted_group_stdout.txt) and SMR (MJ/t/tick, ~7 C)
PRIOR = {"cod": 11338, "haddock": 6205, "plaice": 5825, "dab": 5825, "LRD": 3066, "witch": 2575,
         "grey gurnard": 6205, "lemon sole": 5825, "greater weever": 5825, "starry ray": 4088, "hagfish": 2044}
SMR = {"cod": 3.53, "haddock": 2.50, "plaice": 2.85, "dab": 2.85, "LRD": 1.50, "witch": 1.26,
       "grey gurnard": 2.50, "lemon sole": 2.85, "greater weever": 2.85, "starry ray": 2.00, "hagfish": 1.00}
COD = 11338.0
MGKGH = 119.1                      # MJ/t/yr per mg O2/kg/h
ANCHOR = (11338 / 16252, 11338 / 12830)   # 0.70 (Q10 3) .. 0.88 (Q10 2)
# within-site ratios to cod at the model masses (haddock 275 g, LRD 170 g, cod 1200 g)
R_HAD = {"low": 0.60, "central": 0.70, "high": 0.82}   # equal-mass 0.72-0.76 (CI 0.63-0.85) x (275/1200)^0.03
R_LRD = {"low": 0.65, "central": 0.85, "high": 1.10}   # adult 0.84-0.95 x0.95; all-size 0.88; CI 0.68-1.18
PLA = {"low": 54 * MGKGH * ANCHOR[0], "central": 54 * MGKGH, "high": 54 * MGKGH * 38.7 / 31.2}
WIT = {"low": 36 * MGKGH * ANCHOR[0], "central": 36 * MGKGH, "high": 36 * MGKGH * (170 / 484) ** -0.2}


def wmean(v):
    return sum(W[k] * v[k] for k in W) / sum(W.values())


def case(lab, lrd_mode="otolith"):
    v = dict(PRIOR)
    v["cod"] = COD
    v["haddock"] = v["grey gurnard"] = R_HAD[lab] * COD
    v["plaice"] = v["dab"] = v["lemon sole"] = v["greater weever"] = PLA[lab]
    v["witch"] = WIT[lab]
    v["LRD"] = R_LRD[lab] * COD if lrd_mode == "otolith" else PLA[lab]
    return v


smr_w = wmean({k: SMR[k] * 1460 for k in W})
print(f"weighted SMR {smr_w:.0f} MJ/t/yr = {smr_w/1460:.3f} MJ/t/tick (prior W.2: 2.688)")
print(f"prior group FMR {wmean(PRIOR):.0f} MJ/t/yr = {wmean(PRIOR)/smr_w:.2f} x weighted SMR\n")
print("Route B (cod-anchored) per species and group, MJ/t/yr at ~7.5 C:")
for lrd_mode in ["otolith", "plaice"]:
    print(f"  LRD from {'the Barents otoliths' if lrd_mode == 'otolith' else 'the plaice value (fallback)'}:")
    for lab in ["low", "central", "high"]:
        v = case(lab, lrd_mode)
        g = wmean(v)
        print(f"    {lab:7s} cod {v['cod']:.0f} haddock {v['haddock']:.0f} LRD {v['LRD']:.0f} plaice/dab {v['plaice']:.0f} "
              f"witch {v['witch']:.0f} -> group {g:.0f} MJ/t/yr = {g/1460:.2f} MJ/t/tick = {g/smr_w:.2f} x weighted SMR")
print()
print("Per-species multipliers on the prior SMR denominators (central Route B):")
v = case("central")
for k in ["cod", "haddock", "LRD", "plaice", "witch"]:
    print(f"  {k:8s} FMR {v[k]:6.0f}  SMR {SMR[k]:.2f} MJ/t/tick ({SMR[k]*1460:.0f}/yr)  -> {v[k]/(SMR[k]*1460):.2f} x SMR"
          f"   (prior {PRIOR[k]/(SMR[k]*1460):.2f})")
print()
print("Route A (absolute, Chung calibration; Barents fish extrapolated to 7.5 C with Q10 2 / 2.67 / 3):")
A = {"cod": (12830, 15183, 16252), "haddock": (9386, 10401, 10842), "LRD": (11641, 13347, 14106)}
for i, lab in enumerate(["Q10 2", "Q10 2.67", "Q10 3"]):
    v = case("central")
    for k in A:
        v[k] = A[k][i]
    v["grey gurnard"] = v["haddock"]
    print(f"  {lab:8s} group {wmean(v):.0f} MJ/t/yr")
print()
own = sum(W[k] for k in ["cod", "haddock", "LRD", "plaice", "witch"]) / sum(W.values())
print(f"biomass weight on own-species otolith FMR: {own:.2f} (cod, haddock, LRD, plaice, witch); "
      f"prior note: 0.32 (cod only)")
