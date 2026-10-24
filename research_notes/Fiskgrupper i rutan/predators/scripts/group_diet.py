"""Group diet and Q/B for predatory_fish from per-species central values.

Usage: python3 -I group_diet.py <inputs/diet_consumption_central.csv>
Shares renormalised over the listed species. Diet weighted by consumption (B_i * QB_i), which is
what M2 needs, and by biomass for comparison. dark_ratio weighted by consumption of that prey pool.
"""
import csv, sys
R = list(csv.DictReader(open(sys.argv[1])))
f = lambda r, k: float(r[k])
S = sum(f(r, "biomass_share") for r in R)
pools = ["P", "Bf", "W", "Z", "B", "O"]
for k in ("QB", "QB_low", "QB_high"):
    print(f"group {k} (biomass-weighted): {sum(f(r,'biomass_share')*f(r,k) for r in R)/S:.2f}")
C = {r["species"]: f(r, "biomass_share") * f(r, "QB") for r in R}
Ct = sum(C.values())
print("consumption share by species: " + " ".join(f"{r['species']} {C[r['species']]/Ct:.3f}" for r in R))
print("diet, consumption-weighted: " + " ".join(f"{p} {sum(C[r['species']]*f(r,p) for r in R)/Ct:.3f}" for p in pools))
print("diet, biomass-weighted:     " + " ".join(f"{p} {sum(f(r,'biomass_share')*f(r,p) for r in R)/S:.3f}" for p in pools))
nonW = {r["species"]: 1 - f(r, "W") for r in R}
print("diet without within-group (renormalised, consumption-weighted): " + " ".join(
    f"{p} {sum(C[r['species']]*f(r,p) for r in R)/sum(C[r['species']]*nonW[r['species']] for r in R):.3f}" for p in pools if p != "W"))
for pool, col in (("P", "dark_fish"), ("Bf", "dark_fish"), ("Z", "dark_zoo")):
    w = {r["species"]: C[r["species"]] * f(r, pool) for r in R}
    print(f"dark_ratio for pair on {pool}: {sum(w[r['species']]*f(r,col) for r in R)/sum(w.values()):.2f}  (weights: " +
          " ".join(f"{k} {v/sum(w.values()):.2f}" for k, v in w.items()) + ")")
