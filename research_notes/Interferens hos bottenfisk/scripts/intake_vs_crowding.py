"""Per-individual intake at k x the regional mean own density, relative to intake at 1x.

Puts the evidence and the model on one scale: R(k) = intake(k * mean) / intake(mean).
Evidence forms are power laws (Hassell-Varley, intake ~ P^-m), so R(k) = k^-m.
Model forms use P = k * grid_mean (t/km2), I = w P:
  per-pair BD (engine):  (1 + x + w*mean) / (1 + x + w*k*mean)
  shared BD:             same with x -> S = sum of the pair x's (102.55)
  Crowley-Martin:        (1 + w*mean) / (1 + w*k*mean)
Writes tables/intake_vs_crowding.csv.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tables", "intake_vs_crowding.csv")
K = [1, 3, 5, 13, 30]
MEAN = 1.0  # t/km2, run grid mean (0.7-1.3)
PAIRS = {"infauna": 59.0, "epifauna": 38.0, "herring": 5.3, "whiting": 0.25}
S = sum(PAIRS.values())

rows = []


def add(label, kind, fn):
    rows.append([label, kind] + [round(fn(k), 3) for k in K])


# Evidence (power laws, elasticity m)
add("HV m=0.7 (DeLong & Vasseur 2011 median; tanks, mostly invertebrates)", "evidence/extrapolated",
    lambda k: k ** -0.7)
add("HV m=0.05-0.09 midpoint 0.07 (population-scale growth DD, earlier note)", "evidence",
    lambda k: k ** -0.07)
add("HV m=0.185 (Baltic large cod, total prey, haul scale; incl. local depletion) [D]", "evidence",
    lambda k: k ** -0.185)
add("HV m=0.21 (Baltic flounder, benthic prey, haul scale; incl. depletion) [D]", "evidence",
    lambda k: k ** -0.21)
add("m=0 (Baltic small cod benthic; Barents cod on capelin, Fall et al. 2021)", "evidence",
    lambda k: 1.0)
# Model forms
for p, x in PAIRS.items():
    add(f"model per-pair BD w=0.45, {p} (x={x})", "model",
        lambda k, x=x: (1 + x + 0.45 * MEAN) / (1 + x + 0.45 * k * MEAN))
add("model shared-denominator BD w=0.45 (S=102.55), all pairs", "alternative",
    lambda k: (1 + S + 0.45 * MEAN) / (1 + S + 0.45 * k * MEAN))
for w in (0.02, 0.05):
    add(f"Crowley-Martin w'={w}, all pairs", "alternative",
        lambda k, w=w: (1 + w * MEAN) / (1 + w * k * MEAN))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["form", "kind"] + [f"R_{k}x" for k in K])
    wr.writerows(rows)
for r in rows:
    print("%-88s %s" % (r[0], " ".join("%6.3f" % v for v in r[2:])))

# BD w implied by a field elasticity m at a reference crowding density P_ref and
# predator saturation x: m = wP/(1+x+wP)  =>  w = m (1 + x) / ((1 - m) P_ref).
OUT2 = os.path.join(HERE, "..", "tables", "w_implied_by_field_elasticity.csv")
with open(OUT2, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["m", "P_ref_t_km2", "P_ref_note", "x", "w_km2_per_t"])
    print("\nw implied by m at P_ref and saturation x")
    for m in (0.1, 0.185):
        for P, note in ((0.4, "Baltic large cod crowding (0.11 x Lloyd 3.55)"),
                        (5.1, "Skagerrak benthivore crowding (1.67 x Lloyd 3.06)")):
            for x in (0.0, 1.0, 5.3, 38.0):
                w = m * (1 + x) / ((1 - m) * P)
                wr.writerow([m, P, note, x, round(w, 3)])
                print(f"m={m} P_ref={P} x={x}: w={w:.3f}")
