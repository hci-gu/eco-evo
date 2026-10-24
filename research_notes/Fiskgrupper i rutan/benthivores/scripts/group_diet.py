"""Consumption-weighted diet of the pooled benthivorous_fish group (what the prey M2 needs).
Usage: python3 -I group_diet.py <qb_species.csv> <diet_species.csv> <out_csv>
Consumption share of pool j = sum_i share_i x QB_i(central) x diet_ij / sum_i share_i x QB_i.
Bf (within group) and Oth (no pool) are kept as columns; they are not renormalised away.
Also returns consumption per unit group biomass by pool (t prey / t group / yr).
"""
import csv, sys
qb = {r["species"]: r for r in csv.DictReader(open(sys.argv[1]))}
D = {}
for r in csv.DictReader(open(sys.argv[2])):
    D.setdefault(r["species"], {})[r["variant"]] = r
P = ["B", "Z", "P", "W", "Bf", "Oth"]
def run(choice):
    tot = {p: 0.0 for p in P}; cons = 0.0; bt = 0.0
    for sp, q in qb.items():
        v = choice.get(sp, "central")
        d = D[sp][v]; s = float(q["share_pct"]) / 100; Q = float(q["qb_central"])
        cons += s * Q; bt += s
        for p in P:
            tot[p] += s * Q * float(d[p]) / 100
    return {p: tot[p] / cons for p in P}, cons / bt
variants = {"central (cod Skagerrak, Gu allocated)": {"cod": "skag_gu"},
            "cod Skagerrak, Gu not allocated": {"cod": "skag_nogu"},
            "cod NNS 2023, Gu allocated": {"cod": "nns_gu"},
            "haddock winter only": {"cod": "skag_gu", "haddock": "winter_only"},
            "haddock high Z": {"cod": "skag_gu", "haddock": "high_Z"},
            "LRD high fish": {"cod": "skag_gu", "long_rough_dab": "high_fish"},
            "starry ray ICES 2024": {"cod": "skag_gu", "starry_ray": "ices2024"},
            "high P (cod NNS + LRD high fish)": {"cod": "nns_gu", "long_rough_dab": "high_fish"}}
rows = []
for name, ch in variants.items():
    sh, qbm = run(ch)
    rows.append([name, round(qbm, 2)] + [round(100 * sh[p], 1) for p in P] + [round(qbm * sh[p], 3) for p in P])
with open(sys.argv[3], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "pooled_QB"] + [f"pct_{p}" for p in P] + [f"t_per_t_yr_{p}" for p in P])
    w.writerows(rows)
for r in rows: print(r)
