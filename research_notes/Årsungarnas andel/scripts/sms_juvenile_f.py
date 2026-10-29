"""Biomass-weighted fishing mortality on age 0-1 vs all ages, from the WGSAM SMS North Sea key run 2023.

Usage: python3 -I sms_juvenile_f.py <path/to/summary.out> [first_year last_year]

summary.out columns: Year Quarter Species.n Age M1 M2 M F Z N N.bar C.hat C.obs west weca ...
F is per quarter. Biomass in a quarter = N.bar * west. For a set S of (quarter, age) cells:
    F_S (per year) = 4 * sum_S F * Nbar * west / sum_S Nbar * west
which is the rate a constant per-tick loss on the S tonnes must have to remove the same catch.
J variants:
  A: ages 0 and 1, all quarters (the J of summary.md: age 0 + age 1 in every quarter)
  B: age 0 in Q3-Q4 plus age 1 in Q1-Q2 (graduation at the start of the second summer)
Prints per species: annual F at ages 0-4 (sum of quarterly F), F_J, F_adult, F_all and the ratios.
"""
import sys
from collections import defaultdict

SPECIES = {16: "cod", 17: "whiting", 18: "haddock", 19: "saithe", 21: "herring",
           24: "Norway pout", 25: "sprat", 26: "plaice"}

path = sys.argv[1]
y0, y1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2013, 2022)

rows = []
with open(path) as fh:
    head = fh.readline().split()
    ix = {k: i for i, k in enumerate(head)}
    for line in fh:
        p = line.split()
        if len(p) < len(head) - 1:
            continue
        yr, q, sp, age = int(p[ix["Year"]]), int(p[ix["Quarter"]]), int(p[ix["Species.n"]]), int(p[ix["Age"]])
        if sp not in SPECIES or not (y0 <= yr <= y1):
            continue
        F = float(p[ix["F"]]); nb = float(p[ix["N.bar"]]); w = float(p[ix["west"]])
        C = float(p[ix["C.hat"]]); wc = float(p[ix["weca"]])
        rows.append((yr, q, sp, age, F, nb * w, C * wc))


def inA(q, a):
    return a <= 1


def inB(q, a):
    return (a == 0 and q >= 3) or (a == 1 and q <= 2)


print(f"SMS key run 2023, years {y0}-{y1}; F per year; biomass-weighted over quarter x age cells")
print("species | F0 F1 F2 F3 F4 (annual, mean of years) | F_J(A) F_adult(A) F_all | rJ(A) rA(A) | F_J(B) rJ(B) | "
      "J(A) share of stock biomass | C/B check")
for sp, name in SPECIES.items():
    fa = defaultdict(float)
    acc = defaultdict(float)
    for yr, q, s, a, F, B, Cw in rows:
        if s != sp:
            continue
        if a <= 4:
            fa[a] += F
        for key, inside in (("JA", inA(q, a)), ("AA", not inA(q, a)), ("ALL", True), ("JB", inB(q, a))):
            if inside:
                acc[key + "_FB"] += F * B
                acc[key + "_B"] += B
        acc["C"] += Cw
    ny = y1 - y0 + 1
    f = {k: 4 * acc[k + "_FB"] / acc[k + "_B"] for k in ("JA", "AA", "ALL", "JB") if acc[k + "_B"] > 0}
    fat = " ".join(f"{fa[a] / ny:.3f}" for a in range(5))
    cb = 4 * acc["C"] / acc["ALL_B"]
    print(f"{name:12s} | {fat} | {f['JA']:.3f} {f['AA']:.3f} {f['ALL']:.3f} | {f['JA'] / f['ALL']:.2f} "
          f"{f['AA'] / f['ALL']:.2f} | {f['JB']:.3f} {f['JB'] / f['ALL']:.2f} | {acc['JA_B'] / acc['ALL_B']:.2f} | {cb:.3f}")
