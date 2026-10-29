"""Share of each predator's take of a prey stock that is juvenile (age 0 + 1), SMS 2023 key run.

Usage: python3 -I sms_juvenile_diet.py <partial_m2.out> <summary.out> [Y0 Y1]

Eaten biomass of prey age a by predator p in a quarter = Part.M2(p, a) x N.bar(a) x west(a) (the M2 is
per quarter in SMS). Summed over quarters and years Y0-Y1 (default 2013-22) and predator ages. Prints,
per predator and prey stock, the juvenile share of the eaten tonnes and the juvenile share of the prey
stock's biomass; their ratio is the field selectivity of the predator for juveniles (f_J / f_A follows
from it: share_eaten / share_stock = f_J / (f_J x share_stock + f_A x (1 - share_stock)) ...).
"""
import sys

import pandas as pd

pm, summ = sys.argv[1:3]
y0, y1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (2013, 2022)
P = pd.read_csv(pm, sep=r"\s+")
S = pd.read_csv(summ, sep=r"\s+")
P = P[(P.Year >= y0) & (P.Year <= y1)]
S = S[(S.Year >= y0) & (S.Year <= y1)].copy()
S["B"] = pd.to_numeric(S["N.bar"], errors="coerce") * pd.to_numeric(S.west, errors="coerce")
B = S.set_index(["Year", "Quarter", "Species.n", "Age"]).B
NAMES = [l.strip().strip("_") for l in open(pm.replace("partial_m2.out", "species_names.in"))][:27]
NAME = {i + 1: n for i, n in enumerate(NAMES)}
P = P.join(B.rename("Bprey"), on=["Year", "Quarter", "Prey.no", "Prey.age"])
P["eaten"] = P["Part.M2"] * P.Bprey
PREY = {21: "Herring", 18: "Haddock", 16: "Cod", 17: "Whiting", 25: "Sprat", 24: "Nor.pout"}
print(f"SMS 2023, {y0}-{y1}: juvenile (age 0-1) share of eaten tonnes vs of the stock's biomass")
for prey, pname in PREY.items():
    st = S[S["Species.n"] == prey]
    js = st[st.Age <= 1].B.sum() / st.B.sum()
    d = P[P["Prey.no"] == prey]
    out = []
    for pred, e in d.groupby("Predator.no"):
        tot = e.eaten.sum()
        if tot <= 0:
            continue
        je = e[e["Prey.age"] <= 1].eaten.sum() / tot
        out.append((tot, NAME.get(pred, pred), je))
    out.sort(reverse=True)
    tot_all = sum(o[0] for o in out)
    print(f"\n{pname}: stock juvenile share {js:.3f}")
    for tot, n, je in out[:12]:
        print(f"  {n:12s} {tot / tot_all:6.3f} of the eaten tonnes; juvenile share eaten {je:.3f} (x{je / js:.1f})")
