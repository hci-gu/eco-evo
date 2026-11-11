"""Quarterly predation mortality on the planktivore species in the SMS 2023 North Sea key run.

Usage: python3 -I sms_quarter_m2.py <sms2023 dir> [Y0 Y1]
       (<sms2023 dir> = research_notes/"Fiskgrupper i rutan"/planktivores/data/sms2023)

For herring (21), Norway pout (24) and sprat (25): the partial M2 by predator class and quarter,
biomass-weighted over the prey ages in each stage class (age 0, age 1, ages 2+), annualised (x 4),
mean over Y0-Y1 (default 2013-2022). Biomass = N.bar x west from summary.out.

What it is: SMS M2 = predator abundance and ration by quarter x size suitability x spatial overlap
(quarterly where the key run has it) / available food. It is NOT a measurement of a per-capita refuge;
it says whether the stomach-based model puts herring predation low in winter, with all of predator
presence, ration (temperature) and other food confounded. Nothing is fitted; no model output is read.
"""
import os
import sys

import pandas as pd

D = os.path.abspath(sys.argv[1])
y0, y1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2013, 2022)

names = [ln.strip().strip("_") for ln in open(os.path.join(D, "species_names.in"))][:27]
NAME = {i + 1: n for i, n in enumerate(names)}
FISH = {"Cod", "Whiting", "Haddock", "Saithe", "Hake", "Mackerel", "W.horse.mac", "N.horse.mac",
        "A.radiata", "G.gurnards"}


def pclass(no):
    n = NAME.get(no, "?")
    if n in FISH:
        return "fish:" + n
    if n == "Grey.seal":
        return "seal"
    if n == "H.porpoise":
        return "porpoise"
    return "bird"


S = pd.read_csv(os.path.join(D, "summary.out"), sep=r"\s+")
for c in ("N.bar", "west"):
    S[c] = pd.to_numeric(S[c], errors="coerce")
S = S[(S.Year >= y0) & (S.Year <= y1)]
S["B"] = S["N.bar"] * S.west
P = pd.read_csv(os.path.join(D, "partial_m2.out"), sep=r"\s+")
P = P[(P.Year >= y0) & (P.Year <= y1)]

STAGES = {"age 0": lambda a: a == 0, "age 1": lambda a: a == 1, "ages 2+": lambda a: a >= 2,
          "all ages": lambda a: a >= 0}
for prey in (21, 24, 25):
    print(f"\n=== prey {NAME[prey]} ({y0}-{y1}); annualised partial M2 (x4) by quarter, /yr")
    Sp = S[S["Species.n"] == prey]
    Pp = P[P["Prey.no"] == prey].copy()
    Pp = Pp.merge(Sp[["Year", "Quarter", "Age", "B"]], left_on=["Year", "Quarter", "Prey.age"],
                  right_on=["Year", "Quarter", "Age"], how="left")
    Pp["eaten"] = Pp["Part.M2"] * Pp.B
    Pp["cls"] = Pp["Predator.no"].map(pclass)
    Pp["grp"] = Pp.cls.map(lambda c: "fish" if c.startswith("fish") else c)
    for st, sel in STAGES.items():
        rows = []
        for q in (1, 2, 3, 4):
            b = Sp[(Sp.Quarter == q) & sel(Sp.Age)].B.sum()
            d = Pp[(Pp.Quarter == q) & sel(Pp["Prey.age"])]
            if b <= 0:
                continue
            r = {"Q": q, "B_share": b}
            for g, e in d.groupby("grp").eaten.sum().items():
                r[g] = 4 * e / b
            for g, e in d[d.grp == "fish"].groupby("cls").eaten.sum().items():
                r[g] = 4 * e / b
            rows.append(r)
        T = pd.DataFrame(rows).fillna(0.0).set_index("Q")
        T["B_share"] = T.B_share / T.B_share.sum()
        cols = ["B_share"] + [c for c in ("fish", "seal", "porpoise", "bird") if c in T] + \
            sorted([c for c in T if c.startswith("fish:") and T[c].max() > 0.005])
        print(f"-- {st}")
        print(T[cols].round(3).to_string())
        if "fish" in T:
            ann = (T.fish * T.B_share).sum()
            print(f"   fish M2: biomass-weighted annual {ann:.3f}; Q1/annual {T.fish.get(1, 0) / ann:.2f}; "
                  f"Q4/annual {T.fish.get(4, 0) / ann:.2f}; Q3/annual {T.fish.get(3, 0) / ann:.2f}")
