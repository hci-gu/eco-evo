"""Booked M2 vs SMS 2023 M2 on a model group, by predator group (summary.md, closure caveats).

Usage: python3 -I sms_m2_check.py <summary.out> <partial_m2.out> <who_eats_whom_level2.csv> <group> [Y0 Y1]
group: planktivorous_fish | whiting

Partial M2 per quarter, predator (number) and prey age from partial_m2.out; predator numbers are
named by matching their eaten biomass (Part.M2 x N.bar x west, by year) against the named eatenW of
who_eats_whom_level2.csv. Rates are annual biomass-weighted (x 4 per quarter), for all ages, ages
1+, and the box stage mix (J = ages 0-1 at the box J/B). SMS predators are mapped to model groups.
"""
import sys

import numpy as np
import pandas as pd

summ, part, who, group = sys.argv[1:5]
y0, y1 = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (2013, 2022)
PREY = {"planktivorous_fish": {21: ("Herring", 0.79 / 0.895), 24: ("Nor.pout", 0.075 / 0.895),
                               25: ("Sprat", 0.03 / 0.895)},
        "whiting": {17: ("Whiting", 1.0)}}[group]
JB = {"planktivorous_fish": 0.38, "whiting": 0.30}[group]
MODEL = {"Cod": "benthivorous_fish", "Haddock": "benthivorous_fish", "A.radiata": "benthivorous_fish",
         "G.gurnards": "benthivorous_fish", "Saithe": "predatory_fish", "Mackerel": "predatory_fish",
         "Hake": "predatory_fish", "W.horse.mac": "predatory_fish", "N.horse.mac": "predatory_fish",
         "Whiting": "whiting", "Grey.seal": "seals", "H.porpoise": "porpoises"}
S = pd.read_csv(summ, sep=r"\s+")
for c in ("N.bar", "west", "M2"):
    S[c] = pd.to_numeric(S[c], errors="coerce")
S["B"] = S["N.bar"] * S.west
P = pd.read_csv(part, sep=r"\s+")
W = pd.read_csv(who)
B = S.set_index(["Year", "Quarter", "Species.n", "Age"]).B
# name the predator numbers
P = P[P["Prey.no"].isin(PREY)].copy()
P["B"] = [B.get((y, q, s, a), np.nan) for y, q, s, a in
          zip(P.Year, P.Quarter, P["Prey.no"], P["Prey.age"])]
P["eaten"] = P["Part.M2"] * P.B
names = {}
for no, d in P.groupby("Predator.no"):
    e = d.groupby("Year").eaten.sum()
    best, err = None, np.inf
    for nm, w in W[W.Prey.isin([v[0] for v in PREY.values()])].groupby("Predator"):
        x = w.groupby("Year").eatenW.sum().reindex(e.index).fillna(0)
        r = float(np.abs(np.log((x + 1) / (e + 1))).mean())
        if r < err:
            best, err = nm, r
    names[no] = (best, err)
P["pred"] = P["Predator.no"].map(lambda n: names[n][0])
P = P[(P.Year >= y0) & (P.Year <= y1)]
Sx = S[(S.Year >= y0) & (S.Year <= y1) & S["Species.n"].isin(PREY)]


def rate(stage_sel, col_filter=None):
    """Annual biomass-weighted M2 by predator over the selected prey ages, member-weighted."""
    out = {}
    for sp, (_, wgt) in PREY.items():
        b = Sx[(Sx["Species.n"] == sp) & stage_sel(Sx.Age)].B.sum()
        d = P[(P["Prey.no"] == sp) & stage_sel(P["Prey.age"])]
        for pr, e in d.groupby("pred").eaten.sum().items():
            out[pr] = out.get(pr, 0.0) + wgt * 4 * e / b
    return out


rows = {}
for lab, sel in (("all ages", lambda a: a >= 0), ("age 1+", lambda a: a >= 1),
                 ("J (0-1)", lambda a: a <= 1), ("A (2+)", lambda a: a >= 2)):
    rows[lab] = rate(sel)
T = pd.DataFrame(rows).fillna(0.0)
T["box mix"] = JB * T["J (0-1)"] + (1 - JB) * T["A (2+)"]
T["model group"] = [MODEL.get(i, "birds/other") for i in T.index]
print(f"SMS 2023 key run {y0}-{y1}: M2 on {group} members (" +
      ", ".join(f"{v[0]} {v[1]:.2f}" for v in PREY.values()) + f"), box J/B {JB}")
print("predator numbers named by eaten-biomass match (mean |log ratio|): " +
      ", ".join(f"{k}={v[0]}({v[1]:.2f})" for k, v in sorted(names.items())))
with pd.option_context("display.width", 160):
    print(T.round(3).sort_values("box mix", ascending=False).to_string())
G = T.groupby("model group")[["all ages", "age 1+", "box mix"]].sum()
G.loc["total"] = G.sum()
print("\nby model group:")
print(G.round(3).to_string())

# Scaled to the box: a predator's partial M2 on a prey is ~ (its ration x diet share) x B_pred / B_prey,
# so the box value is the SMS value x (B_pred/B_prey in the box) / (B_pred/B_prey in SMS), at the
# SMS diet (resume 223.28 did this for cod -> whiting). Box biomass = group density x member share
# (library, t/km2); SMS biomass = 2013-22 mean over quarters. Horse mackerel W + N pooled (each part scaled by the pooled
# box / SMS ratio, which splits the box's horse mackerel by the SMS shares); birds out
# (seabirds muted); grey seal stands for the box's harbour seals only by analogy.
BOX_PRED = {"Saithe": 0.754 * 0.651, "Mackerel": 0.754 * 0.128, "Hake": 0.754 * 0.040,
            "W.horse.mac": 0.754 * 0.055, "N.horse.mac": 0.754 * 0.055, "Cod": 1.53 * 0.314,
            "Haddock": 1.53 * 0.386, "A.radiata": 1.53 * 0.015, "G.gurnards": 1.53 * 0.009,
            "Whiting": 0.446, "H.porpoise": 3.0 / 3600, "Grey.seal": 15.0 / 3600}
BOX_PREY = {21: 3.3 * 0.79, 24: 3.3 * 0.075, 25: 3.3 * 0.03, 17: 0.446}
SMS_NO = {v[0]: k for k, v in names.items()}
Bq = S[(S.Year >= y0) & (S.Year <= y1)].groupby("Species.n").apply(
    lambda d: d.B.sum() / len(d.groupby(["Year", "Quarter"])))
Bhm = Bq[SMS_NO["W.horse.mac"]] + Bq[SMS_NO["N.horse.mac"]]
scaled = {}
for sp, (_, wgt) in PREY.items():
    bJ = Sx[(Sx["Species.n"] == sp) & (Sx.Age <= 1)].B.sum()
    bA = Sx[(Sx["Species.n"] == sp) & (Sx.Age >= 2)].B.sum()
    for pr in BOX_PRED:
        d = P[(P["Prey.no"] == sp) & (P.pred == pr)]
        mJ = 4 * d[d["Prey.age"] <= 1].eaten.sum() / bJ
        mA = 4 * d[d["Prey.age"] >= 2].eaten.sum() / bA
        m = JB * mJ + (1 - JB) * mA
        b_ns = Bhm if "horse" in pr else Bq[SMS_NO[pr]]
        box_pred = BOX_PRED[pr] if "horse" not in pr else 0.754 * 0.055
        f = (box_pred / BOX_PREY[sp]) / (b_ns / Bq[sp])
        scaled[pr] = scaled.get(pr, 0.0) + wgt * m * f
Sc = pd.Series(scaled)
Gs = Sc.groupby([MODEL.get(i, "birds/other") for i in Sc.index]).sum()
Gs.loc["total"] = Gs.sum()
print("\nscaled to the box's predator / prey biomass (box stage mix), by predator and model group:")
print(Sc.round(3).sort_values(ascending=False).to_string())
print(Gs.round(3).to_string())
