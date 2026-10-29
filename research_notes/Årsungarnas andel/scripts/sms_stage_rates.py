"""Mortality of the juvenile stage vs the whole stock, SMS 2023 North Sea key run.

Usage: python3 -I sms_stage_rates.py <summary.out> [Y0 Y1]

Per species and quarter: biomass at age B = N.bar x west; quarterly M1, M2 and F per age. The juvenile
stage J is age 0 (Q3-Q4) + age 1 (Q1-Q2), the plan's "age 0 + 1 until the second summer"; the
variant J01 counts age 1 in every quarter. Prints biomass-weighted annualised rates (x 4) for J and
for all ages, and their ratios; the J share of stock biomass per quarter (North Sea scale, not the
box). SMS species: 16 cod, 17 whiting, 18 haddock, 21 herring, 24 Norway pout, 25 sprat.
"""
import sys

import numpy as np
import pandas as pd

path = sys.argv[1]
y0, y1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2013, 2022)
S = pd.read_csv(path, sep=r"\s+")
S = S[(S.Year >= y0) & (S.Year <= y1)].copy()
for c in ("M1", "M2", "F", "N.bar", "west"):
    S[c] = pd.to_numeric(S[c], errors="coerce")
S["B"] = S["N.bar"] * S.west
NAME = {16: "cod", 17: "whiting", 18: "haddock", 21: "herring", 24: "Norway pout", 25: "sprat"}
S = S[S["Species.n"].isin(NAME)]


def is_j(age, q, variant):
    if variant == "J01":
        return age <= 1
    return (age == 0 and q >= 3) or (age == 1 and q <= 2)


print(f"SMS 2023 key run, years {y0}-{y1}; rates /yr (quarterly x 4), biomass-weighted")
rows = []
for sp, d in S.groupby("Species.n"):
    for variant in ("J", "J01"):
        for q, e in d.groupby("Quarter"):
            j = e[[is_j(a, q, variant) for a in e.Age]]
            tot = e.B.sum()
            if tot <= 0:
                continue
            r = dict(species=NAME[sp], variant=variant, quarter=q, J_share=j.B.sum() / tot)
            for c in ("M1", "M2", "F"):
                r[f"{c}_J"] = 4 * float((j[c] * j.B).sum() / j.B.sum()) if j.B.sum() > 0 else np.nan
                r[f"{c}_all"] = 4 * float((e[c] * e.B).sum() / tot)
            rows.append(r)
R = pd.DataFrame(rows)
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(R.round(3).to_string(index=False))
print("\nannual (biomass-weighted over quarters): rate on J / rate on all ages")
for (sp, v), d in R.groupby(["species", "variant"]):
    w = d.J_share
    out = []
    for c in ("M1", "M2", "F"):
        rj = float((d[f"{c}_J"].fillna(0) * w).sum() / w.sum()) if w.sum() > 0 else np.nan
        ra = float(d[f"{c}_all"].mean())
        out.append(f"{c} {rj:.2f}/{ra:.2f} (x{rj / ra if ra > 0 else np.nan:.2f})")
    zj = sum(float((d[f"{c}_J"].fillna(0) * w).sum() / w.sum()) for c in ("M1", "M2", "F"))
    za = sum(float(d[f"{c}_all"].mean()) for c in ("M1", "M2", "F"))
    print(f"  {sp:12s} {v:4s} J share by quarter " + " ".join(f"{x:.2f}" for x in d.sort_values("quarter").J_share)
          + " | " + ", ".join(out) + f" | Z {zj:.2f}/{za:.2f} (x{zj / za:.2f})")

# ------------------------------------------------------------------ somatic growth per tonne, J vs adults
print("\nsomatic growth per tonne (/yr): 4 ln(w(next quarter)/w), age +1 at the year turn; biomass-weighted")
for sp, d in S.groupby("Species.n"):
    w = d.groupby(["Year", "Quarter", "Age"]).west.mean()
    b = d.groupby(["Year", "Quarter", "Age"]).B.sum()
    gj = ga = bj = ba = 0.0
    for (y, q, a), wv in w.items():
        nxt = (y, q + 1, a) if q < 4 else (y + 1, 1, a + 1)
        if nxt not in w.index or wv <= 0 or w[nxt] <= 0:
            continue
        g = 4 * np.log(w[nxt] / wv)
        bb = b[(y, q, a)]
        if is_j(a, q, "J"):
            gj += g * bb
            bj += bb
        else:
            ga += g * bb
            ba += bb
    if bj > 0 and ba > 0:
        print(f"  {NAME[sp]:12s} g_J {gj / bj:.2f}, g_adult {ga / ba:.2f} (x{(gj / bj) / (ga / ba):.1f})")
