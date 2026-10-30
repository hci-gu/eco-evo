"""Open-population closure of each fish group with field rates only (summary.md section 3).

Usage: python3 -I closure.py <sms summary.out> <share_summary.csv> <out_dir>

For each group: is today's box biomass B* an equilibrium of dB/dt = I + (P - L) B?
- P: somatic production per tonne, SMS 2023 key run 2013-22 (per-tonne growth from weight at age,
  biomass-weighted, stage J = ages 0-1 and adults A, as Arsungarnas andel/stage_box_model.py),
  read at the box J/B. Members outside SMS take the group's SMS members' value (stated).
- L: M1 (library) + booked M2 (library growth budget) + F (library, today's mean effort) +
  net emigration e_J J/B + e_A (1 - J/B) (WP2 central values, upper bounds).
- I: imported recruits = SMS stock numbers at the survey recruit stage (age 0 Q3 or age 1 Q1,
  2013-22 mean N.bar, thousands) x SMS weight there x the box's share of the stock's recruits
  (WP3 share_summary.csv, survey low-high, fractions).
Prints I/B against the I/B needed for balance, (L - P), and the gap. B* = box density x 3600 km2.
Nothing here reads model output.
"""
import os
import sys

import numpy as np
import pandas as pd

sms_path, share_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
S = pd.read_csv(sms_path, sep=r"\s+")
S = S[(S.Year >= 2013) & (S.Year <= 2022)].copy()
for c in ("N.bar", "west"):
    S[c] = pd.to_numeric(S[c], errors="coerce")
S["B"] = S["N.bar"] * S.west
SH = pd.read_csv(share_path).set_index("member")
SP = {"cod": 16, "whiting": 17, "haddock": 18, "saithe": 19, "mackerel": 20, "herring": 21,
      "Norway pout": 24, "sprat": 25}


def growth(sp):
    """(g_J, g_A) per tonne per year, ages 0-1 vs 2+, cohort-followed weight at age."""
    d = S[S["Species.n"] == sp]
    w = d.groupby(["Year", "Quarter", "Age"]).west.mean()
    b = d.groupby(["Year", "Quarter", "Age"]).B.sum()
    acc = {"J": [0.0, 0.0], "A": [0.0, 0.0]}
    for (y, q, a), wv in w.items():
        k = "J" if a <= 1 else "A"
        nxt = (y, q + 1, a) if q < 4 else (y + 1, 1, a + 1)
        if nxt in w.index and wv > 0 and w[nxt] > 0:
            acc[k][0] += 4 * np.log(w[nxt] / wv) * b[(y, q, a)]
            acc[k][1] += b[(y, q, a)]
    return tuple(acc[k][0] / acc[k][1] if acc[k][1] > 0 else np.nan for k in ("J", "A"))


def recruits_t(sp, age, quarter):
    d = S[(S["Species.n"] == sp) & (S.Age == age) & (S.Quarter == quarter)]
    return float(d.groupby("Year").B.sum().mean())  # thousands x kg = tonnes


# Groups: box density (t/km2), J/B in the box (Arsungarnas andel; whiting from its age shares, J),
# members (share, SMS species or None, recruit import: (share_summary member, age, quarter) or None),
# library M1, booked M2, F (sections 209, 223.26), WP2 e_J, e_A.
G = {
    "planktivorous_fish": dict(dens=3.3, jb=0.38, M1=0.26, M2=0.764, F=0.10, eJ=0.1, eA=0.6,
        members=[("herring", 0.79, "herring", ("herring NSAS", 1, 1)),
                 ("blue whiting", 0.105, None, None),
                 ("Norway pout", 0.075, "Norway pout", ("Norway pout", 1, 1)),
                 ("sprat", 0.03, "sprat", ("sprat", 1, 1))]),
    "benthivorous_fish": dict(dens=1.53, jb=0.15, M1=0.29, M2=0.221, F=0.17, eJ=1.1, eA=0.17,
        members=[("haddock", 0.386, "haddock", ("haddock", 0, 3)),
                 ("cod", 0.314, "cod", ("cod (whole NS stock)", 1, 1)),
                 ("flatfish etc", 0.30, None, None)]),
    "whiting": dict(dens=0.446, jb=0.30, M1=0.25, M2=0.369, F=0.08, eJ=0.2, eA=0.3,
        members=[("whiting", 1.0, "whiting", ("whiting (whg.27.47d)", 1, 1))]),
    "predatory_fish": dict(dens=0.754, jb=0.0, M1=0.261, M2=0.016, F=0.242, eJ=0.0, eA=0.1,
        members=[("saithe", 0.651, "saithe", None), ("mackerel", 0.128, "mackerel", None),
                 ("others", 0.221, None, None)]),
}
rows = []
for grp, g in G.items():
    B = g["dens"] * 3600.0
    sms = [(m, s, SP[sp]) for m, s, sp, _ in g["members"] if sp]
    wsum = sum(s for _, s, _ in sms)
    gJ = sum(s * growth(sp)[0] for _, s, sp in sms) / wsum
    gA = sum(s * growth(sp)[1] for _, s, sp in sms) / wsum
    P = g["jb"] * gJ + (1 - g["jb"]) * gA
    emig = g["eJ"] * g["jb"] + g["eA"] * (1 - g["jb"])
    L = g["M1"] + g["M2"] + g["F"] + emig
    I_lo = I_hi = 0.0
    detail = []
    for m, s, sp, imp in g["members"]:
        if not imp:
            continue
        key, age, q = imp
        r = recruits_t(SP[sp], age, q)
        lo, hi = SH.loc[key, f"surv_age{age}_lo"], SH.loc[key, f"surv_age{age}_hi"]  # fractions
        I_lo += r * lo
        I_hi += r * hi
        detail.append(f"{m}: stock age{age} Q{q} {r:,.0f} t x {100 * lo:.2f}-{100 * hi:.2f} % "
                      f"= {r * lo:,.0f}-{r * hi:,.0f} t/yr")
    need = (L - P) * B
    rows.append(dict(group=grp, B_t=B, JB=g["jb"], g_J=gJ, g_A=gA, P=P, M1=g["M1"], M2=g["M2"],
                     F=g["F"], emig=emig, L=L, L_minus_P=L - P, I_lo_t=I_lo, I_hi_t=I_hi,
                     IB_lo=I_lo / B, IB_hi=I_hi / B, I_needed_t=need,
                     covered_lo=I_lo / need if need > 0 else np.nan,
                     covered_hi=I_hi / need if need > 0 else np.nan))
    print(f"\n{grp}: B* {B:,.0f} t, J/B {g['jb']:.2f}; SMS g_J {gJ:.2f} g_A {gA:.2f} -> P {P:.2f} /yr "
          f"(members in SMS: {', '.join(m for m, _, _ in sms)}; share {wsum:.2f})")
    print(f"  L = M1 {g['M1']:.2f} + M2 {g['M2']:.3f} + F {g['F']:.3f} + emigration {emig:.2f} = {L:.2f} /yr;"
          f" L - P = {L - P:.2f}")
    for line in detail:
        print("   " + line)
    print(f"  I/B imported recruits {I_lo / B:.3f}-{I_hi / B:.3f} vs needed {L - P:.2f}: "
          f"covers {100 * I_lo / need:.0f}-{100 * I_hi / need:.0f} % (need {need:,.0f} t/yr)"
          if need > 0 else "  needs no import")
pd.DataFrame(rows).to_csv(os.path.join(out, "closure.csv"), index=False)
