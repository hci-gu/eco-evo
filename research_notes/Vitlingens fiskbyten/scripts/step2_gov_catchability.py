"""GOV catchability-at-length for herring, sprat and Norway pout, applied to the box stage lengths.

Usage: python3 -I step2_gov_catchability.py <stage_length_biomass.csv> <walker_EfficiencyTab.csv>

Question: are small herring and sprat (<= ~13 cm) under-measured by the NS-IBTS GOV relative to
larger fish, and does a q(L) correction change the box's share of small planktivore biomass?

Inputs
- stage_length_biomass.csv (../../Arsungarnas andel/tables/): box GOV biomass (kg/h) by 1-cm class
  L_cm (class lower bound) and stage J/A; herring and sprat RAW (q = 1), Norway pout already divided by
  the Fraser et al. 2007 q (<= 11 cm 0.066204, >= 12 cm 0.191855, evaluated at L_cm + 0.5).
- EfficiencyTab.csv: supplementary data file of Walker, Maxwell, Le Quesne & Jennings 2017, ICES J Mar
  Sci 74:1448-1457, doi 10.1093/icesjms/fsw250 (fsw250_supp.zip, downloaded 2026-10-31).

Tags: [F] number taken from the source, [D] derived here, [J] judgement.

q(L) parameters and sources
- [F] Walker 2017 Table 3, herring (species-group 6) efficiency multiplier = IBTS survey numbers-at-age /
  HAWG assessment numbers-at-age, averaged over Q1 and Q3 and years, with mean length-at-age (cm):
  age 0 0.11 (9.1), 1 0.41 (12.6), 2 0.30 (19.5), 3 0.26 (23.1), 4 0.22 (24.6), 5 0.28 (25.2),
  6 0.22 (25.7), 7 0.22 (26.2), 8 0.22 (26.8). Norway pout (group 5): 0.2 (5.5), 0.36 (10.1),
  0.53 (13.7), 0.56 (15.2), 0.4 (15.7). Same points in supplement Fig. S10 (p. 16).
- [F] Walker 2017 text: "Fish smaller than 5cm were excluded from our analysis"; below the smallest
  observed class "a straight line from the origin to the value of the function at the smallest
  observed length class was assumed"; "we account for the horizontal availability of fish but ignore
  the vertical component".
- [F] EfficiencyTab.csv, Gear GOV, column Absolute (assessment-anchored efficiency):
  herring 5-cm classes 5: 0.0456, 10: 0.3748, 15: 0.3707, 20: 0.2643, 25 and up: 0.2167;
  sprat / anchovy 1-cm classes 5-8: 0.0456, 9: 0.1292, 10: 0.2118, 11: 0.2880, 12: 0.3507,
  13: 0.3932, 14: 0.4137, 15: 0.4160, anchovy 18: 0.3572.
- [D] The sprat and anchovy Absolute values equal the herring values where they overlap (5 cm 0.045604
  in all group-6 species) and GOV is the reference gear at every sprat length, so the 1-cm series is
  the group-6 (HERRING) multiplier GAM evaluated at 1-cm classes. Sprat has no own anchor: its GOV q(L)
  is herring's age-based multiplier mapped onto sprat lengths [J: borrowed, not a sprat measurement].
- [F] EfficiencyTab.csv, GOV column Efficiency (the final shrinkage-GAM curve): herring flat 0.2437
  (95 % 0.161-0.327) at every length >= 5 cm, i.e. NO length effect in the published herring curve;
  sprat 0.040 (6 cm) - 0.068 (8) - 0.207 (10) - 0.347 (12) - 0.412 (14) - 0.431 (16+);
  Norway pout 0.314 (6) - 0.371 (10) - 0.427 (14) - 0.509 (20) - 0.574 (25+).
- [F] HAWG 2023 (ICES Sci Rep 5:23, doi 10.17895/ices.pub.22182034), Table 2.6.2.2 p. 102, North Sea
  herring SAM catchabilities at age, IBTS-Q3 (GOV, August): age 0 0.1036, 1 0.04618, 2 0.04158,
  3 0.03802, 4 0.03314, 5 0.02481; IBTS-Q1 age 1 0.1059.
- [F] HAWG 2023 Tables 3.6.8.d (p. 254-255) and 3.6.13 (p. 268-269), WBSS herring: IBTS+BITS Q1 index
  ages 1-3 and SAM stock numbers ages 1-3, 2002-2022 (transcribed below); [D] q(age) = index / N.
- [F] HAWG 2023 p. 679, North Sea + 3.a sprat SMS survey catchability, IBTS Q3 ages 1-3: 0.941, 1.202,
  1.205 (age-0 sprat excluded from IBTS Q3 and HERAS, p. 642-643: "0-group (young-of-the-year) sprat is
  unlikely to be fully recruited by the time of IBTS Q3 and HERAS").
- [J] Age -> length map for the HAWG age-based q in the box: Q3 herring L <= 14 cm age 0 (box Q3 J
  mode 11-13 cm), 15-17 age 1, 18-21 age 2, 22-24 age 3, 25-26 age 4, >= 27 age 5.
- [J] Every q(L) is normalised so that only its SHAPE matters (shares are ratios within a species).
  Blue whiting has no length rows in the box table; it counts as no small biomass in the pool share.
- [J] The herring row L_cm = 160 (Q3 2001-25, A = 0.117) is a bad length class and is dropped.

Variants (herring): H0 raw; H1 Walker Absolute, 1-cm group-6 GAM (5-18 cm) + herring 5-cm classes
>= 20 cm, linear between class midpoints; H2 Walker herring 5-cm Absolute as a step; H3 Walker Table 3
per-age points interpolated on length, flat 0.11 below 9.1 cm; H4 = H3 with Walker's straight line
from the origin below 9.1 cm (largest correction); H5 HAWG 2023 SAM IBTS-Q3 q-at-age (Q3 and Q4
rows only); H6 Walker final Efficiency curve (flat).
Sprat: S0 raw; S1 Walker GOV Efficiency (sprat row). Norway pout: P0 as tabled (Fraser);
P1 Fraser undone, Walker NOP GOV Efficiency applied.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

stage_csv, walker_csv = sys.argv[1:3]

# ---------------------------------------------------------------------------------------------- data
SL = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))   # (period, q, sp) -> L -> [J, A]
dropped = []
with open(stage_csv) as fh:
    for r in csv.DictReader(fh):
        L = int(r["L_cm"])
        if r["species"] == "herring" and L > 60:
            dropped.append((r["period"], r["quarter"], L, float(r["J"]), float(r["A"])))
            continue
        cell = SL[(r["period"], int(r["quarter"]), r["species"])][L]
        cell[0] += float(r["J"])
        cell[1] += float(r["A"])

W = defaultdict(dict)   # (code, gear, column) -> {L: value}
with open(walker_csv) as fh:
    for r in csv.DictReader(fh):
        for col in ("Absolute", "Efficiency"):
            if r[col] != "NA":
                W[(r["Code"], r["Gear"], col)][int(r["Length"])] = float(r[col])


def interp(points, L):
    x = np.array(sorted(points))
    y = np.array([points[k] for k in x])
    return np.interp(L, x, y)          # flat beyond the end points


# ------------------------------------------------------------------------------------------ q curves
# H1: 1-cm group-6 multiplier (sprat/anchovy Absolute rows) + herring 5-cm classes >= 20 cm
h1 = {}
for code in ("SPR", "ANE"):
    for L, v in W[(code, "GOV", "Absolute")].items():
        h1.setdefault(L + 0.5, v)
for L, v in W[("HER", "GOV", "Absolute")].items():
    if L >= 20:
        h1[L + 2.5] = v                 # 5-cm class midpoint
h2_cls = W[("HER", "GOV", "Absolute")]  # {5: .., 10: .., 15: .., 20: .., 25: .., ...}
T3_HER = [(9.1, 0.11), (12.6, 0.41), (19.5, 0.30), (23.1, 0.26), (24.6, 0.22), (25.2, 0.28),
          (25.7, 0.22), (26.2, 0.22), (26.8, 0.22)]
SAM_Q3 = {0: 0.1036, 1: 0.04618, 2: 0.04158, 3: 0.03802, 4: 0.03314, 5: 0.02481}


def q_herring(var, Lmid, quarter):
    if var == "H0":
        return np.ones_like(Lmid)
    if var == "H1":
        return interp(h1, Lmid)
    if var == "H2":
        cls = np.clip((np.floor(Lmid) // 5) * 5, 5, max(h2_cls))
        return np.array([h2_cls.get(int(c), h2_cls[max(k for k in h2_cls if k <= c)]) for c in cls])
    if var in ("H3", "H4"):
        pts = {}
        for L, m in T3_HER:             # mean of duplicates at nearly equal lengths kept as listed
            pts[L] = m
        q = interp(pts, Lmid)
        if var == "H4":
            q = np.where(Lmid < 9.1, 0.11 * Lmid / 9.1, q)
        return q
    if var == "H5":
        if quarter not in (3, 4):
            return None
        age = np.select([Lmid < 15, Lmid < 18, Lmid < 22, Lmid < 25, Lmid < 27], [0, 1, 2, 3, 4], 5)
        return np.array([SAM_Q3[a] for a in age])
    if var == "H6":
        return interp({k + 0.5: v for k, v in W[("HER", "GOV", "Efficiency")].items() if k >= 5}, Lmid)
    raise ValueError(var)


def q_sprat(var, Lmid):
    if var == "S0":
        return np.ones_like(Lmid)
    return interp({k + 0.5: v for k, v in W[("SPR", "GOV", "Efficiency")].items() if k >= 5}, Lmid)


def q_fraser_pout(Lmid):
    return np.where(Lmid <= 11, 0.066204, 0.191855)


def q_pout(var, Lmid):
    if var == "P0":
        return np.ones_like(Lmid)       # table already Fraser-corrected
    walker = interp({k + 0.5: v for k, v in W[("NOP", "GOV", "Efficiency")].items() if k >= 5}, Lmid)
    return walker / q_fraser_pout(Lmid)  # divide raw (= table * q_fraser) by Walker


# ----------------------------------------------------------------------------------------- summaries
def shares(key, qfun):
    d = SL[key]
    L = np.array(sorted(d))
    J = np.array([d[k][0] for k in L])
    A = np.array([d[k][1] for k in L])
    Lm = L + 0.5
    q = qfun(Lm)
    if q is None:
        return None
    J, A = J / q, A / q
    B = J.sum() + A.sum()
    tot = J + A
    return dict(J10=J[Lm <= 10].sum() / J.sum(), J13=J[Lm <= 13].sum() / J.sum(), JB=J.sum() / B,
                B10=tot[Lm <= 10].sum() / B, B13=tot[Lm <= 13].sum() / B, B14=tot[Lm <= 14].sum() / B)


PERIOD = {1: "2001-25", 2: "1991-96", 3: "2001-25", 4: "1991-96"}
POOL_W = {1: {"herring": 0.88, "Norway pout": 0.11, "sprat": 0.01, "blue whiting": 0.0},
          2: {"herring": 0.788, "Norway pout": 0.076, "sprat": 0.030, "blue whiting": 0.106},
          3: {"herring": 0.78, "Norway pout": 0.04, "sprat": 0.03, "blue whiting": 0.15},
          4: {"herring": 0.788, "Norway pout": 0.076, "sprat": 0.030, "blue whiting": 0.106}}

print("Dropped bad herring length rows (period, quarter, L_cm, J, A):", dropped)
print("\nq(L) curves used (absolute GOV efficiency; only the shape matters) at class midpoints:")
Ls = np.array([5.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 16.5, 18.5, 20.5, 22.5, 25.5, 28.5])
print("  L      " + " ".join(f"{x:6.1f}" for x in Ls))
for v in ("H1", "H2", "H3", "H4", "H5", "H6"):
    print(f"  {v:6s} " + " ".join(f"{x:6.3f}" for x in q_herring(v, Ls, 3)))
print(f"  S1     " + " ".join(f"{x:6.3f}" for x in q_sprat('S1', Ls)))
print(f"  NOP-W  " + " ".join(f"{x:6.3f}" for x in
                              interp({k + .5: v for k, v in W[('NOP', 'GOV', 'Efficiency')].items() if k >= 5}, Ls)))
print(f"  NOP-F  " + " ".join(f"{x:6.3f}" for x in q_fraser_pout(Ls)))

print("\nRelative q, small vs adult (q(L) / biomass-weighted q of the box adults, Q3 2001-25 herring A):")
k = ("2001-25", 3, "herring")
LA = np.array([L for L in SL[k] if SL[k][L][1] > 0]) + 0.5
wA = np.array([SL[k][L - 0.5][1] for L in LA])
for v in ("H1", "H2", "H3", "H4", "H5", "H6"):
    qa = np.sum(wA * q_herring(v, LA, 3)) / wA.sum()   # weight of observed (raw) adult catch
    r = q_herring(v, np.array([8.5, 9.5, 11.5, 12.5, 13.5]), 3) / qa
    print(f"  {v}: q_adult {qa:.3f}; q/q_adult at 8.5 {r[0]:.2f}, 9.5 {r[1]:.2f}, 11.5 {r[2]:.2f}, "
          f"12.5 {r[3]:.2f}, 13.5 {r[4]:.2f}")

print("\nBox shares by species, quarter and variant. J10/J13 = share of the species' J biomass at"
      " L+0.5 <= 10 / <= 13 cm;\nJ/B; B10/B13/B14 = share of ALL the species' biomass <= 10/13/14 cm"
      " (adult sprat 11-14 cm included).")
res = {}
for sp, variants, qf in (("herring", ("H0", "H1", "H2", "H3", "H4", "H5", "H6"), None),
                         ("sprat", ("S0", "S1"), None), ("Norway pout", ("P0", "P1"), None)):
    for q in (1, 2, 3, 4):
        key = (PERIOD[q], q, sp)
        if key not in SL:
            continue
        for v in variants:
            if sp == "herring":
                f = (lambda Lm, v=v, q=q: q_herring(v, Lm, q))
            elif sp == "sprat":
                f = (lambda Lm, v=v: q_sprat(v, Lm))
            else:
                f = (lambda Lm, v=v: q_pout(v, Lm))
            s = shares(key, f)
            if s is None:
                continue
            res[(sp, q, v)] = s
            print(f"  {sp:11s} Q{q} ({PERIOD[q]}) {v}: J10 {s['J10']:.3f}  J13 {s['J13']:.3f}  J/B {s['JB']:.2f}"
                  f"  | B10 {s['B10']:.3f}  B13 {s['B13']:.3f}  B14 {s['B14']:.3f}")

print("\nPool-weighted share of planktivore biomass <= 13 cm and <= 10 cm (pool weights per quarter;"
      " blue whiting counted as 0):")
combos = [("raw", "H0", "S0", "P0"), ("H1 (Walker 1-cm GAM)", "H1", "S1", "P1"),
          ("H2 (Walker 5-cm step)", "H2", "S1", "P1"), ("H3 (Walker Table 3)", "H3", "S1", "P1"),
          ("H4 (Table 3 + origin line)", "H4", "S1", "P1"), ("H5 (HAWG SAM Q3 q-at-age)", "H5", "S1", "P1"),
          ("H6 (Walker final, flat)", "H6", "S1", "P1")]
base = {}
for name, hv, sv, pv in combos:
    out = []
    for q in (1, 2, 3, 4):
        if ("herring", q, hv) not in res:
            out.append("   -    ")
            continue
        w = POOL_W[q]
        b13 = (w["herring"] * res[("herring", q, hv)]["B13"] + w["sprat"] * res[("sprat", q, sv)]["B13"]
               + w["Norway pout"] * res[("Norway pout", q, pv)]["B13"])
        b10 = (w["herring"] * res[("herring", q, hv)]["B10"] + w["sprat"] * res[("sprat", q, sv)]["B10"]
               + w["Norway pout"] * res[("Norway pout", q, pv)]["B10"])
        if name == "raw":
            base[q] = (b13, b10)
        out.append(f"Q{q} {b13:.3f}/{b10:.3f} (x{b13 / base[q][0]:.2f}/x{b10 / base[q][1]:.2f})")
    print(f"  {name:28s} " + "  ".join(out))

print("\nCross-check, WBSS herring (Skagerrak-Kattegat stock) IBTS+BITS Q1: q(age) = index / SAM N, "
      "geometric mean 2002-2022 [D from HAWG 2023 Tables 3.6.8.d, 3.6.13]")
I1 = {2002: (1012698, 57144, 11439), 2003: (679132, 131493, 3395), 2004: (289327, 72938, 12960),
      2005: (178201, 113465, 7044), 2006: (143497, 32836, 6406), 2007: (237369, 36605, 3262),
      2008: (162723, 32322, 3946), 2009: (533986, 38010, 1120), 2010: (301646, 78811, 9377),
      2011: (159743, 68881, 12912), 2012: (267873, 74418, 3626), 2013: (170436, 73289, 13112),
      2014: (150560, 21317, 2742), 2015: (260080, 61794, 2046), 2016: (190864, 92649, 5407),
      2017: (465661, 71591, 11029), 2018: (106439, 67129, 2736), 2019: (417826, 39068, 5625),
      2020: (326622, 79385, 5288), 2021: (325379, 127912, 6423), 2022: (182473, 75620, 7168)}
N = {2002: (1974032, 1022498, 948433), 2003: (1968178, 982551, 565762), 2004: (2177660, 995187, 578229),
     2005: (1497932, 1106839, 619893), 2006: (1300362, 754532, 656535), 2007: (984515, 669130, 427801),
     2008: (1072280, 502311, 383468), 2009: (861040, 558253, 280755), 2010: (800250, 434020, 298258),
     2011: (1070880, 424331, 248103), 2012: (983916, 612334, 281654), 2013: (873794, 540704, 408364),
     2014: (1334182, 473332, 352391), 2015: (909022, 774443, 317014), 2016: (732268, 495210, 510106),
     2017: (656006, 393307, 272324), 2018: (681086, 357534, 217378), 2019: (600180, 367595, 197654),
     2020: (619775, 347061, 250692), 2021: (454322, 362378, 233320), 2022: (330972, 272534, 265572)}
lr = np.log(np.array([[I1[y][a] / N[y][a] for a in range(3)] for y in sorted(I1)]))
g = np.exp(lr.mean(0))
lo, hi = np.exp(np.percentile(lr, 10, axis=0)), np.exp(np.percentile(lr, 90, axis=0))
for a in range(3):
    print(f"  age {a + 1} wr: q {g[a]:.4f} (p10-p90 {lo[a]:.4f}-{hi[a]:.4f}); relative to age 2: {g[a] / g[1]:.2f}")
print("NSAS SAM IBTS-Q3 q-at-age relative to age 2: " +
      ", ".join(f"age {a} {SAM_Q3[a] / SAM_Q3[2]:.2f}" for a in SAM_Q3))
print("Walker Table 3 herring multiplier relative to age 2 (0.30): " +
      ", ".join(f"age {a} {m / 0.30:.2f}" for a, (L, m) in enumerate(T3_HER)))
print("Sprat SMS IBTS Q3 q relative to age 2: age 1 %.2f, age 3 %.2f" % (0.941 / 1.202, 1.205 / 1.202))

print("\nScale needed to move a whiting planktivore diet share from 0.07 (Skagerrak April 1991) to 0.40"
      " (North Sea),\nother prey unchanged: odds ratio (0.40/0.60)/(0.07/0.93) = "
      f"{(0.40 / 0.60) / (0.07 / 0.93):.1f}x more accessible planktivore biomass.")
