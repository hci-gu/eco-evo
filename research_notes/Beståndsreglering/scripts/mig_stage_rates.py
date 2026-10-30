"""Seasonal presence of each species in the Skagerrak / box relative to the stock area, and the
cohort net-emigration rates of mig_cohort_rates.py collapsed onto the engine's two stages.

Usage: python3 -I mig_stage_rates.py <cpue_age_all.csv.gz> <CA.csv.gz> <cohort_csv> <out_dir>

<CA.csv.gz> = ../../Torskfiskarnas artsammansattning/data/extract/CA.csv.gz (NS-IBTS 2021-25 Q1/Q3
age-length-weight records): mean individual weight at age and quarter (survey-wide, weighted by
CANoAtLngt; length-stratified sampling makes it approximate). Q2 weight = mean of Q1 and Q3 at the
same age; Q4 = mean of Q3 at age a and Q1 at age a+1.

1. Quarterly presence 1991-97 (the only years with Q2 May and Q4 Nov surveys): biomass index of
   the unit (numbers at age x weight at age) divided by the stock index, pooled over years, then
   normalised to the four-quarter mean = 1. Ages 1+ (age 0 enters the GOV only from Q3) and adults
   (age 2+) separately.
2. Q1/Q3 biomass ratio of unit/stock, 1991-2025 (ages 1+), the long-run version of the winter dip.
3. Stage rates. Engine juvenile J = age 0 from settlement (Q3) to the Q3 survey at age 1; adult A
   = older. e_J = annual3(0) (Q3 age 0 -> Q3 age 1), per year (e_J_from_seasons = 0.5 winter(0->1)
   + 0.5 summer(1), a check). e_A_Q3Q3 = mean over ages a >= 1 of annual3(a), weighted by the
   unit's mean Q3 biomass at age a+1. e_A_annual = the same weighting on the Q1->Q1 rates.
"""
import os
import sys

import numpy as np
import pandas as pd

src, ca_path, coh_path, out = sys.argv[1:5]
SP = {126417: "herring", 126425: "sprat", 126436: "cod", 126437: "haddock", 126438: "whiting",
      126441: "saithe", 126444: "norway_pout", 127023: "mackerel", 127143: "plaice"}
MAXAGE = {"herring": 5, "sprat": 6, "cod": 6, "haddock": 6, "whiting": 6, "saithe": 10,
          "norway_pout": 4, "mackerel": 6, "plaice": 10}
AG = list(range(11))
d = pd.read_csv(src)
d = d[d.Gear.astype(str).str.strip() == "GOV"].copy()
d["sp"] = d.AphiaID.map(SP)
d["rect"] = d.SubArea.astype(str).str.strip()
d["haul"] = list(zip(d.Year, d.Quarter, d.Ship.astype(str).str.strip(), d.HaulNo, d.ShootLat.round(4),
                     d.ShootLon.round(4)))
d = d.drop_duplicates(["haul", "sp"])
box = (d.ShootLat >= 57.81) & (d.ShootLat <= 58.31) & (d.ShootLon >= 10.07) & (d.ShootLon <= 11.15)
UNITS = {"BOX": box, "SKA50": (d.Area == 8) & (d.Depth >= 50), "STOCK": d.Area.between(1, 9)}

ca = pd.read_csv(ca_path)
ca = ca[ca.Valid_Aphia.isin(SP) & (ca.Age >= 0) & (ca.IndWgt > 0)]
ca["sp"] = ca.Valid_Aphia.map(SP)
ca["wn"] = ca.IndWgt * ca.CANoAtLngt
W = (ca.groupby(["sp", "Quarter", "Age"]).wn.sum() / ca.groupby(["sp", "Quarter", "Age"]).CANoAtLngt.sum())


def w(sp, q, a):
    a = min(a, MAXAGE[sp])
    def g(qq, aa):
        aa = min(aa, MAXAGE[sp])
        try:
            return float(W.loc[(sp, qq, aa)])
        except KeyError:
            return np.nan
    if q in (1, 3):
        return g(q, a)
    if q == 2:
        return np.nanmean([g(1, a), g(3, a)])
    return np.nanmean([g(3, a), g(1, a + 1)])


def idx(mask, rect_mean):
    x = d[mask]
    cols = [f"Age_{a}" for a in AG]
    if rect_mean:
        return x.groupby(["sp", "Year", "Quarter", "rect"])[cols].mean().groupby(level=[0, 1, 2]).mean()
    return x.groupby(["sp", "Year", "Quarter"])[cols].mean()


IDX = {u: idx(m, u != "BOX") for u, m in UNITS.items()}


def bio(u, sp, y, q, ages):
    try:
        r = IDX[u].loc[(sp, y, q)]
    except KeyError:
        return np.nan
    v = [r[f"Age_{a}"] * w(sp, q, a) for a in ages if np.isfinite(r[f"Age_{a}"])]
    return np.nansum(v) if v else np.nan


rows = []
for sp in SP.values():
    for u in ("BOX", "SKA50"):
        for lab, ages in (("1+", range(1, MAXAGE[sp] + 1)), ("2+", range(2, MAXAGE[sp] + 1))):
            num = {q: 0.0 for q in (1, 2, 3, 4)}
            den = {q: 0.0 for q in (1, 2, 3, 4)}
            yrs = []
            for y in range(1991, 1997):
                b = {q: (bio(u, sp, y, q, ages), bio("STOCK", sp, y, q, ages)) for q in (1, 2, 3, 4)}
                if all(np.isfinite(b[q][0]) and np.isfinite(b[q][1]) and b[q][1] > 0 for q in b):
                    yrs.append(y)
                    for q in b:
                        num[q] += b[q][0]
                        den[q] += b[q][1]
            if len(yrs) < 3:
                continue
            r = np.array([num[q] / den[q] for q in (1, 2, 3, 4)])
            rel = r / r.mean()
            rows.append(dict(species=sp, unit=u, ages=lab, years=len(yrs), Q1_Feb=rel[0], Q2_May=rel[1],
                             Q3_Aug=rel[2], Q4_Nov=rel[3], ratio_mean=r.mean()))
P = pd.DataFrame(rows)
P.to_csv(os.path.join(out, "mig_quarterly_presence_1991_96.csv"), index=False, float_format="%.3g")

rows = []
for sp in SP.values():
    for u in ("BOX", "SKA50"):
        for lab, ages in (("1+", range(1, MAXAGE[sp] + 1)), ("2+", range(2, MAXAGE[sp] + 1))):
            n1 = d1 = n3 = d3 = 0.0
            per = []
            for y in range(1991, 2026):
                a1, s1 = bio(u, sp, y, 1, ages), bio("STOCK", sp, y, 1, ages)
                a3, s3 = bio(u, sp, y, 3, ages), bio("STOCK", sp, y, 3, ages)
                if all(np.isfinite([a1, s1, a3, s3])) and s1 > 0 and s3 > 0:
                    n1 += a1; d1 += s1; n3 += a3; d3 += s3
                    if a3 > 0 and a1 > 0:
                        per.append((a1 / s1) / (a3 / s3))
            rows.append(dict(species=sp, unit=u, ages=lab, rel_Q1_over_Q3=(n1 / d1) / (n3 / d3),
                             median_year=np.median(per) if per else np.nan, years=len(per),
                             unit_Q1_over_Q3_raw=n1 / n3 if n3 > 0 else np.nan))
Q = pd.DataFrame(rows)
Q.to_csv(os.path.join(out, "mig_q1_over_q3_biomass.csv"), index=False, float_format="%.3g")

C = pd.read_csv(coh_path)


def rate(sp, u, kind, a0):
    x = C[(C.species == sp) & (C.unit == u) & (C.transition == kind) & (C.age_from == a0)]
    return float(x.E_rel.iloc[0]) if len(x) else np.nan


rows = []
for sp in SP.values():
    for u in ("BOX", "SKA50"):
        eJ = rate(sp, u, "annual3", 0)
        eJs = 0.5 * rate(sp, u, "winter", 0) + 0.5 * rate(sp, u, "summer", 1)
        num = den = numa = dena = 0.0
        parts = []
        for a in range(1, MAXAGE[sp] - 1):
            wt = np.nanmean([bio(u, sp, y, 3, [a + 1]) for y in range(1991, 2026)])
            ea = rate(sp, u, "annual3", a)
            ann = rate(sp, u, "annual", a)
            if np.isfinite(ea) and np.isfinite(wt):
                num += wt * ea; den += wt
            if np.isfinite(ann) and np.isfinite(wt):
                numa += wt * ann; dena += wt
            parts.append(f"{a}->{a + 1}:{ea:.2f}|{ann:.2f}|w{wt:.0f}")
        rows.append(dict(species=sp, unit=u, e_J=eJ, e_J_from_seasons=eJs, e_A_Q3Q3=num / den if den else np.nan,
                         e_A_annual=numa / dena if dena else np.nan, detail=" ".join(parts)))
ST = pd.DataFrame(rows)
ST.to_csv(os.path.join(out, "mig_stage_rates.csv"), index=False, float_format="%.3g")

pd.set_option("display.width", 250, "display.max_colwidth", 200)
for name, T in [("quarterly presence 1991-96", P), ("Q1/Q3", Q), ("stage rates", ST)]:
    print(f"\n==== {name}")
    print(T.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
