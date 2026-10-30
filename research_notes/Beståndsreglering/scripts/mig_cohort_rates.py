"""Net emigration of the Mareld box and the Skagerrak relative to the stock area, by age and season,
from NS-IBTS CPUE-at-age per haul (DATRAS getCPUEAge, 1991-2025, Q1 and Q3).

Usage: python3 -I mig_cohort_rates.py <cpue_age_all.csv.gz> <out_dir>

Units (index = numbers per hour of GOV trawling):
  BOX   hauls inside the grid 57.81-58.31 N, 10.07-11.15 E (haul mean; ~5 Q1 and ~4 Q3 hauls a year,
        all in roundfish area 8, 43-253 m)
  SKA   roundfish area 8 (Skagerrak, 57.15-58.86 N, 8.0-11.65 E): mean of rectangle means
  SKA50 SKA hauls at 50 m or deeper (the box's depth range; Q3 has twice the Q1 share of
        hauls shallower than 50 m, which this removes from the seasonal comparison)
  KAT   roundfish area 9 (Kattegat)
  NSEA  roundfish areas 1-7 (North Sea proper)
  STOCK roundfish areas 1-9 (North Sea + Skagerrak + Kattegat): mean of rectangle means
        (each rectangle one weight = an area-weighted density)

Cohort method. With the same gear in both areas, catchability at age and the common mortality
(stock M and F) cancel in the difference of the cohort log-ratios:
  E_rel = -[ln(D_u(dest)/D_u(orig)) - ln(D_STOCK(dest)/D_STOCK(orig))] / dt
is the net emigration rate (emigration minus immigration, per year) of the unit's cohort relative
to the stock's cohort, i.e. the extra loss rate of the unit beyond stock-wide Z. Positive = the
unit loses fish to the rest of the stock area; negative = net immigration. Pooled over cohorts by
summing densities over the cohorts sampled in both surveys (weights = cohort size); 90 % interval
by bootstrap over cohorts; the per-cohort median alongside. Transitions:
  annual  Q1 age a, year y  -> Q1 age a+1, year y+1   dt 1.0
  annual3 Q3 age a, year y  -> Q3 age a+1, year y+1   dt 1.0 (a = 0: the engine's juvenile year)
  summer  Q1 age a, year y  -> Q3 age a,   year y     dt 0.5 (Feb -> Aug)
  winter  Q3 age a, year y  -> Q1 age a+1, year y+1   dt 0.5 (Aug -> Feb; a = 0 included)

Density dependence:
  share elasticity b: OLS slope of ln D_u on ln D_STOCK over years (same age and quarter);
    b = 1 proportional, b > 1 the unit fills disproportionately when the year class is strong
    (spillover / basin), b < 1 the unit is saturated (preferred habitat filled first).
  emigration vs year-class strength: Spearman rho of per-cohort E_rel(annual, age a) with
    ln D_STOCK at the origin (independent of the unit's sampling noise), and with the unit's own
    relative share at an EARLIER survey of the same cohort (Q3 age a-1 or Q1 age a-1), which avoids
    the regression-to-the-mean artefact of regressing E on the share it is computed from.
"""
import os
import sys

import numpy as np
import pandas as pd

src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1)

SP = {126417: "herring", 126425: "sprat", 126436: "cod", 126437: "haddock", 126438: "whiting",
      126441: "saithe", 126444: "norway_pout", 127023: "mackerel", 127143: "plaice"}
MAXAGE = {"herring": 4, "sprat": 5, "cod": 5, "haddock": 5, "whiting": 5, "saithe": 9,
          "norway_pout": 3, "mackerel": 5, "plaice": 9}   # oldest age used (plus groups excluded)
AGES = [f"Age_{a}" for a in range(11)]

d = pd.read_csv(src)
d = d[d.Gear.astype(str).str.strip() == "GOV"].copy()
d["sp"] = d.AphiaID.map(SP)
d["rect"] = d.SubArea.astype(str).str.strip()
d[AGES] = d[AGES].astype(float)
d["haul"] = list(zip(d.Year, d.Quarter, d.Ship.astype(str).str.strip(), d.HaulNo, d.ShootLat.round(4),
                     d.ShootLon.round(4)))
d = d.drop_duplicates(["haul", "sp"])
box = (d.ShootLat >= 57.81) & (d.ShootLat <= 58.31) & (d.ShootLon >= 10.07) & (d.ShootLon <= 11.15)
UNITS = {"BOX": box, "SKA": d.Area == 8, "SKA50": (d.Area == 8) & (d.Depth >= 50), "KAT": d.Area == 9, "NSEA": d.Area.between(1, 7),
         "STOCK": d.Area.between(1, 9)}


def index(mask, rect_mean):
    x = d[mask]
    if rect_mean:
        r = x.groupby(["sp", "Year", "Quarter", "rect"])[AGES].mean()
        g = r.groupby(level=[0, 1, 2]).mean()
    else:
        g = x.groupby(["sp", "Year", "Quarter"])[AGES].mean()
    n = x.groupby(["sp", "Year", "Quarter"]).haul.nunique()
    return g, n


IDX, NH = {}, {}
for u, m in UNITS.items():
    IDX[u], NH[u] = index(m, rect_mean=(u != "BOX"))
print("hauls per survey (median over years):",
      {u: {q: int(NH[u].xs("cod", level=0).xs(q, level=1).median()) for q in (1, 3)} for u in UNITS})


def D(u, sp, y, q, a):
    try:
        return float(IDX[u].loc[(sp, y, q), f"Age_{a}"])
    except KeyError:
        return np.nan


YEARS = range(1991, 2026)


def spearman_p(x, y, n_perm=4000):
    """Two-sided permutation p of Spearman rho."""
    rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
    r0 = np.corrcoef(rx, ry)[0, 1]
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    return r0, (np.abs(perm) >= abs(r0)).mean()

# ---------------------------------------------------------------- 1. density ratio by age
rows = []
for sp in SP.values():
    for q in (1, 3):
        for a in range(0, MAXAGE[sp] + 1):
            if q == 1 and a == 0:
                continue
            for u in ("BOX", "SKA", "SKA50", "KAT"):
                num = np.array([D(u, sp, y, q, a) for y in YEARS])
                den = np.array([D("STOCK", sp, y, q, a) for y in YEARS])
                ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
                if ok.sum() < 5:
                    continue
                pooled = num[ok].sum() / den[ok].sum()
                rows.append(dict(species=sp, quarter=q, age=a, unit=u, years=int(ok.sum()),
                                 ratio_pooled=pooled, ratio_median=np.median(num[ok] / den[ok]),
                                 zero_years=int((num[ok] == 0).sum()), stock_mean=den[ok].mean()))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(out, "mig_density_ratio_by_age.csv"), index=False, float_format="%.4g")


# ---------------------------------------------------------------- 2. cohort net emigration
def transitions(sp):
    t = []
    for a in range(1, MAXAGE[sp]):
        t.append(("annual", a, 1, 0, a + 1, 1, 1, 1.0))
    for a in range(0, MAXAGE[sp]):
        t.append(("annual3", a, 3, 0, a + 1, 3, 1, 1.0))
    for a in range(1, MAXAGE[sp] + 1):
        t.append(("summer", a, 1, 0, a, 3, 0, 0.5))
    for a in range(0, MAXAGE[sp]):
        t.append(("winter", a, 3, 0, a + 1, 1, 1, 0.5))
    return t


def cohort_vals(u, sp, a0, q0, a1, q1, dy):
    o, e, so, se = [], [], [], []
    for y in YEARS:
        v = (D(u, sp, y, q0, a0), D(u, sp, y + dy, q1, a1), D("STOCK", sp, y, q0, a0),
             D("STOCK", sp, y + dy, q1, a1))
        if all(np.isfinite(v)) and v[2] > 0 and v[3] > 0:
            o.append(v[0]); e.append(v[1]); so.append(v[2]); se.append(v[3])
    return [np.array(z) for z in (o, e, so, se)]


def pooled_rate(o, e, so, se, dt):
    if o.sum() <= 0 or e.sum() <= 0:
        return np.nan, np.nan, np.nan
    zu = np.log(o.sum() / e.sum()) / dt
    zs = np.log(so.sum() / se.sum()) / dt
    return zu - zs, zu, zs


rows = []
for sp in SP.values():
    for kind, a0, q0, _, a1, q1, dy, dt in transitions(sp):
        for u in ("BOX", "SKA", "SKA50", "KAT"):
            o, e, so, se = cohort_vals(u, sp, a0, q0, a1, q1, dy)
            if len(o) < 8:
                continue
            er, zu, zs = pooled_rate(o, e, so, se, dt)
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(o), len(o))
                bs.append(pooled_rate(o[i], e[i], so[i], se[i], dt)[0])
            bs = np.array(bs)
            bs = bs[np.isfinite(bs)]
            pc = (o > 0) & (e > 0)
            per = -(np.log(e[pc] / o[pc]) - np.log(se[pc] / so[pc])) / dt
            rows.append(dict(species=sp, transition=kind, age_from=a0, q_from=q0, age_to=a1, q_to=q1,
                             unit=u, cohorts=len(o), E_rel=er, E_lo=np.percentile(bs, 5) if len(bs) else np.nan,
                             E_hi=np.percentile(bs, 95) if len(bs) else np.nan,
                             E_median_cohort=np.median(per) if len(per) else np.nan,
                             cohorts_nonzero=int(pc.sum()), Z_unit=zu, Z_stock=zs))
C = pd.DataFrame(rows)
C.to_csv(os.path.join(out, "mig_cohort_net_emigration.csv"), index=False, float_format="%.4g")

# ---------------------------------------------------------------- 3. density dependence
rows = []
for sp in SP.values():
    for q in (1, 3):
        for a in range(0, MAXAGE[sp] + 1):
            if q == 1 and a == 0:
                continue
            for u in ("BOX", "SKA", "SKA50", "KAT"):
                x = np.array([D("STOCK", sp, y, q, a) for y in YEARS])
                yv = np.array([D(u, sp, y, q, a) for y in YEARS])
                ok = np.isfinite(x) & np.isfinite(yv) & (x > 0)
                pos = ok & (yv > 0)
                if pos.sum() < 8:
                    continue
                lx, ly = np.log(x[pos]), np.log(yv[pos])
                b, c = np.polyfit(lx, ly, 1)
                bb = []
                for _ in range(2000):
                    i = rng.integers(0, pos.sum(), pos.sum())
                    if np.ptp(lx[i]) > 0:
                        bb.append(np.polyfit(lx[i], ly[i], 1)[0])
                # Spearman rank correlation of occurrence-inclusive share with stock abundance
                sh = yv[ok] / x[ok]
                rho = pd.Series(sh).rank().corr(pd.Series(x[ok]).rank())
                rows.append(dict(species=sp, quarter=q, age=a, unit=u, years=int(ok.sum()),
                                 years_pos=int(pos.sum()), slope_b=b, b_lo=np.percentile(bb, 5),
                                 b_hi=np.percentile(bb, 95), r2=np.corrcoef(lx, ly)[0, 1] ** 2,
                                 rho_share_vs_stock=rho))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(out, "mig_share_elasticity.csv"), index=False, float_format="%.4g")

# per-cohort annual emigration vs year-class strength
rows = []
for sp in SP.values():
    for a in range(1, MAXAGE[sp]):
        for u in ("BOX", "SKA", "SKA50"):
            recs = []
            for y in YEARS:
                o, e = D(u, sp, y, 1, a), D(u, sp, y + 1, 1, a + 1)
                so, se = D("STOCK", sp, y, 1, a), D("STOCK", sp, y + 1, 1, a + 1)
                # earlier independent survey of the same cohort: Q3 a-1 of year y-1
                eo, es = D(u, sp, y - 1, 3, a - 1), D("STOCK", sp, y - 1, 3, a - 1)
                if not all(np.isfinite([o, e, so, se])) or min(o, e, so, se) <= 0:
                    continue
                er = -(np.log(e / o) - np.log(se / so))
                prior = np.log(eo / es) if np.isfinite(eo) and np.isfinite(es) and eo > 0 and es > 0 else np.nan
                recs.append((er, np.log(so), prior))
            if len(recs) < 8:
                continue
            r = pd.DataFrame(recs, columns=["E", "lnStock", "lnPriorShare"])
            rho1, p1 = spearman_p(r.E.values, r.lnStock.values)
            rp = r.dropna()
            rho2, p2 = spearman_p(rp.E.values, rp.lnPriorShare.values) if len(rp) >= 8 else (np.nan, np.nan)
            b1 = np.polyfit(r.lnStock, r.E, 1)[0]
            b2 = np.polyfit(rp.lnPriorShare, rp.E, 1)[0] if len(rp) >= 8 else np.nan
            bb = []
            for _ in range(2000):
                if len(rp) < 8:
                    break
                i = rng.integers(0, len(rp), len(rp))
                if np.ptp(rp.lnPriorShare.values[i]) > 0:
                    bb.append(np.polyfit(rp.lnPriorShare.values[i], rp.E.values[i], 1)[0])
            b2lo, b2hi = (np.percentile(bb, 5), np.percentile(bb, 95)) if bb else (np.nan, np.nan)
            rows.append(dict(species=sp, age_from=a, unit=u, cohorts=len(r), E_median=r.E.median(),
                             E_sd=r.E.std(), rho_E_vs_lnStock=rho1, p_lnStock=p1, slope_E_vs_lnStock=b1,
                             cohorts_prior=len(rp), rho_E_vs_lnPriorShare=rho2, p_lnPriorShare=p2,
                             slope_E_vs_lnPriorShare=b2, slope_lo=b2lo, slope_hi=b2hi))
DD = pd.DataFrame(rows)
DD.to_csv(os.path.join(out, "mig_emigration_vs_density.csv"), index=False, float_format="%.4g")

pd.set_option("display.width", 220, "display.max_rows", 500)
for name, T in [("density ratio", R), ("cohort", C), ("elasticity", S), ("E vs density", DD)]:
    print(f"\n==== {name}")
    print(T.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
