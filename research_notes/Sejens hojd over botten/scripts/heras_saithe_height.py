"""Adult saithe (>= 35 cm) in the HERAS / IESSNS pelagic hauls of the whole stored region (54.6-59.1 N,
5.0-12.4 E; June-July): catch by fished window height above the bed, and the scale height Hs of an
exponential profile f(h) = exp(-h/Hs)/Hs fitted to the hauls by two likelihoods.

Usage: python3 -I heras_saithe_height.py <hauls.tsv> <catch.tsv> <individuals.tsv> <haul_meta.tsv>
  research_notes/"Ekolodsdata planktivorer"/data/derived/biotic_region_{hauls,catch,individuals}.tsv
  (ICES acoustic-trawl database, CC BY 4.0) and research_notes/"Krill i Skagerrak"/wc_depth/tables/
  heras_haul_meta.tsv (net opening, duration). Same window convention as heras_saithe_depth.py:
  [headline, headline + net opening], height of the footrope above the bed = bed - (headline + opening).

Model (per haul i): expected adult-saithe kg/h = c_g x O_i(Hs), O_i = integral of f over the window's
heights above the bed (the share of the column's saithe inside the window), c_g one level per gear
(EXP / FOT / PEL mouths differ). Fits:
  qp  quasi-Poisson ML on kg/h: sum_i y_i log(c E_i) - c E_i (dominated by the large hauls);
  pa  presence / absence: P(saithe in haul) = 1 - exp(-k_g O_i) (robust to the large hauls).
Hauls: day (sun > 0 deg) and night separately; bed >= 100 m (where adults live, Sejens djuplage);
only hauls whose window lies in the bottom 150 m or at the surface over any bed are informative.
The hauls were aimed at herring marks: a profile within the hauls, not a survey design [J].
"""
import math
import sys

import numpy as np
import pandas as pd

H = pd.read_csv(sys.argv[1], sep="\t", dtype=str)
C = pd.read_csv(sys.argv[2], sep="\t", dtype=str)
I = pd.read_csv(sys.argv[3], sep="\t", dtype=str)
M = pd.read_csv(sys.argv[4], sep="\t", dtype=str)
SAITHE = "126441"
H = H.merge(M[["cruise", "haul", "duration_min", "netopening_m"]].drop_duplicates(["cruise", "haul"]),
            on=["cruise", "haul"], how="left")
for c in ("lat", "lon", "min_depth", "bottom", "duration_min", "netopening_m"):
    H[c] = pd.to_numeric(H[c], errors="coerce")
C["weight_kg"] = pd.to_numeric(C["weight_kg"], errors="coerce")
I["length_mm"] = pd.to_numeric(I["length_mm"], errors="coerce")
I["weight_kg"] = pd.to_numeric(I["weight_kg"], errors="coerce")


def sun_elev(ts, lat, lon):
    t = pd.Timestamp(ts)
    hr = t.hour + t.minute / 60 + lon / 15.0
    dec = math.radians(23.44) * math.sin(2 * math.pi * (284 + t.dayofyear) / 365)
    p = math.radians(lat)
    return math.degrees(math.asin(math.sin(p) * math.sin(dec)
                                  + math.cos(p) * math.cos(dec) * math.cos(math.radians(15 * (hr - 12)))))


H = H[H.time.notna() & H.bottom.notna() & H.min_depth.notna()].copy()
H["elev"] = [sun_elev(t, la, lo) for t, la, lo in zip(H.time, H.lat, H.lon)]
H["open"] = H.netopening_m.fillna(H.groupby("gear").netopening_m.transform("median")).fillna(20.0)
H["hab_top"] = (H.bottom - H.min_depth).clip(lower=0)
H["hab_foot"] = (H.bottom - H.min_depth - H.open).clip(lower=0)
H["hours"] = H.duration_min.fillna(H.duration_min.median()) / 60.0

# adult share of the saithe catch weight per haul (individual weights; L^3 where weight missing)
s = I[I.species == SAITHE].copy()
s["w"] = s.weight_kg.where(s.weight_kg > 0, 7.5e-9 * s.length_mm ** 3.06)
ad = s.groupby(["cruise", "haul"]).apply(lambda d: pd.Series(dict(
    adult_share=float(d.w[d.length_mm >= 350].sum() / d.w.sum()) if d.w.sum() > 0 else np.nan,
    n_meas=len(d), med_len=float(d.length_mm.median()))), include_groups=False).reset_index()
cs = C[C.species == SAITHE].groupby(["cruise", "haul"]).weight_kg.sum().rename("saithe_kg").reset_index()
H = H.merge(cs, on=["cruise", "haul"], how="left").merge(ad, on=["cruise", "haul"], how="left")
H["saithe_kg"] = H.saithe_kg.fillna(0.0)
H["adult_kg"] = H.saithe_kg * H.adult_share.fillna(0.0)
H.loc[(H.saithe_kg > 0) & H.adult_share.isna(), "adult_kg"] = np.nan     # unmeasured catch: drop
H["y"] = H.adult_kg / H.hours
H["day"] = H.elev > 0
# Adult-saithe density by bed class relative to the box mean (GOV 35-70 cm, 0.1 Q1 + 0.9 Q3; research_notes/
# "Krill i Skagerrak"/wc_depth/evidence.md 1.2: shares 0.024 / 0.268 / 0.646 / 0.062 over cells 0.21 / 0.31 /
# 0.33 / 0.15). Used as a known offset on the expected rate in the "offset" fits, so on-bed hauls over shallower
# beds are not read as saithe standing off the bed.
def bed_offset(b):
    return np.where(b < 100, 0.024 / 0.21, np.where(b < 150, 0.268 / 0.31, np.where(b < 250, 0.646 / 0.33, 0.062 / 0.15)))


H["dens"] = bed_offset(H.bottom.values)
OFFSET = False

print("=" * 110)
print(f"region hauls {len(H)}; with saithe {int((H.saithe_kg > 0).sum())}; adult kg total {H.adult_kg.sum():.0f}")
print(f"gears {H.gear.value_counts().to_dict()}; median opening {H.groupby('gear').open.median().to_dict()}")


def overlap(Hs, h0, h1):
    """share of an exp(-h/Hs) profile (truncated at the surface: column height D) between heights h0, h1."""
    return np.exp(-h0 / Hs) - np.exp(-h1 / Hs)


def fit(d, label):
    d = d[d.y.notna()].copy()
    gears = sorted(d.gear.unique())
    grid = np.concatenate([np.arange(1, 10, 0.5), np.arange(10, 41, 1), np.arange(45, 201, 5)])
    ll_qp, ll_pa = [], []
    for Hs in grid:
        trunc = 1 - np.exp(-d.bottom.values / Hs)
        O = overlap(Hs, d.hab_foot.values, d.hab_top.values) / trunc
        O = np.maximum(O, 1e-300)
        lq = lp = 0.0
        for g in gears:
            m = (d.gear == g).values
            y, o = d.y.values[m], O[m]
            c = y.sum() / o.sum() if o.sum() > 0 else 0.0
            if c > 0:
                lq += float((y * np.log(c * o)).sum() - (c * o).sum())
            pres = (y > 0).astype(float)
            ks = np.exp(np.linspace(-6, 14, 400))
            best = -1e300
            for k in ks:
                p = np.clip(1 - np.exp(-k * o), 1e-12, 1 - 1e-12)
                v = float((pres * np.log(p) + (1 - pres) * np.log(1 - p)).sum())
                best = max(best, v)
            lp += best
        ll_qp.append(lq)
        ll_pa.append(lp)
    ll_qp, ll_pa = np.array(ll_qp), np.array(ll_pa)
    out = {}
    for nm, ll in (("qp", ll_qp), ("pa", ll_pa)):
        i = int(np.argmax(ll))
        # 1.92 log-likelihood units ~ 95 % profile interval (quasi-Poisson: scaled by the dispersion)
        if nm == "qp":
            d2 = d.copy()
            Hs = grid[i]
            trunc = 1 - np.exp(-d2.bottom.values / Hs)
            O = overlap(Hs, d2.hab_foot.values, d2.hab_top.values) / trunc
            mu = np.zeros(len(d2))
            for g in gears:
                m = (d2.gear == g).values
                c = d2.y.values[m].sum() / O[m].sum() if O[m].sum() > 0 else 0
                mu[m] = c * O[m]
            phi = float(((d2.y.values - mu) ** 2 / np.maximum(mu, 1e-9)).sum() / max(len(d2) - len(gears) - 1, 1))
            cut = 1.92 * phi
        else:
            phi, cut = 1.0, 1.92
        ok = grid[ll >= ll[i] - cut]
        out[nm] = (grid[i], ok.min(), ok.max(), phi)
        print(f"  {label:44s} {nm}: Hs {grid[i]:6.1f} m  (95 % profile {ok.min():.1f}-{ok.max():.1f}; dispersion {phi:.0f})"
              f"   n {len(d)}, with adults {int((d.y > 0).sum())}")
    return out


def table(d, label):
    print(f"\n  {label}: hauls by footrope height above the bed (adult saithe, kg/h; per m of opening)")
    bins = [(0, 3), (3, 10), (10, 30), (30, 60), (60, 150), (150, 1e5)]
    for lo, hi in bins:
        m = (d.hab_foot >= lo) & (d.hab_foot < hi)
        x = d[m & d.y.notna()]
        if len(x) == 0:
            continue
        print(f"    footrope {lo:5.0f}-{hi:<6.0f} m: hauls {len(x):3d}, with adults {int((x.y > 0).sum()):3d}, "
              f"mean {x.y.mean():8.2f} kg/h, median {x.y.median():6.2f}, mean per m {float((x.y / x.open).mean()):7.3f};"
              f" bed median {x.bottom.median():.0f} m, opening median {x.open.median():.0f} m")


RES = {}
for light, dm in (("day", H.day), ("night", ~H.day)):
    for bed_lo in (100.0, 140.0):
        d = H[dm & (H.bottom >= bed_lo)]
        lab = f"{light}, bed >= {bed_lo:.0f} m"
        table(d, lab)
        RES[lab] = fit(d, lab)
        if light == "day":
            fit(d[d.gear == "EXP"], lab + ", EXP only")
            fit(d[d.hab_foot < 150], lab + ", window in bottom 150 m")
            dd = d.copy()
            dd.loc[dd.y > 1000, "y"] = dd.y[dd.y < 1000].max()
            fit(dd, lab + ", largest haul capped")


# ------------------------------------------------------------------ other profile shapes (day, bed >= 100 / 140 m)
def cdf_shape(shape, p, h):
    h = np.asarray(h, float)
    if shape == "exp":
        return 1 - np.exp(-h / p)
    if shape == "uniform":
        return np.clip(h / p, 0, 1)
    if shape == "gamma2":           # f = h exp(-h/s) / s^2: mode at s above the bed, little in the bottom metres
        return 1 - (1 + h / p) * np.exp(-h / p)
    raise ValueError(shape)


def fit_shape(d, shape, grid, offset=False):
    d = d[d.y.notna()]
    gears = sorted(d.gear.unique())
    best = None
    for p in grid:
        O = (cdf_shape(shape, p, d.hab_top.values) - cdf_shape(shape, p, d.hab_foot.values)) / cdf_shape(shape, p, d.bottom.values)
        if offset:
            O = O * d.dens.values
        O = np.maximum(O, 1e-300)
        lq = lp = 0.0
        for g in gears:
            m = (d.gear == g).values
            y, o = d.y.values[m], O[m]
            c = y.sum() / o.sum()
            if c > 0:
                lq += float((y * np.log(c * o)).sum() - (c * o).sum())
            pres = (y > 0).astype(float)
            v = -1e300
            for k in np.exp(np.linspace(-6, 14, 300)):
                pp = np.clip(1 - np.exp(-k * o), 1e-12, 1 - 1e-12)
                v = max(v, float((pres * np.log(pp) + (1 - pres) * np.log(1 - pp)).sum()))
            lp += v
        if best is None or lq > best[1]:
            best = (p, lq, best[2] if best else None, best[3] if best else None)
        if best[2] is None or lp > best[3]:
            best = (best[0], best[1], p, lp)
    return best


print("\nDay profile shapes (best parameter by quasi-Poisson / presence; log-likelihoods comparable within a fit type);"
      " share of the profile in the bottom 2 m at the qp parameter")
def dispersion(d, shape, p, offset):
    d = d[d.y.notna()]
    O = (cdf_shape(shape, p, d.hab_top.values) - cdf_shape(shape, p, d.hab_foot.values)) / cdf_shape(shape, p, d.bottom.values)
    if offset:
        O = O * d.dens.values
    mu = np.zeros(len(d))
    for g in sorted(d.gear.unique()):
        m = (d.gear == g).values
        mu[m] = d.y.values[m].sum() / max(O[m].sum(), 1e-300) * O[m]
    return float(((d.y.values - mu) ** 2 / np.maximum(mu, 1e-9)).sum() / max(len(d) - d.gear.nunique() - 1, 1))


for offset in (False, True):
    print(f"  --- bed-class density offset: {'ON' if offset else 'off'}")
    for bed_lo in (60.0, 100.0, 140.0):
        d = H[H.day & (H.bottom >= bed_lo)]
        fits = {}
        for shape, grid in (("exp", np.arange(2, 81, 1.0)), ("uniform", np.arange(5, 121, 1.0)),
                            ("gamma2", np.arange(1, 41, 0.5))):
            fits[shape] = fit_shape(d, shape, grid, offset)
        phi = dispersion(d, "exp", fits["exp"][0], offset)
        for shape, (pq, lq, pp, lp) in fits.items():
            b2 = float(cdf_shape(shape, pq, 2.0))
            b2p = float(cdf_shape(shape, pp, 2.0))
            print(f"  bed >= {bed_lo:.0f} m  {shape:8s} qp: par {pq:5.1f} m, 2 dll/phi vs exp {2 * (lq - fits['exp'][1]) / phi:+7.2f},"
                  f" bottom-2-m share {b2:.3f} | pa: par {pp:5.1f} m, 2 dll vs exp {2 * (lp - fits['exp'][3]):+6.2f},"
                  f" bottom-2-m share {b2p:.3f}")
        print(f"    (exp dispersion phi {phi:.0f}; hauls {int(d.y.notna().sum())}, with adults {int((d.y > 0).sum())})")

print("\nInterpretation aid: share of an exp(Hs) profile within 2 m / 10 m / 35 m of the bed")
for Hs in (5, 10, 15, 20, 30, 40, 60):
    print(f"  Hs {Hs:3d} m: {1 - math.exp(-2 / Hs):.3f} / {1 - math.exp(-10 / Hs):.3f} / {1 - math.exp(-35 / Hs):.3f};"
          f" density at the bed {1 / Hs:.3f} /m")
