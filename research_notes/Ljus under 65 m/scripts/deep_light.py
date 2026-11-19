"""Daytime light at 65-250 m in the box on the prior note's surface convention, with the glider-measured K below 65 m.

Usage: python3 -I deep_light.py > ../tables/deep_light_stdout.txt   (needs ../tables/voto_k.json; writes deep_light.json)

Surface convention and upper layers copied from research_notes/"Ljus under 50 m"/scripts/box_light.py (not
imported): clear-sky illuminance vs sun elevation x 0.5 cloud factor, 58.15 N, two days per month, 15-min steps,
E(z) = E0 T(z); 0-2 m K 0.30; 2-20 m K = 1.56/Secchi(month) (box Secchi); 20-50 m 0.109; 50-65 m 0.091
(SMHI CTD PAR). Statistic: median over sun > 0 (variant B of box_light.py).
Below 65 m (this note): slab K per 15 m from the VOTO glider casts (voto_k.py), by bed-depth class, held below the
deepest measured slab at the value of the deepest slab ("extrapolation", [J]).
lx: the convention attenuates lux with K_PAR; at 54 lx per umol for umol. A spectral correction (lx per umol at
depth, spectral.py) is printed separately and not applied to the headline.
Variants:
  glider PAR median   - MPE-PAR (2024-26) slab medians for the bed class
  glider PAR p25/p75  - the slab quartiles (lower K = brighter) as a bracket
  OCR 490 nm          - OCR-504 (2020-22) K490 slab medians (the rod band), pooled over bed classes
  old K_deep 0.06/0.05/0.09 - the prior judged values, for comparison
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAT = 58.15
DAYS = np.arange(8, 366, 15.2)[:24]
HOURS = np.arange(0.125, 24, 0.25)
dec = np.radians(23.44) * np.sin(2 * np.pi * (284 + DAYS) / 365.0)
ha = np.radians(15 * (HOURS - 12))
phi = math.radians(LAT)
sin_e = (math.sin(phi) * np.sin(dec)[:, None] + math.cos(phi) * np.cos(dec)[:, None] * np.cos(ha)[None, :])
ELEV = np.degrees(np.arcsin(np.clip(sin_e, -1, 1))).ravel()
MONTH = np.repeat(np.minimum(((DAYS - 1) / (365 / 12)).astype(int), 11), len(HOURS))
SECCHI = {1: 5.0, 2: 7.0, 3: 7.0, 4: 8.5, 5: 8.0, 6: 7.0, 7: 10.0, 8: 9.5, 9: 8.0, 10: 9.0, 11: 9.0, 12: 11.5}
LX_PER_UMOL = 54.0
ZS = (65, 80, 100, 120, 150, 175, 200, 250)
SLABS = [(65, 80), (80, 95), (95, 110), (110, 125), (125, 140), (140, 155), (155, 170)]


def surface_lux(h):
    pts = [(-18, 6e-4), (-12, 8e-3), (-6, 3.0), (0, 500.0), (5, 1.3e5 * math.sin(math.radians(5)))]
    if h >= 5:
        return 1.3e5 * math.sin(math.radians(h))
    if h <= -18:
        return 6e-4
    for (h0, e0), (h1, e1) in zip(pts[:-1], pts[1:]):
        if h0 <= h <= h1:
            f = (h - h0) / (h1 - h0)
            return math.exp(math.log(e0) + f * (math.log(e1) - math.log(e0)))
    return 6e-4


E0 = np.array([surface_lux(h) for h in ELEV]) * 0.5
DAY = ELEV > 0


def tau(z, layers):
    t = 0.0
    for top, bot, k in layers:
        if z > top:
            t += k * (min(z, bot) - top)
    return t


def upper(month):
    return [(0, 2, 0.30), (2, 20, 1.56 / SECCHI[month]), (20, 50, 0.109), (50, 65, 0.091)]


def day_median(z, deep_layers, months=range(1, 13)):
    T = np.array([math.exp(-tau(z, upper(m) + deep_layers)) for m in range(1, 13)])
    E = E0 * T[MONTH]
    sel = DAY & np.isin(MONTH, [m - 1 for m in months])
    return float(np.median(E[sel]))


def deep_from(kslabs, k_below):
    """kslabs: list of K for SLABS (None = missing -> stop); below the last one K_below."""
    lay = []
    last = 65
    for (lo, hi), k in zip(SLABS, kslabs):
        if k is None or not np.isfinite(k):
            break
        lay.append((lo, hi, k))
        last = hi
    lay.append((last, 1e4, k_below))
    return lay


def main():
    P = json.load(open(os.path.join(HERE, "..", "tables", "voto_k.json")))["pooled"]
    out = {}

    def slabk(prefix, idx, n_min=20):
        ks = []
        for lo, hi in SLABS:
            v = P.get(f"{prefix} {lo}-{hi}")
            ks.append(v[idx] if v and v[0] >= n_min else None)
        return ks

    variants = {}
    for bc in ("150-250", ">=250", "100-150", "<100"):
        for lab, idx in (("median", 2), ("p25 (bright)", 1), ("p75 (dark)", 3)):
            ks = slabk(f"MPE-PAR bed {bc}", idx)
            good = [k for k in ks if k is not None]
            if not good:
                continue
            variants[f"glider PAR bed {bc} {lab}"] = (ks, good[-1])
    ks = slabk("Ed490", 2)
    good = [k for k in ks if k is not None]
    variants["OCR 490 nm, all beds (2020-22)"] = (ks, good[-1])
    for kd in (0.05, 0.06, 0.09):
        variants[f"old judged K_deep {kd}"] = ([None] * len(SLABS), kd)

    print("Slab K used (65-80, 80-95, ..., 155-170 m; '-' = fewer than 20 casts) and the K held below:")
    for name, (ks, kb) in variants.items():
        print(f"  {name:38s} " + " ".join("  -  " if k is None else f"{k:.3f}" for k in ks) + f" | below {kb:.3f}")

    print("\nDaytime median (sun > 0, all months), lx on the convention (54 lx per umol for umol):")
    print("  variant                                " + " ".join(f"{z:>8d} m" for z in ZS))
    for name, (ks, kb) in variants.items():
        lay = deep_from(ks, kb)
        vals = [day_median(z, lay) for z in ZS]
        out[name] = dict(zip(map(str, ZS), vals))
        print(f"  {name:38s} " + " ".join(f"{v:10.3g}" for v in vals))
    print("  (umol m-2 s-1 = lx / 54)")

    # bed-mid convention: light at the bed for a bed class = light at the class mid-depth
    print("\nLight at the bed by day (median over sun > 0), lx [umol]:")
    for name, z in (("glider PAR bed 150-250", 200), ("glider PAR bed 150-250", 150), ("glider PAR bed 150-250", 250),
                    ("glider PAR bed >=250", 250), ("glider PAR bed 100-150", 125)):
        row = []
        for lab in ("median", "p25 (bright)", "p75 (dark)"):
            key = f"{name} {lab}"
            if key in variants:
                ks, kb = variants[key]
                v = day_median(z, deep_from(ks, kb))
                row.append(f"{lab} {v:.2g} [{v / LX_PER_UMOL:.2g}]")
        print(f"  {name} at {z} m: " + "; ".join(row))

    # seasonal: Apr-Sep vs Oct-Mar, bed 150-250 median
    print("\nBy half-year, bed 150-250 m class, glider PAR median slab K of that half-year (lx):")
    for hy, months in (("Apr-Sep", range(4, 10)), ("Oct-Mar", (10, 11, 12, 1, 2, 3))):
        ks = slabk(f"MPE-PAR bed 150-250 {hy}", 2, n_min=10)
        good = [k for k in ks if k is not None]
        if not good:
            continue
        lay = deep_from(ks, good[-1])
        vals = [day_median(z, lay, months) for z in ZS]
        out[f"half-year {hy}"] = dict(zip(map(str, ZS), vals))
        print(f"  {hy}: slab K " + " ".join("-" if k is None else f"{k:.3f}" for k in ks) + "\n     " +
              " ".join(f"{z} m {v:.3g}" for z, v in zip(ZS, vals)))

    # check of the convention's absolute level against the glider's own daytime medians (umol)
    print("\nCheck: convention (glider PAR median K, all beds) vs glider-measured daytime median PAR, umol m-2 s-1:")
    ks = slabk("MPE-PAR", 2)
    lay_all = deep_from([P.get(f"MPE-PAR {lo}-{hi}", [0, 0, None])[2] for lo, hi in SLABS], 0.09)
    D = json.load(open(os.path.join(HERE, "..", "tables", "voto_k.json"))).get("daymed", {})
    for m in range(1, 13):
        row = []
        for z in (50, 65, 80, 100, 120, 140):
            meas = D.get(f"MPE-PAR {m} {z}")
            if meas is None:
                continue
            conv = day_median(z + 2.5, lay_all, [m]) / LX_PER_UMOL
            row.append(f"{z} m {meas:.2g}/{conv:.2g} (x{meas / conv:.2f})")
        if row:
            print(f"  month {m:2d}: " + "; ".join(row))
    print("\nCheck by bed class, all months pooled: glider-measured daytime median / convention with that class's K "
          "(the glider months: all but July, unequal weights):")
    for bc in ("150-250", ">=250", "100-150"):
        key = f"glider PAR bed {bc} median"
        if key not in variants:
            continue
        lay = deep_from(*variants[key])
        row = []
        for z in (50, 65, 80, 100, 120, 140):
            meas = D.get(f"MPE-PAR bed {bc} {z}")
            if meas is None:
                continue
            conv = day_median(z + 2.5, lay) / LX_PER_UMOL
            row.append(f"{z} m {meas:.2g}/{conv:.2g} (x{meas / conv:.2f})")
        print(f"  bed {bc}: " + "; ".join(row))
    json.dump(out, open(os.path.join(HERE, "..", "tables", "deep_light.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
