"""Audit of the whiting -> planktivorous_fish stage access [f_J, f_A] (section 223.32).

Usage: python3 -I planktivore_access.py <stage_length_biomass.csv> <pred_length_biomass.csv>

Reproduces the library value (window 0.39 x whiting length over the POOLED box whiting length mix,
prey species weighted by the planktivore pool weights; research_notes/"Arsungarnas andel" (A-ring)
scripts/stage_access.py) and then changes one thing at a time:
  A  predator side: only whiting >= 20 cm (the encounter's "adult" stage). The pair's a (layer
     variant, section 223.26) already counts only adult whiting (juvenile score 0.04 < 0.1), so a
     window averaged over the 0-20 cm fish counts the predator-size restriction twice.
  B  prey side: species weighted by their share of the ENCOUNTER (a_sp x pool share), not by the pool
     share alone. Per-unit-biomass a of an adult whiting (section 220 central factors, recomputed
     below): herring and sprat 0.0304 (dispersed, day H 70 m R 0.5 m / night H 40 m R 0.2 m), Norway
     pout 0.0289 (near-bed layer 15 m, R 0.2 m), blue whiting 0.0087 (50 m layer). Blue whiting has no
     stage table in the box; its window share is taken as 0.05 (J; box blue whiting 15-25 cm).
  C  window ratio 0.45 (SMS whiting 25-29 cm on herring p95 0.45; ICES-DB 20-30 cm by mass 0.46-0.48)
     instead of 0.39.
The effective f at J/B (0.38, the box planktivore target) = J/B f_J + (1 - J/B) f_A.
Also printed: the planktivore species mix of whiting's take implied by encounter x window (to compare
with the SMS stomachs: Norway pout 0.61, sprat 0.26, herring 0.12 of the planktivore-pool prey).
"""
import math
import sys

import numpy as np
import pandas as pd

SL = pd.read_csv(sys.argv[1])
PL = pd.read_csv(sys.argv[2])
POOL_W = {1: {"herring": 0.88, "blue whiting": 0.0, "Norway pout": 0.11, "sprat": 0.01},
          2: {"herring": 0.788, "blue whiting": 0.106, "Norway pout": 0.076, "sprat": 0.030},
          3: {"herring": 0.78, "blue whiting": 0.15, "Norway pout": 0.04, "sprat": 0.03},
          4: {"herring": 0.788, "blue whiting": 0.106, "Norway pout": 0.076, "sprat": 0.030}}
PERIOD = {1: "2001-25", 2: "1991-96", 3: "2001-25", 4: "1991-96"}
JB = 0.38
T = 21600.0


def area(R, H):
    h = H / 2.0
    return math.pi * R * R if R <= h else 2.0 * (R * R * math.asin(h / R) + h * math.sqrt(R * R - h * h))


def vrel(v, u):
    return (u * u + 3 * v * v) / (3 * v) if v >= u else (v * v + 3 * u * u) / (3 * u)


def a3d(R, v, u, H, Nt, p=0.2, scan=0.5):
    return scan * area(R, H) * vrel(v, u) * T * Nt * p / H / 1e6


# adult whiting, section 220 central: L 0.2516 m (number mean), v 0.8 BL/s, Nt 7330 per t, night speed 1.0
L, Nt = 0.2516, 7330.39
v = 0.8 * L
a_unit = {
    "herring": 0.5 * a3d(min(0.5, 2 * L), v, 0.2, 70.0, Nt) + 0.5 * a3d(0.2, v, 0.2, 40.0, Nt),
    "sprat": 0.5 * a3d(min(0.5, 2 * L), v, 0.2, 70.0, Nt) + 0.5 * a3d(0.2, v, 0.2, 40.0, Nt),
    "Norway pout": a3d(0.2, v, 0.1, 15.0, Nt),
    "blue whiting": a3d(0.2, v, 0.1, 50.0, Nt),
}
annual_pool = {"herring": 0.79, "blue whiting": 0.105, "Norway pout": 0.075, "sprat": 0.03}
a_adult = sum(a_unit[s] * annual_pool[s] for s in a_unit)
print("Adult whiting a per unit prey biomass (km2/t/tick, annual day/night mean):")
for s, val in a_unit.items():
    print(f"  {s:12s} {val:.4f}  x pool share {annual_pool[s]:.3f} -> {val * annual_pool[s]:.5f}")
print(f"  sum (adult a on the pool) {a_adult:.4f}  (223.26 per-stage table: 0.02802)")
print(f"  per tonne of the whole group (adult biomass share 0.753): {0.753 * a_adult:.4f} (library 0.021)\n")

wl = PL[PL.common == "whiting"]
mids = wl.L5.values + 2.5
wts = wl.pooled.values / wl.pooled.sum()


def pred_mix(min_len):
    m = mids >= min_len
    return mids[m], wts[m] / wts[m].sum()


def share_inside(sp, q, stage, ratio, Ls, ws):
    if sp == "blue whiting":
        return 0.05 if stage == "J" else 0.0
    d = SL[(SL.period == PERIOD[q]) & (SL.quarter == q) & (SL.species == sp)]
    if d.empty or d[stage].sum() <= 0:
        return np.nan
    Lp = d.L_cm.values + 0.5
    vv = d[stage].values / d[stage].sum()
    return float(sum(w * vv[Lp <= ratio * Lw].sum() for Lw, w in zip(Ls, ws)))


def stage_access(ratio, min_len, encounter_weighted):
    Ls, ws = pred_mix(min_len)
    res = {}
    for stage in ("J", "A"):
        fq = []
        for q in (1, 2, 3, 4):
            num = den = 0.0
            for sp, w in POOL_W[q].items():
                if w <= 0:
                    continue
                if sp == "blue whiting" and not encounter_weighted:
                    continue  # stage_access.py: no table -> skipped
                f = share_inside(sp, q, stage, ratio, Ls, ws)
                if np.isnan(f):
                    continue
                k = w * (a_unit[sp] if encounter_weighted else 1.0)
                num += k * f
                den += k
            fq.append(num / den)
        res[stage] = (fq, float(np.mean(fq)))
    return res


variants = [
    ("library (pooled whiting, pool-weighted, 0.39)", 0.39, 0.0, False),
    ("A: whiting >= 20 cm", 0.39, 20.0, False),
    ("B: encounter-weighted species", 0.39, 0.0, True),
    ("A+B", 0.39, 20.0, True),
    ("A+B+C: ratio 0.45", 0.45, 20.0, True),
    ("A+B, ratio 0.49 (SMS p99 / ICES-DB)", 0.49, 20.0, True),
    ("A, ratio 0.45 (pool-weighted)", 0.45, 20.0, False),
]
print("Stage access whiting -> planktivores; f_J, f_A annual mean (Q1-Q4); effective f at J/B 0.38")
out = {}
for name, r, ml, ew in variants:
    res = stage_access(r, ml, ew)
    fJ, fA = res["J"][1], res["A"][1]
    eff = JB * fJ + (1 - JB) * fA
    out[name] = (fJ, fA, eff)
    qs = " ".join(f"{x:.3f}" for x in res["J"][0])
    print(f"  {name:42s} [{fJ:.3f}, {fA:.4f}]  eff {eff:.4f}   f_J by quarter {qs}")

print("\nImplied species mix of whiting's planktivore take (encounter x window x pool share, A+B, 0.39 and 0.45),")
print("annual mean over quarters at each quarter's own stage biomass; SMS stomachs: pout 0.61, sprat 0.26, herring 0.12")
for r in (0.39, 0.45):
    Ls, ws = pred_mix(20.0)
    tot = {s: 0.0 for s in a_unit}
    for q in (1, 2, 3, 4):
        for sp, w in POOL_W[q].items():
            if w <= 0:
                continue
            d = SL[(SL.period == PERIOD[q]) & (SL.quarter == q) & (SL.species == sp)]
            jb = (d.J.sum() / (d.J.sum() + d.A.sum())) if not d.empty else 0.5
            fJ = share_inside(sp, q, "J", r, Ls, ws)
            fA = share_inside(sp, q, "A", r, Ls, ws)
            fJ = 0.0 if np.isnan(fJ) else fJ
            fA = 0.0 if np.isnan(fA) else fA
            tot[sp] += w * a_unit[sp] * (jb * fJ + (1 - jb) * fA)
    s = sum(tot.values())
    print(f"  ratio {r}: " + ", ".join(f"{k} {v / s:.2f}" for k, v in tot.items()))

print("\nBox planktivore stage length check: share of each species' J biomass <= 10 cm and <= 13 cm")
for sp in ("herring", "sprat", "Norway pout"):
    for q in (1, 2, 3, 4):
        d = SL[(SL.period == PERIOD[q]) & (SL.quarter == q) & (SL.species == sp)]
        if d.empty or d.J.sum() <= 0:
            continue
        Lp = d.L_cm.values + 0.5
        J = d.J.values / d.J.sum()
        print(f"  {sp:12s} Q{q} ({PERIOD[q]}): J<=10 cm {J[Lp <= 10].sum():.3f}, J<=13 cm {J[Lp <= 13].sum():.3f}, "
              f"J/B {d.J.sum() / (d.J.sum() + d.A.sum()):.2f}")
print("SMS whiting stomachs (North Sea 1981-91): herring eaten p50 8.5 cm, p95 12.5-13.5; sprat 6.0 / 8.5;"
      " Norway pout 6.0-7.5 / 8.5-9.0 (sms_whiting_diet_stdout.txt)")
