"""Whiting's fish predators by bed-depth class in the box (NS-IBTS GOV, Q1 + Q3, 2000-2025), and the predator-weighted
hazard on whiting per class with the model's class a's.

Usage: python3 -I gov_predators_by_class.py <gov dir> > ../tables/gov_predators_by_class_stdout.txt
  <gov dir> = research_notes/"Sejens djupläge"/data/gov (hauls.csv, catch.csv.gz; DATRAS, ICES data policy, CC BY 4.0;
  "ICES Database of Trawl Surveys (DATRAS), ICES, Copenhagen"). Read in place.

Box 57.7-58.4 N, 9.9-11.3 E; haul key with ShootLat (research_notes/"Vitlingens uteblivna predation" side finding).
Bed class from the haul's Depth (< 100 / 100-150 / 150-250 / >= 250 m); the box's cell shares 0.213 / 0.306 / 0.332 /
0.149 (maps/depth.npz, as in that note). Biomass kg/h from CPUE-at-length with generic W = a L^b (g, cm):
cod 0.0077 L^3.07, saithe 0.0086 L^3.06, whiting 0.0058 L^3.10, hake 0.0051 L^3.13, spurdog 0.0018 L^3.18,
ling 0.0019 L^3.24, anglerfish 0.0240 L^2.85 (FishBase-type values [S]; only within-species class shapes are used).
Predators: cod >= 35 cm (piscivorous size; the benthivores -> whiting pair is cod), whiting >= 30 cm (cannibal size;
unchecked in the model), the water-column members saithe >= 35 cm, hake >= 30 cm, spurdog, ling, anglerfish.
Prey: whiting 10-30 cm. CIs: haul bootstrap within class, 5-95 %, 1000 draws.

Hazard per class (relative): h_c = sum_p a_p,c x B_p x s_p,c, s = the field class shape (class density / box
equal-cell mean), B_p the model group's density (water column 0.617, benthivores' cod part ~ cod share of 1.53 is
not needed: each pair is taken at its own model a and the ratio is per pair), a_p,c = live library a x class
multiplier (research_notes/"Vitlingens placering" tables/pair_depth_mult_live_vs_pre279.txt): WC -> whiting 0.1713 /
0.1104 / 0.0326 / 0.0281, benthivores -> whiting 0.0718 / 0.0466 / 0.0162 / 0.0148 (t^-1 km2 tick^-1 scale, ratio
only). Uniform predators (s = 1) give the per-predator a ratio; field s gives the field-placement ratio.
"""
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
EDGES = [100, 150, 250]
CELLS = np.array([0.213, 0.306, 0.332, 0.149])
LW = {"126436": (0.0077, 3.07), "126441": (0.0086, 3.06), "126438": (0.0058, 3.10), "126484": (0.0051, 3.13),
      "105923": (0.0018, 3.18), "126461": (0.0019, 3.24), "126555": (0.0240, 2.85)}
GROUPS = {
    "cod >= 35": ("126436", 35, 999),
    "saithe >= 35": ("126441", 35, 999),
    "hake >= 30": ("126484", 30, 999),
    "spurdog": ("105923", 0, 999),
    "ling": ("126461", 0, 999),
    "anglerfish": ("126555", 0, 999),
    "whiting >= 30": ("126438", 30, 999),
    "whiting 10-30 (prey)": ("126438", 10, 30),
}
# water column members' shares in the model group (AGENTS.md section 4: saithe 0.80 of the 0.617 t/km2; the rest
# spurdog, hake, anglerfish, ling, velvet belly, pollack). Judged split of the 0.20 rest [J]:
WC_SHARES = {"saithe >= 35": 0.80, "spurdog": 0.06, "hake >= 30": 0.06, "anglerfish": 0.04, "ling": 0.04}
A_WC = np.array([0.1713, 0.1104, 0.0326, 0.0281])
A_BE = np.array([0.0718, 0.0466, 0.0162, 0.0148])

H = {}
with open(d / "hauls.csv") as f:
    for r in csv.DictReader(f):
        lat, lon = float(r["ShootLat"]), float(r["ShootLong"])
        if r["Quarter"] in ("1", "3") and 57.7 <= lat <= 58.4 and 9.9 <= lon <= 11.3:
            H[(r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip())] = dict(
                q=int(r["Quarter"]), c=sum(float(r["Depth"]) >= e for e in EDGES), kg=defaultdict(float))
with gzip.open(d / "catch.csv.gz", "rt") as f:
    for r in csv.DictReader(f):
        h = H.get((r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip()))
        if h is None or r["aphia"] not in LW:
            continue
        L, n = float(r["length_cm"]), float(r["n_per_h"])
        if not (L > 0 and n > 0):
            continue
        a, b = LW[r["aphia"]]
        for g, (aph, lo, hi) in GROUPS.items():
            if aph == r["aphia"] and lo <= L < hi:
                h["kg"][g] += n * a * L ** b / 1000

rng = np.random.default_rng(1)


def shape(hauls, g):
    by = [[h["kg"][g] for h in hauls if h["c"] == k] for k in range(4)]
    m = np.array([np.mean(v) if v else np.nan for v in by])
    boots = []
    for _ in range(1000):
        mb = np.array([np.mean(rng.choice(v, len(v))) if v else np.nan for v in by])
        boots.append(mb / np.nansum(CELLS * mb))
    boots = np.array(boots)
    return m, m / np.nansum(CELLS * m), np.nanpercentile(boots, 5, axis=0), np.nanpercentile(boots, 95, axis=0), \
        [len(v) for v in by]


for qsel, lab in ((None, "Q1 + Q3"), (1, "Q1"), (3, "Q3")):
    hs = [h for h in H.values() if qsel is None or h["q"] == qsel]
    print(f"=== {lab}, box, 2000-2025: hauls per class {[sum(h['c'] == k for h in hs) for k in range(4)]}")
    print("  group                 kg/h < 100 / 100-150 / 150-250 / >= 250   shape (density / box mean) [5-95 %]   "
          "biomass share < 100 m")
    S = {}
    for g in GROUPS:
        m, s, lo, hi, n = shape(hs, g)
        S[g] = (s, lo, hi)
        share = CELLS * m / np.nansum(CELLS * m)
        print(f"  {g:22s} " + " / ".join(f"{x:7.2f}" for x in m) + "   "
              + " / ".join(f"{x:.2f} [{a:.2f}-{b:.2f}]" for x, a, b in zip(s, lo, hi)) + f"   {share[0]:.2f}")
    wc = sum(WC_SHARES[g] * S[g][0] for g in WC_SHARES)
    wc_lo = sum(WC_SHARES[g] * S[g][1] for g in WC_SHARES)
    wc_hi = sum(WC_SHARES[g] * S[g][2] for g in WC_SHARES)
    print("  water column (member-weighted shape): " + " / ".join(f"{x:.2f} [{a:.2f}-{b:.2f}]"
                                                              for x, a, b in zip(wc, wc_lo, wc_hi)))
    for name, a, s in (("WC -> whiting", A_WC, wc), ("benthivores (cod) -> whiting", A_BE, S["cod >= 35"][0])):
        h_uni = a
        h_fld = a * s
        print(f"  {name:30s} per-predator a ratio < 100 : 150-250 = {h_uni[0] / h_uni[2]:.2f}; "
              f"with field predator placement {h_fld[0] / h_fld[2]:.2f} "
              f"(placement factor {s[0] / s[2]:.2f}); class hazard " + " / ".join(f"{x / h_fld[2]:.2f}" for x in h_fld))
    print()
