"""Field check of the bed-class gradient of fish-on-gadoid encounter, from the predator side: juvenile-gadoid
frequency of occurrence (FO) in adult saithe stomachs by bed class (research_notes/"Sejens profil over djupa
bottnar" tables/saithe_stomach_bed_stdout.txt, ICES stomach database) against GOV juvenile-gadoid density by bed class
in the same area. Clearance index = -ln(1 - FO) / density (FO -> an occurrence rate per stomach, Poisson), relative
to the 150-250 m class. If the light gradient of rule A held in the field, the index would be ~3-5x higher over
< 100 m beds than over 150-250 m beds.

Usage: python3 -I saithe_clearance_index.py <gov dir> > ../tables/saithe_clearance_index_stdout.txt
GOV: research_notes/"Sejens djupläge"/data/gov (Q1 + Q3 2000-2025; DATRAS, CC BY 4.0), north of 57 N, NE + NW
North Sea + Skagerrak (lon -4 to 12 E); juvenile gadoids = haddock, cod, whiting 5-25 cm (prey sizes of adult
saithe; Gadidae unidentified left out as in the source table). Stomachs are 1980-87 spread records (rectangle
sea-area shares) - an ecological inference; GOV is 2000-2025 (era mismatch [J]).
FO values (all quarters, spread; 5-95 % in brackets where given):
  N57 (NE + NW + Skagerrak): < 100 0.14 (0.08-0.20), 100-150 0.21 (0.17-0.26), 150-250 0.18 (0.13-0.24), > 250 0.14
  NE (NE + Skagerrak):       < 100 0.16 (0.09-0.24), 100-150 0.19 (0.13-0.26), 150-250 0.18 (0.10-0.27), > 250 0.14
"""
import csv
import gzip
import math
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
FO = {"N57": [0.14, 0.21, 0.18, 0.14], "NE": [0.16, 0.19, 0.18, 0.14]}
FO_LO = {"N57": [0.08, 0.17, 0.13, None], "NE": [0.09, 0.13, 0.10, None]}
FO_HI = {"N57": [0.20, 0.26, 0.24, None], "NE": [0.24, 0.26, 0.27, None]}
LW = {"126437": (0.0060, 3.10), "126436": (0.0077, 3.07), "126438": (0.0058, 3.10)}
EDGES = [100, 150, 250]


def area(lat, lon):
    if lat < 57.0 or not (-4 <= lon <= 12):
        return None
    return "NE" if lon >= 3.0 else "NW"


H = {}
for r in csv.DictReader(open(d / "hauls.csv")):
    lat, lon = float(r["ShootLat"]), float(r["ShootLong"])
    a = area(lat, lon)
    if a is None or r["Quarter"] not in ("1", "3"):
        continue
    H[(r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip())] = dict(
        a=a, c=sum(float(r["Depth"]) >= e for e in EDGES), kg=0.0)
for r in csv.DictReader(gzip.open(d / "catch.csv.gz", "rt")):
    if r["aphia"] not in LW:
        continue
    h = H.get((r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip()))
    L, n = float(r["length_cm"]), float(r["n_per_h"])
    if h is not None and 5 <= L < 25 and n > 0:
        a, b = LW[r["aphia"]]
        h["kg"] += n * a * L ** b / 1000
for key, areas in (("N57", ("NE", "NW")), ("NE", ("NE",))):
    dens = []
    for k in range(4):
        v = [h["kg"] for h in H.values() if h["a"] in areas and h["c"] == k]
        dens.append((np.mean(v) if v else float("nan"), len(v)))
    print(f"=== {key}: GOV juvenile gadoids 5-25 cm, kg/h by class < 100 / 100-150 / 150-250 / >= 250: "
          + " / ".join(f"{m:.1f} (n{n})" for m, n in dens))
    idx = []
    for k in range(4):
        fo = FO[key][k]
        idx.append(-math.log(1 - fo) / dens[k][0])
    ref = idx[2]
    print("  clearance index rel. to 150-250 m (central FO): " + " / ".join(f"{x / ref:.2f}" for x in idx))
    lo = -math.log(1 - FO_LO[key][0]) / dens[0][0] / (-math.log(1 - FO_HI[key][2]) / dens[2][0])
    hi = -math.log(1 - FO_HI[key][0]) / dens[0][0] / (-math.log(1 - FO_LO[key][2]) / dens[2][0])
    print(f"  < 100 : 150-250 m index {idx[0] / ref:.2f} (FO-interval corners {lo:.2f}-{hi:.2f}); "
          f"density ratio < 100 : 150-250 {dens[0][0] / dens[2][0]:.1f}")
