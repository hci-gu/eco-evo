"""Whiting's GOV bed-depth shares in the box, recomputed with a collision-free haul key.

Usage: python3 -I whiting_bed_shares_fixedkey.py <gov dir> > ../tables/whiting_bed_shares_fixedkey_stdout.txt
Input: ../../Sejens djupläge/data/gov/ (hauls.csv, catch.csv.gz; key year, quarter, ship, haul number, ShootLat).
Method exactly as ../../Krill i Skagerrak/a_revisit/scripts/gov_by_bed.py (box mode): mean kg/h per bed class x the
box's cell share per class (0.213, 0.306, 0.332, 0.149), normalised; whiting W = 0.0058 L^3.10 (g, cm); every valid
GOV haul, zeros included. That script keys hauls on (year, quarter, ship, haul number): same-numbered hauls of the
same ship elsewhere in the North Sea add their catch to the box haul. Run for 2000-2019 (that script's period),
2000-2020 and 2000-2025.
"""
import csv, gzip, sys, collections
from pathlib import Path
d = Path(sys.argv[1])
CLASSES = [(0, 100), (100, 150), (150, 250), (250, 10000)]
CELLS = [0.213, 0.306, 0.332, 0.149]
H = {}
with open(d / "hauls.csv") as f:
    for r in csv.DictReader(f):
        lat, lon = float(r["ShootLat"]), float(r["ShootLong"])
        if r["Quarter"] in ("1", "3") and 57.7 <= lat <= 58.4 and 9.9 <= lon <= 11.3:
            H[(r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip())] = dict(
                y=int(r["Year"]), q=int(r["Quarter"]), depth=float(r["Depth"]), kg=0.0)
with gzip.open(d / "catch.csv.gz", "rt") as f:
    for r in csv.DictReader(f):
        if r["aphia"] != "126438": continue
        h = H.get((r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip()))
        if h is None: continue
        L, n = float(r["length_cm"]), float(r["n_per_h"])
        if L > 0 and n > 0: h["kg"] += n * 0.0058 * L ** 3.10 / 1000
for y0, y1 in ((2000, 2019), (2000, 2020), (2000, 2025)):
    for q in (1, 3, None):
        rate, nn = [], []
        for lo, hi in CLASSES:
            s = [h["kg"] for h in H.values() if y0 <= h["y"] <= y1 and lo <= h["depth"] < hi and (q is None or h["q"] == q)]
            rate.append(sum(s) / len(s) if s else 0.0); nn.append(len(s))
        w = [r * c for r, c in zip(rate, CELLS)]; t = sum(w)
        print(f"{y0}-{y1} {'Q' + str(q) if q else 'Q1+Q3'}: shares < 100 / 100-150 / 150-250 / > 250 m "
              + " / ".join(f"{x / t:.3f}" for x in w) + "   kg/h " + " / ".join(f"{x:.1f}" for x in rate)
              + "   hauls " + " / ".join(map(str, nn)))
