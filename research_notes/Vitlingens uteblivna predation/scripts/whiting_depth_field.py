"""Whiting by bed-depth class in GOV hauls in the box (collision-free key): length structure.

Usage: python3 -I whiting_depth_field.py <gov dir> > ../tables/whiting_depth_field_stdout.txt
Input: ../../Sejens djupläge/data/gov/ (hauls.csv, catch.csv.gz). Box, Q1 and Q3, 2000-2020.
Per quarter and class (< 100 / 100-150 / 150-250 / >= 250 m): hauls, mean kg/h of whiting < 20 cm (0-group and
1-group; the model's juvenile stage is 1-2 cm short of that), 20-30 cm and >= 30 cm (W = 0.0058 L^3.10, g, cm), the
class's biomass share per length band at the box's cell shares (0.213, 0.306, 0.332, 0.149). BotTemp is missing in
all but one box haul (checked), so no temperature here; see "Sejens djupläge" tables/hydro_by_depth_stdout.txt.
"""
import csv, gzip, sys, collections
from pathlib import Path
d = Path(sys.argv[1])
EDGES = [100, 150, 250]; CELLS = [0.213, 0.306, 0.332, 0.149]
BANDS = [(0, 20), (20, 30), (30, 999)]
H = {}
with open(d / "hauls.csv") as f:
    for r in csv.DictReader(f):
        lat, lon = float(r["ShootLat"]), float(r["ShootLong"])
        if r["Quarter"] in ("1", "3") and 57.7 <= lat <= 58.4 and 9.9 <= lon <= 11.3 and 2000 <= int(r["Year"]) <= 2020:
            H[(r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip())] = dict(
                q=int(r["Quarter"]), c=sum(float(r["Depth"]) >= e for e in EDGES), kg=[0.0, 0.0, 0.0])
with gzip.open(d / "catch.csv.gz", "rt") as f:
    for r in csv.DictReader(f):
        if r["aphia"] != "126438": continue
        h = H.get((r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip()))
        if h is None: continue
        L, n = float(r["length_cm"]), float(r["n_per_h"])
        if L > 0 and n > 0:
            for i, (lo, hi) in enumerate(BANDS):
                if lo <= L < hi: h["kg"][i] += n * 0.0058 * L ** 3.10 / 1000
for q in (1, 3):
    by = collections.defaultdict(list)
    for h in H.values():
        if h["q"] == q: by[h["c"]].append(h)
    print(f"Q{q}: hauls per class {[len(by[k]) for k in range(4)]}")
    for i, (lo, hi) in enumerate(BANDS):
        kgh = [sum(h["kg"][i] for h in by[k]) / len(by[k]) if by[k] else 0.0 for k in range(4)]
        tot = sum(CELLS[k] * kgh[k] for k in range(4) if by[k])
        share = [CELLS[k] * kgh[k] / tot if by[k] and tot else 0.0 for k in range(4)]
        lab = f"{lo}-{hi} cm" if hi < 999 else f">= {lo} cm"
        print(f"  {lab:9s} kg/h " + " / ".join(f"{v:6.1f}" for v in kgh) + "   shares " + " / ".join(f"{v:.3f}" for v in share))
