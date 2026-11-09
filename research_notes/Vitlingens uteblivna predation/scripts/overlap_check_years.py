"""Check: box Lloyd L cod >= 20 cm x whiting 10-28 cm by period, with the full haul key (incl. ShootLat) and with the
older key (year, quarter, ship, haul number) used by ../../Vitlingens predatorer/scripts/gov_overlap.py, whose
colliding hauls pool their catches into one record. Usage: python3 -I overlap_check_years.py <gov dir>"""
import csv, gzip, sys, collections
from pathlib import Path
d = Path(sys.argv[1])
LW = {126438: (0.0058, 3.10), 126436: (0.0077, 3.07)}
def run(keyfn, y0, y1, q):
    H = {}
    with open(d / "hauls.csv") as f:
        for r in csv.DictReader(f):
            lat, lon = float(r["ShootLat"]), float(r["ShootLong"])
            if r["Quarter"] == str(q) and y0 <= int(r["Year"]) <= y1 and 57.7 <= lat <= 58.4 and 9.9 <= lon <= 11.3:
                H[keyfn(r)] = collections.Counter()
    with gzip.open(d / "catch.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            sp = int(r["aphia"])
            if sp not in LW: continue
            h = H.get(keyfn(r))
            if h is None: continue
            L, n = float(r["length_cm"]), float(r["n_per_h"]); a, b = LW[sp]
            if sp == 126436 and L >= 20: h["c"] += n * a * L ** b / 1000
            if sp == 126438 and 10 <= L < 28: h["w"] += n * a * L ** b / 1000
    x = [h["c"] for h in H.values()]; y = [h["w"] for h in H.values()]; n = len(x)
    L = (sum(a * b for a, b in zip(x, y)) / n) / ((sum(x) / n) * (sum(y) / n))
    return n, L
full = lambda r: (r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip())
old = lambda r: (r["Year"], r["Quarter"], r["Ship"].strip(), int(r["HaulNo"]))
for q in (1, 3):
    for y0, y1 in ((2000, 2020), (2000, 2025), (2021, 2025)):
        print(f"Q{q} {y0}-{y1}: full key n, L = {run(full, y0, y1, q)}; old key n, L = {run(old, y0, y1, q)}")
