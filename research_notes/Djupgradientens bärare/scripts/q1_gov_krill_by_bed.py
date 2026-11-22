"""Q1: presence of Euphausiidae records in NS-IBTS GOV (bottom trawl, daylight) hauls by bed depth.

Reads the compact DATRAS CSVs cached in research_notes/"Lek i rutan"/data/datras (ICES DATRAS, public).
Only hauls by ships/countries and years in which a euphausiid record ever occurs are used as the
denominator (recording effort for invertebrates differs by nation and year).
Usage: python3 -I q1_gov_krill_by_bed.py <hh.csv> <hl.csv>
"""
import csv, sys
from collections import defaultdict

KRILL = {"110671", "110690", "1128", "110673", "110674", "110676", "110700", "110701"}
hh_path, hl_path = sys.argv[1], sys.argv[2]

def key(r):
    return (r["Year"], r["Quarter"], r["Country"], r["Ship"], r["Gear"], r["HaulNo"])

hauls = {}
with open(hh_path) as f:
    for r in csv.DictReader(f):
        if r["HaulVal"] != "V" or r["Gear"] != "GOV":
            continue
        try:
            d = float(r["Depth"]); lat = float(r["ShootLat"]); lon = float(r["ShootLong"])
        except ValueError:
            continue
        hauls[key(r)] = dict(depth=d, lat=lat, lon=lon, dn=r["DayNight"], q=r["Quarter"],
                             y=int(r["Year"]), c=r["Country"], ship=r["Ship"])

pos = set(); aph = defaultdict(int); specval = defaultdict(int)
with open(hl_path) as f:
    for r in csv.DictReader(f):
        if r["Valid_Aphia"] in KRILL:
            k = key(r)
            if k in hauls:
                pos.add(k); aph[r["Valid_Aphia"]] += 1; specval[r.get("SpecVal", "?")] += 1
print("krill Aphia ids found:", dict(aph), "SpecVal codes:", dict(specval))

rec = {(hauls[k]["c"], hauls[k]["y"], hauls[k]["q"]) for k in pos}
print("country-year-quarter strata with any krill record:", sorted(rec))

def cls(d):
    for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 400), (400, 9999)):
        if lo <= d < hi:
            return f"{lo}-{hi}"

for region, (la0, la1, lo0, lo1) in {"box 57.3-59.3N 8-12.2E": (57.3, 59.3, 8.0, 12.2),
                                       "all cached region": (-90, 90, -180, 180)}.items():
    for q in ("1", "3"):
        n = defaultdict(int); p = defaultdict(int)
        for k, h in hauls.items():
            if h["q"] != q or (h["c"], h["y"], h["q"]) not in rec:
                continue
            if not (la0 <= h["lat"] <= la1 and lo0 <= h["lon"] <= lo1):
                continue
            c = cls(h["depth"]); n[c] += 1; p[c] += k in pos
        print(f"\n{region}, Q{q}, hauls in strata that recorded krill at least once (day hauls only in NS-IBTS):")
        for c in sorted(n, key=lambda s: float(s.split('-')[0])):
            print(f"  bed {c:>9} m: hauls {n[c]:4d}  with euphausiids {p[c]:3d}  share {p[c]/n[c]:.2f}")

print("\nPositive hauls: year Q country ship haulno depth lat lon daynight")
for k in sorted(pos):
    h = hauls[k]
    print(" ", h["y"], h["q"], h["c"], h["ship"], k[5], h["depth"], h["lat"], h["lon"], h["dn"])
