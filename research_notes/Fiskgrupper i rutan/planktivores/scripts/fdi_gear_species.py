"""Aggregate FDI Table A rows (3.a.20 extract) by year, gear and target assemblage for given species.
Usage: python3 -I fdi_gear_species.py SPECIES file1.csv [file2.csv ...]"""
import csv, sys
from collections import defaultdict
sp = sys.argv[1]
agg = defaultdict(float); yt = defaultdict(float); qt = defaultdict(float)
for fn in sys.argv[2:]:
    for r in csv.DictReader(open(fn, encoding="utf-8", errors="replace")):
        if r["species"] != sp: continue
        def f(k):
            try: return float(r[k])
            except ValueError: return 0.0
        w = f("total_live_weight_landed") + f("tot_discards_tonnes")
        g = r["gear_type"]
        gg = "bottom_otter(OTB/OTT/PTB)" if g in ("OTB", "OTT", "PTB") else ("pelagic(OTM/PTM/PS)" if g in ("OTM", "PTM", "PS") else "other:" + g)
        agg[(r["year"], gg, r["target_assemblage"], r["mesh_size_range"])] += w
        yt[r["year"]] += w; qt[(r["year"], r["quarter"])] += w
for y in sorted(yt):
    print(f"{sp} {y} total {yt[y]:.1f} t; by quarter:", {q: round(v, 1) for (yy, q), v in sorted(qt.items()) if yy == y})
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        if k[0] == y and v > 0.05:
            print(f"   {k[1]:28s} {k[2]:5s} {k[3]:10s} {v:9.1f}")
tot = defaultdict(float)
for (y, gg, ta, m), v in agg.items(): tot[gg] += v
s = sum(tot.values())
print("pooled all years:", {k: f"{v:.0f} t ({v/s*100:.1f} %)" for k, v in sorted(tot.items(), key=lambda x: -x[1])} if s else "none")
