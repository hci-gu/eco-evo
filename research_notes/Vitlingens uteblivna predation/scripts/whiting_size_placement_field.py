"""Size-specific placement of whiting in GOV hauls in the box (collision-free key): what the field's length bands contribute.

Usage: python3 -I whiting_size_placement_field.py <gov dir> <guilds.csv> <length_weight_all.csv>
       > ../tables/whiting_size_placement_field_stdout.txt
Input as bed_overlap_fixedkey.py. Box, Q1 and Q3, 2000-2020. Whiting W = 0.0058 L^3.10 (g, cm; whiting_depth_field.py),
the other species from length_weight_all.csv. Bands: whiting < 20 cm (the model's juvenile stage J is the fish below
their length at age 1 in Q3, 1-2 cm short of 20 cm), 20-30 cm, >= 30 cm, and >= 20 cm (the model's adults A).
Per quarter:
  - each band's share of the box's whiting biomass (class means x the box's cell shares 0.213/0.306/0.332/0.149);
  - the < 20 cm share of whiting biomass within each bed class (the field's J/B per class);
  - bed-class shares per band (as whiting_depth_field.py);
  - Lloyd's L of each band with cod >= 20 cm and with guild B: per haul / classes only / within class
    (as bed_overlap_fixedkey.py).
"""
import csv, gzip, sys, collections
from pathlib import Path
d = Path(sys.argv[1])
G = {int(r["aphia"]): r["guild"] for r in csv.DictReader(open(sys.argv[2]))}
LW = {}
for r in csv.DictReader(open(sys.argv[3])):
    try: LW[int(r[""] if "" in r else r["aphia"])] = (float(r["a"]), float(r["b"]))
    except (KeyError, ValueError): pass
EDGES = [100, 150, 250]; CELLS = [0.213, 0.306, 0.332, 0.149]
WHITING, COD = 126438, 126436
BANDS = (("j", "< 20 cm"), ("m", "20-30 cm"), ("l", ">= 30 cm"), ("a", ">= 20 cm"), ("w", "all"))
H = {}
with open(d / "hauls.csv") as f:
    for r in csv.DictReader(f):
        lat, lon = float(r["ShootLat"]), float(r["ShootLong"])
        if r["Quarter"] in ("1", "3") and 57.7 <= lat <= 58.4 and 9.9 <= lon <= 11.3 and 2000 <= int(r["Year"]) <= 2020:
            H[(r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip())] = dict(
                q=int(r["Quarter"]), c=sum(float(r["Depth"]) >= e for e in EDGES),
                j=0.0, m=0.0, l=0.0, a=0.0, w=0.0, cod=0.0, B=0.0)
with gzip.open(d / "catch.csv.gz", "rt") as f:
    for r in csv.DictReader(f):
        a = int(r["aphia"])
        if a != WHITING and G.get(a) != "B": continue
        h = H.get((r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip()))
        if h is None: continue
        L, n = float(r["length_cm"]), float(r["n_per_h"])
        if not (L > 0 and n > 0): continue
        if a == WHITING:
            kg = n * 0.0058 * L ** 3.10 / 1000
            h["w"] += kg
            h["j" if L < 20 else "m" if L < 30 else "l"] += kg
            if L >= 20: h["a"] += kg
        elif a in LW:
            kg = n * LW[a][0] * L ** LW[a][1] / 1000
            h["B"] += kg
            if a == COD and L >= 20: h["cod"] += kg
for q in (1, 3):
    by = collections.defaultdict(list)
    for h in H.values():
        if h["q"] == q: by[h["c"]].append(h)
    present = [k for k in range(4) if by[k]]
    wsum = sum(CELLS[k] for k in present); f = {k: CELLS[k] / wsum for k in present}
    mean = lambda k, g: sum(h[g] for h in by[k]) / len(by[k])
    tot = lambda g: sum(f[k] * mean(k, g) for k in present)
    print(f"Q{q}: hauls per class {[len(by[k]) for k in range(4)]} (< 100 / 100-150 / 150-250 / >= 250 m)")
    print("  band share of whiting biomass  " + " | ".join(f"{lab} {tot(g) / tot('w'):.3f}" for g, lab in BANDS[:4]))
    print("  < 20 cm share within class     " + " / ".join(f"{mean(k, 'j') / mean(k, 'w'):.3f}" if k in present and mean(k, 'w') > 0 else "  -  " for k in range(4)))
    for g, lab in BANDS:
        sh = [f[k] * mean(k, g) / tot(g) if k in present else 0.0 for k in range(4)]
        print(f"  {lab:9s} class shares " + " / ".join(f"{v:.3f}" for v in sh))
    for p, plab in (("cod", "cod >= 20 cm"), ("B", "guild B")):
        for g, lab in BANDS:
            mx, my = tot(g), tot(p)
            haul = sum(f[k] * sum(h[g] * h[p] for h in by[k]) / len(by[k]) for k in present) / (mx * my)
            clas = sum(f[k] * mean(k, g) * mean(k, p) for k in present) / (mx * my)
            print(f"  L {lab:9s} x {plab:12s} per haul {haul:.2f} | classes only {clas:.2f} | within class {haul / clas:.2f}")
