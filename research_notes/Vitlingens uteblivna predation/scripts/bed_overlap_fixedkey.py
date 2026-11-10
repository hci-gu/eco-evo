"""Bed-depth classes and the whiting x benthivore overlap in GOV hauls in the box (collision-free haul key).

Usage: python3 -I bed_overlap_fixedkey.py <gov dir> <guilds.csv> <length_weight_all.csv> > ../tables/bed_overlap_fixedkey_stdout.txt
Input: ../../Sejens djupläge/data/gov/ (hauls.csv, catch.csv.gz; key year, quarter, ship, haul number, ShootLat),
"Fiskgrupper i rutan" inputs/guilds.csv (guild B = the benthivores' members) and tables/length_weight_all.csv.
Box, Q1 and Q3, 2000-2020 (and 2000-2025). Groups: whiting (all lengths, as FIELD_BED_SHARES), cod >= 20 cm (the
member that eats whiting), guild B (every benthivore member). Per group: bed-class shares = mean kg/h per class x the
box's cell share per class (0.213, 0.306, 0.332, 0.149), normalised (method of whiting_bed_shares_fixedkey.py).
Overlap, Lloyd's L = mean(x y) / (mean x mean y):
  per haul      - over hauls, each class weighted by its cell share (hauls within a class equally weighted);
  classes only  - the same with each haul replaced by its class mean (what bed depth alone gives);
  within class  - per haul / classes only.
"""
import csv, gzip, sys, collections
from pathlib import Path
d = Path(sys.argv[1])
G = {int(r["aphia"]): r["guild"] for r in csv.DictReader(open(sys.argv[2]))}
LW = {}
for r in csv.DictReader(open(sys.argv[3])):
    try: LW[int(r[""] if "" in r else r["aphia"])] = (float(r["a"]), float(r["b"]))
    except (KeyError, ValueError): pass
EDGES = [100, 150, 250]
CELLS = [0.213, 0.306, 0.332, 0.149]
WHITING, COD = 126438, 126436
def cls(depth):
    return sum(depth >= e for e in EDGES)
H = {}
with open(d / "hauls.csv") as f:
    for r in csv.DictReader(f):
        lat, lon = float(r["ShootLat"]), float(r["ShootLong"])
        if r["Quarter"] in ("1", "3") and 57.7 <= lat <= 58.4 and 9.9 <= lon <= 11.3:
            H[(r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip())] = dict(
                y=int(r["Year"]), q=int(r["Quarter"]), c=cls(float(r["Depth"])), w=0.0, cod=0.0, B=0.0)
missing = collections.Counter()
with gzip.open(d / "catch.csv.gz", "rt") as f:
    for r in csv.DictReader(f):
        a = int(r["aphia"])
        if a != WHITING and G.get(a) != "B": continue
        h = H.get((r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip()))
        if h is None: continue
        L, n = float(r["length_cm"]), float(r["n_per_h"])
        if not (L > 0 and n > 0): continue
        if a not in LW: missing[a] += 1; continue
        kg = n * LW[a][0] * L ** LW[a][1] / 1000
        if a == WHITING: h["w"] += kg
        else:
            h["B"] += kg
            if a == COD and L >= 20: h["cod"] += kg
print("guild B species without length-weight (rows skipped):", dict(missing) or "none")
def run(y0, y1, qs):
    hs = [h for h in H.values() if y0 <= h["y"] <= y1 and h["q"] in qs]
    by = collections.defaultdict(list)
    for h in hs: by[h["c"]].append(h)
    present = [k for k in range(4) if by[k]]
    wsum = sum(CELLS[k] for k in present)
    f = {k: CELLS[k] / wsum for k in present}
    mean = lambda k, g: sum(h[g] for h in by[k]) / len(by[k])
    out = {}
    for g in ("w", "cod", "B"):
        tot = sum(f[k] * mean(k, g) for k in present)
        out[g] = [f[k] * mean(k, g) / tot if k in present else 0.0 for k in range(4)]
    def lloyd(x, y):
        mx = sum(f[k] * mean(k, x) for k in present); my = sum(f[k] * mean(k, y) for k in present)
        haul = sum(f[k] * sum(h[x] * h[y] for h in by[k]) / len(by[k]) for k in present) / (mx * my)
        clas = sum(f[k] * mean(k, x) * mean(k, y) for k in present) / (mx * my)
        return haul, clas, haul / clas
    tag = f"{y0}-{y1} Q{'+'.join(map(str, qs))}"
    print(f"{tag}: hauls per class {[len(by[k]) for k in range(4)]}")
    for g, lab in (("w", "whiting"), ("cod", "cod >= 20 cm"), ("B", "guild B")):
        print(f"  {lab:13s} shares < 100 / 100-150 / 150-250 / >= 250 m " + " / ".join(f"{v:.3f}" for v in out[g]))
    for g, lab in (("cod", "cod >= 20 cm"), ("B", "guild B")):
        hl, cl, wl = lloyd("w", g)
        print(f"  L whiting x {lab:13s} per haul {hl:.2f} | classes only {cl:.2f} | within class {wl:.2f}")
for y0, y1 in ((2000, 2020), (2000, 2025)):
    for qs in ((1,), (3,), (1, 3)):
        run(y0, y1, qs)
