"""Population-level krill ration of whiting (>= 20 cm) and saithe as a fraction of the GER Cmax, all stomachs
(krill-free ones count as zero), against the krill-dominated subset (krill_pool_rations.py). Same method and
caveats as krill_pool_rations.py: ratio = sqrt(S/S_max)/M per record (pool mean S), S_max 2.5 %BW, M 1, krill
0.19 g (the weighed saithe pools' 0.17-0.19 g) when only counted.
Records with krill present but neither counted nor weighed are skipped (n reported).
Only areas where krill is a regular prey: NE North Sea, NW North Sea, Skagerrak, Kattegat.

Usage: python3 -I krill_ration_population.py <records_linked.csv.gz> [all|prey|no2022]
  all: every record; prey: only records with prey rows (an upper bound: true empties dropped); no2022: without the
  2022 IBTS uploads, flagged unusable in research_notes/"Krillparen mot vintermagar" (mostly records without prey rows).
Model-ceiling scale: divide by 1.2 (whiting) / 1.23 (saithe).
"""
import csv, gzip, math, sys
from collections import defaultdict
LW = {"whiting": (0.0058, 3.10), "saithe": (0.0089, 3.05)}
MODE = sys.argv[2] if len(sys.argv) > 2 else "all"   # all | prey (records with prey rows only) | no2022
AREAS = {"NE North Sea", "NW North Sea", "Skagerrak", "Kattegat"}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


acc = defaultdict(lambda: [0.0, 0.0, 0, 0, 0.0])   # key -> [sum ratio*N, sum N, records, skipped, sum N with krill]
with gzip.open(sys.argv[1], "rt") as fh:
    for r in csv.DictReader(fh):
        sp = r["sp"].strip().lower()
        if sp not in LW or r["area"] not in AREAS:
            continue
        L0, N = f(r["L_cm"]), f(r["Number"]) or 1.0
        if not L0:
            continue
        classed = r["year"] in ("1980", "1981", "1982", "1987", "1991") and abs(L0 - round(L0 / 5) * 5) < 1e-6 and N > 1
        L = L0 + (5.0 if sp == "saithe" else 2.5) if classed else L0
        if sp == "whiting" and L < 20:
            continue
        a, b = LW[sp]
        W = a * L ** b
        occ = f(r["occ_krill"]) or 0
        if MODE == "prey" and r["anyprey"] != "True":
            continue
        if MODE == "no2022" and r["year"] == "2022":
            continue
        gk, kc = f(r["g_krill"]), f(r["krill_count"])
        q = r["quarter"]
        key = (sp, r["area"], q)
        if occ and not gk and not kc:
            acc[key][3] += 1
            continue
        S = (gk if gk else (kc * 0.19 if kc else 0.0)) / N
        ratio = math.sqrt(S / (0.025 * W)) if S > 0 else 0.0
        acc[key][0] += ratio * N
        acc[key][1] += N
        acc[key][2] += 1
        acc[key][4] += N if S > 0 else 0
print("species, area, quarter: records, stomachs, share of stomachs in krill-bearing records, mean krill ration / GER Cmax"
      " (S_max 2.5 %, M 1; x0.47 for the low band S_max 5 %, M 1.5), records skipped (krill uncounted)")
tot = defaultdict(lambda: [0.0, 0.0])
for k in sorted(acc):
    s, n, rec, sk, nk = acc[k]
    if n == 0:
        continue
    tot[k[0]][0] += s
    tot[k[0]][1] += n
    print(f"{k[0]:8s} {k[1]:13s} Q{k[2]}: rec {rec:4d} stom {n:6.0f} krill-bearing {nk / n:4.2f}  mean ratio {s / n:5.3f}  skipped {sk}")
for sp, (s, n) in tot.items():
    print(f"{sp}: all areas and quarters, stomach-weighted mean krill ration / GER Cmax {s / n:.3f} over {n:.0f} stomachs")
