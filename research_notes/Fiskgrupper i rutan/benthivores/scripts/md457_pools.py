"""Map the EcoBase 457 (Mackinson & Daskalov 2007, North Sea 1991, as deposited up to 2009) diets to the
split model pools. Usage: python3 -I md457_pools.py <ecobase_groups_diets.csv> <out_csv>
Mapping (prey group seq -> pool) is a judgement, stated here:
  P   22 blue whiting, 23 Norway pout, 28/29 herring, 30 sprat, 48 misc. filter-feeding pelagics
  W   5-8 sharks/spurdog, 15/16 whiting, 19/20 saithe, 21 hake, 24 other gadoids (large), 26 monkfish,
      31 mackerel, 32 horse mackerel
  Bf  9-12 rays, 13/14 cod, 17/18 haddock, 27 gurnards, 34-43 flatfish, 45 wolffish, 46 large demersal
  Sand 33 sandeels;  Ofi 25 other gadoids (small: poor cod, bib...), 44 dragonets, 47 small demersal fish
  Z   51 carnivorous zooplankton (euphausiids, hyperiids...), 52 herbivorous/omnivorous zooplankton
  B   54-62 (crabs, Nephrops, epifauna, infauna, shrimp, small mobile epifauna, polychaetes, sessile, meiofauna)
  Ceph 49;  O 50 fish larvae, 53 gelatinous zooplankton, 63-68 microflora, phytoplankton, detritus, discards
"""
import csv, sys, collections
src, out = sys.argv[1:3]
def pool(s):
    if s in (22, 23, 28, 29, 30, 48): return "P"
    if s in (5, 6, 7, 8, 15, 16, 19, 20, 21, 24, 26, 31, 32): return "W"
    if s in (9, 10, 11, 12, 13, 14, 17, 18, 27, 45, 46) or 34 <= s <= 43: return "Bf"
    if s == 33: return "Sand"
    if s in (25, 44, 47): return "Ofi"
    if s in (51, 52): return "Z"
    if 54 <= s <= 62: return "B"
    if s == 49: return "Ceph"
    return "O"
D = collections.defaultdict(lambda: collections.defaultdict(float)); meta = {}
sub = collections.defaultdict(lambda: collections.defaultdict(float))
for r in csv.DictReader(open(src)):
    if r["model"] != "457" or not r["prey_seq"]:
        continue
    k = int(r["seq"]); meta[k] = (r["group"], r["B"], r["PB"], r["QB"])
    p = int(r["prey_seq"]); q = float(r["proportion"])
    D[k][pool(p)] += q; sub[k][r["prey"]] += q
P = ["B", "Z", "P", "W", "Bf", "Sand", "Ofi", "Ceph", "O"]
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seq", "group", "B_t_km2", "PB", "QB"] + [f"pct_{p}" for p in P] + ["pct_carniv_zoo", "pct_herb_zoo", "pct_norway_pout", "pct_herring", "pct_sprat", "top_prey"])
    for k in sorted(D):
        if not (9 <= k <= 47):
            continue
        t = sum(D[k].values())
        g = sub[k]
        top = "; ".join(f"{n[:24]} {100*v/t:.1f}" for n, v in sorted(g.items(), key=lambda x: -x[1])[:6])
        her = sum(v for n, v in g.items() if n.lower().startswith("herring"))
        w.writerow([k, meta[k][0]] + list(meta[k][1:]) + [round(100 * D[k][p] / t, 1) for p in P] +
                   [round(100 * g.get("Carnivorous zooplankton", 0) / t, 1),
                    round(100 * sum(v for n, v in g.items() if n.startswith("Herbivorous")) / t, 1),
                    round(100 * g.get("Norway pout", 0) / t, 1), round(100 * her / t, 1), round(100 * g.get("Sprat", 0) / t, 1), top])
print("ok")
