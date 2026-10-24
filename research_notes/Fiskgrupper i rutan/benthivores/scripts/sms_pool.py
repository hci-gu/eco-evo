"""Age-pooled, biomass-weighted M1, M2 by predator and F from the SMS North Sea key run 2023.

Usage: python3 -I sms_pool.py <dir with summary.out, partial_m2.out, species_names.in> [y0 y1]

Rates are 'removal / mean biomass' per year: sum over quarters and ages of rate(q,a) * Bbar(q,a),
divided by the mean over the four quarters of sum_a Bbar(q,a). Bbar = N.bar * west (mean biomass
in the quarter). Prints with and without age 0 (age 0 only enters in Q3-Q4).
"""
import sys, os, collections
d = sys.argv[1]
y0, y1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2018, 2022)
names = [l.strip().strip("_") for l in open(os.path.join(d, "species_names.in"))][:27]
sp = {16: "Cod", 18: "Haddock", 26: "Plaice", 17: "Whiting", 19: "Saithe"}
rows = open(os.path.join(d, "summary.out")).read().split("\n")
hdr = rows[0].split()
ix = {h: i for i, h in enumerate(hdr)}
B = {}  # (y,q,s,a) -> mean biomass
rate = collections.defaultdict(float)
for r in rows[1:]:
    f = r.split()
    if len(f) < len(hdr) - 1:
        continue
    y, q, s, a = int(f[0]), int(f[1]), int(f[2]), int(f[3])
    if s not in sp or not (y0 <= y <= y1):
        continue
    nbar, west = float(f[ix["N.bar"]]), float(f[ix["west"]])
    b = nbar * west
    B[(y, q, s, a)] = b
    for k in ("M1", "M2", "F"):
        rate[(k, y, q, s, a)] = float(f[ix[k]])
    rate[("BIO", y, q, s, a)] = float(f[ix["BIO"]])
pm2 = collections.defaultdict(float)
for r in open(os.path.join(d, "partial_m2.out")):
    f = r.split()
    if not f or f[0] == "Year":
        continue
    y, q, pr, pa, s, a, v = int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5]), float(f[6])
    if s in sp and y0 <= y <= y1:
        pm2[(pr, y, q, s, a)] += v
preds = sorted({k[0] for k in pm2})
print(f"SMS key run 2023, years {y0}-{y1}, rates per year on mean biomass (N.bar*west)")
for s, nm in sp.items():
    for minage in (0, 1, 2):
        tot = collections.defaultdict(float)
        bsum_y = collections.defaultdict(float)
        for (y, q, ss, a), b in B.items():
            if ss != s or a < minage:
                continue
            bsum_y[y] += b / 4.0
            for k in ("M1", "M2", "F"):
                tot[(k, y)] += rate[(k, y, q, s, a)] * b
            for p in preds:
                tot[(p, y)] += pm2[(p, y, q, s, a)] * b
        ys = sorted(bsum_y)
        if not ys:
            continue
        def m(k):
            return sum(tot[(k, y)] / bsum_y[y] for y in ys) / len(ys)
        mb = sum(bsum_y[y] for y in ys) / len(ys)
        line = f"{nm:8s} ages>={minage}: meanB {mb/1000:9.1f} kt  M1 {m('M1'):.3f}  M2 {m('M2'):.3f}  F {m('F'):.3f} | M2 by predator:"
        parts = sorted(((m(p), names[p - 1]) for p in preds), reverse=True)
        line += " ".join(f"{n} {v:.3f}" for v, n in parts if v >= 0.0005)
        print(line)
    # age structure of biomass
    ages = sorted({a for (y, q, ss, a) in B if ss == s})
    shares = []
    for a in ages:
        bb = sum(b for (y, q, ss, aa), b in B.items() if ss == s and aa == a) 
        shares.append(bb)
    tt = sum(shares)
    print("   biomass share by age (all quarters):", " ".join(f"{a}:{v/tt*100:.0f}%" for a, v in zip(ages, shares)))
# check BIO vs N*west
k = next(iter(B))
print("check BIO column vs N.bar*west for", k, rate[("BIO",) + k], B[k])
