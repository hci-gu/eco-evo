"""Crab size window for cod / haddock / whiting (J scenario arithmetic).
Gape rule from Arnott & Pihl 2000 (1-group cod, Carcinus): CW <= k * TL, k = 0.06 (no rejection)
and 0.09 (upper acceptance, frequent rejection). Haddock k = cod k x 0.70 (haddock/cod gape 0.33/0.47,
earlier gape table). Whiting: cod k as an upper bound (J).
Crab biomass by CW: two J scenarios for Liocarcinus-type crabs (adults 20-56 mm CW):
 A: numbers uniform on 10-55 mm; B: numbers triangular 15-55 mm, mode 35 mm. Biomass ~ CW^3."""
import csv, sys
import numpy as np
cw = np.arange(10.0, 55.01, 0.5)
numA = np.ones_like(cw)
numB = np.where(cw < 15, 0, np.where(cw <= 35, (cw-15)/20, (55-cw)/20))
def bshare(num):
    b = num * cw**3
    return b / b.sum()
scen = {'A_uniform10-55': bshare(numA), 'B_tri15-35-55': bshare(numB)}
def share_below(x, bs):
    return bs[cw <= x].sum()
benth, pred = sys.argv[1], sys.argv[2]
dist = {'cod': {}, 'haddock': {}}
with open(benth) as f:
    for r in csv.DictReader(f):
        if r['species'] in dist:
            dist[r['species']][float(r['L_cm'])] = float(r['share_2001_25'])
dist['whiting'] = {}
with open(pred) as f:
    for r in csv.DictReader(f):
        if r['aphia'] == '126438':
            dist['whiting'][float(r['L5']) + 2.5] = float(r['pooled'])
fac = {'cod': 1.0, 'haddock': 0.70, 'whiting': 1.0}
for sp, d in dist.items():
    L = np.array(sorted(d)); w = np.array([d[x] for x in L]); w = w / w.sum()
    p = np.cumsum(w)
    q = lambda a: L[np.searchsorted(p, a)]
    print(f"{sp}: biomass length p10/p50/p90 = {q(0.1):.1f}/{q(0.5):.1f}/{q(0.9):.1f} cm")
    for k in (0.06, 0.09):
        kk = k * fac[sp]
        out = []
        for name, bs in scen.items():
            s = sum(wi * share_below(kk * Li * 10, bs) for Li, wi in zip(L, w))
            out.append(f"{name} {s:.2f}")
        print(f"   k={kk:.3f}: max CW at p50 {kk*q(0.5)*10:.0f} mm; biomass-weighted crab share in window: " + ', '.join(out))
for name, bs in scen.items():
    print(name, 'crab biomass share below CW 20/30/40/50 mm:', ' '.join(f"{share_below(x,bs):.2f}" for x in (20,30,40,50)))
