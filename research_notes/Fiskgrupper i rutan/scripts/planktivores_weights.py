"""Biomass-weighted group values for planktivorous_fish (params_planktivorous_fish.md).
Species values come from the evidence notes in ../planktivores/ and the sibling notes cited there.
Usage: python3 planktivores_weights.py"""
# annual-mean biomass, t/km2: central, low-herring, high-herring
B = {'central': dict(her=2.60, bw=0.35, pout=0.25, spr=0.10),
     'low_her': dict(her=2.10, bw=0.50, pout=0.35, spr=0.20),
     'high_her': dict(her=3.40, bw=0.20, pout=0.15, spr=0.05)}
# species values: (central, low, high)
V = {
 'r_max':   dict(her=(0.35, .27, .40), spr=(0.48, .48, .56), pout=(0.46, .38, 1.0), bw=(0.45, .41, .47)),
 'M1':      dict(her=(0.26, .10, .35), spr=(0.42, .20, .49), pout=(0.30, .20, .55), bw=(0.20, .10, .40)),
 'F':       dict(her=(0.10, .005, .19), spr=(0.45, .09, .60), pout=(0.03, .02, .16), bw=(0.18, .0, .21)),
 'max_reserve': dict(her=(7000, 6100, 7400), spr=(6700, 6000, 7900), pout=(2000, 1200, 2400), bw=(3600, 2900, 3600)),
 'dark_ratio': dict(her=(0.1, .1, .1), spr=(0.1, .1, .1), pout=(1.0, .5, 1.0), bw=(1.0, .5, 1.0)),
 'QB':      dict(her=(4.7, 4.3, 5.6), spr=(6.0, 5.3, 6.0), pout=(4.5, 4.0, 5.05), bw=(7.5, 6.0, 9.06)),
 'zoo_share': dict(her=(0.95, .93, .98), spr=(0.80, .79, .80), pout=(0.70, .30, .80), bw=(0.77, .74, .80)),
 'benthos_share': dict(her=(0.03, .02, .05), spr=(0.0, 0, 0), pout=(0.30, .20, .65), bw=(0.12, .10, .15)),
 # Cmax as energy, kJ/g/day, mean of months (evidence_cmax_energy.md 1a, central reading)
 'cmax_kJ': dict(her=(0.47, 0.43, 0.65), spr=(0.75, 0.68, 1.0), pout=(0.176, .17, .25), bw=(0.135, .13, .19)),
}
def wavg(field, comp, k=0):
    b = B[comp]; tot = sum(b.values())
    return sum(b[s] * V[field][s][k] for s in b) / tot
for comp, b in B.items():
    tot = sum(b.values())
    print(f"== {comp}: total {tot:.2f} t/km2 = {tot*3600:.0f} t; shares " + ', '.join(f"{s} {v/tot:.3f}" for s, v in b.items()))
    for f in V:
        print(f"   {f:14s} central {wavg(f, comp):8.3f}   low {wavg(f, comp, 1):8.3f}   high {wavg(f, comp, 2):8.3f}")
c = 'central'; b = B[c]; tot = sum(b.values())
r = wavg('r_max', c); F = wavg('F', c)
print('\nK = B*/(1 - F/r):')
for rr, FF in ((0.34, 0.10), (r, 0.10), (r, F), (0.34, 0.115), (r, 0.115)):
    print(f"   r {rr:.3f} F/P {FF:.3f} -> K {3.33/(1-FF/rr):.2f}")
cm = wavg('cmax_kJ', c)
print(f"\nCmax weighted {cm:.3f} kJ/g/d -> {cm/4.5*100:.1f} %BW/d on 4.5 kJ/g -> h {4/(cm/4.5):.1f}")
zoo_cons_per_B = sum(b[s] * V['QB'][s][0] * V['zoo_share'][s][0] for s in b) / tot
QB = wavg('QB', c)
for Bz, lab in ((35000/3600, 'manifest start 35000 t'), (6.76, 'A17 resident 12-month mean')):
    print(f"zoo M2: Q/B {QB:.2f}, zoo eaten per t group {zoo_cons_per_B:.2f}/yr x 3.33 = {zoo_cons_per_B*3.33:.1f} t/km2/yr / {Bz:.2f} ({lab}) = {zoo_cons_per_B*3.33/Bz:.2f} /yr")
for comp in ('low_her', 'high_her'):
    bb = B[comp]; t = sum(bb.values())
    lo = sum(bb[s]*V['QB'][s][1]*V['zoo_share'][s][1] for s in bb)  # t/km2/yr at that biomass
    hi = sum(bb[s]*V['QB'][s][2]*V['zoo_share'][s][2] for s in bb)
    print(f"   {comp}: consumption {lo:.1f}-{hi:.1f} t/km2/yr -> M2 {lo/9.72:.2f}-{hi/6.76:.2f}")
print(f"benthos consumption {sum(b[s]*V['QB'][s][0]*V['benthos_share'][s][0] for s in b):.2f} t/km2/yr")
