"""Saturation arithmetic for the engine's benthic pairs (diagnostic; no value is set from it).

Run: python3 -I scripts/saturation_arithmetic.py

1. x = a h f B per pair at today's prey (library 223.26) and the elasticity of intake to prey,
   eps = dlnI/dlnB = 1/(1 + x) (Holling II), plus the intake left at 10 % of today's prey.
2. Group sweep rate a P (per tick, per year): how often the predator group, if unsaturated,
   would clear the accessible stock; compared with the biological renewal of exposure k
   (research_notes/"Refugium och utbyte i bentos": infauna ~2.5 /yr, epifauna diel 365-730 /yr).
   In a foraging-arena reading (Walters & Juanes 1993; Ahrens et al. 2012) the realised supply
   is v B_V/(v + aP) of the vulnerable stock; aP/v >> 1 => ratio-dependent intake ~ v B_V / P.
3. Beddington-DeAngelis: the interference w that would make intake fall by 20 % when the
   consumer density rises from 0.18 to 0.29 t/km2 (Kattegat high- vs low-trawl flatfish, q = 1,
   tables/field_density_x_stdout.txt; scale x1/q for q 0.3-0.6) at a given x.
"""
PAIRS = {  # a (km2/t/tick), h (tick t/t), f, B (t/km2), P (t/km2)
    'benthivores -> infauna (A = 0.2 x 94)': (0.036, 99.3, 1.0, 0.2 * 94.0, 1.53),
    'benthivores -> exposed epifauna': (0.15, 143.2, 0.62, 3.78, 1.53),
    'whiting -> exposed epifauna': (0.27, 119.7, 0.29, 3.78, 0.446),
    'benthivores -> Nephrops': (0.44, 137.6, 0.02, 4.57, 1.53),
}
K_EXPOSURE = {  # renewal of the exposed / vulnerable state, per year (refuge note)
    'benthivores -> infauna (A = 0.2 x 94)': (1.5, 3.0),
    'benthivores -> exposed epifauna': (365.0, 730.0),
    'whiting -> exposed epifauna': (365.0, 730.0),
    'benthivores -> Nephrops': (365.0, 730.0),
}
print('1. Saturation and elasticity at today\'s prey')
for k, (a, h, f, B, P) in PAIRS.items():
    x = a * h * f * B
    r10 = 0.1 * (1 + x) / (1 + 0.1 * x)
    print(f'  {k:42s} x {x:6.1f}  eps {1 / (1 + x):.3f}  intake at 10 % prey {r10:.2f} of today'
          f'  half-sat {1 / (a * h * f):.3f} t/km2 total')
print()
print('2. Group sweep rate vs renewal of exposure')
for k, (a, h, f, B, P) in PAIRS.items():
    ap_tick = a * P
    ap_yr = ap_tick * 1460
    lo, hi = K_EXPOSURE[k]
    print(f'  {k:42s} aP {ap_tick:.3f}/tick = {ap_yr:6.0f}/yr; k {lo:.0f}-{hi:.0f}/yr; aP/k {ap_yr / hi:.2f}-{ap_yr / lo:.2f};'
          f' arena V/B_V = k/(k + aP) {lo / (lo + ap_yr):.2f}-{hi / (hi + ap_yr):.2f}')
    cons_ceiling = P / h * 1460
    print(f'  {"":42s} group intake at the ceiling (this prey only) {cons_ceiling:.1f} t/km2/yr ='
          f' {cons_ceiling / (f * B):.1f} x the accessible stock per year')
print()
print('3. Interference w needed for a 20 % intake rise when P falls 0.29 -> 0.18 t/km2 (x1/q)')
for q in (1.0, 0.5, 0.3):
    p_lo, p_hi = 0.18 / q, 0.29 / q
    for x in (0.5, 1, 2, 4, 35, 67):
        # I ~ 1/(1 + x + w P); require (1 + x + w p_hi)/(1 + x + w p_lo) = 1.2
        w = 0.2 * (1 + x) / (p_hi - 1.2 * p_lo)
        print(f'  q {q:.1f}  x {x:5.1f}: w {w:8.2f} km2/t  (w P_hi {w * p_hi:6.2f}; library w 0.45)')
