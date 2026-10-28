"""Q3: is h = E/Cmax (digestion-limited) right for small benthic items, or does per-item
handling add a time cost that rivals digestion?

Run: python3 -I scripts/Q3_item_handling.py   (no inputs; every number is typed in with
its source; see tables/Q2Q3_evidence.csv)

Jeschke, Kopp & Tollrian 2002 (Ecol Monogr 72:95) [F, research_notes/"Fodoniva och mattnad
i falt"/data/B_jeschke2002]: in the SSS model the maximum predation rate is "not determined
by the sum of the time spent for handling and digesting prey, but solely by the larger of
these two terms" -> y_max = 1/max(b, c), b = handling time per item, c = digestion time per
item. Handling only runs while foraging, digestion round the clock, so the handling side is
compared against the foraging hours.

Units of the model's h: ticks x t_pred / t_prey (6 h tick). For one fish of mass M eating items
of mass m with handling time t (s): h_handle = (t / 21600) x M / m.
"""
TICK_S = 21600.0
CMAX_KJ_PER_KG_D = 104.7          # benthivore ceiling, 26.18 MJ/t/tick (library, 223.15)
H_DIGEST = {'infauna': 99.3, 'epifauna': 143.2, 'nephrops': 137.6}   # library h (E / 26.18)
E_KJ_PER_G = {'infauna': 2.6, 'epifauna': 3.75, 'nephrops': 3.6}     # library energy densities

# (item, pool, wet mass g central (low, high), handling s central (low, high), source keys)
ITEMS = [
    ('Amphiura arm tip (cropped)', 'infauna', 0.002, (0.001, 0.01), 3, (1, 10),
     'mass: Duineveld & van Noort 1986 [A] 0.84 g / 420 arms; ICES plaice ophiuroid median 0.016 g [D] '
     '(whole small ophiuroids); t: J (strike+bite, Holmes 1979 internal-only bound <= 8 s median interval)'),
    ('Bivalve siphon tip (cropped)', 'infauna', 0.003, (0.001, 0.01), 3, (1, 10),
     'mass: J from de Vlas 1979 via Lees 2024 [F] (siphon tips, mg scale); t: J'),
    ('Small polychaete / palp', 'infauna', 0.03, (0.01, 0.06), 4, (2, 10),
     'mass: ICES plaice polychaete median 0.030 g, all predators 0.064 [D]; t: J'),
    ('Small bivalve (whole)', 'infauna', 0.05, (0.02, 0.3), 5, (2, 15),
     'mass: ICES bivalve median 0.066 (plaice 0.045) [D]; t: J'),
    ('Amphipod / cumacean', 'infauna', 0.01, (0.005, 0.025), 2, (1, 5),
     'mass: ICES amphipod median 0.010 [D]; t: J (Persson et al. 2012 [F] Neomysis ~1 s, J)'),
    ('Mysid', 'epifauna', 0.03, (0.016, 0.06), 1, (0.5, 3),
     'mass: ICES mysid median 0.030 [D]; t: Persson 2012 h 0.026 s/J x ~40 J (J)'),
    ('Small shrimp (Crangon etc.)', 'epifauna', 0.18, (0.055, 0.38), 7, (3, 40),
     'mass: ICES small_shrimp median 0.18 [D]; t: Arnott & Pihl 2000 [F] ln HT = -1.73 + 21.4 P:C '
     '(+0.78 SF): 4.4 s at P:C 0.15, 7 s at 0.17, 37 s at 0.25'),
    ('Pandalus', 'epifauna', 0.61, (0.2, 1.8), 10, (4, 40),
     'mass: ICES pandalid median 0.61 (cod 0.83) [D]; t: J as Crangon at P:C 0.15-0.25'),
    ('Small crab (Liocarcinus etc.)', 'epifauna', 0.56, (0.14, 3.0), 9, (5, 14),
     'mass: ICES decapod-other median 0.56 (cod 0.97) [D]; t: Arnott & Pihl 2000 [F] median 9 s '
     '(quartiles 5-14, Carcinus, independent of P:C)'),
    ('Hermit crab', 'epifauna', 1.47, (0.55, 3.0), 15, (9, 40),
     'mass: ICES hermit median 1.47 [D]; t: J (crab value up to 3x for the shell)'),
    ('Munida', 'epifauna', 0.42, (0.1, 0.9), 12, (5, 30),
     'mass: ICES squat-lobster median 0.42 [D]; t: J'),
    ('Nephrops', 'nephrops', 7.7, (3.2, 12.6), 60, (20, 180),
     'mass: ICES Nephrops median 7.7 [D]; t: J (no measurement)'),
]

FISH_KG = (0.5, 2.0)
FORAGE_H = (24.0, 12.0)  # foraging hours per day: round the clock, or night only (Rindorf 2003, J)

print('Per-item handling vs digestion at the benthivore ceiling (104.7 kJ/kg/d)')
print('items/d = Cmax x M / (E x m); hours/d = items x t / 3600; h_handle = t/21600 x M/m (model units)')
print()
hdr = f'{"item":32s} {"M kg":>4s} {"items/d":>8s} {"h/d (t lo-hi)":>20s} {"h_handle":>9s} {"h_digest":>8s} {"ratio":>6s}'
print(hdr)
rows = []
for name, pool, m, (mlo, mhi), t, (tlo, thi), src in ITEMS:
    for M in FISH_KG:
        need_g = CMAX_KJ_PER_KG_D * M / E_KJ_PER_G[pool]
        n = need_g / m
        hrs = n * t / 3600
        hrs_lo = need_g / mhi * tlo / 3600
        hrs_hi = need_g / mlo * thi / 3600
        hh = t / TICK_S * M / (m / 1000)
        hd = H_DIGEST[pool]
        rows.append((name, M, n, hrs, hrs_lo, hrs_hi, hh, hd))
        print(f'{name:32s} {M:4.1f} {n:8.0f} {hrs:6.1f} ({hrs_lo:5.2f}-{hrs_hi:5.1f}) {hh:9.1f} {hd:8.1f} {hh / hd:6.2f}')
print()
print('ratio = h_handle / h_digest. Jeschke 2002 SSS: the ceiling is 1/max(b, c); handling binds where')
print('ratio > 1 with 24 h foraging, or > 0.5 with 12 h foraging (handling only runs while foraging).')
print()
print('Where handling would bind (central masses and times):')
for name, M, n, hrs, lo, hi, hh, hd in rows:
    for fh in FORAGE_H:
        if hrs > fh:
            print(f'  {name} at {M} kg: {hrs:.1f} h/d of handling > {fh:.0f} h foraging')
print()
print('Direction (J): a larger h (handling-limited) LOWERS the ceiling on that item class but RAISES')
print('x = a h f B; it does not unsaturate the pair. It matters for which items can supply a')
print('large fish at Cmax: cropped parts (arm tips, siphon tips) cannot.')
