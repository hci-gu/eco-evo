"""Convert Persson et al. 2012 (MEPS 456:245, Table 1, [F]) juvenile-cod functional responses
on crustacean epifauna to the engine's units and to a digestion-limited saturation index.

Persson: 1-yr cod (15-20 cm TL; mass 30-60 g, J), 8-9 C, 150 l tanks with sand bottom,
starved 48 h, time from 1st to 5th prey; Type II fitted with a in cm2/s and a BEHAVIOURAL
handling time h_b in s/J. Prey: Neomysis 13 mm, Crangon 40 mm, Palaemon 35 mm. Natural
densities at the site: brown and grass shrimp < 100 ind/m2, small crustaceans < 500 ind/m2.

Outputs: a in km2 per tonne cod per 6 h tick (engine unit), the behavioural half-saturation
1/(a h_b), and x_dig = (encounter-limited intake rate) / Cmax at field densities, i.e. how many
times over a starved, foraging juvenile cod would cover its DAILY ration per unit time
(x_dig >> 1 = saturated against digestion, the engine's h = E/Cmax convention).
Judgement inputs (J) are listed explicitly; run: python3 -I scripts/Q1_persson2012_conversion.py
"""

TABLE1 = {  # prey: (a sand, a Zostera, a Fucus) cm2/s ; (h sand, h Zos, h Fuc) s/J
    'Neomysis': ((57, 8.8, 13), (0.026, 0.022, 0.024)),
    'Crangon': ((1.8, 0.86, 1.2), (0.026, 0.038, 0.019)),
    'Palaemon': ((21, 2.4, 1.6), (0.033, 0.045, 0.041)),
}
ITEM_G = {'Neomysis': 0.010, 'Crangon': 0.64, 'Palaemon': 0.35}   # g wet per item, J (L-W)
E_KJ_G = 4.0          # kJ/g wet, crustaceans 3.5-4.5 (J)
FIELD_N = {'Neomysis': (50, 500), 'Crangon': (1, 100), 'Palaemon': (1, 100)}  # ind/m2, Persson text
TICK_S = 6 * 3600
for mass_g in (30.0, 60.0):
    fish_per_t = 1e6 / mass_g
    # Cmax: 2-4 % body mass/day in crustacean wet weight at 8-9 C (J; juvenile cod)
    for cmax_frac in (0.02, 0.04):
        cmax_J_s = mass_g * cmax_frac * E_KJ_G * 1000 / 86400
        print(f'cod {mass_g:.0f} g, Cmax {cmax_frac*100:.0f} %BW/d = {cmax_J_s:.3f} J/s')
        for prey, (avals, hvals) in TABLE1.items():
            a_cm2s, hb = avals[0], hvals[0]        # sand (open bed)
            a_eng = a_cm2s * 1e-4 * TICK_S * fish_per_t / 1e6   # km2 / t / tick
            half_J_cm2 = 1 / (a_cm2s * hb)
            half_gm2 = half_J_cm2 * 1e4 / (E_KJ_G * 1000)
            out = []
            for n in FIELD_N[prey]:
                N_J_cm2 = n * ITEM_G[prey] * E_KJ_G * 1000 / 1e4
                rate = a_cm2s * N_J_cm2 / (1 + a_cm2s * hb * N_J_cm2)   # J/s while foraging
                out.append(f'{n} ind/m2: x_dig {rate / cmax_J_s:7.1f}')
            print(f'  {prey:9s} sand: a {a_cm2s:5.1f} cm2/s = {a_eng:6.3f} km2/t/tick; '
                  f'behavioural half-sat {half_gm2:6.2f} g/m2; ' + '; '.join(out))
print('Engine for comparison: benthivores -> epifauna a 0.15 (a_enc 0.44 x e 0.43), '
      'whiting -> epifauna 0.27 km2/t/tick (adult fish, whole-group tonne).')
print('Predator cue (cannibal) cut consumption to 22 % (sand) and 44 % (vegetation) (Persson 2012 [F]).')
