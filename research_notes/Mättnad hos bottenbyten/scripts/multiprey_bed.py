"""Shared search and gut on one bed: multi-prey Holling II vs the engine's per-pair denominators.

Run: python3 -I scripts/multiprey_bed.py

Engine (lib/environments/ecosystem_env/predation.py + interactions.holling_a_eff): each pair has its
own denominator, I_i = P eat_i a_i f_i B_i / (1 + a_i h_i f_i B_i + wP); the policy splits eat over
pairs (time sharing between separate patches). A fish searching ONE bed meets every bed prey in the
same search time and fills ONE gut; the standard multi-prey form (Murdoch 1973; Holling 1959 disc
equation generalised; Jeschke et al. 2002 for the digestion-limited ceiling) is
    I_i = a_i f_i B_i / (1 + sum_j a_j h_j f_j B_j)          (bed prey j only)
so x_i alone is not the pair's saturation: its own-density elasticity is
    eps_i = 1 - x_i / (1 + X),  X = sum_j x_j,
and its effective half-saturation is (1 + X - x_i) / (a_i h_i f_i).
Library numbers (fgconfig/fg_library.yaml, 223.26); B today (start): infauna 94 (exposed 0.2),
exposed epifauna 3.78, Nephrops 4.57 t/km2. Armour (J, tables/Q2Q3_notes.txt section 7): h on
crustaceans x1.7-3; shown as a variant (h x2 on epifauna and Nephrops).
No value here is set from model output; the engine's 223.27 M2 is not used.
"""
BED = {
    'benthivores': {
        'infauna': (0.036, 99.3, 1.0, 0.2 * 94.0, 0.7, 2600.0),  # B = A (refuge exposed share)
        'epifauna': (0.15, 143.2, 0.62, 3.78, 0.8, 3750.0),
        'Nephrops': (0.44, 137.6, 0.02, 4.57, 0.8, 3600.0),
    },
    'whiting (bed prey checked today)': {
        'epifauna': (0.27, 119.7, 0.29, 3.78, 0.8, 3750.0),
    },
}
P = {'benthivores': 1.53, 'whiting (bed prey checked today)': 0.446}


def analyse(name, prey, armour=1.0):
    xs = {}
    for k, (a, h, f, B, ae, E) in prey.items():
        hh = h * (armour if k != 'infauna' else 1.0)
        xs[k] = a * hh * f * B
    X = sum(xs.values())
    tot = sum(a * f * B for (a, h, f, B, ae, E) in prey.values())
    print(f'{name}  (armour h x{armour:g} on crustaceans)   X_bed = {X:.1f}')
    for k, (a, h, f, B, ae, E) in prey.items():
        hh = h * (armour if k != 'infauna' else 1.0)
        x = xs[k]
        eps = 1 - x / (1 + X)
        xeff = x / (1 + X - x)
        share = a * f * B / tot
        r10_multi = 0.1 * (1 + X) / (1 + X - 0.9 * x)
        r10_pair = 0.1 * (1 + x) / (1 + 0.1 * x)
        half = (1 + X - x) / (a * hh * f)
        # per-capita M2 on the pair's TOTAL stock (infauna: total = A / 0.2), per year, at the
        # field feeding level F 0.36-0.54 of the ceiling (Fodoniva note, J)
        tot_stock = 94.0 if k == 'infauna' else B
        m2 = P[name] * a * f * B / (1 + X) * 1460 / tot_stock
        print(f'  {k:9s} x {x:6.1f}  x_eff {xeff:5.2f}  eps {eps:4.2f}  diet share {share:4.2f}  '
              f'half-sat on f*B {half:6.2f} t/km2 (f*B today {f * B:5.2f})  intake at 10 % prey: multi {r10_multi:4.2f}'
              f' / per-pair {r10_pair:4.2f}  M2 on total stock at F 0.36-0.54: {0.36 * m2:5.3f}-{0.54 * m2:5.3f}/yr')
    print()


for nm, pr in BED.items():
    analyse(nm, pr)
    analyse(nm, pr, armour=2.0)

# Whiting with its unchecked field infauna share (7.3 % of diet, library comment) searched in
# the same bed: a_wi chosen so that a f B shares match the field diet ratio infauna/epifauna
# 7.3/13.7 (J; a diet-derived ratio, not a fitted level).
a_e, h_e, f_e, B_e = 0.27, 119.7, 0.29, 3.78
afb_e = a_e * f_e * B_e
afb_i = afb_e * 7.3 / 13.7
h_i = 2600.0 / 31.34  # whiting ceiling 31.34 MJ/t/tick
x_e, x_i = a_e * h_e * f_e * B_e, afb_i * h_i
X = x_e + x_i
print('whiting with a diet-ratio infauna term on the same bed (J):')
print(f'  x_epi {x_e:.1f}, x_inf {x_i:.1f}, X {X:.1f}; epifauna x_eff {x_e / (1 + x_i):.2f}, eps {1 - x_e / (1 + X):.2f},'
      f' intake at 10 % epifauna {0.1 * (1 + X) / (1 + X - 0.9 * x_e):.2f}')
