"""Order-of-magnitude bluefin tuna take of mackerel in Skagerrak-Kattegat [D].
Inputs: Langert 2022 (SLU MSc, 19 tuna): mackerel 17 085 g of a stomach total
~74 300 g (garfish 40 854 g = ~55 %), mean mackerel 407 g, 32 cm; Hellstrom et al.
2025 (via DTU Aqua news): daily ration 3-6 % of body mass, ~90-day stay (Aug-Oct),
example tuna 278 kg (thesis uses 250 kg). Tuna numbers are unknown: the thesis
scenarios 500 / 1500 / 15 000 are used. Area Skagerrak + Kattegat ~54 000 km2 [J,
round figure]. Mackerel density: box mean 0.85 t/km2 (library, research_notes/
'Makrillens tathet'); the season mean while tuna are present is taken as the same
order (the box total peaks ~2.2x in August), so the M2 below is an upper side."""
w_share = 17085 / (40854 / 0.55)          # mackerel share by weight in stomachs
area = 54000.0                            # km2, Skagerrak + Kattegat [J]
dens = 0.85                               # t/km2 mackerel
for n in (500, 1500, 15000):
    for ration in (0.03, 0.06):
        take_t = n * 90 * ration * 250.0 * w_share / 1000.0
        m2 = take_t / area / dens
        print(f"tuna {n:6d} ration {ration:.2f}: mackerel {take_t:8.0f} t/season, "
              f"{take_t / area * 3600:6.1f} t in a 3600 km2 box, M2 ~ {m2:.4f} /yr")
print(f"mackerel share by weight {w_share:.3f}")
