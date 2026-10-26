#!/usr/bin/env python3
"""Field-implied saturation and attack rates from field ration, diet and foraging time.

Research diagnostic, no model output used. Inputs are field data only:
  * the group's field consumption C (Q/B by mass, /yr -> t/t/tick),
  * the consumption-weighted diet shares c_j (stomach-based, the growth
    budget's diet: resume 221, 223.10; params_predatory_fish.md),
  * today's grid-mean prey densities N_j and the library's accessible
    fractions f_j (prey that is reachable at all),
  * a foraging-time fraction tau (share of the 24 h in which the fish
    searches; the literature range is collected separately),
and the engine's per-pair ceilings 1/h_j (gastric-evacuation Cmax).

Model: a multi-prey Type II for a fish that searches all its prey at once
(the standard field form; Murdoch 1973; Rindorf & Gislason 2005),
    C = tau * sum_j a_j f_j N_j / (1 + sum_j a_j h_j f_j N_j).
With a_j f_j N_j = A c_j (diet shares follow encounter):
    sum_j a_j h_j f_j N_j = A hbar,  hbar = sum_j c_j h_j,
    F = C hbar (feeding level on the engine's ceilings),
    x_field = A hbar = F / (tau - F),
    a_j(field) = A c_j / (f_j N_j)  [km2 per t predator per tick, per accessible t].
The ratio a(encounter, section 220) / a(field) is the factor by which the
encounter-theory rate exceeds what the field ration, diet and foraging
time allow. tau <= F has no solution with a per-tick ceiling (the fish
would have to buffer meals in the stomach over several ticks).

Interference (w B_pred) is set to 0 here; with it A is larger by (1 + wB).
"""

QB = {
    # group: (central, low, high) Q/B by mass, /yr
    "benthivorous_fish": (4.46, 3.48, 5.3),   # 223.10 energy budget; published 3.48; Greenstreet 5.3
    "predatory_fish": (3.34, 2.84, 4.32),     # SMS 2023 / Ecopath 1991
}
# pair: (diet share c_j by mass, prey density N_j t/km2, accessible f_j, h_j ticks, a_encounter)
PAIRS = {
    "benthivorous_fish": {
        "benthic_infauna": (0.544, 93.61, 0.20, 95.9, 0.25),
        "mobile_epifauna": (0.314, 7.75, 0.31, 138.4, 0.44),
        "planktivorous_fish": (0.120, 3.333, 1.0, 262.9, 0.019),
        "predatory_fish": (0.023, 1.194, 1.0, 193.8, 0.026),
    },
    "predatory_fish": {
        # renormalised over the checked pairs (0.41, 0.31, 0.072, 0.07 of total food)
        "planktivorous_fish": (0.41 / 0.862, 3.333, 1.0, 279.4, 0.012),
        "zooplankton": (0.31 / 0.862, 9.722, 1.0, 176.5, 0.002),
        "mobile_epifauna": (0.072 / 0.862, 7.75, 0.13, 147.1, 0.33),
        "benthivorous_fish": (0.07 / 0.862, 1.528, 0.15, 176.5, 0.18),
    },
}
TAUS = [1.0, 0.75, 0.5, 0.33, 0.25]

for g, pairs in PAIRS.items():
    hbar = sum(c * h for (c, N, f, h, a) in pairs.values())
    print(f"== {g}: consumption-weighted h = {hbar:.1f} ticks (ceiling {1460 / hbar:.2f} /yr by mass)")
    # Diet electivity: relative a_j f_j implied by the diet vs encounter theory.
    rel_field = {p: c / N for p, (c, N, f, h, a) in pairs.items()}
    rel_enc = {p: a * f for p, (c, N, f, h, a) in pairs.items()}
    ref = max(rel_field, key=lambda p: rel_field[p])
    print("  relative a*f (normalised to the largest field value), field diet vs encounter theory:")
    for p in pairs:
        rf = rel_field[p] / rel_field[ref]
        re = rel_enc[p] / rel_enc[ref]
        print(f"    {p:20s} field {rf:7.3f}   encounter {re:7.3f}   encounter/field {re / rf:6.2f}")
    # Encounter-theory diet if the fish searched all prey at once.
    tot = sum(a * f * N for (c, N, f, h, a) in pairs.values())
    print("  diet a fish searching all prey at once would get from the encounter a's (by mass):")
    print("    " + ", ".join(f"{p} {a * f * N / tot:.2f} (field {c:.2f})" for p, (c, N, f, h, a) in pairs.items()))
    for label, qb in zip(("central", "low", "high"), QB[g]):
        C = qb / 1460.0
        F = C * hbar
        print(f"  Q/B {qb:.2f} ({label}): feeding level on the engine ceilings F = {F:.3f}")
        for tau in TAUS:
            if tau <= F:
                print(f"    tau {tau:4.2f}: no per-tick solution (tau <= F)")
                continue
            x = F / (tau - F)
            A = x / hbar
            parts = []
            for p, (c, N, f, h, a) in pairs.items():
                af = A * c / (f * N)
                parts.append(f"{p.split('_')[0][:5]} {af:.4f} (enc/field {a / af:5.1f})")
            print(f"    tau {tau:4.2f}: x_field {x:5.2f}, share of ceiling while foraging {x / (1 + x):.2f}; a_field: " + "; ".join(parts))
    print()
print("Reading: x_field is the field saturation index a fish would need, at today's prey,")
print("to eat its field ration in the foraging time tau. Encounter theory gives x 44-266 on the")
print("benthic pairs (engine_feeding_level_stdout.txt). A ratio enc/field >> 1 means the encounter")
print("a overstates the realised rate by that factor at the given tau; it is a diagnostic, not a value.")
