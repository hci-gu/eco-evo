"""The benthivores' energy budget against its sources: which term breaks the closure at the field Q/B?

Library values as in research_notes/"Krillens flykt och energibudget"/diagnostics/tables/closure.json
(read from the engine, 2026-11-10); field budget from research_notes/"Bentivorernas energibudget"
tables/literature_evidence.md (section 223.9/223.10) and resume 240.2 / 241.2 / 250.
Per tonne-year, MJ, net of SDA (q is energy_gain_mat over the booked diet, SDA netted).
The needed Q/B is the total need / q. No model output enters any number.
usage: python3 levers.py"""
RM = 4862.0          # resting_metabolism 3.33 MJ/t/tick x 1460 (bottom-layer Q10 annual mean ~1.00)
FC, RC = 1.18, 1.0   # feeding_cost (240.2), resting_cost
EC, ME = 3600.0, 1800.0
Q = 2197.0           # assimilated net SDA per tonne of the booked diet (closure.py)
QB_FIELD = 4.46      # 223.10: the 223.9 energy budget 4.2 on the redistributed diet
LOSS = 0.694         # M1 0.29 /yr (library per tick x 1460; was 0.0002 by a unit bug, fixed 2026-11-11) + booked M2 0.234 + adult emigration 0.17
S_FIELD = 0.62       # mean of the monthly start_fill


def need(e=0.95, c=0.69, P=0.74, s=1.0):
    costs = RM * (e * FC + (1 - e) * RC)
    G = P * (EC + c * (EC + s * ME))
    refill = LOSS * s * ME
    return costs, G, refill, costs + G + refill


rows = [
    ("library, trained fish (e 0.95, fill 1)", {}),
    ("c 0.49 (cod alone, 241.2)", dict(c=0.49)),
    ("c 0 (no growth cost)", dict(c=0.0)),
    ("P 0.64 (Z, the 223.9 budget's production)", dict(P=0.64)),
    ("fill at the field mean 0.62", dict(s=S_FIELD)),
    ("eating share 0.5", dict(e=0.5)),
    ("c 0.49 + fill 0.62", dict(c=0.49, s=S_FIELD)),
    ("c 0.49 + fill 0.62 + P 0.64", dict(c=0.49, s=S_FIELD, P=0.64)),
]
print(f"field-implied net intake at Q/B {QB_FIELD}: {QB_FIELD * Q:.0f} MJ/t/yr")
print(f"{'case':45s} {'costs':>6s} {'G':>6s} {'refill':>6s} {'need':>6s} {'Q/B needed':>10s}")
for name, kw in rows:
    c, g, r, n = need(**kw)
    print(f"{name:45s} {c:6.0f} {g:6.0f} {r:6.0f} {n:6.0f} {n / Q:10.2f}")

# The same budget in the field frame (223.9): metabolism incl. SDA and synthesis + production.
sda_per_t_gross = 0.10 * Q / 0.8 / (1 - 0.10 / 0.8)  # rough gross energy per t x 0.10, for scale only
c, g, r, n = need()
synthesis = 0.74 * 0.69 * (EC + ME)
print()
print("field frame (223.9 table: FMR incl. SDA 8200 (6100-10200), P 2900 = Z 0.64 x 4500):")
print(f"  model metabolism without c (action costs at e 0.95): {c:.0f} + SDA (netted from intake)")
print(f"  model synthesis cost c P (ec + ME): {synthesis:.0f}")
print(f"  model deposition P ec + refill (s ME x losses): {0.74 * EC:.0f} + {r:.0f} = {0.74 * EC + r:.0f}")
print(f"  FMR-derived feeding_cost if the field synthesis at P 0.74 is taken out of the 240.2 FMR 4.97/tick:")
fmr, sda_tick = 4.97, 1.05
syn_tick = 0.69 * 0.74 * (EC + 0.62 * ME) / 1460
print(f"    (4.97 - SDA 1.05 - synthesis {syn_tick:.2f}) / RM 3.33 = {(fmr - sda_tick - syn_tick) / 3.33:.2f}"
      f"  (resting_cost 1.0)")
