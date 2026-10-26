"""Validation arithmetic for predators_encounter.md (no inputs read; numbers from the cited papers).

Usage: python3 -I pred_validation_arith.py
These numbers TEST the encounter-derived a; they were not used to set it.
"""
T_TICK_MIN = 360.0
# 1. Macy et al. 1998: mackerel clearance (L fish-1 min-1), fish 184 g at the end of the trials
fish_kg = 0.184
for label, L_min in (("particulate peak at 3.7 /L", 19.8), ("filter, large copepods 50-58 /L", 6.9),
                     ("filter, small + intermediate copepods", 2.6)):
    m3_per_t_tick = L_min / 1000.0 * T_TICK_MIN / fish_kg * 1000.0
    out = ", ".join(f"H {H} m: {m3_per_t_tick / H / 1e6:.2e}" for H in (30, 50, 100))
    print(f"Macy {label}: {m3_per_t_tick:.3g} m3/t/tick -> a on copepods (km2/t/tick) {out}")
# 2. Engelhard et al. 2008: whiting per-capita intake elasticity 0.08 vs local sandeel density.
#    Under Type II the elasticity is 1/(1 + x), x = a h B.
e = 0.08
x = 1 / e - 1
for B in (1.17, 3.33):
    print(f"Engelhard elasticity {e}: x = {x:.1f}; with h 279.4 and B_vis {B} t/km2 -> a = {x / (279.4 * B):.3g}")
# 3. Lopez-Lopez et al. 2015: hake fullness rises from 100-1000 blue whiting per km2 (Type III threshold).
for n in (100, 1000):
    for w in (0.05, 0.1):
        K = n * w / 1000.0
        print(f"Hake threshold {n} ind/km2 x {w * 1000:.0f} g = {K:.3g} t/km2; read as 1/(a h) with h 279.4 -> a ~ {1 / (279.4 * K):.3g}")
# 4. Feeding level: group ration 0.49 of Cmax. If one pair alone gave it at full-time foraging,
#    x/(1+x) = 0.49 -> x = 0.96, a = 0.96/(h B).
x49 = 0.49 / 0.51
for name, h, B in (("planktivores, B_vis 1.17-3.33", 279.4, (1.17, 3.33)), ("zooplankton, B 0.5-9.72", 176.5, (0.5, 9.72))):
    print(f"Feeding level 0.49 from {name}: a = {x49 / (h * B[1]):.3g}-{x49 / (h * B[0]):.3g}")
