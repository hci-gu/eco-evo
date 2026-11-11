"""One visual rule for the water column's krill and fish pairs: R_krill and R_fish at the box light.

Usage: python3 -I one_rule.py  (no inputs; prints a table; nothing is fitted to model output)

Light [F/D]: daytime medians of research_notes/"Ljus under 50 m" (section 253) at 100 and 150 m (best,
dark bound K_deep 0.09, bright bound K_deep 0.05), extended below 150 m with the same K_deep.

Rule (every quantity per predator, the same eye for both prey):
    R^2(E) = R_nv^2 (1 - g) + R_sat^2 g,   g = E / (E + Ke)               (Aksnes & Utne 1997 small-c limit)
Two ways to get R_sat per prey from ONE rule:
  A  mechanistic: R_sat^2 exp(c R_sat) = |C0| A E'   (A = 0.75 L x 0.2 L), E' from the library's
     k_v (R0 = k_v sqrt(A) at C0 = 1 -> E' = k_v^2), capped at cap_L x L_pred (section 220).
  B  empirical piscivore: R_sat(fish >= 50 mm) = RD50 (salmonid lab, size-free 55-139 mm, Vogel &
     Beauchamp 1999 via Hansen 2014), R_sat(krill) = RD50 (L_k / 50 mm)^b sqrt(C_k / C_f) with b from
     Hansen et al. 2013 (0.42 low light, 0.71 high light, prey 23-50 mm).
Non-visual floor: R_nv krill 0.1 m (library), fish 0.3 m [J] (0.2-0.5; Chinook low-light asymptote
~0.5 m, Rohan et al.; lateral line ~0.4-1 predator body length).
"""
import math

LIGHT = {  # lx, daytime median (best, dark K_deep 0.09, bright K_deep 0.05); 100/150 m from section 253
    100: (0.43, 0.15, 0.84),
    150: (0.022, 0.0017, 0.069),
}
for z in (200, 250):
    b, d, br = LIGHT[150]
    dz = z - 150
    LIGHT[z] = (b * math.exp(-0.06 * dz), d * math.exp(-0.09 * dz), br * math.exp(-0.05 * dz))

KE = {"central 0.02 lx (library Ke_gad)": 0.02, "sensitive 2e-4 lx (pollock, Ryer & Olla 1999) ": 2e-4,
      "insensitive 0.1 lx": 0.1, "salmonid 10 lx (Hansen/Vogel fits, upper bracket)": 10.0}
C_BEAM = 0.25
K_V = 100.0
PRED = {"saithe 43 cm": 0.43, "saithe 70 cm": 0.70}
CAP_L = 2.0


def r_mech(L, C0, Lpred):
    A = 0.75 * L * 0.2 * L
    rhs = C0 * A * K_V ** 2
    r = math.sqrt(rhs)
    for _ in range(200):
        r = math.sqrt(rhs * math.exp(-C_BEAM * r))
    return min(r, CAP_L * Lpred)


def r2(E, ke, rsat, rnv):
    g = E / (E + ke)
    return rnv ** 2 * (1 - g) + rsat ** 2 * g


C_K, C_F = 0.4, 0.7          # [J] |C0| krill (partly transparent, counter-illuminating) / fish silhouette
L_K = 0.030                  # m, M. norvegica adult
FISH = (0.10, 0.15, 0.25)    # m, prey fish lengths
RNV_K, RNV_F = 0.10, 0.30
RD50 = (1.0, 1.9)            # m, salmonid RDmax on 50 mm fish (lake trout 1.0; Chinook 1.22, cutthroat 1.87)
B_LO, B_HI = 0.415, 0.709    # Hansen et al. 2013 prey-size exponents (low / high light)

print("R_sat, rule A (mechanistic, k_v 100, c 0.25, cap 2 L_pred, |C0| krill 0.4 / fish 0.7):")
for pn, lp in PRED.items():
    rk = r_mech(L_K, C_K, lp)
    rf = [r_mech(L, C_F, lp) for L in FISH]
    print(f"  {pn}: krill 30 mm {rk:.2f} m; fish 10/15/25 cm " + " / ".join(f"{x:.2f}" for x in rf) + " m"
          f"  (uncapped fish 15 cm {r_mech(0.15, C_F, 99):.2f} m)")
print("R_sat, rule B (empirical): fish >= 50 mm RD50 1.0-1.9 m; krill 30 mm = RD50 x (30/50)^b x sqrt(0.4/0.7):")
for b in (B_LO, B_HI):
    f = (L_K / 0.05) ** b * math.sqrt(C_K / C_F)
    print(f"  b {b}: krill/fish R ratio {f:.2f}, R^2 ratio {f * f:.2f}; krill R_sat {RD50[0] * f:.2f}-{RD50[1] * f:.2f} m")

print("\nDaytime median light and R (m) at depth; saithe 43 cm, rule A, R_nv krill 0.10 / fish 0.30 m")
print("depth | E best (dark-bright) lx | Ke | g | A: R_krill | A: R_fish 15 cm | A: R2 ratio k/f | "
      "B: R_krill | B: R2 ratio k/f | library daytime R2 ratio (krill rule / fixed fish 0.2 m)")
FB = math.sqrt(((L_K / 0.05) ** B_LO) * ((L_K / 0.05) ** B_HI)) * math.sqrt(C_K / C_F)   # geometric mid of b
rf_b = min(1.2, CAP_L * 0.43)
rk_b = rf_b * FB
rk_s = r_mech(L_K, C_K, 0.43)
rf_s = r_mech(0.15, C_F, 0.43)
for z in (100, 150, 200, 250):
    eb, ed, ebr = LIGHT[z]
    for kn, ke in KE.items():
        g = eb / (eb + ke)
        R2k = r2(eb, ke, rk_s, RNV_K)
        R2f = r2(eb, ke, rf_s, RNV_F)
        # library: krill as this rule but k_v without contrast (R_sat 0.869 capped 0.86), fish fixed 0.2 m
        R2k_lib = r2(eb, ke, min(0.869, CAP_L * 0.43), RNV_K)
        R2kb = r2(eb, ke, rk_b, RNV_K)
        R2fb = r2(eb, ke, rf_b, RNV_F)
        print(f"{z:4d} m | {eb:.2g} ({ed:.2g}-{ebr:.2g}) | {kn[:14]:14s} | {g:.2f} | {math.sqrt(R2k):.2f} | "
              f"{math.sqrt(R2f):.2f} | {R2k / R2f:.2f} | {math.sqrt(R2kb):.2f} | {R2kb / R2fb:.2f} | {R2k_lib / 0.2 ** 2:.1f}")
print(f"rule B: fish R_sat = min(Chinook RDmax 1.2 m, 2 L_pred) = {rf_b:.2f} m, krill R_sat = {rk_b:.2f} m "
      f"(factor {FB:.2f} = (30/50)^b at the geometric mid of b x sqrt(0.4/0.7))")
print("night (E ~ 0): R_krill = R_nv 0.10, R_fish = 0.30 -> R2 ratio %.2f (library %.2f)" %
      (RNV_K ** 2 / RNV_F ** 2, RNV_K ** 2 / 0.2 ** 2))
