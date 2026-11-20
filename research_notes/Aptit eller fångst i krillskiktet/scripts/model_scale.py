"""Field whiting krill and total rations on the model's scale, and the size of each candidate mechanism.

Arithmetic only. No model run and no model output chooses a value: the yardstick (T, X split, need) is the
resume 289 decomposition quoted in research_notes/"Intag under fodosok" section 0, used only to size effects.
Usage: python3 -I model_scale.py

Model inputs (library, read-only):
- whiting ceiling 31.34 MJ/t/tick at the bottom temperature, 4 ticks/d -> 125.4 kJ/kg/d;
- krill E 5000 MJ/t (field 4.3 spring-summer, ~6 autumn-winter; research_notes/"Krillens flykt och
  energibudget" section 2), h_krill = E / ceiling = 159.5;
- net assimilation 0.80 - SDA 0.12 = 0.68 of ingested energy; cost 5.80 MJ/t/tick -> 0.272 of the ceiling
  (ingested basis), cost + P at fill 1 -> 0.388 ("Mattnad hos rovfisk" tables/model_side_stdout.txt).
Yardstick: uncapped potential = T X/(1+X), T 0.905, X 5.45 = krill 4.47 + other 0.98.
"""
import math

CEIL_TICK = 31.34
CEIL_D = CEIL_TICK * 4          # kJ/kg/d (MJ/t/d)
T, XK, XO = 0.905, 4.47, 0.98
NEED, COST = 0.388, 0.272


def pot(xk, xo=XO, t=T):
    x = xk + xo
    return t * x / (1 + x)


def level_from_pct(pct_bw_d, e_kj_g):
    """%BW/d of wet prey at e kJ/g -> fraction of the model ceiling (energy, ingested)."""
    return pct_bw_d / 100 * e_kj_g * 1000 / CEIL_D


print("=== A. Units ===")
print(f"ceiling {CEIL_TICK} MJ/t/tick = {CEIL_D:.1f} kJ/kg/d; krill-equivalent at 5.0 / 6.0 / 4.3 kJ/g: "
      f"{CEIL_D/50:.2f} / {CEIL_D/60:.2f} / {CEIL_D/43:.2f} %BW/d")
print(f"maintenance (cost only) {COST:.3f}; need (cost + P at fill 1) {NEED:.3f} of the ceiling")
print(f"krill alone at the yardstick X_k: T X/(1+X) = {T*XK/(1+XK):.3f}; other prey alone {pot(0):.3f}; both {pot(XK):.3f}")

print()
print("=== B. Field whiting on the model scale (fraction of the ceiling, energy; krill at 5 kJ/g, the E the GER")
print("    rations already assume; SMS totals at a diet E of 4.0-5.5 kJ/g [J]) ===")
rows = [
    # label, low %BW/d, high %BW/d, kJ/g low, kJ/g high, tag, note
    ("whiting Oslofjord Nov 2000, krill only", 0.23, 0.54, 5.0, 5.0, "D from F", "Onsrud 2004; GER 7 C; onsrud_three_months.py"),
    ("whiting Oslofjord Mar 2001, krill only", 0.65, 1.51, 5.0, 5.0, "D from F", "as above"),
    ("whiting Oslofjord Jun 2001, krill only", 0.20, 0.46, 5.0, 5.0, "D from F", "as above; thin, see evidence E1"),
    ("whiting Oslofjord Nov, total (krill 0.82 by mass)", 0.23 / 0.82, 0.54 / 0.82, 5.0, 5.0, "D", "krill/0.82, other prey at krill E [J]"),
    ("whiting Oslofjord Mar, total", 0.65 / 0.82, 1.51 / 0.82, 5.0, 5.0, "D", "as above"),
    ("whiting North Sea Q1 1981, krill only (counts)", 0.34, 0.52, 5.0, 5.0, "D", "Krillparen E4 upper bounds NE/NW"),
    ("whiting Skagerrak Q1 1981, krill only (counts)", 0.11, 0.16, 5.0, 5.0, "D", "Krillparen E4, 51 stomachs"),
    ("whiting SMS key run, North Sea Q1 total, 0.2 kg", 0.78, 0.78, 4.0, 5.5, "S", "WGSAM 2024 via Krillparen E3; North Sea Q1 T ~6-7 C"),
    ("whiting SMS key run, North Sea Q2 total", 0.83, 0.83, 4.0, 5.5, "S", "T ~7-9 C"),
    ("whiting SMS key run, North Sea Q3 total", 1.24, 1.24, 4.0, 5.5, "S", "T ~9-12 C: above the box bottom layer"),
    ("whiting SMS key run, North Sea Q4 total", 1.05, 1.05, 4.0, 5.5, "S", "T ~9-11 C"),
]
for lab, lo, hi, e1, e2, tag, note in rows:
    a, b = level_from_pct(lo, e1), level_from_pct(hi, e2)
    print(f"  {lab:52s} {lo:5.2f}-{hi:5.2f} %BW/d -> {a:4.2f}-{b:4.2f} of ceiling [{tag}] ({note})")
print("  On the GER Cmax scale (model ceiling = 1.2 GER Cmax for whiting, 'Krillintag och jaktsatt'):")
for lab, lo, hi in [("whiting krill-dominated stomachs >= 20 cm, low / central band", 0.38, 0.90),
                    ("whiting krill-dominated, weighed only", 0.41, 0.88),
                    ("whiting all stomachs, population krill ration, NE / NW North Sea Q1", 0.279, 0.306),
                    ("whiting Skagerrak Q2 (April 1991), population krill ration", 0.365, 0.365),
                    ("whiting North Sea Q2-Q4, population krill ration", 0.0, 0.084)]:
    print(f"  {lab:64s} GER {lo:4.2f}-{hi:4.2f} -> model {lo/1.2:4.2f}-{hi/1.2:4.2f} [D]")

print()
print("=== C. Mechanisms, each in its own model slot (yardstick only) ===")
print("C1. a-side factor f on the krill pair only (capture p, escape, vigilance, detection, patch occupancy,")
print("    local voids): potential T (XO + f XK)/(1 + XO + f XK); floor at f = 0 is", f"{pot(0):.3f} (> need {NEED})")
def krill_intake(f):
    return T * f * XK / (1 + XO + f * XK)


for f in (1.5, 1.0, 0.8, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0):
    print(f"    f {f:5.3f}: potential {pot(f*XK):.3f}, of which krill {krill_intake(f):.3f}")
for tgt in (0.55, 0.50, 0.47, 0.46):
    # solve T x/(1+x) = tgt -> x = tgt/(T - tgt)
    x = tgt / (T - tgt)
    f = (x - XO) / XK
    print(f"    target {tgt:.2f}: f = {f:.4f}" + ("  (impossible, below the floor)" if f < 0 else ""))
print("    Sized inputs: capture p 0.4-0.75 vs 0.5 -> f 0.8-1.5 (Krillens flykt) [S]; field-implied krill a,")
print("    library/field 76-203 on the shelf, 104-261 in the fjord (Krillparen, JANUARY, 100-150 m beds) -> f 0.004-0.013 [S];")
print("    patch occupancy p 0.77-0.9 (acoustic) or 0.1-0.3 (stomach) (Krillens flackighet) [S].")
for lab, f in [("field-implied a, f 0.004", 0.004), ("field-implied a, f 0.013", 0.013),
               ("stomach occupancy p 0.1", 0.1), ("stomach occupancy p 0.3", 0.3), ("acoustic p 0.77", 0.77)]:
    print(f"    {lab:30s} potential {pot(f*XK):.3f}, of which krill {krill_intake(f):.3f}")
print("    NOTE: the January f 0.004-0.013 is a local inversion on 100-150 m beds; on the box-aggregate X it gives a")
print("    krill intake of 0.01-0.03 of the ceiling, far below whiting's own field krill intake. Solve f instead from")
print("    the field krill intake I_k on this yardstick: f = I_k (1 + XO) / (XK (T - I_k)).")
print("    (Assumes the field krill intake is acquisition-limited; if appetite also acts, f is a lower bound.)")
# Field krill intake I_k (fraction of the model ceiling), derived, by region:
ns_q = {"Q1": (0.23, 0.26), "Q2": (0.0, 0.0), "Q3": (0.01, 0.01), "Q4": (0.006, 0.07)}   # North Sea population means [D]
ns_mean = (sum(v[0] for v in ns_q.values()) / 4, sum(v[1] for v in ns_q.values()) / 4)
osl = {"Nov": (0.09, 0.22), "Mar": (0.26, 0.60), "Jun": (0.08, 0.18)}                       # Oslofjord layer [D]
osl_mean = (sum(v[0] for v in osl.values()) / 3, sum(v[1] for v in osl.values()) / 3)
booked = 0.1873 * 5.0 / 0.446 * 5000 / (CEIL_D * 365)    # library M2 x B_krill / B_whiting, t/t/yr x E / ceiling
print(f"    booked field flow: M2 0.1873 x 5.0 t/km2 / 0.446 t/km2 = {0.1873*5.0/0.446:.2f} t/t/yr of krill"
      f" -> {booked:.3f} of the ceiling (booked in 245 from the in-area diet share 0.50 x field Q/B 4.2) [S/D]")
cells = [("North Sea Q2-Q4 population mean (not in-area)", 0.02),
         (f"North Sea annual, quarterly mean low (not in-area)", ns_mean[0]),
         (f"North Sea annual, quarterly mean high (not in-area)", ns_mean[1]),
         ("IN-AREA Skagerrak Q1 1981 (51 stomachs)", 0.05),
         (f"Oslofjord layer, Nov/Mar/Jun mean low", osl_mean[0]),
         ("IN-AREA booked field flow (diet share x Q/B)", booked),
         ("North Sea Q1 1981 population mean", 0.25),
         ("IN-AREA Skagerrak Apr 1991 (36 records)", 0.30),
         (f"Oslofjord layer, Nov/Mar/Jun mean high", osl_mean[1]),
         ("Oslofjord Mar alone, ~0.40", 0.40),
         ("krill-dominated stomachs, central 0.75 (selected fish)", 0.75)]
for lab, ik in cells:
    f = ik * (1 + XO) / (XK * (T - ik))
    print(f"    I_k {ik:5.3f} ({lab:52s}): f {f:6.3f}, potential {pot(f*XK):.3f}")
print("    Field refill-season bound on the potential ('Mattnad hos rovfisk'): 0.44-0.55.")

print()
print("C2. krill-specific ceiling c x the model ceiling (armour, slower evacuation per J, a per-prey plateau")
print("    written as handling): h_k -> h_k / c, so with shared handling and fixed attack (p = 1 on every prey)")
print("    level = T (XK + XO) / (1 + XK/c + XO): krill handling crowds out the other prey.")
for c in (1.0, 0.8, 0.67, 0.6, 0.5, 0.45, 0.42, 0.35, 0.25):
    lev = T * (XK + XO) / (1 + XK / c + XO)
    print(f"    c {c:4.2f}: level {lev:.3f}")
for tgt in (0.50, 0.45, NEED):
    c = XK / ((T * (XK + XO)) / tgt - 1 - XO)
    print(f"    target {tgt:.3f}: c = {c:.3f}")
print("    Sized inputs: whiting evacuate krill like fish of equal energy (Andersen 1999 [A]) -> c ~1; GER")
print("    armament M 1-1.5 -> c 0.67-1 [J]; the krill-dominated stomachs reach 0.32-0.75 of the model ceiling")
print("    on krill alone ('Krillintag och jaktsatt') -> a krill plateau below ~0.3 is contradicted centrally.")

print()
print("C3. exclusive foraging modes: share tau of the foraging time in the krill layer, 1 - tau on the bed and")
print("    fish prey (no krill met outside the layer, no bed prey inside it):")
print("    level = T [tau XK/(1+XK) + (1-tau) XO/(1+XO)]")
for tau in (1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.0):
    lev = T * (tau * XK / (1 + XK) + (1 - tau) * XO / (1 + XO))
    print(f"    tau {tau:3.1f}: level {lev:.3f}")
print("    Note [J]: the yardstick XK already carries the band overlap and the bed-depth access of the krill")
print("    pair, so a mode share stacked on it would count the vertical separation twice.")

print()
print("C4. appetite term (lipostatic / seasonal / light-cued): it caps REALISED intake, not the potential.")
print("    Under reserve_room the realised intake is already capped at the reserve room; an appetite term")
print("    would act earlier, below fill 1. Field winter whiting rations on this scale:")
for lab, lo, hi in [("SMS Q1 total 0.78 %BW/d at 4.0-5.5 kJ/g", level_from_pct(0.78, 4.0), level_from_pct(0.78, 5.5)),
                    ("Oslofjord Nov total", level_from_pct(0.23 / 0.82, 5.0), level_from_pct(0.54 / 0.82, 6.0)),
                    ("Oslofjord Mar total", level_from_pct(0.65 / 0.82, 4.3), level_from_pct(1.51 / 0.82, 6.0))]:
    print(f"    {lab:40s} {lo:4.2f}-{hi:4.2f} vs maintenance {COST:.3f}, need {NEED:.3f}")
days = 120
for cap in (COST, 0.30, 0.34):
    deficit = (NEED - cap) * CEIL_D * 0.68 * days        # MJ/t of net energy not gained vs the need
    print(f"    winter cap at {cap:.3f} for {days} d: {deficit:6.0f} MJ/t of net energy below the need (cost + P at"
          f" fill 1) = {deficit/2000:.2f} x the 2000 MJ/t reserve (whiting max_energy_reserve); at the cap = cost"
          f" this is production not paid, not a reserve loss")

print()
print("C5. krill escape / vigilance as local voids around predators (Onsrud 2004: 'voids' near deep fish")
print("    assemblages by day; Christiansen 2024: -68 % under schools > 10 m by day, ~0 at night): an a-side factor.")
for f in (0.32, 0.6, 0.88):
    for dshare in (0.5, 0.33):
        ff = dshare * f + (1 - dshare)
        print(f"    void factor {f:.2f} in a daytime share {dshare:.2f} -> f {ff:.2f}, potential {pot(ff*XK):.3f}")

print()
print("C6. spawning-bound intake cut (the library's whiting spawning entry, section 279): share 0.344 of the group,")
print("    fast_cut 0.5 of normal intake, 40 spawning days per fish, window Feb-Jun (8.0 h rising to 17.8 h falling).")
print("    In the engine it closes the reserve room (lib/world/spawning.py), so it binds only where the room binds.")
for days_in_quarter in (40, 20):
    cut = 0.344 * 0.5 * days_in_quarter / 91
    print(f"    {days_in_quarter} of a fish's spawning days inside one quarter: group intake x{1 - cut:.3f} in that quarter")
print("    Field, whiting: SMS Q1 / Q3 ration/Cmax 0.43 / 0.49 (x0.88); Robb et al. 1994, 1991 adults 200-400 mm:")
print("    Q1 0.42 vs Q2-Q4 0.78-0.83 (x0.52; x0.60 with regurgitated as empty); Hislop 1983, 1981: no between-quarter")
print("    difference in food weights. Mature adults only: 0.5 x (40-70 d)/91 -> x0.62-0.78 in the quarter [D].")
for d in (40, 70):
    print(f"    mature adults fasting at fast_cut 0.5 for {d} d of the quarter: x{1 - 0.5 * d / 91:.2f}; full fast: x{1 - d / 91:.2f}")
