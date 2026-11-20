"""Arithmetic for research_notes/"Intag under fodosok" (o-umlaut). Standard library only.

Run:  python3 -I intake_arithmetic.py > ../tables/intake_arithmetic_stdout.txt

Sections
 1. Yardstick: the multiplier on X = sum a h B_vis that takes the model's uncapped potential
    (resume 289 decomposition, frozen fix105, used as a yardstick only) to 0.40-0.50 of the
    ceiling; fish-only versus krill-only versus generic factors.
 2. Pursuit / handling time per item against digestion time per item (Staniland 1995,
    Ellis & Gibson 1997, Fall & Fiksen 2020 bounds); breakeven handling time for a 10 % loss.
 3. Twilight window share at 58.5 N (sun between -12 and +6 deg), by month.
 4. Field intake levels on the model scale (Wmax = 1.10-1.34 model ceilings, "Mattnad hos
    rovfisk" M4).
 5. Forward combination of the independently sized factors (never back-solved).
"""
import math

# ---------------------------------------------------------------- 1. yardstick
# Resume 289 (fill_decomp.py, frozen fix105, 1 year, seeds 0/1): T = share of ticks foraging,
# sat = saturation while foraging = X/(1+X) under shared handling (fixed attack, p = 1 on the menu),
# krill = krill share of the uncapped handling time = X_krill / X.  Interference w B_pred ignored [J]
# (water column w 0.1 x 0.62 t/km2 = 0.06 in the denominator).
GROUPS = {
    "whiting":      dict(T=(0.90, 0.905, 0.91), sat=(0.83, 0.845, 0.86), krill=(0.80, 0.82, 0.84), need=0.388),
    "water column": dict(T=(0.97, 0.97, 0.97), sat=(0.78, 0.78, 0.78), krill=(0.46, 0.46, 0.46), need=0.435),
}
TARGETS = (0.40, 0.45, 0.50)


def x_of(s):
    return s / (1.0 - s)


def s_of(x):
    return x / (1.0 + x)


print("=== 1. Yardstick (289 decomposition; yardstick only) ===")
YARD = {}
for g, d in GROUPS.items():
    T, sat, kr = d["T"][1], d["sat"][1], d["krill"][1]
    X = x_of(sat)
    Xk, Xo = kr * X, (1 - kr) * X
    YARD[g] = (T, X, Xk, Xo)
    print(f"{g}: T {T:.3f}  sat {sat:.3f}  X {X:.2f}  (krill {Xk:.2f}, other {Xo:.2f})  uncapped {T*sat:.3f}  need {d['need']}")
    print(f"   all non-krill prey removed (factor 0): uncapped {T*s_of(Xk):.3f}")
    print(f"   krill removed (factor 0):              uncapped {T*s_of(Xo):.3f}")
    for tgt in TARGETS:
        s_t = tgt / T
        X_t = x_of(s_t)
        f_gen = X_t / X
        f_kr = (X_t - Xo) / Xk if X_t > Xo else float('nan')
        f_ot = (X_t - Xk) / Xo if X_t > Xk else float('nan')
        f_T = tgt / (T * sat)
        print(f"   target {tgt:.2f}: generic factor on every a {f_gen:.3f}; krill-only {f_kr:.3f}; "
              f"non-krill-only {'impossible' if f_ot != f_ot else f'{f_ot:.3f}'}; or a time share x{f_T:.2f}")
    # ranges over the 289 spans
    lo = min(d["T"][i] * s_of(x_of(d["sat"][i]) * d["krill"][i]) for i in (0, 2))
    hi = max(d["T"][i] * s_of(x_of(d["sat"][i]) * d["krill"][i]) for i in (0, 2))
    print(f"   krill alone over the 289 spans: {lo:.3f}-{hi:.3f}")
print()

# ---------------------------------------------------------- 2. handling vs digestion
print("=== 2. Pursuit/handling time per item vs digestion time per item ===")
# Ceilings (model, AGENTS.md / Mattnad M0): whiting 31.34 MJ/t/tick = 125.4 kJ/kg/d; water column
# 29.79 MJ/t/tick = 119.2 kJ/kg/d.
CEIL = {"whiting": 125.4, "water column": 119.2}       # kJ per kg per day
# Predator masses [J]: whiting 25 cm ~130 g, 30 cm ~210 g; saithe 45 cm ~0.85 kg; hake 40 cm ~0.5 kg.
PRED = [("whiting 25 cm", "whiting", 0.13), ("whiting 30 cm", "whiting", 0.21),
        ("saithe 45 cm", "water column", 0.85), ("hake 40 cm", "water column", 0.50)]
# Prey items: mean wet mass at ingestion in whiting stomachs (Rindorf 2003 Table 4) and library energy
# densities (kJ/g = MJ/t / 1000; planktivores lean 4.0 + reserve at fill ~0.4 of 4.99 -> ~6 [J]).
PREY = [("krill (Meganyctiphanes)", 0.12, 5.0), ("shrimp", 0.80, 3.75), ("juvenile whiting", 3.36, 4.8),
        ("herring", 8.26, 6.0), ("sprat", 9.27, 6.0), ("Thysanoessa-size krill [J]", 0.03, 5.0)]
# Handling time per captured item, seconds: Staniland 1995 whiting on shrimp (hunt median 1.9-2.5 s;
# handling median 0-4.3 s; tank, lit); Ellis & Gibson 1997 0-group cod on 0-group flatfish (pursuit +
# ingestion, least-square means 15.1 / 20.3 / 30.3 s for prey 1/2/3); Fall & Fiksen 2020 model bounds
# 3 s minimum, 10 min maximum.  A capture at per-encounter p 0.55 needs ~1.8 attempts; each miss adds
# a pursuit of ~1-2 s (Ellis & Gibson pursuit 1.0-2.0 s).
H_LOW, H_CEN, H_HIGH = 4.0, 30.0, 600.0   # s per capture, failures included
for name, grp, w in PRED:
    rate = CEIL[grp] * w                   # kJ per day at the ceiling
    for pname, m, e in PREY:
        c_s = m * e / rate * 86400.0       # digestion time per item at the ceiling, s
        loss = [h / (h + c_s) for h in (H_LOW, H_CEN, H_HIGH)]
        be = c_s / 9.0                     # handling time per item that would cost 10 % (additive disc)
        print(f"{name:14s} {pname:28s} item {m*e:6.2f} kJ  c {c_s/60:8.1f} min  "
              f"items/d at ceiling {86400/c_s:6.1f}  loss at b 4/30/600 s: "
              f"{loss[0]*100:.3f}/{loss[1]*100:.2f}/{loss[2]*100:.1f} %  breakeven(10%) {be/60:.1f} min")
print("Jeschke et al. 2002 SSS form: asymptote 1/max(b, c), so with b << c the loss is ~0, smaller")
print("than the additive disc figures above.")
print()

# ------------------------------------------------------------- 3. twilight window
print("=== 3. 'Moderate light' window at 58.5 N: sun between -12 and +6 deg (morning + evening) ===")
LAT = math.radians(58.5)


def hours_above(elev_deg, doy):
    decl = math.radians(-23.44) * math.cos(2 * math.pi * (doy + 10) / 365.0)
    h0 = math.radians(elev_deg)
    cosw = (math.sin(h0) - math.sin(LAT) * math.sin(decl)) / (math.cos(LAT) * math.cos(decl))
    if cosw <= -1:
        return 24.0
    if cosw >= 1:
        return 0.0
    return 2 * math.degrees(math.acos(cosw)) / 15.0


for mon, doy in zip(("Jan", "Mar", "May", "Jun", "Aug", "Oct", "Dec"), (15, 74, 135, 172, 227, 288, 349)):
    win = hours_above(-12, doy) - hours_above(6, doy)
    day = hours_above(-0.833, doy)
    print(f"{mon}: day length {day:5.1f} h; window (-12..+6 deg) {win:4.1f} h = {win/24:.2f} of the day; "
          f"share of lit+twilight hours {win/max(hours_above(-12, doy),1e-9):.2f}")
print()

# ------------------------------------------------------ 4. field levels on model scale
print("=== 4. Field intake while foraging, on the model ceiling scale (1 Wmax = 1.10-1.34 ceilings) ===")
W2C = (1.10, 1.20, 1.34)
rows = [("whiting, one fish-prey group plateau (Rindorf & Gislason 2005), Wmax share", 0.06, 0.13, 0.22)]
for lab, lo, ce, hi in rows:
    print(f"{lab} {lo:.2f}/{ce:.2f}/{hi:.2f} -> model scale "
          f"{lo*W2C[0]:.2f}/{ce*W2C[1]:.2f}/{hi*W2C[2]:.2f}")
print("whiting, 'other food' observed (krill, crustaceans; R&G 2005), model scale (Mattnad M4): 0.10/0.18/0.26")
print("whiting, three fish groups saturated + other food, model scale (Mattnad M4): 0.29/0.45/0.61")
print("largemouth bass at high prey density: ~0.5 Cmax (Essington et al. 2000 via R&G 2005) [S]")
print("whiting on krill, Oslofjord Nov/Mar: ration 0.1-0.7 of field Cmax (Krillparen mot vintermagar) [D from F]")
print("wild / farm-fed saithe intake 0.43-0.65 (Dempster 2011; Mattnad M4) [D]")
print()

# ------------------------------------------------------- 5. forward combination
print("=== 5. Forward combination of sized factors on a (low / central / high) ===")
# Each factor is independently sized in evidence.md; none is chosen to hit a target.
F = {
    # mechanism: (fish-prey factor, krill factor) at (low, central, high); same for both groups
    "1 pursuit/handling (acts on the ceiling side; <0.1 % here)": ((1, 1, 1), (1, 1, 1)),
    "2 field capture / pre-strike escape beyond p (q 0.6 vs 0.8; krill p 0.3 vs 0.5)": ((0.75, 1, 1), (0.6, 1, 1.4)),
    "4 twilight vulnerability window beyond F and the vision rule [J]": ((0.25, 0.5, 1), (0.5, 0.75, 1)),
    "5 satiation-independent (in Holling h under SSS; adding it double counts)": ((1, 1, 1), (1, 1, 1)),
    "6 interference / prey avoidance at km scale": ((0.9, 1, 1), (0.9, 1, 1)),
}
# 3 search speed, group-specific, on every pair of the group. The chain (Attackrater ur motesteori
# scripts/pred_encounter_a.py) uses whiting 0.8 BL/s (0.5-1.3), saithe 1.0 (0.5-1.4); in situ whiting in the
# krill layer swam ~0.6 BL/s, night ~0.7 of day (Onsrud et al. 2005 via addendum_validation_behaviour) [S].
SPEED = {"whiting": (0.6 * 0.7 / 0.8, 0.6 / 0.8, 1.0),   # 0.525 / 0.75 / 1
         "water column": (0.5, 1.0, 1.0)}                # in-situ gadoids mostly < 1 BL/s [S]; no saithe datum
for k in F:
    print(f"  {k}: fish {F[k][0]}, krill {F[k][1]}")
for g in SPEED:
    print(f"  3 in-situ search speed vs the chain, {g}: {tuple(round(x, 3) for x in SPEED[g])} on every pair")
for i, lab in enumerate(("low corner", "central", "high corner")):
    ff = kk = 1.0
    for v in F.values():
        ff *= v[0][i]
        kk *= v[1][i]
    print(f" {lab}: fish-prey factor {ff:.3f}, krill factor {kk:.3f} (before speed)")
    for g, (T, X, Xk, Xo) in YARD.items():
        sp = SPEED[g][i]
        Xn = (Xk * kk + Xo * ff) * sp
        print(f"    {g}: speed x{sp:.3f}; X {X:.2f} -> {Xn:.2f}; uncapped {T*s_of(X):.3f} -> {T*s_of(Xn):.3f} (need {GROUPS[g]['need']})")
print()
print("Field-implied krill a (Krillparen: library / field 50-260; JANUARY, 100-150 m beds, planktivores and whiting;")
print("applying it to the annual X and to the water column is an extrapolation) applied to krill only:")
for g, (T, X, Xk, Xo) in YARD.items():
    for r in (50, 100, 260):
        Xn = Xk / r + Xo
        print(f"    {g}: krill a / {r}: uncapped {T*s_of(Xn):.3f}")
