#!/usr/bin/env python3
"""Independent field tests of the encounter-theory a versus the field-implied a.

Research diagnostic. Neither test datum below was used to derive the
field-implied rates (those come from field ration, diet shares and a
foraging-time fraction tau in field_implied_a.py).

Test 1 - density response at the km scale (Hiddink et al. 2016, Kattegat
[A]: food intake per plaice and the condition of plaice and long rough dab
fell above ~5 trawl passes/yr; dab showed no effect). Under a single-pair
Type II, the intake ratio when prey falls to a fraction q of today is
  R(q) = q (1 + x) / (1 + q x),  x = saturation index today.
A measurable intake/condition effect at moderate prey reduction requires
x of order 1-5; encounter theory gives x 87-266 (no effect until q < ~0.05).
Prey reduction at 5 passes/yr is estimated with the library's own trawl
mortality and r (AGENTS.md section 5: infauna d 0.06 per pass, so F = 0.3
/yr at 5 passes; r_max 1.0) as the logistic equilibrium q = 1 - F/r = 0.7
(J: whole-infauna value; the sensitive polychaete/crustacean prey of plaice
fall more, q ~0.5).

Test 2 - half-saturation of gadoids on fish prey at the field scale:
  * Rindorf & Gislason 2005, whiting on clupeids, pool-equivalent 0.57
    t/km2 (corner span 0.035-7.1), addendum_validation_behaviour.md;
  * MULTSPEC (Bogstad, Hauge & Ulltang 1997) cod on suitability-weighted
    food 1.57 t/km2, fitted on ~10^2 km areas (benthivores_encounter.md).
Compared with 1/(a h) (1 + w B) at the encounter a and at the field-implied a.
"""

print("Test 1: intake ratio R(q) when the prey falls to q of today")
xs = {"encounter infauna (x 266)": 266.0, "encounter epifauna, benthivores (x 87)": 87.0,
      "field-implied tau 0.5 (x 4.1)": 4.11, "field-implied tau 0.75 (x 1.2)": 1.16,
      "field-implied tau 1.0 (x 0.67)": 0.67}
for q in (0.7, 0.5, 0.3):
    print(f"  q = {q}: " + "; ".join(f"{k} {q * (1 + x) / (1 + q * x):.3f}" for k, x in xs.items()))
print("  Field: a measurable fall in food per plaice and condition above ~5 passes/yr (q ~0.5-0.7, J).")
print("  Encounter theory predicts a 0.1-0.6 % fall; the field-implied rates a 7-35 % fall.")
print()

print("Test 2: half-saturation of fish-prey pairs (t/km2 of the prey pool)")
pairs = [
    # label, h, a_enc, a_field(tau 1.0, 0.75, 0.5) from field_implied_a_stdout.txt, I = w B_pred
    ("predators -> planktivores", 279.4, 0.012, (0.0007, 0.0014, None), 0.1 * 1.194),
    ("benthivores -> planktivores", 262.9, 0.019, (0.0002, 0.0003, 0.0011), 0.45 * 1.528),
]
for label, h, ae, af, I in pairs:
    enc = (1 + I) / (ae * h)
    fld = ", ".join(f"tau {t}: {(1 + I) / (a * h):.1f}" for t, a in zip((1.0, 0.75, 0.5), af) if a)
    print(f"  {label}: encounter {enc:.2f}; field-implied {fld}")
print("  Field: Rindorf & Gislason 2005 whiting on clupeids 0.57 (0.035-7.1);")
print("         MULTSPEC cod on suitable fish food 1.57 (10^2 km areas, patchiness inflates it).")
print("  Reading: on fish prey the encounter a sits inside the field span (within 2-10x);")
print("  the field-implied a at tau 0.75-1 is too low there. The large gap is on the benthic pairs.")
