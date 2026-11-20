"""Mechanism sizes on the model's feeding-level scale (fraction of the model ceiling), and field
reserve series expressed as fills. Research only; all inputs are field numbers from lit_*.md or
earlier notes and library constants. Run: python3 -I scripts/mechanism_sizes.py
"""
import math

CEIL_WHITING = 125.4            # kJ/kg/d, model whiting ceiling at 7.4 C (31.34 MJ/t/tick)
NEED = {"whiting": (0.275, 0.388), "water_column": (0.350, 0.435)}   # fill 0.1 .. 1.0 (model_side.py s.2)
UNCAPPED = {"whiting": 0.664, "water_column": 0.731}                 # fix104, yardstick only

print("1. Andersen & Riis-Vestergaard whiting Wmax vs the model ceiling")
w120, w250 = 165.3, 136.0       # kJ/kg/d at 7.17 C (A_ration_over_cmax_stdout.txt)
b = math.log(w250 / w120) / math.log(250 / 120)
for m in (120, 180, 250):
    wm = w120 * (m / 120) ** b * math.exp(0.075 * (7.4 - 7.17))
    print(f"  {m} g: Wmax {wm:.1f} kJ/kg/d at 7.4 C = {wm / CEIL_WHITING:.2f} x model ceiling")
R = 150.6 / CEIL_WHITING        # 180 g, used below
print(f"  conversion used: 1 Wmax = {R:.2f} model ceilings (180 g)\n")

print("2. Mechanism 2: per-prey-group plateaus (Rindorf & Gislason 2005 [F]) on the model scale")
groups = {"clupeids (planktivores)": (0.10, 0.17), "gadoids (benthivore juveniles)": (0.06, 0.12),
          "other food (krill, epifauna, copepods)": (0.08, 0.22)}
lo = hi = 0.0
for k, (a, c) in groups.items():
    print(f"  {k:40s} {a:.2f}-{c:.2f} Wmax = {a*R:.2f}-{c*R:.2f} of the model ceiling")
    lo += a; hi += c
print(f"  all three saturated and additive: {lo:.2f}-{hi:.2f} Wmax = {lo*R:.2f}-{hi*R:.2f} of the ceiling")
print(f"  observed total (5 stations x 3 sizes): 0.14-0.35 Wmax = {0.14*R:.2f}-{0.35*R:.2f}")
print(f"  'other food' is an OBSERVED intake range (8-22 % of Wmax), not a fitted plateau: krill alone <= {0.22*R:.2f} only if it behaves like the fish plateaus [J]")
print(f"  whiting need band (fill 0.1-1): {NEED['whiting'][0]:.3f}-{NEED['whiting'][1]:.3f}; "
      f"uncapped today {UNCAPPED['whiting']:.3f}\n")

print("3. Dempster et al. 2011 [F]: wild vs farm-associated saithe (summer, coastal Norway)")
s_ratio = 20.2 / 8.6
for beta, lab in ((1.0, "linear evacuation"), (0.5, "square-root evacuation (Andersen)")):
    i_ratio = s_ratio ** beta
    print(f"  stomach content ratio {s_ratio:.2f}; {lab}: intake ratio {i_ratio:.2f} -> wild = "
          f"{1/i_ratio:.2f} of the farm-fed intake")
print(f"  if farm-fed saithe feed near their potential, wild potential = 0.43-0.65 of it; on the water")
print(f"  column's scale (farm fish at the model ceiling, J): 0.43-0.65 vs need band "
      f"{NEED['water_column'][0]:.3f}-{NEED['water_column'][1]:.3f} and uncapped {UNCAPPED['water_column']:.3f}\n")

print("4. Field reserve series as fills, library convention (Svaltgrans och startreserv/model/derive_u.py):")
print("   fill = HSI x liver lipid share x 39.5 MJ/kg / ME 2000 MJ/t  (= HSI% x lipid x 0.1975)")
def fill(hsi, lip):
    return hsi * lip * 39.5 * 10 / 2000
rows = [("saithe NIFES Norw. Sea Jan-Mar (61 cm) [F via Svaltgrans]", 5.6, 0.57),
        ("saithe NIFES Apr-Jun (56 cm)", 6.1, 0.56), ("saithe NIFES Sep (64 cm)", 7.3, 0.71),
        ("saithe NIFES Oct-Dec (43 cm, box size)", 4.0, 0.60),
        ("saithe Mathers 1992 Aug 1989 (41-43 cm, box size; lipid 0.6 J)", 5.4, 0.60),
        ("saithe Mathers 1992 Aug 1989, second group", 5.95, 0.60),
        ("saithe Dempster 2011 wild summer low (lipid 0.6 J)", 5.2, 0.60),
        ("saithe Dempster 2011 wild summer high", 6.0, 0.60),
        ("saithe Dempster 2011 farm-fed", 9.2, 0.60),
        ("saithe Uglem 2020 control spring (62-69 cm; lipid 0.6 J)", 6.2, 0.60),
        ("saithe Uglem 2020 control autumn", 6.85, 0.60)]
for lip, lab in ((0.50, "0.50 J"), (0.60, "0.60 J")):
    rows += [(f"whiting Black Sea F Nov-Apr (lipid {lab})", 5.95, lip),
             (f"whiting Black Sea F May-Aug (lipid {lab})", 2.7, lip),
             (f"whiting Black Sea F Sep-Oct (lipid {lab})", 4.1, lip),
             (f"whiting Black Sea M Aug min (lipid {lab})", 1.85, lip)]
for n, h, l in rows:
    print(f"  {n:62s} HSI {h:5.2f} lipid {l:.2f} -> fill {fill(h, l):.2f}")
print("  Reading: on the library's own convention the saithe cycle is ~0.63 (Q1) -> ~1.0 (Sep) in the")
print("  Norwegian Sea, ~0.62-0.81 in coastal controls, 0.47-0.71 at the box's size class; farm-fed")
print("  saithe exceed the library maximum (1.1). Whiting (Black Sea; lipid share unknown, J) ~0.27-0.32")
print("  in May-Aug, 0.41-0.49 Sep-Oct, 0.59-0.71 Nov-Apr: the haddock proxy's 0.30 -> 0.73 is in range.")
