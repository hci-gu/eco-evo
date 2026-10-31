"""Whiting -> mobile_epifauna attack rate with the whiting's vertical overlap with the bed layer.

Usage: python3 -I epifauna_overlap.py

The section 220/223.15 a (0.27 per tonne of whiting, x e 0.39 inside) is the 3-D encounter of a whiting
that searches INSIDE the 0.5-m near-bed prey layer (H_bed_inv) for all of its foraging time. The same
library describes whiting as living in a 5-20 m (central 10 m) layer above the bed (H_whiting, used for
the water column's and the benthivores' pairs ON whiting; research_notes/"Rovfiskens uppdelning"
scripts/split_encounter_223.py). For prey confined to the bottom H_b and predators spread over H_w > H_b,
the encounter integral (N/H_w)(B/H_b)H_b = N B / H_w, against N B / H_b in the 220 form: the a is
multiplied by the overlap H_b / H_w. (When 2R <= H_b the corrected a no longer depends on H_b at all:
it is scan pi R^2 v_rel T N_t p / H_w.) Stage values from split_encounter_223_stdout.txt lines 153-154
(central, annual; per tonne of each stage, already x e 0.39): juvenile (< 20 cm) 0.6064, adult 0.1574.
"""

A_J, A_A = 0.6064, 0.1574
F_ACC = 0.29
H_EPI, B_EPI = 119.7, 3.78


def group_a(wj, ov_j, ov_a):
    return wj * A_J * ov_j + (1 - wj) * A_A * ov_a


print("Group a (km2 per t whiting per 6-h tick) = w_J a_J ov_J + (1 - w_J) a_A ov_A")
print(f"  library: w_J 0.247 (section 220 stage table), no overlap: {group_a(0.247, 1, 1):.4f} (library 0.27)")
print(f"  w_J 0.165 (box J/B target, section 230), no overlap:      {group_a(0.165, 1, 1):.4f}")
print(f"  juvenile share of the epifauna a at w_J 0.247: {0.247 * A_J / group_a(0.247, 1, 1):.2f}; at 0.165: "
      f"{0.165 * A_J / group_a(0.165, 1, 1):.2f}\n")
rows = []
for label, hb, hw_a, hw_j in [
    ("central: H_b 0.5, H_w 10 (both stages)", 0.5, 10.0, 10.0),
    ("H_w 5 (whiting hug the bed)", 0.5, 5.0, 5.0),
    ("H_w 20", 0.5, 20.0, 20.0),
    ("juveniles pelagic: H_w adult 10, juvenile 30", 0.5, 10.0, 30.0),
    ("H_w 5, juveniles 10", 0.5, 5.0, 10.0),
]:
    a = group_a(0.165, hb / hw_j, hb / hw_a)
    x = a * F_ACC * H_EPI * B_EPI
    rows.append((label, a, x))
    print(f"  {label:46s} a {a:.4f}  x = a f h B {x:5.2f}  share of ceiling {x / (1 + x):.2f}  "
          f"half-saturation 1/(a f h) {1 / (a * F_ACC * H_EPI):.2f} t/km2 (pool 3.78)")
print("\nThe a_enc factors themselves (R_bed_inv, p_bed_inv, speed, scan) keep their own MC spread x0.35-4.5"
      " (0.095-1.21 around 0.268, split_encounter_223_stdout.txt); the field clearance test (clearance_test.py)"
      " brackets the product at 0.006-0.023.")
