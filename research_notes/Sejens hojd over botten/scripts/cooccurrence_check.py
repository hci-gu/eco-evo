"""The gate's WC -> benthivores prey-side ratio with the juvenile part scaled by the field co-occurrence of adult saithe
and in-gape juvenile haddock (gov_juv_overlap.py), the adult part left as the gate has it. A CHECK-side correction:
budget_gate places the benthivores (and so their juveniles, a fixed share of every cell) uniformly, while the
booked share comes from stomachs of saithe caught where saithe are. Nothing here enters a or the library.

Usage: python3 -I cooccurrence_check.py <gate_hs.json>
corrected = ratio x (share_J x F + (1 - share_J)), share_J = JB f_J / (JB f_J + (1 - JB) f_A), JB 0.15.
F readings: box class-level presence-weighted 0.225; Skagerrak/Trench class-level 0.490; haul-level box 0.086,
Skagerrak/Trench 0.296 (gov_juv_overlap_stdout.txt).
"""
import json
import sys

G = json.load(open(sys.argv[1]))
FS = {"box class": 0.225, "wide class": 0.490, "box haul": 0.086, "wide haul": 0.296}
JB = 0.15
print(f"{'run':24s} {'form':6s} {'ratio':>6s} {'J share':>8s} " + " ".join(f"{k:>11s}" for k in FS))
for run in ("library", "combined_central", "exp_Hs10", "exp_Hs20", "exp_Hs40", "gamma2_s6", "juv_H0.5", "juv_H0.5_had",
            "exp_Hs10_juvH0.5_had", "gamma2_s6_juvH0.5_had"):
    fj, fa = G["edits"][run]["predatory_fish_preys_on_benthivorous_fish"]["stage_access"]
    sj = JB * fj / (JB * fj + (1 - JB) * fa)
    for form, sec in (("equal", "prey_pair"), ("field", "diet_field")):
        r = G["gate"][run][sec]["predatory_fish->benthivorous_fish"][0]
        print(f"{run:24s} {form:6s} {r:6.2f} {sj:8.2f} " + " ".join(f"{r * (sj * F + 1 - sj):11.2f}" for F in FS.values()))
print("\n'field' is the predator-side encounter/booked share at the field spread (saithe placed as in GOV, benthivores"
      " uniform): the like-for-like form for the co-occurrence factor. 'equal' is the prey-side ratio at equal cells,"
      " shown for scale only (a uniform predator meets the mean juvenile density, F = 1).")
