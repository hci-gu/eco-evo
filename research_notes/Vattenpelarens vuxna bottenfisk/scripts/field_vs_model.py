"""Field flow of each water-column member onto juvenile (J) and adult (A) benthivores against the model's
encounter-implied flow per member, at the combined central with the recruit shape (gate baseline, field spread).

Usage: python3 -I field_vs_model.py <decompose_members.json> <gate_adult.json>

Model side: the pair's implied M2 (gate prey side, field spread) x the booked flow; split J / A by the encounter
flow at the field spread (member windows x J / A field co-occurrence F_eff from research_notes/"Rekryternas
bottendjup"/scripts/decompose_wc_pair_stdout.txt: J import shape 0.655, A annual 0.882), and by member with the
decomposition's J / A flow shares (decompose_members.json).
Field side: member biomass (library member weights x 0.617 t/km2) x Q/B x diet share on the model's benthivores,
split J / A. Every field input is listed in FIELD with its source (evidence.md E-numbers); ranges are carried.
Nothing is fitted.
"""
import json
import sys

DM = json.load(open(sys.argv[1]))
GA = json.load(open(sys.argv[2]))
B_WC, B_BEN, M2_BOOK = 0.617, 1.53, 0.0833
BOOK = M2_BOOK * B_BEN
W = {"saithe": 0.797, "spurdog": 0.097, "hake": 0.048, "anglerfish": 0.027, "ling": 0.014, "velvet belly": 0.009,
     "pollack": 0.008}
# (Q/B central, lo, hi), (benthivore diet share central, lo, hi), (A fraction of that share central, lo, hi)
FIELD = {
    "saithe": ((3.0, 2.87, 3.6), (0.05, 0.02, 0.09), (0.08, 0.0, 0.17)),       # E3 E4 E5; A: stomachs ~0, SMS 0.17
    "hake": ((3.5, 2.2, 5.8), (0.01, 0.0, 0.03), (0.0, 0.0, 0.1)),             # E1 E2 E3 E4 E11 E16
    "anglerfish": ((1.8, 1.5, 2.2), (0.19, 0.08, 0.34), (0.85, 0.5, 0.95)),    # E1 E2 E6 E11
    "ling": ((2.0, 1.5, 3.0), (0.10, 0.03, 0.20), (0.5, 0.2, 0.9)),            # E1 E11 E12 E13; Q/B and A [J]
    "spurdog": ((2.0, 1.5, 2.5), (0.06, 0.02, 0.15), (0.85, 0.6, 1.0)),        # E1 E2 E6
    "pollack": ((3.0, 2.5, 3.6), (0.0, 0.0, 0.02), (0.0, 0.0, 0.5)),           # E1
    "velvet belly": ((2.0, 1.5, 3.0), (0.0, 0.0, 0.02), (0.0, 0.0, 0.5)),      # [J]
}
r = GA["summary"]["baseline"]["prey"]["benthivorous_fish"]["predatory_fish"]
ratio_field = r[2] / r[0]
tot = ratio_field * BOOK
FJ, FA = DM["FJ"], DM["FA"]          # encounter flows at uniform prey (library windows, member shares)
FJe, FAe = FJ * 0.655, FA * 0.882
jshare = FJe / (FJe + FAe)
print(f"Model pair flow at the field spread: implied/booked {ratio_field:.2f} x booked {BOOK:.4f} = {tot:.4f} t/km2/yr;"
      f" J share {jshare:.2f} (uniform prey {FJ / (FJ + FA):.2f})")
mJ, mA = tot * jshare, tot * (1 - jshare)
print(f"  J {mJ:.4f}, A {mA:.4f} t/km2/yr (M2 on J {mJ / (B_BEN * DM['JS']):.3f}, on A {mA / (B_BEN * (1 - DM['JS'])):.4f} /yr)")
print(f"\n{'member':13s} {'B t/km2':>8s} {'Q t/km2/yr':>11s} | {'model J':>8s} {'model A':>8s} {'A/Q':>6s} | "
      f"{'field J':>15s} {'field A':>15s} | {'A model/field':>13s}")
fjt = fat = 0.0
fjl = fal = fjh = fah = 0.0
for sp, w in W.items():
    b = w * B_WC
    (qb, qbl, qbh), (d, dl, dh), (fa, fal_, fah_) = FIELD[sp]
    q = b * qb
    fl = DM["flows"].get(sp, dict(J_flow=0, A_flow=0))
    mj, ma = mJ * fl["J_flow"], mA * fl["A_flow"]
    fA = q * d * fa
    fJ = q * d * (1 - fa)
    lo_A, hi_A = b * qbl * dl * fal_, b * qbh * dh * fah_
    lo_J, hi_J = b * qbl * dl * (1 - fah_), b * qbh * dh * (1 - fal_)
    fjt += fJ
    fat += fA
    fjl += lo_J
    fal += lo_A
    fjh += hi_J
    fah += hi_A
    rat = f"{ma / fA:13.1f}" if fA > 1e-6 else ("     inf" if ma > 1e-6 else "       -")
    print(f"{sp:13s} {b:8.4f} {q:11.4f} | {mj:8.4f} {ma:8.4f} {ma / q:6.2f} | {fJ:6.4f} ({lo_J:.4f}-{hi_J:.4f}) "
          f"{fA:6.4f} ({lo_A:.4f}-{hi_A:.4f}) | {rat}")
print(f"{'total':13s} {'':8s} {'':11s} | {mJ:8.4f} {mA:8.4f} {'':6s} | {fjt:6.4f} ({fjl:.4f}-{fjh:.4f}) "
      f"{fat:6.4f} ({fal:.4f}-{fah:.4f}) | {mA / fat:13.1f}")
print(f"\nField J + A {fjt + fat:.4f} t/km2/yr ({fjl + fal:.4f}-{fjh + fah:.4f}) vs booked {BOOK:.4f} and model {tot:.4f}")
print(f"Field A share of the field flow {fat / (fjt + fat):.2f}; model A share {1 - jshare:.2f}")
print(f"Model A as a multiple of the field A: {mA / fat:.1f} (range {mA / fah:.1f}-{mA / fal if fal > 0 else float('inf'):.1f})")

print("\nPer-member A flow in the corrected gate runs (same split; the corrections' own windows):")
for run in ("stage_p+Aprof", "pA+Aprof+hake_out", "pA+Aprof+hake_out+ling0.40", "pA+Aprof+hake_out+ling_out",
            "bracket_no_bed_A"):
    rr = GA["summary"][run]["prey"]["benthivorous_fish"]["predatory_fish"]
    ed = GA["edits"][run]
    fj, fa = ed["stage_access"]
    js = DM["JS"]
    eJ, eA = js * fj * 0.655, (1 - js) * fa * 0.882
    t = rr[2] / rr[0] * BOOK
    print(f"  {run:30s} field-spread ratio {rr[2] / rr[0]:.2f}: J {t * eJ / (eJ + eA):.4f}, A {t * eA / (eJ + eA):.4f} t/km2/yr"
          f" (A / field A {t * eA / (eJ + eA) / fat:.1f}); J / field J {t * eJ / (eJ + eA) / fjt:.1f}")
