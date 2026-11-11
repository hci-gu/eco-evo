"""SMS North Sea key run 2023 (ICES WGSAM; files on disk in research_notes/"Fiskgrupper i rutan"/predators/data/
population): biomass flow from each predator to cod and haddock by prey age, 2001-22, split into the model's
juvenile (J, < 27 cm) and adult (A) compartments with the SMS mean length at sea (Lsea) per age and quarter and a
normal length spread (CV 0.10 [J]). Also the predators SMS has: hake eats no cod or haddock there; anglerfish, ling,
spurdog are not SMS predators.

flow(pred, prey, age, q) = Part.M2(q) x N.bar(q) x west(q)   (SMS: N in thousands, west in kg -> tonnes)

Usage: python3 -I sms_adult_share.py <dir with sms2023_partial_m2.out, sms2023_summary.out, sms2023_species_names.in>
"""
import math
import os
import sys
from collections import defaultdict

D = sys.argv[1]
names = [l.strip().strip("_") for l in open(os.path.join(D, "sms2023_species_names.in"))][:27]
summ = {}
hdr = None
for line in open(os.path.join(D, "sms2023_summary.out")):
    p = line.split()
    if not p:
        continue
    if p[0] == "Year":
        hdr = p
        continue
    r = dict(zip(hdr, p))
    summ[(int(r["Year"]), int(r["Quarter"]), int(r["Species.n"]), int(r["Age"]))] = r
CUT = {16: 271.0, 18: 270.0}


def adult_frac(L_mm, cut, cv=0.10):
    if L_mm <= 0:
        return 0.0
    z = (math.log(cut) - math.log(L_mm)) / cv
    return 0.5 * math.erfc(z / math.sqrt(2))


flow = defaultdict(float)
for line in open(os.path.join(D, "sms2023_partial_m2.out")):
    p = line.split()
    if not p or not p[0].isdigit():
        continue
    y, q, pr, pa, py, ya, m2 = int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), float(p[6])
    if not (2001 <= y <= 2022) or py not in CUT:
        continue
    r = summ.get((y, q, py, ya))
    if r is None:
        continue
    t = m2 * float(r["N.bar"]) * float(r["west"])
    fa = adult_frac(float(r["Lsea"]), CUT[py])
    flow[(pr, py, "A")] += t * fa
    flow[(pr, py, "J")] += t * (1 - fa)
print("Mean length at sea (mm) by age and quarter, 2001-22 mean:")
for py in CUT:
    for a in range(0, 4):
        ls = []
        for q in (1, 2, 3, 4):
            v = [float(summ[(y, q, py, a)]["Lsea"]) for y in range(2001, 2023) if (y, q, py, a) in summ]
            ls.append(sum(v) / len(v) if v else float("nan"))
        print(f"  {names[py - 1]:8s} age {a}: " + " / ".join(f"{x:.0f}" for x in ls))
print("\nFlow to cod and haddock by predator, t/yr 2001-22 mean, J (< 27 cm) / A (>= 27 cm); A share")
tot = defaultdict(float)
for (pr, py, st), v in flow.items():
    tot[(pr, st)] += v / 22
for pr in sorted(set(k[0] for k in tot)):
    j, a = tot[(pr, "J")], tot[(pr, "A")]
    if j + a < 1:
        continue
    print(f"  {names[pr - 1]:12s} J {j:9.0f}  A {a:9.0f}  A share {a / (j + a):.3f}"
          + "   [" + ", ".join(f"{names[py - 1]} A {flow[(pr, py, 'A')] / 22:.0f} J {flow[(pr, py, 'J')] / 22:.0f}" for py in CUT) + "]")
print("\nHake as a predator in SMS: prey species", sorted(set(names[py - 1] for (pr, py, st) in flow if pr == 15)) or "none among cod/haddock")
