"""Q/B (annual consumption / mean biomass) from the ICES WGSAM North Sea SMS key run 2023.

Usage: python3 -I sms_qb.py <dir with NS2023_summary.out and NS2023_other_predators.out>

Units in SMS: N in thousands, west in kg, ration/consum in kg per individual per quarter.
Reads plain text only.
"""
import sys
from collections import defaultdict

d = sys.argv[1]
NAMES = {11: "W.horse.mac", 12: "N.horse.mac", 15: "Hake", 16: "Cod", 17: "Whiting",
         18: "Haddock", 19: "Saithe", 20: "Mackerel"}
PERIODS = [(1974, 2022), (2000, 2022), (2013, 2022)]

# rows: (sp, year, q, age, N, west, ration)
rows = []
with open(d + "/NS2023_summary.out") as f:
    h = f.readline().split()
    ix = {k: i for i, k in enumerate(h)}
    for line in f:
        p = line.split()
        sp = int(p[ix["Species.n"]])
        if sp not in NAMES or sp in (11, 12, 15):
            continue
        rows.append((sp, int(p[0]), int(p[1]), int(p[ix["Age"]]), float(p[ix["N.bar"]]),
                     float(p[ix["west"]]), float(p[ix["ration"]]), float(p[ix["prop.in"]])))
with open(d + "/NS2023_other_predators.out") as f:
    h = f.readline().split()
    for line in f:
        p = line.split()
        sp = int(p[0])
        if sp not in (11, 12, 15):
            continue
        n, w, c = float(p[4]), float(p[5]), float(p[6])
        if n <= 0 or w <= 0:
            continue
        rows.append((sp, int(p[1]), int(p[2]), int(p[3]), n, w, c, 1.0))

for sp in sorted(NAMES):
    print("=" * 70)
    print(NAMES[sp])
    for y0, y1 in PERIODS:
        R = [r for r in rows if r[0] == sp and y0 <= r[1] <= y1 and r[4] > 0 and r[5] > 0 and r[6] > 0]
        if not R:
            continue
        years = sorted(set(r[1] for r in R))
        ny = len(years)
        # aggregate
        C = defaultdict(float)   # (q) consumption t
        B = defaultdict(float)   # (q) biomass t
        Ca = defaultdict(float)  # (age) consumption over all quarters
        Ba = defaultdict(float)  # (age) sum over quarters of biomass
        nqa = defaultdict(set)
        rat = defaultdict(list)  # (age,q) ration
        wst = defaultdict(list)  # (age,q) west
        Cp = Bp = 0.0
        for sp_, y, q, a, n, w, c, pin in R:
            Cp += n * pin * c
            Bp += n * pin * w
            C[q] += n * c
            B[q] += n * w
            Ca[a] += n * c
            Ba[a] += n * w
            nqa[a].add(q)
            rat[(a, q)].append(c)
            wst[(a, q)].append(w)
        qs = sorted(C)
        Ctot = sum(C.values())
        # mean biomass over the 4 quarters of the year (absent quarters count as 0)
        Bmean4 = sum(B.values()) / 4.0
        # mean biomass over the quarters present
        Bmeanp = sum(B.values()) / len(qs)
        print(f"  {y0}-{y1} ({ny} y) quarters present {qs}")
        print("    by quarter: C/B per quarter = " + ", ".join(f"Q{q} {C[q]/B[q]:.3f}" for q in qs))
        print(f"    annual Q/B, biomass-weighted (sum C / mean B over 4 q): {Ctot/Bmean4:.2f} /yr")
        print(f"    Q/B weighted by the North Sea share (N x prop.in): {4*Cp/Bp:.2f} /yr")
        if len(qs) < 4:
            print(f"    Q/B while present, annualised (4 x sum C / sum B): {4*Ctot/sum(B.values()):.2f} /yr;"
                  f" sum C / mean B over present quarters: {Ctot/Bmeanp:.2f} per {len(qs)} q")
        print(f"    mean annual consumption {Ctot/ny/1000:.0f} kt; mean biomass (4q mean) {Bmean4/ny/1000:.0f} kt;"
              f" biomass share by age (%): " +
              ", ".join(f"{a}:{100*Ba[a]/sum(Ba.values()):.0f}" for a in sorted(Ba)))
        if (y0, y1) == (2000, 2022):
            print("    per age: west(kg, mean of q present) | annual ration kg/ind (sum over q present) |"
                  " Q/B_age (= 4 x mean_q ration/west) | biomass-weighted C/B (4 x sum C / sum B)")
            for a in sorted(Ba):
                aq = sorted(set(q for (aa, q) in rat if aa == a))
                mw = sum(sum(wst[(a, q)]) / len(wst[(a, q)]) for q in aq) / len(aq)
                ar = sum(sum(rat[(a, q)]) / len(rat[(a, q)]) for q in aq)
                qb = 4 * sum((sum(rat[(a, q)]) / len(rat[(a, q)])) / (sum(wst[(a, q)]) / len(wst[(a, q)]))
                             for q in aq) / len(aq)
                print(f"      age {a:2d} q{''.join(map(str,aq))}: W {mw:7.3f} | ration {ar:7.3f} | "
                      f"Q/B {qb:5.2f} | {4*Ca[a]/Ba[a]:5.2f}")
