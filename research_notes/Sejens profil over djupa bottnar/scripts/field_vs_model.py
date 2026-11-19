"""Field test by bed class: adult-saithe stomach occurrence of krill, Norway pout and juvenile gadoids against what the
water column's class a (library and variants) predicts at the field prey densities.

Usage: python3 -I field_vs_model.py <saithe_stomach_bed.csv> <class_terms.json> <class_diet.json>

Field side (per class, relative to 100-150 m = 1):
  occurrence index  -ln(1 - FO), FO = share of fed predator records holding the prey (rectangle-spread 1980s records,
                    NE North Sea + Skagerrak and all N of 57 N; Q3 and all quarters) [D]
  prey density      NS-IBTS GOV kg/h by bed class in the box, 2000-25, read from research_notes/"Sejens djuplage"
                    (a-umlaut)/tables/gov_prey_overlap_stdout.txt [F data, D there]: pout, haddock < 25 cm + whiting
                    < 25 cm. Krill has no field density by bed class: the library's open-pool depth shape (class means
                    2.08 / 5.02 / 6.11 / 7.12 t/km2, research_notes/"Vattenpelarens par per djupklass") is used [model
                    input].
  clearance index   occurrence index / prey density.
Model side: a(c)/a(100-150) from class_terms.json (krill pair; the planktivore pair's pout term; the benthivore pair
for juvenile gadoids), and the predicted intake ratio a x D x (1 + S_ref)/(1 + S_c), S the policy-free saturation
of class_diet.json at the start densities (shared handling: a saturated predator's intake of one prey is
a B / (1 + S)); "no sat" omits that factor. The occurrence index is ~ proportional to intake only while FO is
well below 1 and records are single stomachs; the 1980s records are pooled samples, so it is a weak, compressed
measure [J]. Capture bed is not feeding bed (a saithe moves tens of km a day; gastric evacuation 1-3 days), which
pulls every field ratio towards 1 [J].
"""
import json
import math
import sys

import numpy as np
import pandas as pd

O = pd.read_csv(sys.argv[1])
T = json.load(open(sys.argv[2]))
D = json.load(open(sys.argv[3]))
CLS = ["<100", "100-150", "150-250", ">250"]
GOV = {  # box kg/h by class, Q3 and Q1 (gov_prey_overlap_stdout.txt)
    ("pout", "Q3"): [73.5, 111.3, 60.4, 16.3], ("pout", "Q1"): [5.0, 2.7, 22.5, np.nan],
    ("juv gadoids", "Q3"): [55.9 + 265.2, 16.2 + 22.6, 2.6 + 1.0, 0.0 + 0.2],
    ("juv gadoids", "Q1"): [29.8 + 71.5, 9.3 + 20.5, 7.3 + 1.4, np.nan],
}
KRILL = [2.08, 5.02, 6.11, 7.12]
MODEL = {"krill": "krill", "pout": "plk[pout]", "juv gadoids": "predatory_fish->benthivorous_fish"}
VARS = ["exp15|A", "expHsz|A", "freerange|A", "exp15|Ke2e-4", "exp15|kdeep0.05", "exp15|kdeep0.09"]
DIETVAR = {v: v for v in VARS}


def fidx(fo):
    return -math.log(max(1 - fo, 1e-6))


for aset in ("NE", "N57"):
    for season in ("Q3", "all quarters"):
        d = O[(O.kind == "spread") & (O.areaset == aset) & (O.season == season) & (O.Lc == "all")].set_index("cls")
        if "100-150" not in d.index:
            continue
        print("=" * 110)
        print(f"{aset}, {season}: stations " + " / ".join(f"{int(d.loc[c, 'stations']) if c in d.index else 0}" for c in CLS)
              + "; ratios to 100-150 m for < 100 / 150-250 / > 250 m; prey densities: GOV Q3 in every row (Q1 pout ratio 150-250 / 100-150 is 8.3)")
        print("=" * 110)
        gq = "Q3" if season == "Q3" else "Q3"
        for prey in ("krill", "pout", "juv gadoids"):
            fo = [float(d.loc[c, f"fo_{prey}"]) if c in d.index else np.nan for c in CLS]
            fi = np.array([fidx(x) if np.isfinite(x) else np.nan for x in fo])
            dens = np.array(KRILL if prey == "krill" else GOV[(prey, gq)], float)
            fr = fi / fi[1]
            dr = dens / dens[1]
            print(f"  {prey:12s} FO " + " / ".join(f"{x:.2f}" for x in fo)
                  + f" | occurrence index ratio " + " / ".join(f"{fr[k]:.2f}" for k in (0, 2, 3))
                  + f" | density ratio " + " / ".join(f"{dr[k]:.2f}" for k in (0, 2, 3))
                  + f" | field clearance ratio " + " / ".join(f"{fr[k] / dr[k]:.2f}" for k in (0, 2, 3)))
            for v in VARS:
                cl = np.array(T["runs"][v][MODEL[prey]]["classes"])
                ar = cl / cl[1]
                S = np.array(D[v]["S"])
                sat = (1 + S[1]) / (1 + S)
                pred = ar * dr * sat
                print(f"      model {v:16s} a ratio " + " / ".join(f"{ar[k]:.3f}" for k in (0, 2, 3))
                      + "  predicted intake ratio (no sat) " + " / ".join(f"{ar[k] * dr[k]:.2f}" for k in (0, 2, 3))
                      + "  (with sat) " + " / ".join(f"{pred[k]:.2f}" for k in (0, 2, 3)))
print("\nSaturation S = sum a h B per class (policy-free, start densities; class_diet.json):")
for v in ["library"] + VARS:
    print(f"  {v:16s} " + " / ".join(f"{x:.1f}" for x in D[v]["S"]))
