"""Flow-preserving annual a in the ENGINE's frame: the engine applies a(t) = a m(t) (daylight lever, annual mean 1),
so the a that reproduces the derived annual flow sum_t a_der(t) B_pred(t) B_prey(t) is

    a_engine = a_eq x sum_m der_m w_m / sum_m eng_m w_m,   w_m = prey presence x predator presence (month m)

(der_m = derived monthly a after schooling F, / its equal-time mean; eng_m = monthly mean m(t)). With w = 1 it is the
equal-time a (both series have mean 1). Reads ../tables/seasonal_pairs.json (written by seasonal_pairs.py); presence
tables as there (research_notes/"Bestandsreglering" migration.md section 4, fraction of August). Nothing fitted.

Usage: python3 -I flow_engine.py ../tables/seasonal_pairs.json > ../tables/flow_engine_stdout.txt
"""
import json
import sys

import numpy as np

D = json.load(open(sys.argv[1]))
PRES = {"herring": [0.60, 0.60, 0.60, 0.65, 0.80, 0.95, 1, 1, 0.85, 0.70, 0.60, 0.60],
        "predatory_fish": [0.12, 0.09, 0.13, 0.29, 0.54, 0.83, 1, 1, 0.99, 0.85, 0.54, 0.29],
        "mackerel": [0.01, 0.01, 0.02, 0.03, 0.10, 0.50, 1, 1, 0.90, 0.50, 0.10, 0.03],
        "whiting": [1.0] * 12}
# a lower true winter column presence of herring (if the GOV-based 0.6 carries the doubled near-bed availability):
PRES_LOW = [0.30, 0.30, 0.30, 0.45, 0.70, 0.95, 1, 1, 0.80, 0.55, 0.35, 0.30]
print("a_engine / a_equal-time (flow-preserving, engine frame) and, for contrast, the naive flow mean without m(t)")
for key, v in D["swing"].items():
    name, rn = key.split("|")
    pred = name.split("->")[0]
    der, eng = np.array(v["derived"]), np.array(v["engine"])
    for lab, ph in (("presence 0.6", PRES["herring"]), ("presence 0.3", PRES_LOW)):
        w = np.array(ph) * np.array(PRES[pred])
        f_eng = (der * w).sum() / (eng * w).sum()
        f_naive = (der * w).sum() / w.sum()
        print(f"  {name:36s} {rn[:12]:12s} herring winter {lab}: engine-frame x{f_eng:.3f}   naive x{f_naive:.3f}")
