"""Daytime light and the rule-A vision factor g = E/(E + Ke) at the bed and 15 / 50 m above it, per bed class, on
the section 253 measured light (the light the class tables were derived with).

Usage: python3 -I light_at_bed.py <repo_root> <scratch_dir>

research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py is exec'd unchanged up to its section 0 block (trans_month,
E0, the time rows, the bed bins and cell weights come from there). Light below ~65 m is the judged K_deep (0.06;
0.05-0.09), not a measurement ("Ljus under 50 m"). Also the day-mean R^2 for a 50 cm saithe on a 15 cm fish:
R^2 = R_nv^2 (1 - g) + R_sat^2 g, R_nv 0.3 m, R_sat = min(R0, 2 L) = 1.0 m.
"""
import contextlib
import io
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
BP = os.path.join(ROOT, "research_notes", "Overlapp i fiskparen", "scripts", "band_pairs.py")
src = open(BP).read()
head = src[:src.index("# ================================================================== 0. overlap geometry")]
ARGV = list(sys.argv)
sys.argv = [BP, ROOT, SCR, os.path.join(SCR, "_unused_band.json")]
B = {"__name__": "band_pairs_head", "__file__": BP}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, BP, "exec"), B)
sys.argv = ARGV
O = B["O"]
ZG, BEDS, WCELL, CLS_IDX, MONTH, ELEV = O["ZG"], O["BEDS"], O["WCELL"], O["CLS_IDX"], O["MONTH"], O["ELEV"]
E0, qk0, trans_month = O["E0"], O["qk0"], O["trans_month"]
day = ELEV > 0
print("Per class (cell-weighted over the class's 5 m bed bins): daytime median light (lx) and day-mean g at the bed /"
      " 15 m / 50 m above it; day-mean R^2 (m2) at the bed for a 50 cm saithe; Ke 0.02 lx unless stated")
for kd in (0.05, 0.06, 0.09):
    E = E0[:, None] * trans_month(dict(qk0, k_deep=kd))[MONTH]
    print(f"\n  K_deep {kd}")
    for name, idx in zip(["< 100", "100-150", "150-250", ">= 250"], CLS_IDX):
        out = []
        for up in (0.0, 15.0, 50.0):
            med, gm, r2, g4 = [], [], [], []
            for b, w in zip(BEDS[idx], WCELL[idx]):
                iz = int(np.clip(np.searchsorted(ZG, max(b - 0.5 - up, 0.5)), 0, len(ZG) - 1))
                e = E[day, iz]
                med.append(np.median(e))
                g = e / (e + 0.02)
                gm.append(g.mean())
                g4.append((e / (e + 2e-4)).mean())
                r2.append((0.09 * (1 - g) + 1.0 * g).mean())
            ww = WCELL[idx] / WCELL[idx].sum()
            out.append((float(np.dot(ww, med)), float(np.dot(ww, gm)), float(np.dot(ww, r2)), float(np.dot(ww, g4))))
        print(f"    {name:8s} " + "  |  ".join(f"{lab}: E {o[0]:.2g} lx, g {o[1]:.2f} (Ke 2e-4: {o[3]:.2f}), R2 {o[2]:.2f}"
                                          for lab, o in zip(("bed", "15 m up", "50 m up"), out)))
