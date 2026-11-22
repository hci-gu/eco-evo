"""Per-bed view of the library WC -> krill pair (M2, M3 in ../evidence.md): the section 262 krill day layer and its gap
above the bed, the WC daytime and night a per bed relative to the 100-150 m class mean (not presence-weighted), and
the light at the bed (median over all sun-up ticks of the year, the K_deep 0.09 profile).

Usage: python3 -I per_bed.py <repo_root> <scratch_dir>   (variants.py's set-up, no variant run)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(HERE, "variants.py"), sys.argv[1], sys.argv[2], os.devnull, "__none__"]
src = open(os.path.join(HERE, "variants.py")).read()
src = src[:src.index('H = dict(layer_anchor=')]
ns = {"__name__": "variants_head", "__file__": os.path.join(HERE, "variants.py")}
exec(compile(src, "variants.py", "exec"), ns)
K, B, BEDS, W, ELEV = ns["K"], ns["B"], ns["BEDS"], ns["WCELL"], ns["ELEV"]
q = ns["_OG"]["meas"](dict(K["level"](1), Ke_gad=0.02, k_deep=0.09))
a = K["evaluate"](q)["predatory_fish"]["a"]
day = ELEV > 0
ref = np.average(a[:, (BEDS >= 100) & (BEDS < 150)].mean(0), weights=W[(BEDS >= 100) & (BEDS < 150)])
E = K["lux"]("meas")
print("bed (m)  cell share  krill day layer (m)  gap (m)  WC a day / night (rel. to the 100-150 m class mean)"
      "  light at the bed (lx, median of sun-up ticks)")
for ib, b in enumerate(BEDS):
    if b > 300 and b % 50:
        continue
    t_, b_ = K["krill_day_layer"](b, q)
    iz = int(min(b - 1, len(K["ZG"]) - 1))
    print(f"{b:6.0f}  {W[ib]:.4f}  {t_:6.1f}-{b_:6.1f}  {b - b_:6.1f}  {a[day, ib].mean() / ref:8.4f} / "
          f"{a[~day, ib].mean() / ref:7.4f}  {np.median(E[day, iz]):.2e}")
