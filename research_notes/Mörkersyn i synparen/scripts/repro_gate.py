"""Reproduction gate: the chain run through ke_step.py at the live settings (Ke 0.02 lx, K_deep 0.09), with the
section 296 factors applied (one per predator: planktivorous_fish x0.36, whiting and predatory_fish x0.16, the
others x1; a only), against the live fgconfig/fg_library.yaml, key by key: max_intake_rate, dark_ratio, every
stage_access entry and every depth_access_table row. Tolerance: relative 1e-9 (exact up to float products).

Usage: python3 repro_gate.py <repo_root> <final.json at Ke 0.02, K_deep 0.09> <out_txt>
"""
import json
import os
import sys

import yaml

ROOT = os.path.abspath(sys.argv[1])
FIN = json.load(open(sys.argv[2]))["final"]
OUT = sys.argv[3]
LIVE = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8-sig"))["interaction_definitions"]
F296 = {"planktivorous_fish": 0.36, "whiting": 0.16, "predatory_fish": 0.16}
TOL = 1e-9


def rel(a, b):
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    if b == 0:
        return 0.0 if a == 0 else float("inf")
    return abs(a / b - 1.0)


rows, worst = [], 0.0
for key, f in FIN.items():
    live = LIVE[key]
    pred = key.split("_preys_on_")[0]
    items = [("max_intake_rate", f["max_intake_rate"] * F296.get(pred, 1.0), live.get("max_intake_rate"))]
    items.append(("dark_ratio", f.get("dark_ratio"), live.get("dark_ratio")))
    if "stage_access" in f or "stage_access" in live:
        for i in (0, 1):
            items.append((f"stage_access[{i}]", (f.get("stage_access") or [None, None])[i],
                          (live.get("stage_access") or [None, None])[i]))
    ft, lt = f.get("depth_access_table") or [], live.get("depth_access_table") or []
    for i in range(max(len(ft), len(lt))):
        dm = (ft[i] if i < len(ft) else lt[i])["depth_m"]
        dl = lt[i]["depth_m"] if i < len(lt) else None
        assert dl is None or dl == dm, (key, i, dm, dl)
        items.append((f"table {dm:g} m", ft[i]["factor"] if i < len(ft) else None, lt[i]["factor"] if i < len(lt) else None))
    for k, mine, lv in items:
        r = rel(mine, lv)
        worst = max(worst, r)
        rows.append((key, k, lv, mine, r))
with open(OUT, "w") as fh:
    fh.write("Reproduction gate: chain at Ke 0.02 lx, K_deep 0.09 (scripts/run_chain_ke.sh via ke_step.py) x the section 296\n")
    fh.write("factors (planktivorous_fish 0.36, whiting 0.16, predatory_fish 0.16; a only) against the live library.\n\n")
    fh.write(f"{'pair':48s} {'key':18s} {'live':>12s} {'reproduced':>12s} {'rel':>9s}\n")
    fmt = lambda v: "None" if v is None else f"{v:.6g}"
    for key, k, lv, mine, r in rows:
        fh.write(f"{key:48s} {k:18s} {fmt(lv):>12s} {fmt(mine):>12s} {r:9.2e}{'  <-- MISMATCH' if r > TOL else ''}\n")
    n_bad = sum(1 for r in rows if r[4] > TOL)
    fh.write(f"\n{len(rows)} values, {len(FIN)} pairs; largest relative difference {worst:.2e}; {n_bad} above {TOL:g}"
             f" -> {'PASS' if n_bad == 0 else 'FAIL'}\n")
print(open(OUT).read()[-400:])
