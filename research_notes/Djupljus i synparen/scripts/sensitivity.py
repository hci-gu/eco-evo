"""Sensitivity (NOT applied): bed-class K and the near-bed layer of research_notes/"Ljus under 65 m", as a per-class
composite of whole-chain runs. The chain's light is one transmission profile for every bed (gmat is (time, depth));
a bed-dependent profile would need g(t, z, bed) through point_series, slab_series, benth_V, member_A and the krill
lux. The composite instead takes each bed class's a from the chain run at that class's K (every link re-run at that
K, so F, the ling decomposition and the windows are those of a world with that K everywhere [J: approximation]):
    a_c = class row_c(K_c) x a_final(K_c) / a_269(K_c)     (the 270 factors act on the whole pair)
    a = sum_c share_c a_c                                   (equal-cell shares of the four classes)
Readings (K below 65 m by bed class < 100 / 100-150 / 150-250 / >= 250 m):
    class        0.16 / 0.13 / 0.092 / 0.082   glider mid-water K by class (voto_k by EMODnet bed class; >= 100 m
                                               above the bed for 150-250 and >= 250 m)
    class+bed    0.16 / 0.14 / 0.11  / 0.092   bed-light effective K: the 0-100 m near-bed excess (extra optical depth
                                               2.15 over 100 m, near_bed.py) spread over bed - 65 m (150-250 m beds:
                                               +0.018-0.024; >= 250 m: assumed layer [J], +0.009); < 150 m classes are
                                               within the layer already
Night/day is not composited (printed per K).

Usage: python3 sensitivity.py <repo_root> <scratch_dir with final_K / rows269_K jsons> <out_json>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k280_common import run_patched  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
S = os.path.abspath(sys.argv[2])
OUTJ = os.path.abspath(sys.argv[3])
ORP = os.path.join(ROOT, "research_notes", "En synregel for fiskparen", "scripts", "one_rule_pairs.py")
O = run_patched(ORP, [], [ORP, ROOT, os.path.join(S, "tmp"), os.path.join(S, "tmp", "_u.json")], name="orp_head",
                upto="# =============================================================== run", quiet=True)
SHARE = np.array(O["SHARE_CLS"], float)
SHARE = SHARE / SHARE.sum()
CLS_ROWS = (0, 1, 3, 5)
READ = {"flat 0.09": (0.09, 0.09, 0.09, 0.09), "class": (0.16, 0.13, 0.092, 0.082),
        "class+bed": (0.16, 0.14, 0.11, 0.092)}


def load(k):
    f = json.load(open(os.path.join(S, f"final_{k}.json")))["final"]
    r = json.load(open(os.path.join(S, f"rows269_{k}.json")))["rows"]
    return f, r


cache = {}
print(f"equal-cell class shares < 100 / 100-150 / 150-250 / >= 250 m: " + " / ".join(f"{x:.3f}" for x in SHARE))
F06, R06 = load("0.06")
res = {}
for key in F06:
    if "depth_access_table" not in R06[key]:
        continue
    res[key] = {}
    for rn, ks in READ.items():
        ac = []
        for c, k in enumerate(ks):
            kk = f"{k:g}" if k != 0.09 else "0.09"
            if kk not in cache:
                cache[kk] = load(kk)
            f, r = cache[kk]
            row = r[key]["depth_access_table"][CLS_ROWS[c]]["factor"]
            ac.append(row * f[key]["max_intake_rate"] / r[key]["max_intake_rate"])
        a = float(SHARE @ np.array(ac))
        res[key][rn] = dict(a=a, classes=ac)
    # consistency of the composite formula at a flat K: sum share x rows reproduces the pair a within table rounding
    f, r = cache["0.09"]
    chk = float(SHARE @ np.array([r[key]["depth_access_table"][i]["factor"] for i in CLS_ROWS])) / r[key]["max_intake_rate"]
    res[key]["check_flat"] = chk
print(f"\n{'pair':46s} {'live a':>9s} " + " ".join(f"{rn:>11s}" for rn in READ) + "   class a's (class+bed)  [chk]")
for key, d in res.items():
    print(f"{key:46s} {F06[key]['max_intake_rate']:9.4g} " + " ".join(f"{d[rn]['a']:11.4g}" for rn in READ)
          + "   " + " / ".join(f"{x:.3g}" for x in d["class+bed"]["classes"]) + f"  [{d['check_flat']:.3f}]")
print("\nnight/day per K (pair's chain value):")
ks = sorted(cache, key=float)
print(f"{'pair':46s} " + " ".join(f"{k:>7s}" for k in ks))
for key in res:
    print(f"{key:46s} " + " ".join(f"{str(cache[k][0][key].get('dark_ratio')):>7s}" for k in ks))
json.dump(dict(shares=SHARE.tolist(), readings=READ, pairs=res), open(OUTJ, "w"), indent=1)
