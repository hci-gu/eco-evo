"""The per-pair parameter table that WOULD go into fg_library.yaml under rule A + band overlap (A_band, central,
library capture p), with the review-p and profile brackets beside it. Prints only; writes nothing to the library.

Usage: python3 -I proposal_table.py <repo_root> <band_pairs.json>

max_intake_rate = library a x (rule-A band equal-cell annual mean / reproduced library derivation);
depth_access_table = the pair's own class a's (< 100 / 100-150 / 150-250 / >= 250 m; the loader renormalises to an
equal-cell mean of 1, so only the shape acts); dark_ratio = stomach value where one exists (user decision
2026-11-11), else the band rule's own night/day at the pair's library threshold.
"""
import json
import os
import sys

import yaml

ROOT = sys.argv[1]
R = json.load(open(sys.argv[2]))["variants"]
LIB = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8-sig"))["interaction_definitions"]
STOMACH = {"planktivorous_fish_preys_on_krill": (0.57, "stomachs (Onsrud et al. 2004 / 245)"),
           "mackerel_preys_on_krill": (0.42, "stomachs (245)"),
           "mackerel_preys_on_planktivorous_fish": (0.16, "stomachs (228)"),
           "whiting_preys_on_krill": (None, "stomachs: no diel difference (Onsrud et al. 2004; 245) [J: not on the user's list]")}
P_REV = {"krill": 1.2, "fish220": 2.25, "benth": 1.5}
BENTH = {"benthivorous_fish->planktivorous_fish", "benthivorous_fish->whiting"}


def g(x):
    return float(f"{x:.3g}")


rows = []
for kind in ("krill", "fish"):
    for name, r in R["A_band"][kind].items():
        key = f"{name}_preys_on_krill" if kind == "krill" else name.replace("->", "_preys_on_")
        pair = name if kind == "fish" else f"{name}->krill"
        lib = LIB[key]
        a = r["a_new"]
        cls = r["classes_lib_units"] if kind == "krill" else [a * c for c in r["cls_rel"]]
        nd_rule = r["nd"]
        if key in STOMACH:
            dr, why = STOMACH[key]
        else:
            dr, why = g(nd_rule), "band rule A"
        pf = P_REV["krill"] if kind == "krill" else (P_REV["benth"] if name in BENTH else P_REV["fish220"])
        br = []
        for v in ("A_band_Hs5", "A_band_Hs40", "A_band_Hw5", "A_band_Hw30", "A_band_Ke2e-4", "A_band_Ke0.1"):
            br.append(R[v][kind][name]["a_new"])
        rows.append((key, lib.get("max_intake_rate"), g(a), [g(c) for c in cls], lib.get("dark_ratio"), dr, why,
                     lib.get("light_threshold_deg"), g(nd_rule), g(a * pf), g(min(br)), g(max(br)),
                     R["lib_band"][kind][name]["a_new"]))
print(f"{'pair':42s} {'a lib':>9s} {'a new':>9s}  depth_access_table class a (<100/100-150/150-250/>=250)"
      f"   dark lib -> new (thr)  [rule n/d]  a review p   a bracket (Hs, Hw, Ke)   a lib vision band")
for k, al, an, cl, dl, dn, why, thr, ndr, arev, blo, bhi, alb in rows:
    print(f"{k:42s} {al:9.4g} {an:9.4g}  {' / '.join(f'{c:.3g}' for c in cl):40s}  {str(dl):>5s} -> {str(dn):5s} ({thr})"
          f"  [{ndr}]  {arev:.3g}  {blo:.3g}-{bhi:.3g}  {alb:.3g}   <- {why}")
