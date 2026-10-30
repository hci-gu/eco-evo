"""Stage access [f_J, f_A] of a split-out mackerel group (mackerel + horse mackerel) on the planktivore
and benthivore juvenile compartments (resume 223.32), with the same method, data and windows as
../../Årsungarnas andel/scripts/stage_access.py (imported, not copied), plus the remaining water column.

Usage: python3 -I mackerel_stage_access.py <Årsungarnas andel/tables/stage_length_biomass.csv> <out_dir> [a_shares]

[a_shares] = "mackerel=0.964,horse mackerel=0.036": the members' shares of the group's planktivore a at
the field densities (research_notes/"Makrillens tathet" (Swedish letters: a-umlaut), resume 228);
default the GOV shares 0.80 / 0.20. With it the outputs get a "_field" suffix and a JSON of the
central p95 [f_J, f_A] is written.

Member weights inside a split pair = the members' shares of today's pooled pair a, renormalised
(planktivores: mackerel 0.49, horse mackerel 0.12 -> 0.80 / 0.20; saithe 0.33, hake 0.03, spurdog 0.02,
velvet belly 0.01 -> rest). Mackerel is not a member of the pooled water column -> benthivores pair;
its benthivore row uses gadoid identity (SMS: small gadoids, no flatfish) at the same 0.80 / 0.20.
Sensitivity: the gape ratio (p90 / p95 / ICES-DB 30-50 cm check).
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sl, out = sys.argv[1:3]
ASH = dict((k, float(v)) for k, v in (kv.split("=") for kv in sys.argv[3].split(","))) if len(sys.argv) > 3 else None
SUFFIX = "_field" if ASH else ""
here = os.path.dirname(os.path.abspath(__file__))
sa_path = os.path.join(here, "..", "..", "Årsungarnas andel", "scripts", "stage_access.py")
sys.argv = [sa_path, sl, out]
spec = importlib.util.spec_from_file_location("stage_access", sa_path)
SA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SA)

MACK = ([("mackerel", ASH["mackerel"], None), ("horse mackerel", ASH["horse mackerel"], None)] if ASH
        else [("mackerel", 0.80, None), ("horse mackerel", 0.20, None)])
REST_PLK = [("saithe", 0.33), ("hake", 0.03), ("spurdog", 0.02), ("velvet belly", 0.01)]
t = sum(w for _, w in REST_PLK)
REST_PLK = [(m, w / t, None) for m, w in REST_PLK]
PAIRS = {
    "mackerel group -> planktivores": ("plk", MACK),
    "rest water column -> planktivores": ("plk", REST_PLK),
    "mackerel group -> benthivores (gadoids)": ("ben", [(m, w, SA.GADOIDS) for m, w, _ in MACK]),
    "mackerel group -> benthivores (all)": ("ben", MACK),
    "pooled water column -> planktivores (check)": SA.PAIRS["water column -> planktivores"],
}
# Model J/B by month for the planktivores (library start share 0.354-0.403); box 2001-25 Q1/Q3 0.45/0.30;
# 1991-96 age 0+1 Q1-Q4 0.77/0.51/0.75/0.87 is not the stage definition (age 1 to 18.2 cm), so use
# the field box mean 0.38 and the model's range.
JB = {"plk": (0.34, 0.38, 0.45), "ben": (0.06, 0.15, 0.17)}


MACK_W = [(m, w) for m, w, _ in MACK]


def run(ratios):
    saved = dict(SA.RATIO)
    SA.RATIO.update(ratios)
    for m, r in SA.RATIO.items():
        SA.WINDOWS[m] = ("ratio", r, m)
    rows = []
    for pair, (group, members) in PAIRS.items():
        for stage in ("J", "A"):
            fq = [sum(w * SA.share_inside(q, group, ident, SA.WINDOWS[m], stage) for m, w, ident in members)
                  for q in (1, 2, 3, 4)]
            rows.append(dict(pair=pair, group=group, stage=stage, Q1=fq[0], Q2=fq[1], Q3=fq[2], Q4=fq[3],
                             mean=float(np.mean(fq)), summer=float(np.mean(fq[1:3]))))
    SA.RATIO.clear()
    SA.RATIO.update(saved)
    for m, r in SA.RATIO.items():
        SA.WINDOWS[m] = ("ratio", r, m)
    return pd.DataFrame(rows)


def main():
    cases = {"p95 (central)": {}, "p90": {"mackerel": 0.31, "horse mackerel": 0.29},
             "ICES-DB 0.28": {"mackerel": 0.28, "horse mackerel": 0.28},
             "p95 + 0.04": {"mackerel": 0.37, "horse mackerel": 0.35}}
    allr = []
    for name, rr in cases.items():
        R = run(rr)
        R.insert(0, "case", name)
        allr.append(R)
        print(f"\n== gape {name} {rr or ''}")
        with pd.option_context("display.width", 200):
            print(R.drop(columns="case").round(4).to_string(index=False))
        print("  [f_J, f_A] annual mean (Q1-Q4) | Q2-Q3 (presence) | effective f = f_J J/B + f_A (1-J/B)")
        for pair, g in R.groupby("pair", sort=False):
            fj, fa = g[g.stage == "J"].iloc[0], g[g.stage == "A"].iloc[0]
            eff = ", ".join(f"{jb:.2f}: {fj['mean'] * jb + fa['mean'] * (1 - jb):.4f}" for jb in JB[g.group.iloc[0]])
            print(f"  {pair:45s} [{fj['mean']:.4f}, {fa['mean']:.4f}] | [{fj.summer:.4f}, {fa.summer:.4f}]"
                  f" | J/B {eff}")
    os.makedirs(out, exist_ok=True)
    pd.concat(allr).to_csv(os.path.join(out, f"mackerel_stage_access{SUFFIX}.csv"), index=False)
    c = allr[0]
    get = lambda pair, st: float(c[(c.pair == pair) & (c.stage == st)]["mean"].iloc[0])
    import json
    json.dump({"mackerel_planktivores": [get("mackerel group -> planktivores", "J"), get("mackerel group -> planktivores", "A")],
               "rest_planktivores": [get("rest water column -> planktivores", "J"), get("rest water column -> planktivores", "A")],
               "a_shares": dict(MACK_W)},
              open(os.path.join(out, f"mackerel_stage_access{SUFFIX}.json"), "w"), indent=1)


main()
