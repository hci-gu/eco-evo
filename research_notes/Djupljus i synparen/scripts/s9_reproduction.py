"""Reproduction gate: the chain's K_deep 0.06 output (step 8) against the live fgconfig/fg_library.yaml and the git
HEAD copy, pair by pair and key by key (max_intake_rate, dark_ratio, every depth_access_table row, stage_access).
Pairs that carry a light term but no light below 65 m are listed separately with the reason.

Usage: python3 s9_reproduction.py <repo_root> <step8 json at 0.06> <HEAD library yaml> <out_txt>
"""
import json
import os
import sys

import yaml

ROOT = os.path.abspath(sys.argv[1])
FIN = json.load(open(sys.argv[2]))["final"]
HEAD = yaml.safe_load(open(sys.argv[3]))["interaction_definitions"]
LIVE = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml")))["interaction_definitions"]
OUT = sys.argv[4]


def rel(a, b):
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    if b == 0:
        return 0.0 if a == 0 else float("inf")
    return abs(a / b - 1.0)


rows = []
worst = 0.0
for key, f in FIN.items():
    live, head = LIVE[key], HEAD[key]
    items = [("max_intake_rate", f["max_intake_rate"], live.get("max_intake_rate"), head.get("max_intake_rate"))]
    items.append(("dark_ratio", f.get("dark_ratio"), live.get("dark_ratio"), head.get("dark_ratio")))
    if "stage_access" in f or "stage_access" in live:
        for i in (0, 1):
            items.append((f"stage_access[{i}]", (f.get("stage_access") or [None, None])[i],
                          (live.get("stage_access") or [None, None])[i], (head.get("stage_access") or [None, None])[i]))
    if "depth_access_table" in f or "depth_access_table" in live:
        ft = f.get("depth_access_table") or []
        lt = live.get("depth_access_table") or []
        ht = head.get("depth_access_table") or []
        for i in range(max(len(ft), len(lt))):
            dm = (ft[i] if i < len(ft) else lt[i])["depth_m"]
            items.append((f"table {dm:g} m", ft[i]["factor"] if i < len(ft) else None,
                          lt[i]["factor"] if i < len(lt) else None, ht[i]["factor"] if i < len(ht) else None))
    for k, mine, lv, hv in items:
        r = rel(mine, lv)
        rh = rel(mine, hv)
        worst = max(worst, r, rh)
        rows.append((key, k, lv, hv, mine, r, rh))

INDEP = [
    ("planktivorous_fish_preys_on_zooplankton", "prey and predator in the mixed layer (section 162/243.3 encounter); no light below 65 m"),
    ("predatory_fish_preys_on_zooplankton", "copepods 0-25 m (section 245, Vattenpelarens par i uppmatt ljus); scaled a, no deep light"),
    ("whiting_preys_on_zooplankton", "section 245: a x 0.06 of the old pair on the copepod share; no light below 65 m in its derivation"),
    ("mackerel_preys_on_zooplankton", "Macy 1998 clearance, daytime biting 0.62 (section 245); mackerel 0-50 m"),
    ("seals_preys_on_* / porpoises_preys_on_*", "no light term in the engine (no dark_ratio; Daggdjurens attackrater: the mammal pairs carry no light)"),
    ("benthivorous_fish_preys_on_{benthic_infauna,mobile_epifauna,norway_lobster}", "a x bed-contact tau only (section 269.3); reproduced above, K-independent"),
    ("predatory_fish_preys_on_mackerel, mackerel_preys_on_planktivorous_fish", "in the chain; prey in 0-50 m, so the K_deep run returns the same values"),
]
with open(OUT, "w") as fh:
    fh.write("Reproduction gate (section 280): the derivation chain run at K_deep 0.06 against the live library and git HEAD\n")
    fh.write("chain: band_pairs -> benth_profile + wc_benth_window -> final_gate rows (269) -> pair_terms -> school_factor_pout\n")
    fh.write("       x p_ppr (combined.py) -> decompose_ling -> gate_ling central (270); scripts/run_chain.sh\n")
    fh.write("rel = |reproduced / library - 1| (live; HEAD in brackets). Live and HEAD differ only outside these keys.\n\n")
    fh.write(f"{'pair':48s} {'key':18s} {'live':>11s} {'HEAD':>11s} {'reproduced':>11s} {'rel':>9s} {'(HEAD)':>9s}\n")
    for key, k, lv, hv, mine, r, rh in rows:
        fmt = lambda v: "None" if v is None else f"{v:.4g}"
        fh.write(f"{key:48s} {k:18s} {fmt(lv):>11s} {fmt(hv):>11s} {fmt(mine):>11s} {r:9.2e} {rh:9.2e}\n")
    fh.write(f"\n{len(rows)} values, {len(FIN)} pairs; largest relative difference {worst:.2e}"
             f" -> {'PASS' if worst <= 5e-3 else 'FAIL'} (tolerance 0.5 %)\n")
    fh.write("\nPairs with a light term (or listed in the task) that hold no light below 65 m, not re-derived here:\n")
    for k, why in INDEP:
        fh.write(f"  {k}: {why}\n")
print(open(OUT).read()[-2500:])
