"""What a re-booking of predatory_fish_preys_on_benthivorous_fish.predation_mortality moves: the derived growth
rates (lib/world/growth_budget.py) of every group on in-memory library copies, the benthivores' booked M2 total
(habitat.predation_m2, used only by the equilibrium spawn), and the water column's booked ration.

Usage: python3 growth_check.py <repo_root>     (imports lib/, so not -I; the library is read, never written)
"""
import copy
import os
import sys

import yaml

ROOT = os.path.abspath(sys.argv[1])
sys.path.insert(0, ROOT)
from lib.world import growth_budget as gb  # noqa: E402
from lib.world import habitat  # noqa: E402

LIB = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8-sig"))
MAN = yaml.safe_load(open(os.path.join(ROOT, "mareld2.yaml"), encoding="utf-8-sig"))
PAIR = "predatory_fish_preys_on_benthivorous_fish"
muted = {e["group_id"] for k in ("decision_makers", "non_decision_makers") for e in MAN.get(k, [])
         if e.get("muted")}
active = [g for g in LIB["species_definitions"] if g not in muted]
start = {e["group_id"]: float(e["inference_initial_biomass"]) / 3600.0
         for k in ("decision_makers", "non_decision_makers") for e in MAN[k] if "inference_initial_biomass" in e}
print(f"muted in mareld2.yaml: {sorted(muted)}")
ben = LIB["species_definitions"]["benthivorous_fish"]
print(f"benthivorous_fish: somatic_production {ben.get('somatic_production')}, r_max {ben.get('r_max')}, spawn "
      f"{ben.get('spawn')}; predatory_fish somatic_production "
      f"{LIB['species_definitions']['predatory_fish'].get('somatic_production')}")
print("groups with spawn mode equilibrium (the only engine use of the booked M2 sum, habitat.predation_m2): "
      + ", ".join(g for g, s in LIB["species_definitions"].items()
                  if isinstance(s, dict) and (s.get("spawn") or {}).get("mode") == "equilibrium"))


def rates(lib):
    sd, inter = lib["species_definitions"], lib["interaction_definitions"]
    out = {}
    for g in active:
        try:
            out[g] = gb.derived_growth_rate(g, sd, inter, active_predators=active)
        except Exception as e:  # noqa: BLE001
            out[g] = f"err {e}"
    return out


base = rates(LIB)
print(f"\n{'booking':>8s} {'changed derived g':>40s}  benthivore M2 sum  WC booked ration (t/t/yr)")
for m2 in (0.0833, 0.0578, 0.0414, 0.0910, 0.0266, 0.1308):
    lib = copy.deepcopy(LIB)
    lib["interaction_definitions"][PAIR]["predation_mortality"] = m2
    r = rates(lib)
    changed = [g for g in active if r[g] != base[g]]
    m2sum = habitat.predation_m2("benthivorous_fish", lib, active)
    inter = lib["interaction_definitions"]
    ration = sum(float(v.get("predation_mortality") or 0.0) * start.get(k[len("predatory_fish_preys_on_"):], 0.0)
                 for k, v in inter.items() if k.startswith("predatory_fish_preys_on_") and v.get("preys_on")
                 ) / start["predatory_fish"]
    print(f"{m2:8.4f} {', '.join(changed) if changed else 'none':>40s}  {m2sum:.4f}             {ration:.3f}")
print(f"\nbenthivorous_fish g {base['benthivorous_fish']} (= P 0.74 / ((s* - u) x 1460)); predatory_fish g "
      f"{base['predatory_fish']}")
b = gb.growth_budget("benthivorous_fish", LIB["species_definitions"], LIB["interaction_definitions"],
                     active_predators=active)
print(f"benthivore budget: {b.describe()}; booked predation terms (not in g) {b.predation}")
