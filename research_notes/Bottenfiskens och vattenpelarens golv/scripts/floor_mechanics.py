"""What the benthivores' and the water column's hide floors act on, and how large the cut is.

Usage: python3 -I floor_mechanics.py <repo_root> [<run dir with training.jsonl>]

Prints, from fgconfig/fg_library.yaml (read only):
- every checked predation pair on benthivorous_fish and predatory_fish with its pair floor (or the species
  floor it falls back to), stage_access, a, night/day;
- every observer of the two groups (`<obs>_observes_<grp>`) and the floor it sees them through;
- the shared availability cap (max over checked predators of the visible share; interactions.py
  visible_prey_biomass);
- the visible share 1 - rest (1 - floor) at the brief's annual-mean rest shares (fix97: benthivores 0.026,
  water column 0.032; research_notes/"Gom och sinnen" section 4) and, if a run dir is given, at the
  all-update mean and last-20-update mean of training.jsonl (descriptive only; nothing is set from it);
- the same at a fully resting cell (rest 1) and the candidate floors of this note.
"""
import json
import os
import statistics
import sys

import yaml

ROOT = os.path.abspath(sys.argv[1])
LIB = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml")))
SP, IN = LIB["species_definitions"], LIB["interaction_definitions"]
MUTED = {"seabirds"}
DMS = ["zooplankton", "planktivorous_fish", "benthivorous_fish", "predatory_fish", "whiting", "mackerel",
       "porpoises", "seals"]

rest = {"benthivorous_fish": {"brief (fix97 annual mean)": 0.026},
        "predatory_fish": {"brief (fix97 annual mean)": 0.032}}
if len(sys.argv) > 2:
    rows = [json.loads(x) for x in open(os.path.join(sys.argv[2], "training.jsonl"))]
    # action order in training.jsonl: benthivores, mackerel, planktivores, porpoises, water column, seals,
    # whiting, zooplankton; columns entropy, move, rest, eat
    idx = {"benthivorous_fish": 0, "predatory_fish": 4}
    for g, i in idx.items():
        allv = [r["actions"][i][2] for r in rows]
        last = [r["actions"][i][2] for r in rows[-20:]]
        rest[g][f"training.jsonl all {len(rows)} updates (mean)"] = statistics.mean(allv)
        rest[g]["training.jsonl last 20 updates (mean)"] = statistics.mean(last)
        rest[g]["training.jsonl max over updates"] = max(allv)

for g in ("benthivorous_fish", "predatory_fish"):
    sf = float(SP[g].get("visibility_floor", 0.0))
    print(f"\n=== {g}: species visibility_floor {sf}")
    print("  checked predation pairs on it:")
    preds = []
    for p in DMS:
        d = IN.get(f"{p}_preys_on_{g}", {})
        if d.get("preys_on") and p not in MUTED:
            fl = d.get("visibility_floor")
            preds.append((p, sf if fl is None else float(fl), fl is not None))
            print(f"    {p:20s} floor {'pair ' + str(fl) if fl is not None else 'species ' + str(sf):14s}"
                  f" a {d.get('max_intake_rate')}  stage_access {d.get('stage_access')}"
                  f"  night/day {d.get('dark_ratio')}  M2 {d.get('predation_mortality')}")
    cap = max(f for _, f, _ in preds)
    print(f"  shared availability cap = best detector's floor: {cap} "
          f"({', '.join(p for p, f, _ in preds if f == cap)})")
    print("  observers (observes: true) and the floor they see a resting share through:")
    for o in DMS:
        d = IN.get(f"{o}_observes_{g}", {})
        if not d.get("observes") or o in MUTED or o == g:
            continue
        pd_ = IN.get(f"{o}_preys_on_{g}", {})
        rd = IN.get(f"{g}_preys_on_{o}", {})
        role = "predator of it" if pd_.get("preys_on") else ("its prey (it eats the observer)" if rd.get("preys_on")
                                                             else "no food link")
        fl = pd_.get("visibility_floor") if pd_.get("preys_on") else None
        print(f"    {o:20s} {role:32s} floor {fl if fl is not None else sf} "
              f"({'pair key' if fl is not None else 'species floor'})")
    print("  visible share 1 - rest (1 - floor) of the fish pairs / non-predator observers (species floor):")
    cands = {"benthivorous_fish": [sf, 0.8, 0.9, 0.95, 1.0], "predatory_fish": [sf, 0.9, 1.0]}[g]
    for lab, r in list(rest[g].items()) + [("fully resting cell", 1.0)]:
        print(f"    rest {r:.4f} [{lab}]: " + ", ".join(f"floor {f:.2f} -> {1 - r * (1 - f):.4f}" for f in cands))

print("\nMammal pair floors (decided 2026-11-11, research_notes/'Sal och tumlare golv'; every one an explicit key,")
print("so a species-floor change reaches none of them):")
for p in ("seals", "porpoises"):
    for g in ("benthivorous_fish", "predatory_fish"):
        d = IN[f"{p}_preys_on_{g}"]
        print(f"  {p}_preys_on_{g}: preys_on {d.get('preys_on')}, visibility_floor key {d.get('visibility_floor')}")
print("  decided: seals -> benthivores 0.9, porpoises -> benthivores 0.75 (keep), seals -> water column 0.7,"
      " porpoises -> water column 0.9 (keep)")
