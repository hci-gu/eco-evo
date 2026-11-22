"""Write a library COPY with one variant's tables (and a) from variants.py's json.

Usage: python3 -I make_lib.py <repo_root> <variants.json> <variant> <scope wc|all|wcshape> <out_library.yaml>

The live library is read only. Pairs written:
  scope wc   the four water-column (WC) pairs: krill, planktivores, benthivores, whiting
  scope wcshape  the same four pairs, tables only: max_intake_rate and dark_ratio stay at the live values (the
             section 296 level kept; only the class shape moves)
  scope all  also whiting, planktivores and mackerel -> krill (a krill-layer variant moves every krill pair);
             benthivores -> krill is never written (derived by benth_profile.py, not by this band run)
Per pair:
  depth_access_table  the variant's class a's in library units (g3 rounding, as final_gate.py's table())
  max_intake_rate     live a x a_variant / a_library-variant, so the section 270 capture p x schooling F, the
                      ling decomposition and the section 296 x0.16 stay as they are in the live value (none of them
                      is re-derived at the variant's R_eff: flagged in summary.md)
  dark_ratio          WC -> krill only: the variant's rule night/day (the library-variant run reproduces the live
                      0.0257); every other pair keeps its live value (stomach-based, or the 270 F on top)
stage_access is left at the live value (the WC windows are not re-derived).
"""
import json
import os
import sys

import yaml

ROOT, VJ, NAME, SCOPE, OUT = sys.argv[1:6]
R = json.load(open(VJ))["runs"]
lib = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8"))
inter = lib["interaction_definitions"]


def g3(x):
    return float(f"{x:.3g}") if x > 0 else 0.0


def table(c):
    rows = [(99.9, 0), (100.0, 1), (149.9, 1), (150.0, 2), (249.9, 2), (250.0, 3)]
    return [{"depth_m": d, "factor": g3(max(c[k], 0.0))} for d, k in rows]


PAIRS = [("krill", "predatory_fish", "predatory_fish_preys_on_krill"),
         ("fish", "plk", "predatory_fish_preys_on_planktivorous_fish"),
         ("fish", "benth", "predatory_fish_preys_on_benthivorous_fish"),
         ("fish", "whit", "predatory_fish_preys_on_whiting")]
if SCOPE == "all":
    PAIRS += [("krill", g, f"{g}_preys_on_krill") for g in ("whiting", "planktivorous_fish", "mackerel")]
elif SCOPE not in ("wc", "wcshape"):
    raise SystemExit("scope must be wc, all or wcshape")
for kind, key, pair in PAIRS:
    v, l0 = R[NAME][kind][key], R["library"][kind][key]
    p = inter[pair]
    a_old = float(p["max_intake_rate"])
    a_new = a_old * v["a"] / l0["a"] if SCOPE != "wcshape" else a_old
    p["max_intake_rate"] = float(f"{a_new:.6g}")
    p["depth_access_table"] = table(v["classes"])
    extra = ""
    if pair == "predatory_fish_preys_on_krill" and SCOPE != "wcshape":
        p["dark_ratio"] = g3(v["nd"])
        extra = f"; dark_ratio {p['dark_ratio']}"
    print(f"{pair:45s} a {a_old} -> {p['max_intake_rate']}; table "
          + " / ".join(str(r["factor"]) for r in p["depth_access_table"][::2]) + extra)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
yaml.safe_dump(lib, open(OUT, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
print(f"written {OUT}")
