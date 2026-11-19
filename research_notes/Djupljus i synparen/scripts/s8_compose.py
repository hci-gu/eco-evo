"""Step 8 (the library values) at a given K_deep.

(a) The section 269 state: the library at commit 7153a6c (section 269 applied) with the 17 pair rows of step 4
    at this K_deep written over it (at 0.06 this is that library unchanged; checked by s4 / the reproduction).
(b) The section 270 pair values: research_notes/"Langans indata"/scripts/gate_ling.py's library builder (LIB0,
    pair_edit, other_pairs, lib_copy; executed from its own file) for its "central" run (stage p + adult profile +
    hake out, ling act0.35s1 with its own window ling@0.3), on (a), with step 6's combined factors (was the
    then-live library and the "Fangst och stim i fiskparen" combined.json) and step 7's decomposition. One more
    replacement: the section 269 window [0.392, 0.0388] that pair_edit scales from is step 3's window at this K_deep.
(c) Night/day: combined.py's night/day' (step 6) as its stdout printed it (4 decimals), then at 3 significant
    figures (section 270's yaml_edits_270.txt took 0.1155 -> 0.116 this way), where section 270 moved it (F_night /
    F_day != 1), else the section 269 value; the stomach ratios kept.
The depth_access_table rows are section 269's (270 changed no table).

Usage: python3 s8_compose.py <repo_root> <K_deep> <step4 json> <step6 json> <step7 json> <post-269 library yaml>
                             <scratch_dir> <out_json>
"""
import copy
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k280_common import patched_source  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
KD = float(sys.argv[2])
R269 = json.load(open(sys.argv[3]))["rows"]
COMB = json.load(open(sys.argv[4]))
DM = json.load(open(sys.argv[5]))
LIB269 = os.path.abspath(sys.argv[6])
SCR = os.path.abspath(sys.argv[7])
OUTJ = os.path.abspath(sys.argv[8])
os.makedirs(SCR, exist_ok=True)


def g3(x):
    return float(f"{x:.3g}")


# (a) section 269 state at K_deep
lib = yaml.safe_load(open(LIB269, encoding="utf-8-sig"))
inter = lib["interaction_definitions"]
for key, e in R269.items():
    for k, v in e.items():
        if v is None:
            inter[key].pop(k, None)
        else:
            inter[key][k] = v
lib269_kd = os.path.join(SCR, f"fg_library_269_k{KD}.yaml")
with open(lib269_kd, "w") as fh:
    yaml.safe_dump(lib, fh, sort_keys=False, allow_unicode=True)

# (b) gate_ling's library builder
GL = os.path.join(ROOT, "research_notes", "Langans indata", "scripts", "gate_ling.py")
W269 = R269["predatory_fish_preys_on_benthivorous_fish"]["stage_access"]
src = patched_source(GL, [
    ('LIB0 = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8-sig"))',
     f'LIB0 = yaml.safe_load(open({lib269_kd!r}, encoding="utf-8-sig"))', 1),
    ("    fj_new = 0.392 * (nj / am) / nj0", f"    fj_new = {W269[0]!r} * (nj / am) / nj0", 1),
    ("    fa_new = 0.0388 * (na / am) / na0", f"    fa_new = {W269[1]!r} * (na / am) / na0", 1),
])
part = src[src.index("LIB0 = yaml.safe_load("):src.index("def S(key")]
ns = {"ROOT": ROOT, "DM": DM, "FAC": COMB["factors"], "SCR": SCR, "os": os, "copy": copy, "yaml": yaml, "np": np,
      "json": json}
exec(compile(part, GL, "exec"), ns)
BASE = dict(stage_p=True, aprof=True, g={"hake": 0.0})
path, info = ns["lib_copy"](f"280_k{KD}", dict(BASE, ling_var="act0.35s1", ling_win="ling@0.3"))
L270 = yaml.safe_load(open(path))["interaction_definitions"]

# (c) compose every visual pair: 269 rows, 270 a / stage_access from gate_ling's copy, night/day from step 6
final = {}
for key, e in R269.items():
    f = {k: v for k, v in e.items()}
    f["max_intake_rate"] = L270[key]["max_intake_rate"]
    if "stage_access" in L270[key]:
        f["stage_access"] = L270[key]["stage_access"]
    name = key.replace("_preys_on_", "->")
    if name in COMB["pairs"]:
        c = COMB["pairs"][name]
        if c["nd269"] is not None and abs(c["F_night"] / c["F_day"] - 1) > 1e-12 and name != "mackerel->planktivorous_fish":
            f["dark_ratio"] = g3(round(c["dark_ratio_new"], 4))   # combined_stdout's 4 decimals, then 3 s.f.
    final[key] = f
json.dump(dict(K_deep=KD, final=final, gate_ling_info=info, lib269=lib269_kd, lib270_gate_ling=path),
          open(OUTJ, "w"), indent=1, default=float)
print(f"K_deep {KD}: WC -> benthivores a {info['a']} (x{info['a_mult']:.4f}), stage_access {info['stage_access']}")
for p_, r_ in info.get("other", {}).items():
    print(f"   {p_:36s} a {r_['a']}" + (f", stage_access {r_['stage_access']}" if "stage_access" in r_ else ""))
for key, f in final.items():
    print(f"  {key:48s} a {f['max_intake_rate']}  dark_ratio {f.get('dark_ratio', '-')}"
          + (f"  stage_access {f['stage_access']}" if "stage_access" in f else ""))
