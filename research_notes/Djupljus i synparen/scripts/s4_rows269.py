"""Step 4 (section 269 per-pair values) at a given K_deep: the row builders of research_notes/"Bottenfiskens
hojdprofil"/scripts/final_gate.py (g3 rounding, class tables, stomach night/day ratios, the WC -> benthivores
window) executed from their own file on the step 1-3 outputs, FINAL = overlap rows + benthivore rows ("central")
+ the window, exactly as final_gate.py builds it. One replacement: the window key "k+1.7" (the K_C final_gate.py
was run with) becomes the key of the K_C that step 3 derived at this K_deep. The gate part is not run here.

Usage: python3 s4_rows269.py <repo_root> <step1 json> <step2 json> <step3 json> <out_json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k280_common import patched_source  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
BAND = json.load(open(sys.argv[2]))["variants"]
BPJ = json.load(open(sys.argv[3]))
WIN = json.load(open(sys.argv[4]))
OUTJ = os.path.abspath(sys.argv[5])
FG = os.path.join(ROOT, "research_notes", "Bottenfiskens hojdprofil", "scripts", "final_gate.py")
src = patched_source(FG, [('WIN["A_band|ling anglerfish|k+1.7"]', 'WIN[_WKEY]', 1)])
part = src[src.index("DEPTHS = [(99.9"):src.index("FINAL = merge(")]
kc = WIN["k_central"][0]
ns = {"BAND": BAND, "BPJ": BPJ, "WIN": WIN, "_WKEY": f"A_band|ling anglerfish|k{kc:+.1f}", "copy": __import__("copy")}
exec(compile(part, FG, "exec"), ns)
final = ns["merge"](ns["overlap_rows"](), ns["benth_rows"]("central"), ns["WC_WIN"])
out = {}
for key, e in final.items():
    out[key] = {k: (None if v is ns["REMOVE"] else v) for k, v in e.items()}
json.dump(dict(rows=out, window_key=ns["_WKEY"], k_central=WIN["k_central"]), open(OUTJ, "w"), indent=1)
print(f"{len(out)} pairs; window key {ns['_WKEY']}")
for key, e in out.items():
    print(f"  {key:48s} a {e['max_intake_rate']}  dark_ratio {e.get('dark_ratio', '-')}"
          + (f"  stage_access {e['stage_access']}" if "stage_access" in e else "")
          + ("  table " + "/".join(str(r['factor']) for r in e['depth_access_table'][::2])
             if "depth_access_table" in e else ""))
