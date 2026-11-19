"""Step 1 (section 269 band stage) at a given K_deep: research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py
executed whole, with ONE change: its variant list is replaced by the single variant the library came from
("A_band": rule A on the band overlap, Ke 0.02) carrying kdeep = K_deep ("lib_band", library vision, is kept only
because the script's printout reads it). That variant's loop already passes
kdeep to the fish pairs (gmat), the benthivore vision ratio (benth_V) and the krill pairs (run_krill, k_deep into
the measured-light transmission below 65 m). Left at 0.06 on purpose: the library-rule krill reproduction
KR_REPRO (level(1), the section 253/262 derivation the krill a's came from: it is the ratio's denominator) and
the benthivores' old library light TR_LIB (the section 220 derivation's own light: also a denominator).

Usage: python3 s1_band.py <repo_root> <K_deep> <scratch_dir> <out_json>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k280_common import run_patched  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
KD = float(sys.argv[2])
SCR = os.path.abspath(sys.argv[3])
OUTJ = os.path.abspath(sys.argv[4])
os.makedirs(SCR, exist_ok=True)
BP = os.path.join(ROOT, "research_notes", "Overlapp i fiskparen", "scripts", "band_pairs.py")
VAR_OLD = '''VAR = {
    "lib_band": dict(vision="lib"),
    "A_band": dict(vision="A"),
'''
VAR_NEW = f'''VAR = {{
    "lib_band": dict(vision="lib"),
    "A_band": dict(vision="A", kdeep={KD!r}),
}}
_VAR_DROPPED = {{
'''
NS = run_patched(BP, [(VAR_OLD, VAR_NEW, 1)], [BP, ROOT, SCR, OUTJ], name="band_pairs_280")
# check (advisor): the library-rule krill reproduction against the section 262 library a's and class a's
import json  # noqa: E402
rep = {g: dict(eq=r["eq"], classes=r["classes"], nd=r["nd"], a_lib=NS["KRILL_LIB"][g][0],
               classes_lib=NS["KRILL_LIB"][g][1]) for g, r in NS["KR_REPRO"].items()}
print("\nKrill library-rule reproduction (K_deep 0.06, section 262 derivation) vs the section 262 library a / classes:")
for g, r in rep.items():
    print(f"  {g:20s} eq {r['eq']:.4g} vs a {r['a_lib']}; classes " + " / ".join(f"{c:.3g}" for c in r["classes"])
          + f" vs {r['classes_lib']}")
d = json.load(open(OUTJ))
d["krill_repro_lib_rule_0.06"] = rep
json.dump(d, open(OUTJ, "w"), indent=1, default=float)
