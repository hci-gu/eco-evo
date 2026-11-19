"""Step 2 (section 269 benthivore pairs) at a given K_deep: research_notes/"Bottenfiskens hojdprofil"/scripts/
benth_profile.py executed whole, with these exact replacements:
  * the rule-A vision factor V_A is read from step 1's json at the same K_deep (was the Overlapp i fiskparen table);
  * G_LIGHT (rule A light factor at every depth for the member profiles) at K_deep (was 0.06);
  * the rule-A krill runs (KR_A_BAND and krill_pair) carry k_deep = K_deep.
Left at 0.06: KR_REPRO (the library-rule reproduction of the section 262 krill a: the ratio's denominator).

Usage: python3 s2_benth.py <repo_root> <K_deep> <step1 json> <scratch_dir> <out_json>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k280_common import run_patched  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
KD = float(sys.argv[2])
BANDJ = os.path.abspath(sys.argv[3])
SCR = os.path.abspath(sys.argv[4])
OUTJ = os.path.abspath(sys.argv[5])
BPF = os.path.join(ROOT, "research_notes", "Bottenfiskens hojdprofil", "scripts", "benth_profile.py")
SUBS = [
    ('os.path.join(RN, "Overlapp i fiskparen", "tables", "band_pairs.json")', repr(BANDJ), 1),
    ('G_LIGHT = gmat("gadoid", 0.02, 0.06)', f'G_LIGHT = gmat("gadoid", 0.02, {KD!r})', 1),
    ('KR_A_BAND = run_krill(dict(K["level"](1)))["benthivorous_fish"]',
     f'KR_A_BAND = run_krill(dict(K["level"](1), k_deep={KD!r}))["benthivorous_fish"]', 1),
    ('    r = run_krill(dict(K["level"](1)))["benthivorous_fish"]',
     f'    r = run_krill(dict(K["level"](1), k_deep={KD!r}))["benthivorous_fish"]', 1),
]
run_patched(BPF, SUBS, [BPF, ROOT, SCR, OUTJ], name="benth_profile_280")
