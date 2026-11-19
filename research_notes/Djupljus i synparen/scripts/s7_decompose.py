"""Step 7 (section 270 member decomposition of the water-column pairs, ling's inputs) at a given K_deep:
research_notes/"Langans indata"/scripts/decompose_ling.py executed whole, with its two light calls at K_deep:
the rule-A light factor gmat(eye, 0.02, 0.06) of its re-stated band loop and the reference band_fish_pair(...,
0.02, 0.06, True) it is checked against. Its window exponent KEXP = 1.73 (research_notes/"Sejens hojd over botten"
window_hs.py central: the band-A saithe a(L) exponent + 2.11) is the same quantity step 3 derives from the band a
at this K_deep; here KEXP = round(k_band + 2.11, 2) from step 3's json (1.73 at 0.06, as the original). Output (member shares of the pair a, A-profile factors, stage windows,
ling variants and windows) to <out_json>; its own note's table is not touched.

Usage: python3 s7_decompose.py <repo_root> <K_deep> <step3 json> <scratch_dir> <out_json>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k280_common import run_patched  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
KD = float(sys.argv[2])
import json  # noqa: E402
KEXP = round(json.load(open(sys.argv[3]))["k_central"][1] + 2.11, 2)
SCR = os.path.abspath(sys.argv[4])
OUTJ = os.path.abspath(sys.argv[5])
DL = os.path.join(ROOT, "research_notes", "Langans indata", "scripts", "decompose_ling.py")
SUBS = [
    ('G = B["gmat"](t["eye"], 0.02, 0.06)[rows]', f'G = B["gmat"](t["eye"], 0.02, {KD!r})[rows]', 1),
    ('ref = B["band_fish_pair"](name, q0, "A", 0.02, 0.06, True)',
     f'ref = B["band_fish_pair"](name, q0, "A", 0.02, {KD!r}, True)', 1),
    ("KEXP = 1.73          # window_hs.py central", f"KEXP = {KEXP!r}          # window_hs.py central", 1),
]
print(f"K_deep {KD}: KEXP {KEXP} (printed labels below still read 1.73)")
run_patched(DL, SUBS, [DL, ROOT, SCR, OUTJ], name="decompose_ling_280")
