"""Step 5 (section 270 term shares and light-weighted detection radii R_eff) at a given K_deep:
research_notes/"Stimskydd i motes-a"/scripts/pair_terms.py executed whole, with these exact replacements:
  * the band_pairs.py head it execs gets kdeep = K_deep as band_fish_pair's default (so both the reference
    band_fish_pair and the term-split copy band_terms run at K_deep);
  * its rule-A krill state run carries k_deep = K_deep (state only, as in the original);
  * the band_pairs.json it prints next to its own eq is step 1's json at the same K_deep.

Usage: python3 s5_terms.py <repo_root> <K_deep> <step1 json> <scratch_dir> <out_json>
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
PT = os.path.join(ROOT, "research_notes", "Stimskydd i motes-a", "scripts", "pair_terms.py")
BF_DEF = "def band_fish_pair(name, q, vision, Ke=0.02, kdeep=0.06, clip=True):"
SUBS = [
    ("src = open(BP).read()\n",
     f"src = open(BP).read()\nassert src.count({BF_DEF!r}) == 1\n"
     f"src = src.replace({BF_DEF!r}, {BF_DEF.replace('kdeep=0.06', f'kdeep={KD!r}')!r})\n", 1),
    ('G["run_krill"](dict(K["level"](1), Ke_gad=0.02, k_deep=0.06))',
     f'G["run_krill"](dict(K["level"](1), Ke_gad=0.02, k_deep={KD!r}))', 1),
    ('os.path.join(ROOT, "research_notes", "Overlapp i fiskparen", "tables", "band_pairs.json")', repr(BANDJ), 1),
]
run_patched(PT, SUBS, [PT, ROOT, SCR, OUTJ], name="pair_terms_280")
