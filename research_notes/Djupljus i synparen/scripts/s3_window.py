"""Step 3 (section 269 water column -> benthivores stage_access) at a given K_deep: research_notes/"Bottenfiskens
hojdprofil"/scripts/wc_benth_window.py executed whole; the band a's member shares are read from step 1's json at
the same K_deep, and the output json goes to <out_json> (the original writes into its own note's tables/).
The window algebra (research_notes/"Fonsterviktning" aw_stage_access.py) holds no light term; K_deep reaches the
window only through the members' shares of the band a and the saithe a(L) exponent kexp derived from them.

Usage: python3 s3_window.py <repo_root> <step1 json> <scratch_dir> <out_json>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k280_common import run_patched  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
BANDJ = os.path.abspath(sys.argv[2])
SCR = os.path.abspath(sys.argv[3])
OUTJ = os.path.abspath(sys.argv[4])
WB = os.path.join(ROOT, "research_notes", "Bottenfiskens hojdprofil", "scripts", "wc_benth_window.py")
SUBS = [
    ('os.path.join(RN, "Overlapp i fiskparen", "tables", "band_pairs.json")', repr(BANDJ), 1),
    ('''open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                                     "tables", "wc_benth_window.json"), "w")''',
     f'open({OUTJ!r}, "w")', 1),
    # only the central overlap variant is run at K_deep (the Hs 5 / 40 m brackets are not part of the library value)
    ('for var in ("A_band", "A_band_Hs5", "A_band_Hs40"):', 'for var in ("A_band",):', 1),
]
run_patched(WB, SUBS, [WB, ROOT, SCR], name="wc_benth_window_280")
