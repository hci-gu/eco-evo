"""One step of the section 280 chain (research_notes/"Djupljus i synparen"/scripts/s<n>_*.py) at a given gadoid-eye
half-saturation light Ke (lx), in addition to that driver's K_deep.

The Djupljus driver is read from its own file and executed with exact string replacements (each anchor asserted,
k280_common.run_patched); nothing in that note is written. Only the rule-A (new-value) side of each ratio takes Ke:

  s1  the "A_band" variant gets Ke (band_pairs.py passes it to run_krill as Ke_gad, to band_fish_pair and to
      benth_pair's rule-A side);
  s2  G_LIGHT = gmat("gadoid", Ke, K) and Ke_gad = Ke on the two rule-A krill runs (KR_A_BAND, krill_pair);
  s5  band_fish_pair's default Ke (as its default kdeep) and the rule-A krill state run's Ke_gad;
  s7  decompose_ling's re-stated band loop gmat(eye, Ke, K) and its reference band_fish_pair(..., Ke, K, True).
  s3, s4, s6, s8 hold no Ke (they read the step jsons) and run unchanged.

Kept at 0.02 on purpose (ratio denominators that reproduce how the old a's were derived, as K_deep 0.06 was kept
in section 280): KR_REPRO (library-rule krill reproduction, level(1)'s Ke_gad 0.02), benth_V(pair, "lib", 0.02,
0.06) with TR_LIB, the "lib_band" variant. The clupeid eye (herring, mackerel: Ke_clup 1 lx with the 0.016 lx
biting threshold) is a different parameter and is not touched.

Usage: python3 ke_step.py <step 1..8> <Ke> <the Djupljus driver's own arguments ...>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DJ = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Djupljus i synparen", "scripts")
sys.path.insert(0, DJ)
from k280_common import run_patched  # noqa: E402

STEP = int(sys.argv[1])
KE = float(sys.argv[2])
ARGS = sys.argv[3:]
FILES = {1: "s1_band.py", 2: "s2_benth.py", 3: "s3_window.py", 4: "s4_rows269.py", 5: "s5_terms.py",
         6: "s6_combined.py", 7: "s7_decompose.py", 8: "s8_compose.py"}
SUBS = {
    1: [('    "A_band": dict(vision="A", kdeep={KD!r}),\n',
         '    "A_band": dict(vision="A", kdeep={KD!r}, Ke={KE!r}),\n', 1)],
    2: [("""f'G_LIGHT = gmat("gadoid", 0.02, {KD!r})'""", """f'G_LIGHT = gmat("gadoid", {KE!r}, {KD!r})'""", 1),
        ("""dict(K["level"](1), k_deep={KD!r})""", """dict(K["level"](1), Ke_gad={KE!r}, k_deep={KD!r})""", 2)],
    5: [("""BF_DEF.replace('kdeep=0.06', f'kdeep={KD!r}')""",
         """BF_DEF.replace('kdeep=0.06', f'kdeep={KD!r}').replace('Ke=0.02', f'Ke={KE!r}')""", 1),
        ("""f'G["run_krill"](dict(K["level"](1), Ke_gad=0.02, k_deep={KD!r}))'""",
         """f'G["run_krill"](dict(K["level"](1), Ke_gad={KE!r}, k_deep={KD!r}))'""", 1)],
    7: [("""f'G = B["gmat"](t["eye"], 0.02, {KD!r})[rows]'""", """f'G = B["gmat"](t["eye"], {KE!r}, {KD!r})[rows]'""", 1),
        ("""f'ref = B["band_fish_pair"](name, q0, "A", 0.02, {KD!r}, True)'""",
         """f'ref = B["band_fish_pair"](name, q0, "A", {KE!r}, {KD!r}, True)'""", 1)],
}
path = os.path.join(DJ, FILES[STEP])
print(f"[ke_step] step {STEP} at Ke {KE} lx ({FILES[STEP]}, {len(SUBS.get(STEP, []))} Ke anchor(s))", flush=True)
run_patched(path, SUBS.get(STEP, []), [path] + ARGS, name=f"ke_step{STEP}", extra={"KE": KE})
