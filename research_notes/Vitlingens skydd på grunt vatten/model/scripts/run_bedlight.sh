#!/bin/bash
# Bed-class light composite (resume 286). Not applied to the live library.
# usage: run_bedlight.sh <repo_root> <scratch_dir>
# 1. Mirror of the repo in <scratch>/root: symlinks to everything, except a COPY of
#    research_notes/"En synregel for fiskparen"/scripts/one_rule_pairs.py whose trans_month() reads the env var
#    MARELD_K_LAYERS='k20_50,k50_65' (the two mid-water layers; K below 65 m stays the chain's K argument).
# 2. research_notes/"Djupljus i synparen"/scripts/run_chain.sh on the mirror: base (live profile) at K 0.06 and 0.09;
#    A (< 100 m beds, VOTO gliders) MARELD_K_LAYERS=0.17,0.15 at K 0.16; B (100-150 m) 0.139,0.121 at K 0.14
#    (A and B get the base band_0.06.json). The base 0.09 run reproduces the 280 chain exactly (max rel diff 0).
# 3. compose.py <scratch> <live library> <scratch>/lib_bedlight.yaml; gate_280.py <repo> <copy> <out>;
#    frozen_compare.py / cf_whiting.py <frozen policies> <seed> <library>.
set -e
R="$1"; SC="$2"; E="research_notes/En synregel for fiskparen"
mkdir -p "$SC/root/research_notes"; cd "$SC/root"
for x in fgconfig maps lib mareld2.yaml tools; do ln -sfn "$R/$x" "$x"; done
for d in "$R"/research_notes/*/; do n=$(basename "$d"); [ "$n" = "En synregel for fiskparen" ] || ln -sfn "$d" "research_notes/$n"; done
mkdir -p "$E/scripts"
for f in "$R/$E"/*; do n=$(basename "$f"); [ "$n" = scripts ] || ln -sfn "$f" "$E/$n"; done
for f in "$R/$E/scripts"/*; do n=$(basename "$f"); [ "$n" = one_rule_pairs.py ] || ln -sfn "$f" "$E/scripts/$n"; done
python3 - "$R/$E/scripts/one_rule_pairs.py" "$E/scripts/one_rule_pairs.py" <<'PY'
import sys
s = open(sys.argv[1]).read()
old = "def trans_month(q):\n    z = ZG\n"
new = ("def trans_month(q):\n    _ov = os.environ.get('MARELD_K_LAYERS')\n    if _ov:\n        q = dict(q)\n"
       "        q['k20_50'], q['k50_65'] = (float(v) for v in _ov.split(','))\n    z = ZG\n")
assert s.count(old) == 1
open(sys.argv[2], 'w').write(s.replace(old, new))
PY
CH="$R/research_notes/Djupljus i synparen/scripts/run_chain.sh"
for s in base A B; do mkdir -p "$SC/S_$s/libs"; (cd "$R" && git show 7153a6c:fgconfig/fg_library.yaml > "$SC/S_$s/libs/fg_library_post269.yaml"); done
bash "$CH" "$SC/root" "$SC/S_base" 0.06 0.09
cp "$SC/S_base/band_0.06.json" "$SC/S_A/"; cp "$SC/S_base/band_0.06.json" "$SC/S_B/"
MARELD_K_LAYERS=0.17,0.15 bash "$CH" "$SC/root" "$SC/S_A" 0.16 &
MARELD_K_LAYERS=0.139,0.121 bash "$CH" "$SC/root" "$SC/S_B" 0.14 &
wait
