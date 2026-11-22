#!/bin/bash
# Whole pipeline: class tables of every variant (variants.py), then per variant a library COPY (make_lib.py), the
# diet gate on it (gate_copy.py: tools/probes/budget_gate.py at 16x16, its own grid) and the policy-free intake by
# bed class (intake_by_class.py at 60x60). CPU only. Usage (from scripts/): bash run_all.sh <repo_root> <scratch_dir>
# Outputs: ../tables/variants_stdout.txt, ../tables/variants.json, ../tables/gate/<variant>_<scope>.txt,
#          ../tables/intake/<variant>_<scope>.txt, then tabulate.py -> ../tables/summary_stdout.txt
set -e
R="$1"; S="$2"
export CUDA_VISIBLE_DEVICES=
T="$(cd .. && pwd)/tables"
mkdir -p "$T/gate" "$T/intake" "$S/libs" "$S/tmp"
python3 -I variants.py "$R" "$S/tmp" "$T/variants.json" > "$T/variants_stdout.txt"
for V in $(python3 -I -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['runs']))" "$T/variants.json"); do
  for SC in wc all wcshape; do
    # wc: the four WC pairs, a and table; all: also the other krill pairs; wcshape: WC tables only, live a
    python3 -I make_lib.py "$R" "$T/variants.json" "$V" "$SC" "$S/libs/lib_${V}_${SC}.yaml" > "$T/gate/${V}_${SC}.txt"
    python3 gate_copy.py "$R" "$S/libs/lib_${V}_${SC}.yaml" >> "$T/gate/${V}_${SC}.txt" 2>/dev/null
    python3 intake_by_class.py 60 "$S/libs/lib_${V}_${SC}.yaml" > "$T/intake/${V}_${SC}.txt" 2>/dev/null
  done
  echo "$V done"
done
python3 -I tabulate.py "$T" > "$T/summary_stdout.txt"
python3 -I per_bed.py "$R" "$S/tmp" > "$T/per_bed_stdout.txt"
