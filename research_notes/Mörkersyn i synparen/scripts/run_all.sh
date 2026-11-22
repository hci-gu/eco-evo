#!/bin/bash
# Whole pipeline, CPU only (~10 min). usage (from scripts/): bash run_all.sh <repo_root> <scratch_dir>
# 1. step 1 at Ke 0.02, K_deep 0.09 -> 0.06 (step 6's bed-radius reference); 2. the chain at Ke 0.02 (gate) and
# 0.002 (scope B), K_deep 0.09; 3. reproduction gate; 4. scope A band-stage decomposition; 5. the scope B copy and
# edit list; 6. loader growth_rate check; 7. budget_gate on live and copy; 8. intake by bed class; 9. comparison.
set -e
R="$1"; S="$2"
export CUDA_VISIBLE_DEVICES=
T="$(cd .. && pwd)/tables"
mkdir -p "$S/base/tmp" "$S/libs" "$T/chain" "$T/gate" "$T/intake"
python3 ke_step.py 1 0.02 "$R" 0.06 "$S/base/tmp" "$S/base/band_0.06.json" > "$T/chain/base_ke0.02_k0.06_s1.txt" 2>&1
bash run_chain_ke.sh "$R" "$S" 0.02 0.09 "$S/base/band_0.06.json" &
bash run_chain_ke.sh "$R" "$S" 0.002 0.09 "$S/base/band_0.06.json" &
wait
for c in ke0.02_k0.09 ke0.002_k0.09; do mkdir -p "$T/chain/$c"; cp "$S/$c"/*.json "$S/$c"/s*.txt "$T/chain/$c/"; done
python3 repro_gate.py "$R" "$S/ke0.02_k0.09/final.json" "$T/reproduction_gate.txt"
python3 scope_a_shares.py "$R" "$S/scopeA" > "$T/scope_a_shares_stdout.txt"
LIB="$S/libs/fg_library_ke0.002_scopeB.yaml"
python3 make_copy.py "$R" "$S/ke0.002_k0.09/final.json" "$S/ke0.02_k0.09/final.json" "$LIB" \
    "$T/yaml_edits_ke0.002_scopeB.txt" "Ke 0.002 scope B"
python3 growth_check.py "$R" "$LIB" > "$T/growth_check_stdout.txt" 2>&1
python3 gate_copy.py "$R" "$R/fgconfig/fg_library.yaml" > "$T/gate/live.txt" 2>/dev/null
python3 gate_copy.py "$R" "$LIB" > "$T/gate/ke0.002_scopeB.txt" 2>/dev/null
for P in predatory_fish whiting benthivorous_fish; do
  python3 intake_by_class.py 60 live "$P" > "$T/intake/live_$P.txt" 2>/dev/null
  python3 intake_by_class.py 60 "$LIB" "$P" > "$T/intake/ke0.002_scopeB_$P.txt" 2>/dev/null
done
python3 compare_gates.py "$T/gate/live.txt" "$T/gate/ke0.002_scopeB.txt" "$T/intake" ke0.002_scopeB \
    > "$T/compare_scopeB_stdout.txt"
