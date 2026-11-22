#!/bin/bash
# The section 269 -> 270 derivation chain of every visual fish pair (research_notes/"Djupljus i synparen", section
# 280) at one gadoid-eye Ke and one K_deep, through ke_step.py (each Djupljus driver with Ke anchors).
# usage: scripts/run_chain_ke.sh <repo_root> <scratch_dir> <Ke> <K_deep> <band json at Ke 0.02, K_deep 0.06>
#   The last argument is step 6's bed-radius reference (the 0.6 / 0.3 m radii were judged at Ke 0.02, K_deep 0.06);
#   make it with: python3 ke_step.py 1 0.02 <repo> 0.06 <tmp> <scratch>/base/band_0.06.json
# Outputs in <scratch_dir>/ke<Ke>_k<K>/: band, benth, window, rows269, terms, comb, dec, final (.json) and s<n>.txt.
set -e
R="$1"; S="$2"; KE="$3"; K="$4"; B0="$5"
D="$(cd "$(dirname "$0")" && pwd)"
O="$S/ke${KE}_k${K}"; T="$O/tmp"
mkdir -p "$T" "$S/libs"
LIB269="$S/libs/fg_library_post269.yaml"
[ -f "$LIB269" ] || (cd "$R" && git show 7153a6c:fgconfig/fg_library.yaml > "$LIB269")
export CUDA_VISIBLE_DEVICES=
P() { python3 "$D/ke_step.py" "$@"; }
P 1 "$KE" "$R" "$K" "$T" "$O/band.json" > "$O/s1.txt" 2>&1
P 2 "$KE" "$R" "$K" "$O/band.json" "$T" "$O/benth.json" > "$O/s2.txt" 2>&1
P 3 "$KE" "$R" "$O/band.json" "$T" "$O/window.json" > "$O/s3.txt" 2>&1
P 4 "$KE" "$R" "$O/band.json" "$O/benth.json" "$O/window.json" "$O/rows269.json" > "$O/s4.txt" 2>&1
P 5 "$KE" "$R" "$K" "$O/band.json" "$T" "$O/terms.json" > "$O/s5.txt" 2>&1
P 6 "$KE" "$R" "$K" "$O/terms.json" "$O/rows269.json" "$O/band.json" "$B0" "$O/benth.json" "$T" "$O/comb.json" \
    > "$O/s6.txt" 2>&1
P 7 "$KE" "$R" "$K" "$O/window.json" "$T" "$O/dec.json" > "$O/s7.txt" 2>&1
P 8 "$KE" "$R" "$K" "$O/rows269.json" "$O/comb.json" "$O/dec.json" "$LIB269" "$O/libs" "$O/final.json" > "$O/s8.txt" 2>&1
echo "Ke $KE K_deep $K done"
