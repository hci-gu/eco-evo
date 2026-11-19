#!/bin/bash
# The section 269 -> 270 derivation chain of every visual fish pair at one or more K_deep values (light below 65 m).
# usage: scripts/run_chain.sh <repo_root> <scratch_dir> K [K ...]
#   K 0.06 first: step 6 scales the benthivore bed radii by R_eff(K) / R_eff(0.06), so band_0.06.json must exist.
# Per K (outputs <scratch>/<step>_<K>.json, stdout <scratch>/s<n>_<K>.txt):
#   s1 band_pairs (269 band, rule A)       -> band_K.json
#   s2 benth_profile (269 benthivores)     -> benth_K.json
#   s3 wc_benth_window (269 WC window)     -> window_K.json
#   s4 final_gate rows (269 values)        -> rows269_K.json
#   s5 pair_terms (270 term shares, R_eff) -> terms_K.json
#   s6 combined (270 p x F)                -> comb_K.json
#   s7 decompose_ling (270 members, ling)  -> dec_K.json      (~2 min)
#   s8 gate_ling central (270 values)      -> final_K.json
# then (not here): s9_reproduction.py (K 0.06), s10_edits.py + apply_280.py (K 0.09), gate_280.py, sensitivity.py.
set -e
R="$1"; S="$2"; shift 2
D="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$S/tmp" "$S/libs"
LIB269="$S/libs/fg_library_post269.yaml"
[ -f "$LIB269" ] || (cd "$R" && git show 7153a6c:fgconfig/fg_library.yaml > "$LIB269")
one() {
  K="$1"; T="$S/tmp_$K"; mkdir -p "$T"
  python3 "$D/s1_band.py" "$R" "$K" "$T" "$S/band_$K.json" > "$S/s1_$K.txt" 2>&1
  python3 "$D/s2_benth.py" "$R" "$K" "$S/band_$K.json" "$T" "$S/benth_$K.json" > "$S/s2_$K.txt" 2>&1
  python3 "$D/s3_window.py" "$R" "$S/band_$K.json" "$T" "$S/window_$K.json" > "$S/s3_$K.txt" 2>&1
  python3 "$D/s4_rows269.py" "$R" "$S/band_$K.json" "$S/benth_$K.json" "$S/window_$K.json" "$S/rows269_$K.json" > "$S/s4_$K.txt" 2>&1
  python3 "$D/s5_terms.py" "$R" "$K" "$S/band_$K.json" "$T" "$S/terms_$K.json" > "$S/s5_$K.txt" 2>&1
  python3 "$D/s6_combined.py" "$R" "$K" "$S/terms_$K.json" "$S/rows269_$K.json" "$S/band_$K.json" "$S/band_0.06.json" \
      "$S/benth_$K.json" "$T" "$S/comb_$K.json" > "$S/s6_$K.txt" 2>&1
  python3 "$D/s7_decompose.py" "$R" "$K" "$S/window_$K.json" "$T" "$S/dec_$K.json" > "$S/s7_$K.txt" 2>&1
  python3 "$D/s8_compose.py" "$R" "$K" "$S/rows269_$K.json" "$S/comb_$K.json" "$S/dec_$K.json" "$LIB269" "$S/libs" \
      "$S/final_$K.json" > "$S/s8_$K.txt" 2>&1
  echo "K_deep $K done"
}
for K in "$@"; do
  if [ "$K" = "0.06" ]; then one "$K"; fi
done
pids=()
for K in "$@"; do
  if [ "$K" != "0.06" ]; then one "$K" & pids+=($!); fi
done
for p in "${pids[@]}"; do wait "$p"; done
