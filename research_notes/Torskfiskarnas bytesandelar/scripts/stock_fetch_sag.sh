#!/bin/bash
# Download ICES SAG summary tables (JSON; parse only). Usage: stock_fetch_sag.sh <out_dir> <assessmentKey> ...
set -u
out=$1; shift; mkdir -p "$out"
for k in "$@"; do
  curl -s -m 120 --max-filesize 50000000 -o "$out/SummaryTable_$k.json" "https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=$k"
  echo "$k $(wc -c < "$out/SummaryTable_$k.json") bytes"
done
