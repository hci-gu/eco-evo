#!/bin/bash
# Download SAM input data and result tables of public stockassessment.org runs (parse only,
# never execute). Usage: stock_fetch_sam.sh <out_dir> <run_name> [<run_name> ...]
# Files: data/*.dat (cn, cw, sw, dw, lw, nm, mo, pf, pm, survey), res/*tab*.html, res/titles.cfg.
set -u
out=$1; shift
base=https://stockassessment.org/datadisk/stockassessment/userdirs/user3
for run in "$@"; do
  d="$out/sam_$run"; mkdir -p "$d/data" "$d/res"
  for f in $(curl -s -m 60 "$base/$run/data/" | grep -o 'href="[^"?/][^"]*"' | sed 's/href="//;s/"//' | grep -E '\.(dat|txt)$'); do
    curl -s -m 120 --max-filesize 50000000 -o "$d/data/$f" "$base/$run/data/$f"
  done
  for f in $(curl -s -m 60 "$base/$run/res/" | grep -o 'href="[^"?/][^"]*"' | sed 's/href="//;s/"//' | grep -E '(tab[0-9]+\.html|titles\.cfg)$'); do
    curl -s -m 120 --max-filesize 50000000 -o "$d/res/$f" "$base/$run/res/$f"
  done
  echo "$run: $(ls "$d/data" | wc -l) data files, $(ls "$d/res" | wc -l) result files"
done
