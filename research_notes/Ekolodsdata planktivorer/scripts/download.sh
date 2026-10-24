#!/bin/bash
# Download ICES acoustic-trawl submissions (acoustic + biotic CSV zips) by submission id.
# Usage: download.sh OUTDIR id1 id2 ...   (files are untrusted data; never executed)
out="$1"; shift
mkdir -p "$out"
for id in "$@"; do
  for kind in acoustic biotic; do
    f="$out/${kind}_${id}.zip"
    [ -s "$f" ] || curl -s -f -m 300 -o "$f" "https://acoustic.ices.dk/submissions/download${kind}csv/$id" || echo "fail $kind $id"
  done
done
