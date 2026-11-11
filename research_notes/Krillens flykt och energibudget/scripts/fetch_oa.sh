#!/bin/bash
# Fetch an open-access PDF for a DOI via Unpaywall and convert it to text.
# Usage: UNPAYWALL_EMAIL=... fetch_oa.sh <doi> <outdir>
# The email is read from the environment only; it is never written to disk or echoed.
set -u
DOI="$1"; OUT="$2"
mkdir -p "$OUT"
J=$(curl -s "https://api.unpaywall.org/v2/${DOI}?email=${UNPAYWALL_EMAIL}")
URL=$(printf '%s' "$J" | python3 -I -c "import sys,json
try:
    d=json.load(sys.stdin)
except Exception:
    print(''); sys.exit()
locs=[l for l in (d.get('oa_locations') or []) if l.get('url_for_pdf')]
print(locs[0]['url_for_pdf'] if locs else '')")
LIC=$(printf '%s' "$J" | python3 -I -c "import sys,json
try:
    d=json.load(sys.stdin); b=d.get('best_oa_location') or {}
    print(b.get('license'), b.get('host_type'), b.get('version'))
except Exception:
    print('none')")
echo "doi=$DOI pdf=$URL licence=$LIC"
if [ -n "$URL" ]; then
  curl -sL -A "Mozilla/5.0" -o "$OUT/paper.pdf" "$URL"
  if file "$OUT/paper.pdf" | grep -q PDF; then
    pdftotext -layout "$OUT/paper.pdf" "$OUT/paper.txt"
    echo "ok $(wc -c < "$OUT/paper.txt") chars"
  else
    echo "not a pdf"; rm -f "$OUT/paper.pdf"
  fi
fi
printf 'doi: %s\nsource: %s\nlicence/host/version: %s\n' "$DOI" "$URL" "$LIC" > "$OUT/SOURCE.txt"
