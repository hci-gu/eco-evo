"""Print title, year, venue and reconstructed abstract for DOIs from OpenAlex.

Usage: python3 -I openalex_abstracts.py <out_dir> DOI [DOI ...]
Saves each raw JSON to <out_dir>/<doi with / -> _>.json (untrusted data, only parsed).
"""
import json
import os
import sys
import urllib.parse
import urllib.request

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for doi in sys.argv[2:]:
    url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi, safe="/.")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read()
    except Exception as e:  # record and continue
        print("==", doi, "FAILED", e)
        continue
    with open(os.path.join(out, doi.replace("/", "_") + ".json"), "wb") as f:
        f.write(raw)
    j = json.loads(raw)
    inv = j.get("abstract_inverted_index") or {}
    words = sorted((p, w) for w, ps in inv.items() for p in ps)
    venue = ((j.get("primary_location") or {}).get("source") or {}).get("display_name")
    print("==", doi)
    print("  ", j.get("title"), "|", j.get("publication_year"), "|", venue)
    print("  ", " ".join(w for _, w in words) if words else "(no abstract)")
    oa = j.get("open_access") or {}
    print("   oa:", oa.get("is_oa"), oa.get("oa_url"))
