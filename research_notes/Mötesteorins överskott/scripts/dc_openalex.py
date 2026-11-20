"""Fetch abstract (inverted index) from OpenAlex for DOIs; print title + abstract. No contact details sent.
Usage: python3 -I dc_openalex.py <out_dir> <doi> ..."""
import json, os, re, sys, urllib.request, urllib.parse
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
import time
for doi in sys.argv[2:]:
    time.sleep(3)
    url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi)
    try:
        rec = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client/1.0"}), timeout=30))
    except Exception as e:
        print("==", doi, "ERROR", e); continue
    inv = rec.get("abstract_inverted_index") or {}
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    ab = " ".join(w for _, w in pos)
    open(os.path.join(out, re.sub(r"[^A-Za-z0-9._-]", "_", doi) + ".txt"), "w").write(ab)
    print("==", doi, "|", rec.get("title"), "|", rec.get("publication_year"), "| OA:", (rec.get("open_access") or {}).get("oa_url"))
    print(ab or "(no abstract)")
