"""Semantic Scholar batch lookup by DOI; print title, year, abstract, OA pdf, tldr.
Usage: python3 -I st_s2batch.py <out_json> <doi> ...
"""
import json, sys, urllib.request
HDR = {"User-Agent": "research-client/1.0", "Content-Type": "application/json"}
out = sys.argv[1]
ids = ["DOI:" + d for d in sys.argv[2:]]
req = urllib.request.Request("https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,year,abstract,openAccessPdf,tldr,citationCount",
                             data=json.dumps({"ids": ids}).encode(), headers=HDR, method="POST")
rec = json.load(urllib.request.urlopen(req, timeout=60))
json.dump(rec, open(out, "w"))
for d, w in zip(sys.argv[2:], rec):
    print("===", d)
    if not w:
        print("  not found"); continue
    print("  ", w.get("year"), w.get("title"), (w.get("openAccessPdf") or {}).get("url"))
    print("   ABS:", (w.get("abstract") or "")[:2500])
    if w.get("tldr"): print("   TLDR:", w["tldr"].get("text"))
