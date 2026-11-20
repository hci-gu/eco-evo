"""Crossref bibliographic lookup: top 2 hits per query. Usage: python3 -I st_crossref.py <out_dir> "<q>" ..."""
import json, os, sys, urllib.request, urllib.parse
HDR = {"User-Agent": "research-client/1.0"}
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for q in sys.argv[2:]:
    try:
        rec = json.load(urllib.request.urlopen(urllib.request.Request(
            "https://api.crossref.org/works?rows=2&select=DOI,title,issued,container-title,volume,page&query.bibliographic=" + urllib.parse.quote(q), headers=HDR), timeout=30))
    except Exception as e:
        print(q, "ERR", e); continue
    json.dump(rec, open(os.path.join(out, "cr_" + "".join(c if c.isalnum() else "_" for c in q)[:50] + ".json"), "w"))
    print("=== Q:", q)
    for w in rec["message"]["items"]:
        print("-", w.get("DOI"), (w.get("issued") or {}).get("date-parts", [[None]])[0][0], (w.get("container-title") or [""])[0], w.get("volume"), w.get("page"), "|", (w.get("title") or [""])[0][:110])
