"""Look up DOIs on Crossref by bibliographic query (no contact details sent).
Usage: python3 -I scripts/ov_crossref.py "<query>" [...]"""
import json, sys, urllib.request, urllib.parse
for q in sys.argv[1:]:
    url = "https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(q)
    r = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client/1.0"}), timeout=30))
    print("Q:", q)
    for it in r["message"]["items"]:
        print("  ", it.get("DOI"), "|", (it.get("title") or [""])[0][:100], "|", it.get("container-title"), it.get("issued", {}).get("date-parts"))
