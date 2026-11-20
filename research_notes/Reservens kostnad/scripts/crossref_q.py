"""Crossref bibliographic search: print top hit DOI, title, abstract for each query arg."""
import json, re, sys, urllib.request, urllib.parse, html
for q in sys.argv[1:]:
    u = "https://api.crossref.org/works?rows=2&query.bibliographic=" + urllib.parse.quote(q)
    try:
        items = json.load(urllib.request.urlopen(u, timeout=30))["message"]["items"]
    except Exception as e:
        print(q, "ERR", e); continue
    for d in items[:1]:
        ab = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", d.get("abstract", "") or ""))).strip()
        yr = (d.get("issued", {}).get("date-parts") or [[None]])[0][0]
        au = ", ".join(a.get("family", "") for a in d.get("author", [])[:4])
        print("== Q:", q); print("  ", d.get("DOI"), "|", au, yr, "|", (d.get("title") or [""])[0], "|", (d.get("container-title") or [""])[0], d.get("volume"), d.get("page"))
        print("   ABSTRACT:", ab[:2500] if ab else "(none)")
