"""Search Crossref (query.bibliographic) for each query string; print top hits with DOI, year, container."""
import json, sys, time, urllib.request, urllib.parse
for q in sys.argv[1:]:
    url = "https://api.crossref.org/works?rows=4&select=DOI,title,issued,container-title,abstract&query.bibliographic=" + urllib.parse.quote(q)
    req = urllib.request.Request(url, headers={"User-Agent": "lit-review-script"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            d = json.load(r)
    except Exception as e:
        print("##", q, "ERR", e); continue
    print("## " + q)
    for w in d["message"]["items"]:
        y = (w.get("issued", {}).get("date-parts") or [[None]])[0][0]
        print("  ", w.get("DOI"), "|", y, "|", (w.get("title") or [""])[0][:110], "|", (w.get("container-title") or [""])[0], "| ABS" if w.get("abstract") else "")
    time.sleep(1)
