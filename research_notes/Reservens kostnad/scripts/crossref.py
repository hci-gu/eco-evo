"""Print title, journal, year and abstract (if any) from Crossref for each DOI arg."""
import json, re, sys, urllib.request, urllib.parse, html
for doi in sys.argv[1:]:
    try:
        d = json.load(urllib.request.urlopen("https://api.crossref.org/works/" + urllib.parse.quote(doi), timeout=30))["message"]
    except Exception as e:
        print(doi, "ERR", e); continue
    ab = html.unescape(re.sub(r"<[^>]+>", " ", d.get("abstract", "") or ""))
    ab = re.sub(r"\s+", " ", ab).strip()
    yr = (d.get("issued", {}).get("date-parts") or [[None]])[0][0]
    au = ", ".join(a.get("family", "") for a in d.get("author", [])[:5])
    print("==", doi, "|", au, yr, "|", (d.get("title") or [""])[0], "|", (d.get("container-title") or [""])[0], d.get("volume"), d.get("page"))
    print("   ABSTRACT:", ab[:2500] if ab else "(none)")
