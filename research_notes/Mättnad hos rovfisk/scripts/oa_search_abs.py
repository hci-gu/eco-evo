"""Search OpenAlex by title words; print top hit DOI, year, title and abstract."""
import json, sys, urllib.request, urllib.parse
for q in sys.argv[1:]:
    url = "https://api.openalex.org/works?per-page=2&search=" + urllib.parse.quote(q)
    for w in json.load(urllib.request.urlopen(url, timeout=30)).get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        pos = sorted((p, t) for t, ps in inv.items() for p in ps)
        print("==", q, "->", w.get("doi"), w.get("publication_year"), w.get("title"))
        print("ABS:", " ".join(t for _, t in pos)[:1800] or "(none)"); print()
