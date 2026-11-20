"""Print OpenAlex metadata + reconstructed abstract for DOIs given on argv."""
import json, sys, urllib.request
for doi in sys.argv[1:]:
    url = "https://api.openalex.org/works/https://doi.org/" + doi
    try:
        w = json.load(urllib.request.urlopen(url, timeout=30))
    except Exception as e:
        print("==", doi, "ERR", e); continue
    inv = w.get("abstract_inverted_index") or {}
    pos = sorted((p, t) for t, ps in inv.items() for p in ps)
    auth = ", ".join(a["author"]["display_name"] for a in w.get("authorships", [])[:5])
    src = (w.get("primary_location") or {}).get("source") or {}
    print("==", doi, "|", w.get("publication_year"), "|", auth, "|", src.get("display_name"))
    print("TITLE:", w.get("title"))
    print("OA:", (w.get("open_access") or {}).get("oa_url"))
    print("ABS:", " ".join(t for _, t in pos) or "(none)")
    print()
