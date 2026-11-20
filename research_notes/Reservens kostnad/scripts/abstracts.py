"""Print title, year, venue and abstract for each DOI via OpenAlex (fallback Semantic Scholar)."""
import json, sys, urllib.request, urllib.parse
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "lit-review-script"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)
for doi in sys.argv[1:]:
    ab, title, year, venue = None, "", "", ""
    try:
        d = get("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi))
        title, year = d.get("title"), d.get("publication_year")
        venue = ((d.get("primary_location") or {}).get("source") or {}).get("display_name")
        inv = d.get("abstract_inverted_index")
        if inv:
            pos = {}
            for w, idx in inv.items():
                for i in idx: pos[i] = w
            ab = " ".join(pos[i] for i in sorted(pos))
    except Exception as e:
        title = f"OpenAlex ERR {e}"
    if not ab:
        try:
            d = get("https://api.semanticscholar.org/graph/v1/paper/DOI:" + urllib.parse.quote(doi) + "?fields=title,year,abstract,venue")
            ab = d.get("abstract"); title = title or d.get("title")
        except Exception as e:
            ab = f"(no abstract; S2 ERR {e})"
    print(f"### {doi} | {title} | {year} | {venue}\n{ab}\n")
