"""OpenAlex search helper (no email, no key). Usage:
python3 -I oa.py search "query terms" [n]
python3 -I oa.py doi 10.xxxx/yyy      -> title, year, venue, OA url, abstract
"""
import json, sys, urllib.request, urllib.parse

def get(url):
    with urllib.request.urlopen(url, timeout=40) as r:
        return json.load(r)

def abstract(w):
    inv = w.get("abstract_inverted_index") or {}
    pos = {}
    for word, idx in inv.items():
        for i in idx:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))

def show(w, full=False):
    loc = (w.get("best_oa_location") or {})
    print(w.get("publication_year"), w.get("doi"), "|", w.get("title"))
    if full:
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        print("  venue:", src, "| OA:", loc.get("pdf_url") or loc.get("landing_page_url"))
        print("  authors:", ", ".join(a["author"]["display_name"] for a in w.get("authorships", [])))
        print("  abstract:", abstract(w))

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "search":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 15
        q = urllib.parse.quote(sys.argv[2])
        d = get(f"https://api.openalex.org/works?search={q}&per-page={n}")
        for w in d["results"]:
            show(w)
    elif mode == "doi":
        for doi in sys.argv[2:]:
            try:
                w = get("https://api.openalex.org/works/https://doi.org/" + doi)
                show(w, full=True)
            except Exception as e:
                print(doi, "ERROR", e)
            print()
