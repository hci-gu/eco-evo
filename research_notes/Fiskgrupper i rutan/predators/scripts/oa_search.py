"""Query OpenAlex for works and print title, year, DOI, OA URL and reconstructed abstract.
Usage: python3 -I oa_search.py "query" [n]   or   python3 -I oa_search.py doi:10.xxxx/yyy
"""
import json, sys, urllib.request, urllib.parse

def abstract(inv):
    if not inv:
        return ""
    pos = {}
    for w, idx in inv.items():
        for i in idx:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))

q = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
if q.startswith("doi:"):
    url = "https://api.openalex.org/works/https://doi.org/" + q[4:]
    items = [json.load(urllib.request.urlopen(url, timeout=30))]
else:
    url = "https://api.openalex.org/works?per-page=%d&search=%s" % (n, urllib.parse.quote(q))
    items = json.load(urllib.request.urlopen(url, timeout=30))["results"]
for w in items:
    oa = (w.get("open_access") or {}).get("oa_url")
    print("==", w.get("publication_year"), "|", w.get("title"))
    print("   doi:", w.get("doi"), "| oa:", oa)
    a = abstract(w.get("abstract_inverted_index"))
    if a:
        print("   ABS:", a[:1800])
