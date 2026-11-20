"""Fetch titles/abstracts from the Semantic Scholar graph API and OpenAlex for DOIs given as arguments.
Usage: python3 -I q1_abstracts.py DOI [DOI ...]"""
import json, sys, time, urllib.request, urllib.parse
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research-client/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)
for doi in sys.argv[1:]:
    print("=" * 100); print(doi)
    try:
        d = get("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi))
        inv = d.get("abstract_inverted_index")
        print("TITLE:", d.get("title"), "|", d.get("publication_year"))
        if inv:
            pos = {}
            for w, idx in inv.items():
                for i in idx: pos[i] = w
            print("OPENALEX ABSTRACT:", " ".join(pos[i] for i in sorted(pos)))
        else:
            print("OPENALEX: no abstract")
    except Exception as e:
        print("OPENALEX ERROR", e)
    time.sleep(1)
