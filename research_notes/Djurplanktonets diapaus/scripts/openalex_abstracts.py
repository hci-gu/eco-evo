"""Print OpenAlex records (title, year, venue, abstract, OA url) for DOIs or title searches.

Usage: python3 -I openalex_abstracts.py doi:10.xxxx/yyy "search:title words" ...
Network only (api.openalex.org); nothing is written to disk.
"""
import json
import sys
import urllib.parse
import urllib.request


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research-note"})
    with urllib.request.urlopen(req, timeout=30) as fh:
        return json.load(fh)


def show(d):
    venue = ((d.get("primary_location") or {}).get("source") or {}).get(
        "display_name")
    print(f"TITLE: {d.get('title')} ({d.get('publication_year')}), {venue}")
    print(f"DOI: {d.get('doi')}")
    ii = d.get("abstract_inverted_index")
    if ii:
        pos = {p: w for w, ps in ii.items() for p in ps}
        print("ABSTRACT: " + " ".join(pos[i] for i in sorted(pos)))
    else:
        print("ABSTRACT: (none in OpenAlex)")
    print("OA:", (d.get("open_access") or {}).get("oa_url"))
    print()


for arg in sys.argv[1:]:
    kind, _, val = arg.partition(":")
    try:
        if kind == "doi":
            show(get("https://api.openalex.org/works/doi:" + val))
        else:
            q = urllib.parse.quote(val)
            res = get("https://api.openalex.org/works?per-page=2&search=" + q)
            for d in res.get("results", [])[:2]:
                show(d)
    except Exception as exc:  # network / 404
        print(f"{arg}: failed ({exc})\n")
