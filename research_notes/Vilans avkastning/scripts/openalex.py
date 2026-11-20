"""Fetch OpenAlex records (title, year, venue, doi, abstract) and save them as JSON.

Usage: python3 -I openalex.py <new data dir> doi <doi> [<doi> ...]
       python3 -I openalex.py <new data dir> search "<query>" [n]
No email or other identifier is sent.
"""
import json, os, sys, urllib.request, urllib.parse

HDR = {"User-Agent": "research-client/1.0"}

def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=30))

def abstract(rec):
    inv = rec.get("abstract_inverted_index")
    if not inv:
        return None
    pos = {}
    for w, idx in inv.items():
        for i in idx:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))

def show(rec):
    src = ((rec.get("primary_location") or {}).get("source") or {}).get("display_name")
    out = {"title": rec.get("title"), "year": rec.get("publication_year"), "venue": src,
           "doi": rec.get("doi"), "biblio": rec.get("biblio"), "abstract": abstract(rec),
           "oa_url": (rec.get("open_access") or {}).get("oa_url")}
    return out

def main():
    d, mode = sys.argv[1], sys.argv[2]
    os.makedirs(d, exist_ok=True)
    recs = []
    if mode == "doi":
        for doi in sys.argv[3:]:
            try:
                recs.append(show(get("https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi))))
            except Exception as e:
                recs.append({"doi": doi, "error": str(e)})
    else:
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        r = get("https://api.openalex.org/works?per-page=%d&search=%s" % (n, urllib.parse.quote(sys.argv[3])))
        recs = [show(x) for x in r.get("results", [])]
    with open(os.path.join(d, "openalex_%d.json" % len(os.listdir(d))), "w") as f:
        json.dump(recs, f, indent=1)
    for x in recs:
        print("==", x.get("year"), x.get("venue"), x.get("doi"), x.get("biblio"))
        print("  ", x.get("title"))
        print("  OA:", x.get("oa_url"))
        print("  ABS:", x.get("abstract") or x.get("error"))

if __name__ == "__main__":
    main()
