"""Fetch title + abstract from the Semantic Scholar graph API and OpenAlex for DOIs; save raw JSON; print as data.

Usage: python3 -I app_s2.py <new_dir> <doi> [<doi> ...]
"""
import json, os, re, sys, time, urllib.request, urllib.parse

HDR = {"User-Agent": "research-client/1.0"}

def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=40))

def oa_abstract(inv):
    if not inv:
        return None
    pos = []
    for w, idx in inv.items():
        for i in idx:
            pos.append((i, w))
    return " ".join(w for _, w in sorted(pos))

def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for doi in sys.argv[2:]:
        slug = re.sub(r"[^A-Za-z0-9._-]", "_", doi)
        print("=====", doi)
        try:
            r = get("https://api.semanticscholar.org/graph/v1/paper/DOI:" + urllib.parse.quote(doi) + "?fields=title,year,abstract,tldr")
            json.dump(r, open(os.path.join(out, slug + "_s2.json"), "w"))
            print("S2 TITLE:", r.get("title"), r.get("year"))
            print("S2 ABSTRACT:", r.get("abstract"))
        except Exception as e:
            print("s2 ERR", e)
        time.sleep(3)
        try:
            r = get("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi))
            json.dump(r, open(os.path.join(out, slug + "_oa.json"), "w"))
            print("OA ABSTRACT:", oa_abstract(r.get("abstract_inverted_index")))
        except Exception as e:
            print("openalex ERR", e)
        time.sleep(2)

if __name__ == "__main__":
    main()
