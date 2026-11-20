"""Print OpenAlex metadata and the reconstructed abstract for DOIs or a title search.

Usage:
  python3 -I openalex_abstract.py doi 10.1111/xyz ...
  python3 -I openalex_abstract.py search "title words" [n]
No contact address is sent.
"""
import json
import sys
import urllib.parse
import urllib.request

BASE = "https://api.openalex.org/works"


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def abstract(w):
    inv = w.get("abstract_inverted_index") or {}
    pos = {}
    for word, idx in inv.items():
        for i in idx:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def show(w):
    auth = ", ".join(a["author"]["display_name"] for a in w.get("authorships", [])[:4])
    src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
    print(f"== {w.get('doi')} | {w.get('publication_year')} | {auth} | {src}")
    print(f"   {w.get('title')}")
    oa = (w.get("open_access") or {}).get("oa_url")
    print(f"   oa_url: {oa}")
    print("   ABSTRACT:", abstract(w) or "(none)")


def main():
    mode = sys.argv[1]
    if mode == "doi":
        for d in sys.argv[2:]:
            try:
                show(get(BASE + "/https://doi.org/" + urllib.parse.quote(d)))
            except Exception as exc:
                print("==", d, "error", exc)
    else:
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        q = urllib.parse.quote(sys.argv[2])
        for w in get(f"{BASE}?search={q}&per-page={n}")["results"]:
            show(w)


if __name__ == "__main__":
    main()
