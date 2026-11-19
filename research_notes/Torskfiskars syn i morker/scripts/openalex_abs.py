"""Print OpenAlex records (title, DOI, OA status, OA URL, licence, abstract).

Usage: python3 -I openalex_abs.py "doi:10.xxxx/yyy" ["search:free text" ...]
No email or API key is sent.
"""
import json
import sys
import urllib.parse
import urllib.request

BASE = "https://api.openalex.org/works"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "literature-lookup/0.1"})
    with urllib.request.urlopen(req, timeout=30) as fh:
        return json.load(fh)


def _abstract(inv):
    if not inv:
        return "NO ABSTRACT"
    pos = {}
    for word, idx in inv.items():
        for i in idx:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def _show(w):
    loc = w.get("best_oa_location") or {}
    print("TITLE:", w.get("title"))
    print("DOI:", w.get("doi"), "YEAR:", w.get("publication_year"))
    print("OA:", (w.get("open_access") or {}).get("oa_status"),
          "URL:", loc.get("pdf_url") or loc.get("landing_page_url"),
          "LICENCE:", loc.get("license"))
    print("ABSTRACT:", _abstract(w.get("abstract_inverted_index")))
    print("-" * 60)


def main():
    for arg in sys.argv[1:]:
        kind, _, val = arg.partition(":")
        try:
            if kind == "doi":
                _show(_get(BASE + "/doi:" + urllib.parse.quote(val, safe="/")))
            else:
                d = _get(BASE + "?per-page=3&search=" + urllib.parse.quote(val))
                for w in d.get("results", []):
                    _show(w)
        except Exception as exc:  # noqa: BLE001
            print("ERROR", arg, exc)
            print("-" * 60)


if __name__ == "__main__":
    main()
