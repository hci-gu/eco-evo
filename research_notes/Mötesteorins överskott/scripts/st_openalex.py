"""Search OpenAlex by free text; save JSON hits to <out_dir>/<n>.json and print DOI, year, OA, title, abstract.

Usage: python3 -I st_openalex.py <out_dir> "<query>" ["<query>" ...]
No contact details are sent.
"""
import json, os, sys, urllib.request, urllib.parse

HDR = {"User-Agent": "research-client/1.0"}

def abstract(inv):
    if not inv:
        return ""
    pos = []
    for w, idx in inv.items():
        for i in idx:
            pos.append((i, w))
    return " ".join(w for _, w in sorted(pos))

def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for qi, q in enumerate(sys.argv[2:]):
        url = "https://api.openalex.org/works?per-page=4&search=" + urllib.parse.quote(q)
        try:
            rec = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=30))
        except Exception as e:
            print("Q", q, "ERROR", e); continue
        fn = os.path.join(out, "q_%s.json" % "".join(c if c.isalnum() else "_" for c in q)[:60])
        json.dump(rec, open(fn, "w"))
        print("=== Q:", q)
        for w in rec.get("results", []):
            oa = (w.get("open_access") or {})
            print("-", w.get("doi"), w.get("publication_year"), "OA" if oa.get("is_oa") else "closed", oa.get("oa_url"))
            print("  ", (w.get("title") or "")[:150])
            ab = abstract(w.get("abstract_inverted_index"))
            if ab and "--abs" in os.environ.get("ST_FLAGS", "--abs"):
                print("   ABS:", ab[:1500])

if __name__ == "__main__":
    main()
