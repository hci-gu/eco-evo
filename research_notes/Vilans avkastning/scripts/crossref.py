"""Crossref bibliographic search (title, DOI, venue, abstract if deposited); saves JSON.

Usage: python3 -I crossref.py <new data dir> "<query>" [n]
No email or other identifier is sent.
"""
import json, os, re, sys, urllib.request, urllib.parse

HDR = {"User-Agent": "research-client/1.0"}

def main():
    d, q = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    os.makedirs(d, exist_ok=True)
    url = "https://api.crossref.org/works?rows=%d&query.bibliographic=%s" % (n, urllib.parse.quote(q))
    try:
        items = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=30))["message"]["items"]
    except Exception as e:
        print("ERROR", e); return
    out = []
    for it in items:
        rec = {"doi": it.get("DOI"), "title": (it.get("title") or [""])[0],
               "venue": (it.get("container-title") or [""])[0],
               "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
               "vol": it.get("volume"), "page": it.get("page"),
               "abstract": re.sub(r"<[^>]+>", "", it.get("abstract") or "") or None}
        out.append(rec)
        print("==", rec["year"], rec["venue"], rec["vol"], rec["page"], rec["doi"])
        print("  ", rec["title"])
        if rec["abstract"]:
            print("  ABS:", rec["abstract"])
    with open(os.path.join(d, "crossref_%d.json" % len(os.listdir(d))), "w") as f:
        json.dump(out, f, indent=1)

if __name__ == "__main__":
    main()
