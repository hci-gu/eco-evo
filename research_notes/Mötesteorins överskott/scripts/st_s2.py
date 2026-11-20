"""Search Semantic Scholar (fallback Crossref) by free text; print DOI, year, OA pdf, title, abstract.

Usage: python3 -I st_s2.py <out_dir> "<query>" ...
"""
import json, os, sys, time, urllib.request, urllib.parse

HDR = {"User-Agent": "research-client/1.0"}

def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=30))

def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for q in sys.argv[2:]:
        print("=== Q:", q)
        tag = "".join(c if c.isalnum() else "_" for c in q)[:60]
        try:
            rec = get("https://api.semanticscholar.org/graph/v1/paper/search?limit=4&fields=title,year,externalIds,abstract,openAccessPdf,venue&query=" + urllib.parse.quote(q))
            json.dump(rec, open(os.path.join(out, "s2_" + tag + ".json"), "w"))
            for w in rec.get("data", []):
                print("-", (w.get("externalIds") or {}).get("DOI"), w.get("year"), w.get("venue"), (w.get("openAccessPdf") or {}).get("url"))
                print("  ", (w.get("title") or "")[:150])
                if w.get("abstract"):
                    print("   ABS:", w["abstract"][:1500])
        except Exception as e:
            print("  S2 error", e, "-> crossref")
            try:
                rec = get("https://api.crossref.org/works?rows=4&select=DOI,title,issued,container-title,abstract&query.bibliographic=" + urllib.parse.quote(q))
                json.dump(rec, open(os.path.join(out, "cr_" + tag + ".json"), "w"))
                for w in rec["message"]["items"]:
                    print("-", w.get("DOI"), (w.get("issued") or {}).get("date-parts"), (w.get("container-title") or [""])[0])
                    print("  ", (w.get("title") or [""])[0][:150])
                    if w.get("abstract"):
                        print("   ABS:", w["abstract"][:1200])
            except Exception as e2:
                print("  CR error", e2)
        time.sleep(1.5)

if __name__ == "__main__":
    main()
