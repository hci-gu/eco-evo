"""Look up DOIs on Crossref by bibliographic query; print top hits. No contact details are sent.

Usage: python3 -I app_crossref.py <out_dir> "<query>" ["<query>" ...]
Saves the raw JSON of each query under <out_dir>/q<i>.json (out_dir must be new/empty or own).
"""
import json, os, sys, time, urllib.request, urllib.parse

HDR = {"User-Agent": "research-client/1.0"}

def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for i, q in enumerate(sys.argv[2:]):
        time.sleep(4)
        url = "https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(q)
        try:
            rec = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=40))
        except Exception as e:
            print("ERR", q, e); continue
        json.dump(rec, open(os.path.join(out, f"q{i}.json"), "w"))
        print("== ", q)
        for it in rec["message"]["items"]:
            t = (it.get("title") or [""])[0][:110]
            yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
            cont = (it.get("container-title") or [""])[0][:40]
            print("   ", it.get("DOI"), yr, cont, "|", t)

if __name__ == "__main__":
    main()
