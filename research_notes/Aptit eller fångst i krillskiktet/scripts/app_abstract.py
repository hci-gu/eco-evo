"""Fetch Crossref and Europe PMC records for DOIs; save raw JSON; print title and abstract text as data.

Usage: python3 -I app_abstract.py <new_dir> <doi> [<doi> ...]
"""
import json, os, re, sys, time, urllib.request, urllib.parse

HDR = {"User-Agent": "research-client/1.0"}

def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=40))

def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for doi in sys.argv[2:]:
        slug = re.sub(r"[^A-Za-z0-9._-]", "_", doi)
        print("=====", doi)
        try:
            cr = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
            json.dump(cr, open(os.path.join(out, slug + "_crossref.json"), "w"))
            m = cr["message"]
            print("TITLE:", (m.get("title") or [""])[0])
            ab = m.get("abstract")
            print("CROSSREF ABSTRACT:", re.sub(r"<[^>]+>", " ", ab) if ab else None)
        except Exception as e:
            print("crossref ERR", e)
        time.sleep(2)
        try:
            ep = get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=core&query=DOI:" + urllib.parse.quote('"' + doi + '"'))
            json.dump(ep, open(os.path.join(out, slug + "_epmc.json"), "w"))
            res = ep.get("resultList", {}).get("result", [])
            for r in res[:1]:
                print("EPMC ABSTRACT:", re.sub(r"<[^>]+>", " ", r.get("abstractText", "") or ""))
            if not res:
                print("EPMC: none")
        except Exception as e:
            print("epmc ERR", e)
        time.sleep(2)

if __name__ == "__main__":
    main()
