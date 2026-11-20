"""Query Unpaywall for DOIs; save the record and any OA PDF under <data_dir>/upw_<doi>/.

Usage: UPW_EMAIL=<address> python3 -I unpaywall.py <data_dir> <doi> [<doi> ...]
The address is read from the environment only and is never written to disk.
"""
import json, os, re, sys, urllib.request, urllib.parse

HDR = {"User-Agent": "research-client/1.0"}

def slug(doi):
    return re.sub(r"[^A-Za-z0-9._-]", "_", doi)

def main():
    email = os.environ["UPW_EMAIL"]
    out = sys.argv[1]
    for doi in sys.argv[2:]:
        d = os.path.join(out, "upw_" + slug(doi))
        os.makedirs(d, exist_ok=True)
        url = "https://api.unpaywall.org/v2/" + urllib.parse.quote(doi) + "?email=" + urllib.parse.quote(email)
        try:
            rec = json.load(urllib.request.urlopen(url, timeout=30))
        except Exception as e:
            print(doi, "ERROR", e); continue
        keep = {k: rec.get(k) for k in ("doi", "title", "year", "journal_name", "is_oa", "best_oa_location")}
        with open(os.path.join(d, "record.json"), "w") as f:
            json.dump(keep, f, indent=1)
        loc = rec.get("best_oa_location") or {}
        pdf = loc.get("url_for_pdf")
        print(doi, rec.get("is_oa"), (rec.get("title") or "")[:80], pdf)
        if pdf:
            try:
                data = urllib.request.urlopen(urllib.request.Request(pdf, headers=HDR), timeout=60).read()
                if data[:4] == b"%PDF":
                    open(os.path.join(d, "paper.pdf"), "wb").write(data)
                    print("   saved pdf", len(data))
                else:
                    print("   not a pdf")
            except Exception as e:
                print("   pdf error", e)

if __name__ == "__main__":
    main()
