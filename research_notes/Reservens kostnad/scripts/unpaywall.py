"""Query Unpaywall for an OA PDF URL. Email from env UNPAYWALL_EMAIL (never stored).
Usage: UNPAYWALL_EMAIL=... python3 -I unpaywall.py <doi> [<doi> ...]"""
import json, os, sys, urllib.request, urllib.parse
email = os.environ["UNPAYWALL_EMAIL"]
for doi in sys.argv[1:]:
    url = "https://api.unpaywall.org/v2/%s?email=%s" % (urllib.parse.quote(doi), urllib.parse.quote(email))
    try:
        d = json.load(urllib.request.urlopen(url, timeout=30))
    except Exception as e:
        print(doi, "ERROR", e); continue
    locs = d.get("oa_locations") or []
    pdfs = [l.get("url_for_pdf") for l in locs if l.get("url_for_pdf")]
    lands = [l.get("url") for l in locs]
    print(doi, "| oa:", d.get("is_oa"), "| title:", (d.get("title") or "")[:80])
    for p in pdfs: print("   pdf:", p)
    if not pdfs:
        for p in lands[:2]: print("   land:", p)
