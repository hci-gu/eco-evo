"""Query Unpaywall for OA locations. Usage: python3 -I q1_unpaywall.py EMAIL DOI [DOI ...]
The email is passed on the command line only and never stored."""
import json, sys, urllib.request, urllib.parse
email = sys.argv[1]
for doi in sys.argv[2:]:
    url = "https://api.unpaywall.org/v2/" + urllib.parse.quote(doi) + "?email=" + urllib.parse.quote(email)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
    except Exception as e:
        print(doi, "ERROR", e); continue
    locs = d.get("oa_locations") or []
    print(doi, "| is_oa:", d.get("is_oa"), "|", (d.get("title") or "")[:80])
    for l in locs:
        print("   ", l.get("url_for_pdf") or l.get("url"), "|", l.get("host_type"), l.get("version"))
