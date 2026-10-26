"""Look up DOIs (Crossref) and open-access PDF links (Unpaywall) for a list of titles.

Usage: python3 -I members_lookup.py "<title query>" [...]
The Unpaywall contact address is read from the environment variable UPW_EMAIL (never stored).
Output: one line per query: DOI | year | journal | title | best OA url.
"""
import json
import os
import sys
import urllib.parse
import urllib.request

EMAIL = os.environ.get("UPW_EMAIL", "")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research-notes-lookup/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


for q in sys.argv[1:]:
    try:
        if q.startswith("10."):
            items = [get("https://api.crossref.org/works/" + urllib.parse.quote(q))["message"]]
        else:
            u = "https://api.crossref.org/works?rows=2&query.bibliographic=" + urllib.parse.quote(q)
            items = get(u)["message"]["items"]
        for it in items[:2]:
            doi = it.get("DOI", "")
            yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
            jr = (it.get("container-title") or [""])[0]
            ti = (it.get("title") or [""])[0]
            oa = ""
            if EMAIL and doi:
                try:
                    d = get("https://api.unpaywall.org/v2/" + urllib.parse.quote(doi) + "?email=" + EMAIL)
                    loc = d.get("best_oa_location") or {}
                    oa = loc.get("url_for_pdf") or loc.get("url") or "no OA"
                except Exception as e:  # noqa
                    oa = "unpaywall error " + type(e).__name__
            print(f"{q[:40]!r} -> {doi} | {yr} | {jr[:40]} | {ti[:90]} | {oa}")
    except Exception as e:  # noqa
        print(f"{q[:40]!r} -> ERROR {e}")
