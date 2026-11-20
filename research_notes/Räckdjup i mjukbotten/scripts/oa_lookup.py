"""Look up open-access locations for DOIs via Unpaywall.

Usage: UNPAYWALL_EMAIL=... python3 -I oa_lookup.py DOI [DOI ...]
The contact address is read from the environment only and never stored.
"""
import json
import os
import sys
import urllib.parse
import urllib.request


def lookup(doi, email):
    url = ("https://api.unpaywall.org/v2/" + urllib.parse.quote(doi)
           + "?email=" + urllib.parse.quote(email))
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
    except Exception as exc:  # network or 404
        return f"{doi} | error {exc}"
    locs = d.get("oa_locations") or []
    pdfs = [l.get("url_for_pdf") or l.get("url") for l in locs]
    return f"{doi} | oa={d.get('is_oa')} | {d.get('title')} | {pdfs[:3]}"


def main():
    email = os.environ.get("UNPAYWALL_EMAIL")
    if not email:
        sys.exit("set UNPAYWALL_EMAIL")
    for doi in sys.argv[1:]:
        print(lookup(doi, email))


if __name__ == "__main__":
    main()
