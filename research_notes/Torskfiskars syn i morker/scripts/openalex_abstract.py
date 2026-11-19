"""Print OpenAlex metadata, OA locations and reconstructed abstract for DOIs (no email sent).

Usage: python3 -I openalex_abstract.py DOI [DOI ...]
"""
import json
import sys
import time
import urllib.parse
import urllib.request


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research-client"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


for d in sys.argv[1:]:
    print("==", d)
    try:
        j = get("https://api.openalex.org/works/doi:" + urllib.parse.quote(d, safe="/"))
    except Exception as e:  # noqa: BLE001
        print("  error:", e)
        time.sleep(2)
        continue
    print("  title:", j.get("title"), "|", j.get("publication_year"))
    oa = j.get("open_access") or {}
    print("  oa:", oa.get("is_oa"), oa.get("oa_status"), oa.get("oa_url"))
    for loc in (j.get("locations") or [])[:6]:
        print("   loc:", loc.get("pdf_url"), "|", loc.get("landing_page_url"), "|", loc.get("license"))
    inv = j.get("abstract_inverted_index")
    if inv:
        pos = {}
        for w, idx in inv.items():
            for i in idx:
                pos[i] = w
        print("  abstract:", " ".join(pos[i] for i in sorted(pos)))
    else:
        print("  abstract: none")
    time.sleep(2)
