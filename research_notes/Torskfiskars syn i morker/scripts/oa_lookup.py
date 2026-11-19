"""Look up open-access locations of DOIs via OpenAlex (no email sent).

Usage: python3 -I oa_lookup.py DOI [DOI ...]
       python3 -I oa_lookup.py --search "title words"
"""
import json
import sys
import urllib.parse
import urllib.request


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research-client"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def show(j):
    print("  title:", j.get("title"), "|", j.get("publication_year"), "|", j.get("doi"))
    oa = j.get("open_access") or {}
    print("  oa:", oa.get("is_oa"), oa.get("oa_status"), oa.get("oa_url"))
    for loc in (j.get("locations") or [])[:8]:
        print("   loc:", loc.get("pdf_url"), "|", loc.get("landing_page_url"), "|", loc.get("license"))


def main():
    args = sys.argv[1:]
    if args and args[0] == "--search":
        q = urllib.parse.quote(" ".join(args[1:]))
        j = get(f"https://api.openalex.org/works?search={q}&per-page=6")
        for w in j.get("results", []):
            show(w)
        return
    for d in args:
        print("==", d)
        try:
            show(get("https://api.openalex.org/works/doi:" + urllib.parse.quote(d, safe="/")))
        except Exception as e:  # noqa: BLE001
            print("  error:", e)


if __name__ == "__main__":
    main()
