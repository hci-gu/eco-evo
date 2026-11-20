"""Search OpenAlex for each query string; print top hits with DOI, year, venue."""
import json, sys, time, urllib.request, urllib.parse
for q in sys.argv[1:]:
    url = "https://api.openalex.org/works?per-page=4&search=" + urllib.parse.quote(q)
    req = urllib.request.Request(url, headers={"User-Agent": "lit-review-script"})
    d = {}
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            break
        except Exception as e:
            print("  retry", e); time.sleep(5 * (attempt + 1))
    time.sleep(1.5)
    print("## " + q)
    for w in d.get("results", []):
        v = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        print("  ", w.get("doi"), "|", w.get("publication_year"), "|", (w.get("title") or "")[:110], "|", v, "| OA" if (w.get("open_access") or {}).get("is_oa") else "")
