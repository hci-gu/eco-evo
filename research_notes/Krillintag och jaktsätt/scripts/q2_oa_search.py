"""Q2 helper: OpenAlex title/abstract search, prints year, DOI, title, OA url, and saves the JSON.

Usage: python3 -I q2_oa_search.py OUT_JSON "search terms" [per_page]
No email or personal data is sent.
"""
import json
import sys
import urllib.parse
import urllib.request
import urllib.error


def inverted_to_text(inv):
    if not inv:
        return ""
    pos = []
    for word, idx in inv.items():
        for i in idx:
            pos.append((i, word))
    return " ".join(w for _, w in sorted(pos))


def main():
    out, query = sys.argv[1], sys.argv[2]
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    url = ("https://api.openalex.org/works?search=" + urllib.parse.quote(query)
           + f"&per-page={per_page}")
    req = urllib.request.Request(url, headers={"User-Agent": "research-client/1.0"})
    import time
    data = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(10 * (attempt + 1))
    if data is None:
        sys.exit("OpenAlex: still rate limited")
    with open(out, "w") as f:
        json.dump(data, f)
    for w in data.get("results", []):
        oa = (w.get("open_access") or {}).get("oa_url")
        print(w.get("publication_year"), w.get("doi"), "|", (w.get("title") or "")[:120], "| OA:", oa)
        ab = inverted_to_text(w.get("abstract_inverted_index"))
        if ab and "--abs" in sys.argv:
            print("   ABS:", ab[:1500])


if __name__ == "__main__":
    main()
