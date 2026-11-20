"""Query OpenAlex (search or DOI), save raw JSON in its own data subdir, print title/doi/abstract.
Usage: python3 -I oa_search.py <outdir_root> <tag> "<query or doi:10.x/...>" [n]"""
import sys, json, os, urllib.request, urllib.parse
root, tag, q = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 5
if q.startswith("doi:"):
    url = "https://api.openalex.org/works/https://doi.org/" + q[4:]
else:
    url = "https://api.openalex.org/works?per-page=%d&search=%s" % (n, urllib.parse.quote(q))
req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
data = json.load(urllib.request.urlopen(req, timeout=60))
d = os.path.join(root, "bud_oa_" + tag)
os.makedirs(d, exist_ok=True)
json.dump(data, open(os.path.join(d, "result.json"), "w"))
works = data.get("results", [data])
def abstract(w):
    inv = w.get("abstract_inverted_index")
    if not inv: return "(no abstract)"
    pos = {}
    for word, ps in inv.items():
        for p in ps: pos[p] = word
    return " ".join(pos[i] for i in sorted(pos))
for w in works:
    print("##", w.get("publication_year"), "|", w.get("title"), "|", w.get("doi"),
          "| OA:", (w.get("open_access") or {}).get("oa_url"))
    print("   ", abstract(w)[:2500])
