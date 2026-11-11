"""Print OpenAlex abstracts (reconstructed from the inverted index) for DOIs given as arguments.

Usage: python3 -I oa_abstracts.py <doi> [<doi> ...]   (no email or key is sent)
"""
import json
import sys
import urllib.parse
import urllib.request

dois = sys.argv[1:]
flt = "|".join("https://doi.org/" + d.lower() for d in dois)
url = ("https://api.openalex.org/works?per-page=50&select=doi,display_name,publication_year,"
       "authorships,primary_location,abstract_inverted_index&filter=doi:" + urllib.parse.quote(flt, safe=":/|"))
with urllib.request.urlopen(url) as r:
    data = json.load(r)
for w in data["results"]:
    inv = w.get("abstract_inverted_index") or {}
    pos = sorted((p, word) for word, ps in inv.items() for p in ps)
    abstract = " ".join(word for _, word in pos)
    au = ", ".join(a["author"]["display_name"] for a in w["authorships"][:4])
    src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
    print(f"## {w['doi']} | {au} ({w['publication_year']}) {w['display_name']} | {src}")
    print(abstract or "[no abstract]")
    print()
