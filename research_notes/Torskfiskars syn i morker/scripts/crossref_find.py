"""Find DOIs by bibliographic query via Crossref (no email sent).

Usage: python3 -I crossref_find.py "query" [rows]
"""
import json
import sys
import urllib.parse
import urllib.request

q = sys.argv[1]
rows = sys.argv[2] if len(sys.argv) > 2 else "4"
url = "https://api.crossref.org/works?rows=" + rows + "&query.bibliographic=" + urllib.parse.quote(q)
with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client"}), timeout=40) as r:
    j = json.load(r)
for it in j["message"]["items"]:
    t = (it.get("title") or [""])[0]
    c = (it.get("container-title") or [""])[0]
    y = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
    a = ", ".join(x.get("family", "") for x in it.get("author", [])[:3])
    lic = [l.get("URL") for l in it.get("license", [])]
    print(f"- {y} | {a} | {t} | {c} | {it.get('DOI')} | lic {lic}")
