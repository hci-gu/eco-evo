"""Search Europe PMC (no email sent). Usage: python3 -I epmc_search.py "query" [n]"""
import json
import sys
import urllib.parse
import urllib.request

q = sys.argv[1]
n = sys.argv[2] if len(sys.argv) > 2 else "8"
url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=lite&pageSize="
       + n + "&query=" + urllib.parse.quote(q))
with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client"}), timeout=30) as r:
    j = json.load(r)
print("hits:", j.get("hitCount"))
for w in j.get("resultList", {}).get("result", []):
    print(f"- {w.get('pubYear')} | {w.get('authorString','')[:60]} | {w.get('title')} | {w.get('journalTitle')} | "
          f"doi {w.get('doi')} | pmcid {w.get('pmcid')} | OA {w.get('isOpenAccess')}")
