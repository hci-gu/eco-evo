"""Query OpenAlex by free-text search; save raw JSON and print title, DOI, OA URL, abstract.
Run: python3 -I scripts/Q2Q3_openalex.py <out_dir> <tag> "<query>" [n]
"""
import json
import os
import sys
import urllib.parse
import urllib.request

out, tag, q = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 4
os.makedirs(out, exist_ok=True)
url = 'https://api.openalex.org/works?' + urllib.parse.urlencode({'search': q, 'per-page': n})
with urllib.request.urlopen(url, timeout=60) as r:
    d = json.load(r)
json.dump(d, open(os.path.join(out, f'{tag}.json'), 'w'))
for w in d['results']:
    inv = w.get('abstract_inverted_index') or {}
    pos = sorted((p, k) for k, ps in inv.items() for p in ps)
    ab = ' '.join(k for _, k in pos)
    oa = (w.get('best_oa_location') or {})
    print('##', w.get('publication_year'), w.get('display_name'))
    print('   doi', w.get('doi'), '| oa', oa.get('pdf_url') or oa.get('landing_page_url'))
    print('   ', ab[:1800])
