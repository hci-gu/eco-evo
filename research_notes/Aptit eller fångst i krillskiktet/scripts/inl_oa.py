"""Print OpenAlex metadata + reconstructed abstract for DOIs or a search query (copied from Krillintag och jaktsatt q1_oa.py).
Usage: python3 -I q1_oa.py doi 10.xxx/yyy ... | python3 -I q1_oa.py search "query" [n]"""
import json, sys, urllib.parse, urllib.request, urllib.error

import time
def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == 3:
                raise
            time.sleep(10 * (k + 1))

def abstract(w):
    inv = w.get('abstract_inverted_index')
    if not inv:
        return '(no abstract)'
    pos = {}
    for word, idx in inv.items():
        for i in idx:
            pos[i] = word
    return ' '.join(pos[i] for i in sorted(pos))

def show(w):
    oa = w.get('open_access') or {}
    print('###', w.get('publication_year'), w.get('doi'), '|', w.get('title'))
    print('   OA:', oa.get('is_oa'), oa.get('oa_url'))
    print('   ', abstract(w))
    print()

mode = sys.argv[1]
if mode == 'doi':
    for d in sys.argv[2:]:
        try:
            show(get('https://api.openalex.org/works/doi:' + d))
        except Exception as e:
            print('ERR', d, e)
else:
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    q = urllib.parse.quote(sys.argv[2])
    for w in get(f'https://api.openalex.org/works?search={q}&per-page={n}')['results']:
        show(w)
