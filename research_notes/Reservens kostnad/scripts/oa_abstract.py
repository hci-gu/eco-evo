"""Print title/year/venue and abstract from OpenAlex for given DOIs (no email sent).
Usage: python3 -I oa_abstract.py <doi> [...]"""
import json, sys, urllib.request, urllib.parse
for doi in sys.argv[1:]:
    url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi, safe='/()')
    try:
        d = json.load(urllib.request.urlopen(url, timeout=30))
    except Exception as e:
        print(doi, 'ERROR', e); continue
    inv = d.get('abstract_inverted_index') or {}
    pos = {}
    for w, ps in inv.items():
        for p in ps: pos[p] = w
    ab = ' '.join(pos[i] for i in sorted(pos))
    src = ((d.get('primary_location') or {}).get('source') or {}).get('display_name')
    auth = ', '.join(a['author']['display_name'] for a in d.get('authorships', [])[:4])
    print('==', doi, '|', d.get('title'), '|', d.get('publication_year'), '|', src, '|', auth)
    print(ab or '(no abstract)')
    print()
