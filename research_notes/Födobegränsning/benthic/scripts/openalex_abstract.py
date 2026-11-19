"""Print title, OA locations and reconstructed abstract of OpenAlex work JSON files (paths as args)."""
import json, sys
for p in sys.argv[1:]:
    j = json.load(open(p))
    inv = j.get('abstract_inverted_index') or {}
    pos = sorted((i, w) for w, ii in inv.items() for i in ii)
    print('====', j.get('doi'), '|', j.get('title'), j.get('publication_year'))
    for l in j.get('locations', []):
        if l.get('pdf_url') or l.get('is_oa'):
            print('  loc', l.get('pdf_url'), l.get('landing_page_url'))
    print(' '.join(w for _, w in pos) or '(no abstract)')
