"""Print OpenAlex title + reconstructed abstract for DOIs. Usage: python3 -I A_openalex.py DOI [DOI...]"""
import json, sys, subprocess, urllib.parse
for doi in sys.argv[1:]:
    url = 'https://api.openalex.org/works/https://doi.org/' + urllib.parse.quote(doi, safe='/()')
    r = subprocess.run(['curl', '-s', '-m', '40', url], capture_output=True, text=True).stdout
    try:
        d = json.loads(r)
    except Exception:
        print('==', doi, 'no record'); continue
    inv = d.get('abstract_inverted_index') or {}
    pos = {}
    for w, ps in inv.items():
        for p in ps: pos[p] = w
    ab = ' '.join(pos[i] for i in sorted(pos))
    print('==', doi, '|', d.get('display_name'), '|', d.get('publication_year'))
    print('   ABS:', ab if ab else '(none)')
    print()
