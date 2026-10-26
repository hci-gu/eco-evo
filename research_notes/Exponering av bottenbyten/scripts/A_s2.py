"""Semantic Scholar abstract by DOI. Usage: python3 -I A_s2.py DOI [DOI...]"""
import json, sys, subprocess, time
for doi in sys.argv[1:]:
    url = 'https://api.semanticscholar.org/graph/v1/paper/DOI:' + doi + '?fields=title,year,abstract,openAccessPdf,tldr'
    r = subprocess.run(['curl', '-s', '-m', '40', url], capture_output=True, text=True).stdout
    try:
        d = json.loads(r)
    except Exception:
        print('==', doi, 'bad', r[:200]); continue
    print('==', doi, '|', d.get('title'), d.get('year'))
    print('   ABS:', d.get('abstract'))
    print('   OA:', d.get('openAccessPdf'), '| TLDR:', (d.get('tldr') or {}).get('text'))
    time.sleep(1.5)
