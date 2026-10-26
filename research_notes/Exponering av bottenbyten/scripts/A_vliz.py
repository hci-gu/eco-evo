"""Fetch VLIZ/marineinfo publication JSON records and print title, abstract, availability.
Usage: python3 -I A_vliz.py OUTDIR ID [ID ...]"""
import json, sys, os, re, subprocess
out = sys.argv[1]
for pid in sys.argv[2:]:
    fn = os.path.join(out, pid + '.json')
    if not os.path.exists(fn):
        subprocess.run(['curl', '-sL', '-m', '40', 'https://marineinfo.org/id/publication/' + pid, '-o', fn])
    try:
        d = json.load(open(fn))
    except Exception as e:
        print('==', pid, 'ERR', e); continue
    r = d.get('refrec', {})
    tag = lambda s: re.sub('<[^>]+>', '', s or '')
    print('==', pid, tag(r.get('RR')))
    print('   ABS:', tag(r.get('Englishabstract'))[:4000])
    for a in d.get('availability') or []:
        print('   AV:', a.get('Description'), a.get('URL'), a.get('URLLocation'), a.get('FileName', ''))
    print('   URLS:', d.get('urls'))
