"""Download the open files of an NVA (Sikt) registration. Usage: q1_nva.py <registration-id> <outdir>"""
import json, sys, os, urllib.request
rid, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
def get(u, js=True):
    req = urllib.request.Request(u, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r) if js else r.read()
pub = get(f'https://api.nva.unit.no/publication/{rid}')
for a in pub.get('associatedArtifacts', []):
    if a.get('type') != 'OpenFile':
        continue
    link = get(f"https://api.nva.unit.no/publication/{rid}/filelink/{a['identifier']}")['id']
    data = urllib.request.urlopen(link, timeout=120).read()
    p = os.path.join(out, a.get('name') or a['identifier'] + '.pdf')
    open(p, 'wb').write(data)
    print('saved', p, len(data))
