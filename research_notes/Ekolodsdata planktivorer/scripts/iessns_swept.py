"""Surface-trawl swept-area density of mackerel, herring and other pelagics in the DK
IESSNS hauls in the Skagerrak (57-59 N, 8-12 E). Untrusted ICES biotic zips as input.

Usage: python3 -I iessns_swept.py OUT_CSV H_OPENING_M zip1 zip2 ...
density (t/km2) = catch (kg) / (HaulDistance m * horizontal opening m) * 1000.
Catchability 1 and the 0-~30 m layer only (surface trawl): mackerel below and herring
deeper than the headline are not counted."""
import csv, io, sys, zipfile, collections
NAMES = {'127023': 'mackerel', '126417': 'herring', '126425': 'sprat', '126439': 'blue_whiting',
         '126822': 'horse_mackerel', '126375': 'garfish', '126444': 'norway_pout'}
out, hopen = sys.argv[1], float(sys.argv[2])
rows = []
for zp in sys.argv[3:]:
    with zipfile.ZipFile(zp) as z:
        t = z.read([n for n in z.namelist() if n.endswith('.csv')][0]).decode('utf-8-sig', errors='replace')
    hdr = {}; hauls = {}; catch = collections.defaultdict(float); seen = set()
    for row in csv.reader(io.StringIO(t)):
        if len(row) < 2:
            continue
        if row[1] == 'Header':
            hdr[row[0]] = row[2:]; continue
        if row[0] == 'Haul':
            r = dict(zip(hdr['Haul'], row[2:]))
            try:
                lat, lon = float(r['HaulStartLatitude']), float(r['HaulStartLongitude'])
            except ValueError:
                continue
            if 57.0 <= lat <= 59.0 and 8.0 <= lon <= 12.0 and r.get('HaulValidity', 'V') == 'V':
                hauls[r['HaulNumber']] = (r['CruiseLocalID'], r['HaulStartTime'], lat, lon, r['HaulStatisticalRectangle'], r.get('HaulDistance'), r.get('HaulBottomDepth'))
        elif row[0] == 'Catch':
            r = dict(zip(hdr['Catch'], row[2:]))
            k = (r['HaulNumber'], r['CatchSpeciesCode'], r['CatchSpeciesCategory'])
            if r['HaulNumber'] in hauls and k not in seen and r['CatchSpeciesCode'] in NAMES:
                seen.add(k)
                w = float(r['CatchSpeciesCategoryWeight'] or 0)
                if r.get('CatchWeightUnit') in ('g', 'gr'):
                    w /= 1000
                catch[(r['HaulNumber'], NAMES[r['CatchSpeciesCode']])] += w
    for h, (cr, tm, lat, lon, rc, dist, bd) in hauls.items():
        d = float(dist) if dist else None
        row = [cr, h, tm, lat, lon, rc, bd, dist]
        for sp in ['mackerel', 'herring', 'sprat', 'blue_whiting', 'horse_mackerel', 'garfish']:
            kg = catch.get((h, sp), 0.0)
            row += [f'{kg:.1f}', f'{kg / (d * hopen) * 1000:.3f}' if d else '']
        rows.append(row)
with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    hd = ['cruise', 'haul', 'time', 'lat', 'lon', 'rect', 'bottom_m', 'distance_m']
    for sp in ['mackerel', 'herring', 'sprat', 'blue_whiting', 'horse_mackerel', 'garfish']:
        hd += [sp + '_kg', sp + '_t_km2']
    w.writerow(hd); w.writerows(rows)
