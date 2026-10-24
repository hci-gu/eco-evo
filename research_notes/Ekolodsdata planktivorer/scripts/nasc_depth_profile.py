"""Vertical distribution of NASC inside the box per cruise (untrusted ICES acoustic zips).

Usage: python3 -I nasc_depth_profile.py OUT_CSV zip1 zip2 ...
Bins: 0-10, 10-20, 20-30, 30-50, 50-100, 100-150, 150-200, 200-300, >300 m (channel upper depth).
Also reports the deepest channel lower bound seen and the share of NASC from channels
whose lower bound is within 10 m of the deepest channel in that log (a near-bottom proxy)."""
import csv, io, sys, zipfile, collections
BOX = (57.7, 58.4, 9.9, 11.3)
BINS = [0, 10, 20, 30, 50, 100, 150, 200, 300, 10000]
out = open(sys.argv[1], 'w', newline='')
w = csv.writer(out)
w.writerow(['cruise', 'year', 'logs', 'max_channel_lower_m'] + [f'share_{BINS[i]}_{BINS[i+1]}' for i in range(len(BINS) - 1)] + ['share_last10m_of_log'])
for zp in sys.argv[2:]:
    with zipfile.ZipFile(zp) as z:
        t = z.read([n for n in z.namelist() if n.endswith('.csv')][0]).decode('utf-8-sig', errors='replace')
    hdr = {}; cruise = None
    perlog = collections.defaultdict(list)
    for row in csv.reader(io.StringIO(t)):
        if len(row) < 2:
            continue
        if row[1] == 'Header':
            hdr[row[0]] = row[2:]; continue
        if row[0] == 'Cruise':
            cruise = dict(zip(hdr['Cruise'], row[2:]))
        if row[0] != 'Data':
            continue
        r = dict(zip(hdr['Data'], row[2:]))
        try:
            lat, lon = float(r['LogLatitude']), float(r['LogLongitude'])
            up, lo, v = float(r['SampleChannelDepthUpper']), float(r['SampleChannelDepthLower']), float(r['DataValue'] or 0)
        except ValueError:
            continue
        if r.get('LogValidity') not in ('V', ''):
            continue
        if BOX[0] <= lat <= BOX[1] and BOX[2] <= lon <= BOX[3]:
            perlog[r['LogTime']].append((up, lo, v))
    if not perlog or cruise is None:
        continue
    acc = collections.Counter(); tot = 0.0; near = 0.0; mx = 0
    for lt, chans in perlog.items():
        deepest = max(c[1] for c in chans); mx = max(mx, deepest)
        for up, lo, v in chans:
            for i in range(len(BINS) - 1):
                if BINS[i] <= up < BINS[i + 1]:
                    acc[i] += v
            tot += v
            if lo >= deepest - 10:
                near += v
    if tot <= 0:
        continue
    w.writerow([cruise.get('CruiseLocalID'), cruise.get('CruiseStartDate', '')[:4], len(perlog), mx] + [f'{acc[i] / tot:.3f}' for i in range(len(BINS) - 1)] + [f'{near / tot:.3f}'])
