"""Extract NASC per log and SaCategory from ICES acoustic CSV zips (untrusted data).

Usage: python3 -I acoustic_box.py OUT_TSV zip1 zip2 ...
Writes one row per (cruise, log, category) for logs within a generous region
(54.5-59.5 N, 5-13 E); NASC summed over depth channels of the log.
Also prints, per cruise, the categories and channel layout seen."""
import csv, io, sys, zipfile, collections

LAT0, LAT1, LON0, LON1 = 54.5, 59.5, 5.0, 13.0

def records(zpath):
    with zipfile.ZipFile(zpath) as z:
        name = [n for n in z.namelist() if n.endswith('.csv')][0]
        text = z.read(name).decode('utf-8-sig', errors='replace')
    hdr = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2:
            continue
        if row[1] == 'Header':
            hdr[row[0]] = row[2:]
        elif row[1] == 'Record' and row[0] in hdr:
            yield row[0], dict(zip(hdr[row[0]], row[2:]))

out = open(sys.argv[1], 'w')
out.write('zip\tcruise\tsurvey\tcountry\tplatform\tlog_time\tlat\tlon\tinterval_nmi\tbottom\tvalidity\tcategory\tn_channels\tch_upper_min\tch_lower_max\tnasc\n')
for zp in sys.argv[2:]:
    acc = collections.defaultdict(lambda: [0.0, 0, 1e9, -1e9])
    meta = {}; cruise = {}; echo = []
    for kind, r in records(zp):
        if kind == 'Cruise':
            cruise = r
        elif kind == 'EchoType':
            echo.append(r)
        elif kind == 'Data':
            try:
                lat = float(r['LogLatitude']); lon = float(r['LogLongitude'])
            except (ValueError, KeyError):
                continue
            if not (LAT0 <= lat <= LAT1 and LON0 <= lon <= LON1):
                continue
            key = (r.get('LogTime'), r['LogLatitude'], r['LogLongitude'], r.get('DataSaCategory') or r.get('EchoTypeID') or '')
            a = acc[key]
            try:
                a[0] += float(r.get('DataValue') or 0)
            except ValueError:
                pass
            a[1] += 1
            try:
                a[2] = min(a[2], float(r.get('SampleChannelDepthUpper') or 'nan'))
                a[3] = max(a[3], float(r.get('SampleChannelDepthLower') or 'nan'))
            except ValueError:
                pass
            meta[key[:3]] = (r.get('SamplePingAxisInterval'), r.get('LogBottomDepth'), r.get('LogValidity'))
    cats = collections.Counter(k[3] for k in acc)
    sys.stderr.write(f"{zp}\t{cruise.get('CruiseSurvey')}\t{cruise.get('CruiseCountry')}\t{cruise.get('CruiseLocalID')}\tlogs_in_region={len(meta)}\tcats={dict(cats)}\techotypes={[(e.get('EchoTypeID'), e.get('EchoTypeSpeciesCode'), e.get('EchoTypeSpeciesCategory')) for e in echo][:12]}\n")
    for key, (v, n, lo, hi) in acc.items():
        iv, bd, va = meta[key[:3]]
        out.write('\t'.join(map(str, [zp.split('/')[-1], cruise.get('CruiseLocalID'), cruise.get('CruiseSurvey'), cruise.get('CruiseCountry'), cruise.get('CruisePlatform'), key[0], key[1], key[2], iv, bd, va, key[3], n, lo, hi, v])) + '\n')
