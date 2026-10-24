"""Distance-weighted mean NASC per ICES rectangle and year (Danish HERAS, RV Dana),
plus the box mean, the Skagerrak + Kattegat (IIIa) mean and the box / IIIa ratio.

Usage: python3 -I nasc_rectangles.py NASC_LOGS OUTDIR
NASC_LOGS: table from acoustic_box.py (derived from untrusted ICES downloads)."""
import csv, math, os, sys, collections

BOX = (57.7, 58.4, 9.9, 11.3)


def rect(lat, lon):
    row = int(math.floor((lat - 36.0) * 2)) + 1
    col = ('F%d' % int(lon)) if lon < 10 else ('G%d' % int(lon - 10))
    return f'{row:02d}{col}'


def in_IIIa(lat, lon):
    """Skagerrak + Kattegat: east of the Hanstholm (57.12 N, 8.6 E) - Lindesnes (58.0 N, 7.0 E)
    line, north of 55.6 N (Belts / Sound boundary, approximate), west of 13 E."""
    if lat < 55.6 or lon > 13.0 or lat > 59.5:
        return False
    # line through (8.6, 57.12) and (7.0, 58.0): points east/north of it are in IIIa
    lat_line = 57.12 + (lon - 8.6) * (58.0 - 57.12) / (7.0 - 8.6)
    return lon >= 7.0 and lat >= lat_line and not (lat < 57.12 and lon < 9.5)


def main():
    path, outdir = sys.argv[1:3]
    tmp = collections.defaultdict(float); meta = {}
    for r in csv.DictReader(open(path), delimiter='\t'):
        if r['survey'] != 'HERAS' or r['country'] != 'DK' or r['validity'] not in ('V', ''):
            continue
        key = (r['cruise'], r['log_time'], r['lat'], r['lon'])
        tmp[key] += float(r['nasc'])
        try:
            iv = float(r['interval_nmi'])
        except ValueError:
            iv = 1.0
        meta[key] = iv
    acc = collections.defaultdict(lambda: [0.0, 0.0])
    for key, v in tmp.items():
        y = int(key[1][:4]); lat, lon = float(key[2]), float(key[3]); iv = meta[key]
        areas = [rect(lat, lon)]
        if BOX[0] <= lat <= BOX[1] and BOX[2] <= lon <= BOX[3]:
            areas.append('BOX')
        if in_IIIa(lat, lon):
            areas.append('IIIa')
        for a in areas:
            acc[(y, a)][0] += v * iv; acc[(y, a)][1] += iv
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'nasc_by_rectangle.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['year', 'area', 'nmi', 'mean_nasc_m2_nmi2'])
        for (y, a), (s, d) in sorted(acc.items()):
            w.writerow([y, a, f'{d:.0f}', f'{s / d:.1f}'])
    years = sorted({y for y, _ in acc})
    with open(os.path.join(outdir, 'nasc_box_vs_IIIa.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        cols = ['44G0', '45G0', '44G1', '45G1', '44F9', '45F9', '46G0', 'BOX', 'IIIa']
        w.writerow(['year'] + [c + '_nasc' for c in cols] + ['IIIa_nmi', 'box_over_IIIa'])
        for y in years:
            vals = []
            for c in cols:
                s, d = acc.get((y, c), (0, 0))
                vals.append(f'{s / d:.0f}' if d else '')
            b = acc.get((y, 'BOX')); t = acc.get((y, 'IIIa'))
            ratio = (b[0] / b[1]) / (t[0] / t[1]) if b and t and t[0] > 0 else float('nan')
            w.writerow([y] + vals + [f'{t[1]:.0f}' if t else '', f'{ratio:.2f}'])


if __name__ == '__main__':
    main()
