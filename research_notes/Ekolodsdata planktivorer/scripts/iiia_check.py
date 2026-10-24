"""Validation run: the box estimator applied rectangle by rectangle over the whole
Skagerrak + Kattegat (IIIa) Dana coverage, to compare with published HERAS IIIa totals.

Usage: python3 -I iiia_check.py NASC_LOGS HAULS CATCH INDIVIDUALS OUTDIR
Per year and ICES rectangle: box-style allocation with the hauls within +-0.5 deg lat,
+-1 deg lon of the rectangle centre (widened once to +-1 / +-2 if fewer than 3 hauls).
Writes iiia_rectangles.csv (t/km2 per rectangle) and iiia_summary.csv (rectangle-mean
density and the total over the surveyed rectangles, using each rectangle's geometric
sea area times the given water fraction)."""
import csv, math, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estimate_box as eb  # noqa: E402  (our own script, not downloaded code)
from nasc_rectangles import rect, in_IIIa  # noqa: E402

# Approximate water fraction per IIIa rectangle (land masked by eye from the chart;
# only used for the total in tonnes, not for densities).
WATER = {'41G1': 0.6, '41G2': 0.6, '42G1': 0.6, '42G2': 0.5, '43G1': 0.7, '43G2': 0.3, '43G0': 0.5,
         '44G0': 0.8, '44G1': 0.5, '45G0': 1.0, '45G1': 0.4, '46G0': 0.8, '46G1': 0.3, '43F8': 0.6,
         '43F9': 0.4, '44F8': 1.0, '44F9': 0.9, '45F8': 0.8, '45F9': 1.0, '44F7': 1.0, '45F7': 0.9,
         '46F9': 0.6, '46F8': 0.3, '47G0': 0.3}


def rect_centre(code):
    row = int(code[:2]); lat = 36.0 + (row - 1) * 0.5 + 0.25
    lon = (int(code[3]) if code[2] == 'F' else 10 + int(code[3])) + 0.5
    return lat, lon


def main():
    nasc_p, hp, cp, ip, outdir = sys.argv[1:6]
    eb.HREG = (54.5, 59.5, 5.0, 13.0)
    allbox = (54.5, 59.5, 5.0, 13.0)
    logs = eb.load_logs(nasc_p, allbox)
    hauls, ind = eb.load_hauls(hp, cp, ip)
    tables = eb.haul_tables(hauls, ind)
    pos = {}
    for r in csv.DictReader(open(hp), delimiter='\t'):
        pos[(r['cruise'], r['haul'])] = (float(r['lat']), float(r['lon']))
    rows = []; summ = []
    for y in sorted(logs):
        hl_y = [k for k in tables if hauls[k]['year'] == y]
        if not hl_y:
            continue
        byrect = collections.defaultdict(list)
        for rec in logs[y]:
            lat, lon = rec[3], rec[4]
            if in_IIIa(lat, lon):
                byrect[rect(lat, lon)].append(rec)
        dens = {}
        for rc, recs in byrect.items():
            nasc, nmi = eb.mean_nasc([recs])
            if nmi < 10:
                continue
            clat, clon = rect_centre(rc)
            for dl, dn in ((0.5, 1.0), (1.0, 2.0)):
                hl = [k for k in hl_y if abs(pos[k][0] - clat) <= dl and abs(pos[k][1] - clon) <= dn]
                if len(hl) >= 3:
                    break
            res = eb.allocate(nasc, hl, tables)
            dens[rc] = (res, nmi, len(hl))
            rows.append([y, rc, f'{nmi:.0f}', len(hl), f"{sum(nasc.values()):.0f}", f"{res.get('herring', 0):.2f}", f"{res.get('sprat', 0):.2f}"])
        if not dens:
            continue
        her = [d[0].get('herring', 0) for d in dens.values()]
        spr = [d[0].get('sprat', 0) for d in dens.values()]
        tot_h = tot_s = area = 0.0
        for rc, (res, _, _) in dens.items():
            clat, _ = rect_centre(rc)
            a = 30 * 1.852 * 60 * 1.852 * math.cos(math.radians(clat)) * WATER.get(rc, 0.7)
            area += a; tot_h += res.get('herring', 0) * a; tot_s += res.get('sprat', 0) * a
        summ.append([y, len(dens), f'{sum(her) / len(her):.2f}', f'{sum(spr) / len(spr):.2f}', f'{area:.0f}',
                     f'{tot_h / 1000:.0f}', f'{tot_s / 1000:.0f}'])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'iiia_rectangles.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['year', 'rect', 'nmi', 'hauls', 'nasc', 'herring_t_km2', 'sprat_t_km2']); w.writerows(rows)
    with open(os.path.join(outdir, 'iiia_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['year', 'n_rect', 'herring_rect_mean_t_km2', 'sprat_rect_mean_t_km2',
                                       'surveyed_sea_area_km2_approx', 'herring_total_kt', 'sprat_total_kt']); w.writerows(summ)


if __name__ == '__main__':
    main()
