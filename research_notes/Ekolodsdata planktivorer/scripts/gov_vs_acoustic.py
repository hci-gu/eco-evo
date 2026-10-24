"""Year-by-year comparison: GOV herring kg/h (../Fiskgrupper i rutan long series) vs the
acoustic box herring density. Both are our own derived tables.
Usage: python3 -I gov_vs_acoustic.py LONG_SERIES_BY_YEAR ACOUSTIC_BY_YEAR OUT_CSV"""
import csv, math, sys
gov = {int(r['year']): r for r in csv.DictReader(open(sys.argv[1]))}
ac = {int(r['year']): r for r in csv.DictReader(open(sys.argv[2]))}
rows = []
for y in sorted(set(gov) & set(ac)):
    g = gov[y].get('herring_kgh'); n = gov[y].get('Norway pout_kgh')
    if not g:
        continue
    rows.append((y, float(g), float(n or 0), float(ac[y]['herring_t_km2']), float(ac[y]['norway_pout_t_km2'])))
w = csv.writer(open(sys.argv[3], 'w', newline=''))
w.writerow(['year', 'gov_herring_kgh', 'gov_norway_pout_kgh', 'acoustic_herring_t_km2', 'acoustic_norway_pout_t_km2'])
w.writerows(rows)

def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((x - mb) ** 2 for x in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)
la = [math.log(r[1]) for r in rows]; lb = [math.log(r[3]) for r in rows]
w.writerow(['pearson_r_log_herring', f'{pearson(la, lb):.2f}', 'n', len(rows), ''])
