"""Validation: run the box estimator on HERAS stratum 31 (eastern Skagerrak, geometry read off
the Dana 2019 strata map, see evidence/heras_strata_areas.txt) and compare with the official
international StoX herring biomass for stratum 31 (evidence/heras_per_stratum_3a_2015_2025.txt).

Usage: python3 -I stratum31_check.py NASC_LOGS HAULS CATCH INDIVIDUALS OUT_CSV"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estimate_box as eb  # noqa: E402  (our own script)

OFFICIAL_31_T_KM2 = {2016: 3.92, 2017: 1.94, 2018: 4.31, 2019: 8.67, 2021: 2.84, 2022: 4.84,
                     2023: 3.15, 2024: 0.47, 2025: 4.65}   # herring NSAS+WBSS kt / 6195 km2
OFFICIAL_31_KT = {2016: 24.3, 2017: 12, 2018: 26.7, 2019: 53.7, 2021: 17.6, 2022: 30.0,
                  2023: 19.5, 2024: 2.9, 2025: 28.8}


def in31(lat, lon):
    west = 10.0 + (58.8 - lat) * (0.6 / 1.1)          # 41/31 line (10.0E 58.8N - 10.6E 57.7N)
    south = 57.7 + (lon - 10.6) * (0.1 / 0.8)         # 31/21 line (Skagen - 11.4E 57.8N)
    return 57.6 <= lat <= 59.3 and lon >= west and lat >= south and lon <= 12.0


def main():
    nasc_p, hp, cp, ip, out = sys.argv[1:6]
    eb.HREG = (57.4, 59.4, 9.6, 12.0)
    logs = eb.load_logs(nasc_p, (57.4, 59.4, 9.6, 12.0))
    hauls, ind = eb.load_hauls(hp, cp, ip)
    tables = eb.haul_tables(hauls, ind)
    pos = {}
    for r in csv.DictReader(open(hp), delimiter='\t'):
        pos[(r['cruise'], r['haul'])] = (float(r['lat']), float(r['lon']))
    w = csv.writer(open(out, 'w', newline=''))
    w.writerow(['year', 'logs_in_31', 'hauls_in_31', 'nasc_31', 'ours_herring_t_km2', 'ours_sprat_t_km2',
                'ours_herring_kt_6195km2', 'official_herring_t_km2', 'official_kt', 'ours_over_official',
                'box_herring_t_km2', 'box_over_official31'])
    box = {int(r['year']): float(r['herring_t_km2']) for r in csv.DictReader(open(os.path.join(os.path.dirname(out), 'acoustic_box_by_year.csv')))}
    for y in sorted(logs):
        recs = [r for r in logs[y] if in31(r[3], r[4])]
        hl = [k for k in tables if hauls[k]['year'] == y and in31(*pos[k])]
        if not recs or not hl:
            continue
        nasc, nmi = eb.mean_nasc(eb.segments(recs))
        res = eb.allocate(nasc, hl, tables)
        h = res.get('herring', 0.0); off = OFFICIAL_31_T_KM2.get(y)
        w.writerow([y, len(recs), len(hl), f'{sum(nasc.values()):.0f}', f'{h:.2f}', f"{res.get('sprat', 0):.2f}",
                    f'{h * 6195 / 1000:.1f}', off if off else '', OFFICIAL_31_KT.get(y, ''),
                    f'{h / off:.2f}' if off else '', box.get(y, ''), f'{box[y] / off:.2f}' if off and y in box else ''])


if __name__ == '__main__':
    main()
