"""IESSNS July mackerel density in and around the model box, the North Sea stratum series, and
the survey-vs-assessment ratio (catchability bound).

Usage: python3 -I iessns_box.py HAUL_CSV OUT_CSV
HAUL_CSV = research_notes/"Ekolodsdata planktivorer"/tables/iessns_dk_skagerrak_swept.csv
(DK IESSNS surface-trawl hauls 57-59 N, 8-12 E, 2018-2026, computed there from the ICES
acoustic-trawl database biotic files with a 65 m horizontal opening).

All numbers typed in below are quoted in ../summary.md with their source.
"""
import csv, sys, math, statistics as st

H_OLD, H_NEW = 65.0, 68.0   # m; Ceton wingspread 63.5-72.5 m (DK cruise reports 2018, 2024, 2025)
SCALE = H_OLD / H_NEW

# model grid corners (lat, lon), SWEREF 99 TM E 211-271 km, N 6414-6474 km
BOX = [(57.774, 10.138), (57.810, 11.146), (58.346, 11.086), (58.312, 10.066)]


def inside(lat, lon, poly=BOX):
    n, c = len(poly), False
    for i in range(n):
        y1, x1 = poly[i]
        y2, x2 = poly[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            x = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x:
                c = not c
    return c


def dist_km(lat, lon):
    """Distance to the nearest box corner/edge sample (coarse, 0 inside)."""
    if inside(lat, lon):
        return 0.0
    best = 1e9
    for i in range(4):
        y1, x1 = BOX[i]
        y2, x2 = BOX[(i + 1) % 4]
        for t in [k / 50 for k in range(51)]:
            y, x = y1 + t * (y2 - y1), x1 + t * (x2 - x1)
            d = math.hypot((lat - y) * 111.2, (lon - x) * 111.2 * math.cos(math.radians(58.1)))
            best = min(best, d)
    return best


rows = list(csv.DictReader(open(sys.argv[1], newline='')))
out = []
for r in rows:
    lat, lon, bd = float(r['lat']), float(r['lon']), float(r['bottom_m'])
    d = dist_km(lat, lon)
    mac = float(r['mackerel_t_km2']) * SCALE
    hom = float(r['horse_mackerel_t_km2']) * SCALE
    if d == 0:
        cls = 'in_box'
    elif d <= 15:
        cls = 'edge_<=15km'
    elif bd >= 400:
        cls = 'trench_>=400m'
    else:
        cls = 'shelf_other'
    hour = int(r['time'][11:13])
    out.append(dict(year=r['cruise'][-4:], haul=r['haul'], time=r['time'], lat=lat, lon=lon,
                    rect=r['rect'], bottom_m=bd, dist_box_km=round(d, 1), cls=cls,
                    day='day' if 4 <= hour < 20 else 'night',
                    mackerel_t_km2=round(mac, 3), horse_mackerel_t_km2=round(hom, 4)))

with open(sys.argv[2], 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)


def summ(vals, label):
    vals = sorted(vals)
    gm = math.exp(st.mean(math.log(max(v, 0.01)) for v in vals))
    print(f'{label:34s} n={len(vals):2d} mean {st.mean(vals):5.2f} median {st.median(vals):5.2f} '
          f'geomean {gm:5.2f} min {vals[0]:5.2f} max {vals[-1]:5.2f}')


print(f'Horizontal opening rescaled {H_OLD} -> {H_NEW} m (x{SCALE:.3f}).')
print('\n== DK IESSNS early-July surface-trawl mackerel, t/km2 (0-~30 m, q = 1) ==')
for o in out:
    print(f"  {o['year']} h{o['haul']} {o['time'][5:16]} {o['rect']} {o['bottom_m']:5.0f} m "
          f"{o['dist_box_km']:5.1f} km {o['cls']:14s} {o['day']:5s} mac {o['mackerel_t_km2']:6.3f}"
          f" hom {o['horse_mackerel_t_km2']:.4f}")
summ([o['mackerel_t_km2'] for o in out if o['cls'] == 'in_box'], 'in box')
summ([o['mackerel_t_km2'] for o in out if o['cls'] in ('in_box', 'edge_<=15km')], 'in box + edge <=15 km')
summ([o['mackerel_t_km2'] for o in out if o['cls'] == 'trench_>=400m'], 'Trench >=400 m (west of box)')
summ([o['mackerel_t_km2'] for o in out if o['cls'] == 'shelf_other'], 'shelf, other (Danish side)')
summ([o['mackerel_t_km2'] for o in out], 'all Skagerrak hauls 57-59N 8-12E')
summ([o['mackerel_t_km2'] for o in out if 100 <= o['bottom_m'] <= 320], 'bottom 100-320 m (box depths)')
print('horse mackerel, all hauls: max', max(o['horse_mackerel_t_km2'] for o in out),
      'mean', round(st.mean(o['horse_mackerel_t_km2'] for o in out), 4))

# ---------- North Sea stratum (IESSNS stratum 13, 50 m isobath to 60 N, Skagerrak to 11 E) ----------
# Ceton survey-mean density, kg/km2 (IESSNS reports 2022 App.1, 2024 App.1, 2025 App.1)
ceton_kg = {2018: 1743, 2019: 1009, 2020: 1318, 2021: 2429, 2022: 1689, 2023: 2362, 2024: 2004, 2025: 2060}
# StoX biomass (t) and stratum area (km2): 2018-2023 IESSNS 2024 Table A1-2 (baseline);
# 2024 baseline 556 655 t, 2025 bootstrap 586 020 t, area 285 781 km2 (2022: 278 525 km2)
stox_t = {2018: 387613, 2019: 223112, 2020: 257079, 2021: 560198, 2022: 472627, 2023: 648919,
          2024: 556655, 2025: 586020}
area = {y: (278525 if y <= 2022 else 285781) for y in stox_t}
print('\n== IESSNS North Sea stratum (July) ==')
print('year  Ceton mean t/km2  StoX t/km2')
for y in sorted(stox_t):
    print(y, f'{ceton_kg[y] / 1000:6.2f}', f'{stox_t[y] / area[y]:10.2f}')
summ([v / 1000 for v in ceton_kg.values()], 'North Sea stratum mean 2018-25')
summ([v / 1000 for y, v in ceton_kg.items() if y >= 2021], 'North Sea stratum mean 2021-25')
# 2026 preliminary: incl. NS TSB bootstrap mean 3.3 Mt (Table 10b) vs excl. NS 1.8 Mt (Table 10a)
print('2026 prelim. North Sea implied (3.3 - 1.8 Mt) / 285 781 km2 =', round(1.5e6 / 285781, 2), 't/km2')

# ---------- survey vs assessment: whole-stock ratio ----------
# IESSNS excl. NS (Mt): 2022 7.37, 2024 2.5, 2025 1.7 (IESSNS reports); NS from stox_t.
# Assessment TSB 1 Jan (Mt), ICES SG mac.27.nea key 25875 (2026 assessment): 2022 6.716, 2024 4.071, 2025 3.729.
iessns = {2022: 7.37, 2024: 2.5, 2025: 1.7}
tsb = {2022: 6.716465, 2024: 4.070715, 2025: 3.728611}
print('\n== IESSNS July swept-area total (Nordic + North Sea) / assessment TSB (1 Jan) ==')
ratios = []
for y in sorted(iessns):
    tot = iessns[y] + stox_t[y] / 1e6
    ratios.append(tot / tsb[y])
    print(y, f'IESSNS {tot:.2f} Mt / TSB {tsb[y]:.2f} Mt = {tot / tsb[y]:.2f}')
print('mean ratio', round(st.mean(ratios), 2), '-> assessment-scaled factor on survey densities',
      ' / '.join(f'{1 / r:.2f}' for r in ratios))
