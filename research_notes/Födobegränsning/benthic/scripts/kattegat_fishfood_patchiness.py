"""Kattegat trawling gradient (Hiddink et al. 2016; Dryad doi:10.5061/dryad.04dv4, CC0):
(1) Hiddink's own 'fish food' class (individuals 0.01-0.2 g WW, from his 2013 ICES symposium
    slides) per station, and food per plaice; does plaice / LRD condition track it?
(2) Patchiness: CV between the 0.1 m2 grabs within a station and CV between stations (km scale),
    for total prey (excl. large infauna), the fish-food class and Amphiura.

Run: <venv python with xlrd> -I scripts/kattegat_fishfood_patchiness.py <dryad_04dv4 dir>
Individual mass = WEIGHT / NO. per row (a row-level mean; rows are species x grab, J).
Consumers: catch per haul (relative; swept area and q unknown, J). Condition: relative weight
residual of log10 W on log10 L per species over all stations (as in the earlier note's script).
Trawling frequency per station: Hinz et al. 2017 Table S2 (copied from the earlier note).
"""
import collections
import csv
import math
import os
import statistics as stt
import sys

import xlrd

D = sys.argv[1]
SWEPT = {'A': 0.4, 'B': 1.5, 'C': 0.8, 'D': 1.5, 'E': 0.2, 'F': 2.6, 'G': 4.6, 'H': 4.6,
         'I': 4.5, 'J': 7.9, 'K': 5.4, 'L': 5.7, 'M': 0.4, 'N': 4.5, 'O': 8.0, 'P': 5.4,
         'Q': 9.1, 'R': 7.4, 'S': 8.6}
COARSE = {'K', 'O', 'Q', 'S'}
LARGE = ('ARCTICA', 'SPATANGUS', 'ECHINODERM', 'BRISSOPSIS')

g_prey = collections.defaultdict(collections.Counter)   # station -> grab -> g
g_food = collections.defaultdict(collections.Counter)
g_amph = collections.defaultdict(collections.Counter)
with open(os.path.join(D, 'Infauna Kattegat Biomass all stations!!.csv'), encoding='latin-1') as fh:
    for r in csv.DictReader(fh):
        st = r['STATION'].strip()
        if not st:
            continue
        gr = r['GRAB'].strip()
        try:
            w = float(r['WEIGHT (G)'] or 0)
            n = float(r['NO.'] or 1) or 1
        except ValueError:
            continue
        sp = r['SPECIES'].upper()
        g_prey[st][gr] += 0.0
        g_food[st][gr] += 0.0
        g_amph[st][gr] += 0.0
        if any(k in sp for k in LARGE) or w / n > 5.0:
            continue
        g_prey[st][gr] += w
        if 0.01 <= w / n <= 0.2:
            g_food[st][gr] += w
        if 'AMPHIURA' in sp:
            g_amph[st][gr] += w


def dens(c):  # g per 0.1 m2 grab -> g/m2 = t/km2
    return [v / 0.1 for v in c.values()]


sh = xlrd.open_workbook(os.path.join(D, 'Kattegat otter trawl catches.xls')).sheet_by_index(0)
hauls = collections.defaultdict(set)
catch = collections.defaultdict(collections.Counter)
for i in range(1, sh.nrows):
    v = sh.row_values(i)
    st, sp, w = v[1].strip(), v[3].strip(), v[5]
    hauls[st].add(v[2])
    try:
        catch[st][sp] += float(w)
    except (TypeError, ValueError):
        pass
PL = {s: catch[s]['Pleuronectes platessa'] / len(hauls[s]) / 1000 for s in hauls}
LRD = {s: catch[s]['Hippoglossoides platessoides'] / len(hauls[s]) / 1000 for s in hauls}
FLAT = {s: (catch[s]['Pleuronectes platessa'] + catch[s]['Limanda limanda']
            + catch[s]['Hippoglossoides platessoides']) / len(hauls[s]) / 1000 for s in hauls}

sh = xlrd.open_workbook(os.path.join(D, 'Kattegat Length weight data.xls')).sheet_by_name('Sheet1')
fish = collections.defaultdict(list)
for i in range(1, sh.nrows):
    v = sh.row_values(i)
    if v[8]:
        continue
    try:
        w, l = float(v[5]), float(v[6])
    except (TypeError, ValueError):
        continue
    if w > 0 and l > 0:
        fish[v[4].strip()].append((v[2].strip(), math.log10(l), math.log10(w)))


def ols(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    syy = sum((b - my) ** 2 for b in y)
    b = sxy / sxx
    return my - b * mx, b, (sxy * sxy / (sxx * syy) if syy > 0 else 0.0)


cond = {}
for sp, rows in fish.items():
    a, b, _ = ols([r[1] for r in rows], [r[2] for r in rows])
    res = collections.defaultdict(list)
    for st, ll, lw in rows:
        res[st].append(10 ** (lw - (a + b * ll)) - 1)
    cond[sp] = {st: (sum(v) / len(v), len(v)) for st, v in res.items()}

stations = sorted(s for s in g_prey if s in hauls)
print('st  swept ngrab  prey t/km2 (CVgrab)  food0.01-0.2g t/km2 (CVgrab)  Amph (CVgrab)  '
      'plaice kg/haul  food/plaice  cond_plaice  cond_LRD')
rows = []
for s in stations:
    p, f, am = dens(g_prey[s]), dens(g_food[s]), dens(g_amph[s])
    cv = lambda x: stt.pstdev(x) / stt.mean(x) if stt.mean(x) > 0 else float('nan')
    cp = cond.get('Pleuronectes platessa', {}).get(s, (float('nan'), 0))
    cl = cond.get('Hippoglossoides platessoides', {}).get(s, (float('nan'), 0))
    fpp = stt.mean(f) / PL[s] if PL[s] > 0 else float('inf')
    rows.append(dict(s=s, swept=SWEPT[s], prey=stt.mean(p), food=stt.mean(f), amph=stt.mean(am),
                     cvp=cv(p), cvf=cv(f), cva=cv(am), pl=PL[s], lrd=LRD[s], flat=FLAT[s],
                     cp=cp, cl=cl, n=len(p)))
    print(f'{s}{"*" if s in COARSE else " "} {SWEPT[s]:5.1f} {len(p):4d}  {stt.mean(p):7.1f} ({cv(p):.2f})'
          f'        {stt.mean(f):7.2f} ({cv(f):.2f})              {stt.mean(am):6.1f} ({cv(am):.2f})'
          f'   {PL[s]:6.2f}      {fpp:8.2f}    {cp[0]:+.3f}({cp[1]:3d}) {cl[0]:+.3f}({cl[1]:3d})')
print('* coarse sediment (excluded below, as in the earlier note; J)')
used = [r for r in rows if r['s'] not in COARSE]

print('\nPatchiness (fine-sediment stations)')
for k, lab in (('cvp', 'prey excl. large'), ('cvf', 'fish food 0.01-0.2 g'), ('cva', 'Amphiura')):
    v = [r[k] for r in used if not math.isnan(r[k])]
    print(f'  within-station CV between 0.1 m2 grabs, {lab:22s}: median {stt.median(v):.2f} '
          f'(range {min(v):.2f}-{max(v):.2f})')
for k, lab in (('prey', 'prey excl. large'), ('food', 'fish food 0.01-0.2 g'), ('amph', 'Amphiura')):
    v = [r[k] for r in used]
    print(f'  between-station CV of station means (km scale, ~45 km box), {lab:22s}: '
          f'{stt.pstdev(v) / stt.mean(v):.2f}  mean {stt.mean(v):.1f} t/km2, range {min(v):.1f}-{max(v):.1f}')
share = [r['food'] / r['prey'] for r in used]
print(f'  fish-food class share of prey excl. large: median {stt.median(share):.3f} '
      f'(range {min(share):.3f}-{max(share):.3f})')

print('\nCondition vs fish food and food per plaice (fine-sediment stations, station means, n fish >= 5)')
for sp, lab, P in (('Pleuronectes platessa', 'plaice', PL), ('Hippoglossoides platessoides', 'LRD', LRD)):
    ss = [r for r in used if cond[sp].get(r['s'], (0, 0))[1] >= 5 and P[r['s']] > 0]
    y = [cond[sp][r['s']][0] for r in ss]
    for xl, X in (('ln food', [math.log(r['food']) for r in ss]),
                  ('ln food/own-catch', [math.log(r['food'] / P[r['s']]) for r in ss]),
                  ('ln food/flatfish', [math.log(r['food'] / r['flat']) for r in ss])):
        _, b, r2 = ols(X, y)
        print(f'  cond_{lab:6s} ~ {xl:18s}: slope {b:+.4f} per ln unit, r2 {r2:.2f}, n {len(ss)}')
lo = [r for r in used if r['swept'] < 2]
hi = [r for r in used if r['swept'] > 4]
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
print(f'\nLow (<2/yr, {len(lo)} st) vs high (>4/yr, {len(hi)} st) trawling, geometric means: fish food '
      f'{gm([r["food"] for r in lo]):.2f} vs {gm([r["food"] for r in hi]):.2f} t/km2 '
      f'(x{gm([r["food"] for r in hi]) / gm([r["food"] for r in lo]):.2f}); plaice catch '
      f'{gm([r["pl"] for r in lo if r["pl"] > 0]):.2f} vs {gm([r["pl"] for r in hi if r["pl"] > 0]):.2f} kg/haul')
