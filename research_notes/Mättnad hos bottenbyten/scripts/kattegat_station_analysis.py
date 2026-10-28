"""Kattegat trawling gradient (Hiddink et al. 2016 / Hinz et al. 2017 station data, Dryad
doi:10.5061/dryad.04dv4): does flatfish condition follow benthic prey biomass B, consumer
catch P, or the ratio B/P?

Run: <python with xlrd> -I scripts/kattegat_station_analysis.py <dryad_04dv4 dir>
Inputs (data/hiddink2016/dryad_04dv4): infauna grabs (5 x 0.1 m2 per station, Hinz 2017
methods), otter-trawl catches (2 hauls per station; swept area not in the files -> catch per
haul is a RELATIVE density, J), length-weight per fish. Trawling frequency per station from
Hinz et al. 2017 Table S2 (data/hinz2017/moesm1.txt), typed in below.

Prey definitions (J, following Hinz 2017 "excluding large infauna ... e.g. Arctica
islandica, not sampled well by grabs and not eaten"): total minus Arctica, Spatangus,
Brissopsis/'ECHINODERM' (large irregular urchins) and any single specimen > 5 g.
Condition: residual of log10 W on log10 L per species over all stations (Fulton-free).
No stomach data are in the Dryad package (stomach contents of Hiddink 2016 not public).
"""
import csv
import collections
import math
import os
import sys

import xlrd

D = sys.argv[1]
SWEPT = {  # times trawled per year, Hinz et al. 2017 Table S2 (Kattegat)
    'A': 0.4, 'B': 1.5, 'C': 0.8, 'D': 1.5, 'E': 0.2, 'F': 2.6, 'G': 4.6, 'H': 4.6,
    'I': 4.5, 'J': 7.9, 'K': 5.4, 'L': 5.7, 'M': 0.4, 'N': 4.5, 'O': 8.0, 'P': 5.4,
    'Q': 9.1, 'R': 7.4, 'S': 8.6}
COARSE = {'K', 'O', 'Q', 'S'}  # median grain > 50 um (Table S2); J: excluded set is a judgement
LARGE = ('ARCTICA', 'SPATANGUS', 'ECHINODERM', 'BRISSOPSIS')

# ---------- prey ----------
grabs = collections.defaultdict(set)
tot = collections.Counter()
prey = collections.Counter()
amph = collections.Counter()
with open(os.path.join(D, 'Infauna Kattegat Biomass all stations!!.csv'), encoding='latin-1') as fh:
    for r in csv.DictReader(fh):
        st = r['STATION'].strip()
        if not st:
            continue
        grabs[st].add(r['GRAB'])
        try:
            w = float(r['WEIGHT (G)'] or 0)
            n = float(r['NO.'] or 1) or 1
        except ValueError:
            continue
        sp = r['SPECIES'].upper()
        tot[st] += w
        if any(k in sp for k in LARGE) or w / n > 5.0:
            continue
        prey[st] += w
        if 'AMPHIURA' in sp:
            amph[st] += w
B = {s: prey[s] / (0.1 * len(grabs[s])) for s in grabs}       # g/m2 = t/km2
Btot = {s: tot[s] / (0.1 * len(grabs[s])) for s in grabs}

# ---------- consumers (catch per haul, relative) ----------
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
CONS = ('Pleuronectes platessa', 'Limanda limanda', 'Hippoglossoides platessoides',
        'Nephrops norvegicus')
P = {s: sum(catch[s][c] for c in CONS) / len(hauls[s]) / 1000 for s in hauls}  # kg/haul
Ppl = {s: catch[s]['Pleuronectes platessa'] / len(hauls[s]) / 1000 for s in hauls}
Pcod = {s: catch[s]['Gadus morhua'] / len(hauls[s]) / 1000 for s in hauls}

# ---------- condition ----------
sh = xlrd.open_workbook(os.path.join(D, 'Kattegat Length weight data.xls')).sheet_by_name('Sheet1')
fish = collections.defaultdict(list)
for i in range(1, sh.nrows):
    v = sh.row_values(i)
    if v[8]:
        continue  # flagged 'Dodgy'
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
    r2 = sxy * sxy / (sxx * syy) if syy > 0 else 0
    return my - b * mx, b, r2


cond = {}
for sp, rows in fish.items():
    a, b, _ = ols([r[1] for r in rows], [r[2] for r in rows])
    res = collections.defaultdict(list)
    for st, ll, lw in rows:
        res[st].append(10 ** (lw - (a + b * ll)) - 1)  # relative weight deviation
    cond[sp] = {st: (sum(v) / len(v), len(v)) for st, v in res.items()}

stations = sorted(s for s in grabs if s in hauls)
used = [s for s in stations if s not in COARSE]
print('Station table (B prey g/m2 = t/km2; P = plaice+dab+LRD+Nephrops kg/haul, relative)')
print('st  swept  Btot   Bprey  Amph  P_cons  P_plaice  P_cod  B/P   cond_plaice(n)  cond_LRD(n)  cond_dab(n)')
out = []
for s in stations:
    cp = cond.get('Pleuronectes platessa', {}).get(s, (float('nan'), 0))
    cl = cond.get('Hippoglossoides platessoides', {}).get(s, (float('nan'), 0))
    cd = cond.get('Limanda limanda', {}).get(s, (float('nan'), 0))
    flag = '*' if s in COARSE else ' '
    print(f'{s}{flag} {SWEPT[s]:5.1f} {Btot[s]:6.1f} {B[s]:6.1f} {amph[s] / (0.1 * len(grabs[s])):5.1f} '
          f'{P[s]:6.1f} {Ppl[s]:7.2f} {Pcod[s]:6.2f} {B[s] / P[s]:5.2f} '
          f'{cp[0]:+.3f}({cp[1]:3d})  {cl[0]:+.3f}({cl[1]:3d})  {cd[0]:+.3f}({cd[1]:3d})')
    out.append((s, SWEPT[s], Btot[s], B[s], P[s], Ppl[s], Pcod[s], cp[0], cp[1], cl[0], cl[1], cd[0], cd[1]))
print('* coarse-sediment station (excluded in the regressions, J)')

with open(sys.argv[2], 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['station', 'swept_per_yr', 'B_total_gm2', 'B_prey_gm2', 'P_consumers_kg_haul',
                'P_plaice_kg_haul', 'P_cod_kg_haul', 'cond_plaice', 'n_plaice', 'cond_LRD', 'n_LRD',
                'cond_dab', 'n_dab'])
    for r in out:
        w.writerow([r[0]] + [f'{x:.4f}' if isinstance(x, float) else x for x in r[1:]])

print()
print('Regressions over the', len(used), 'fine-sediment stations (station means; weighted equally)')
lnB = {s: math.log(B[s]) for s in used}
lnP = {s: math.log(P[s]) for s in used}
for name, X in (('swept', {s: SWEPT[s] for s in used}), ('ln B_prey', lnB), ('ln P_cons', lnP),
                ('ln B/P', {s: lnB[s] - lnP[s] for s in used})):
    _, b, r2 = ols([X[s] for s in used], [X[s] for s in used]) if False else (0, 0, 0)
    for sp, lab in (('Pleuronectes platessa', 'plaice'), ('Hippoglossoides platessoides', 'LRD'),
                    ('Limanda limanda', 'dab')):
        ss = [s for s in used if s in cond[sp] and cond[sp][s][1] >= 5]
        a, b, r2 = ols([X[s] for s in ss], [cond[sp][s][0] for s in ss])
        print(f'  cond_{lab:6s} ~ {name:9s}: slope {b:+.4f}  r2 {r2:.2f}  n {len(ss)}')
_, bB, r2B = ols([SWEPT[s] for s in used], [lnB[s] for s in used])
_, bP, r2P = ols([SWEPT[s] for s in used], [lnP[s] for s in used])
print(f'  ln B_prey ~ swept: slope {bB:+.3f}/pass (r2 {r2B:.2f}); ln P_cons ~ swept: slope {bP:+.3f}/pass (r2 {r2P:.2f})')
lo = [s for s in used if SWEPT[s] < 2]
hi = [s for s in used if SWEPT[s] > 4]
gm = lambda d, ss: math.exp(sum(math.log(d[s]) for s in ss) / len(ss))
print(f'  geometric means, <2 passes ({len(lo)} st) vs >4 passes ({len(hi)} st): '
      f'B_prey {gm(B, lo):.1f} vs {gm(B, hi):.1f} g/m2 (x{gm(B, hi) / gm(B, lo):.2f}); '
      f'P_cons {gm(P, lo):.1f} vs {gm(P, hi):.1f} kg/haul (x{gm(P, hi) / gm(P, lo):.2f})')
