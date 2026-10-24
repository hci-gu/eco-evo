"""Biomass-weighted M1 and M2 by predator class for SMS prey species (NS key run 2023).
Usage: python3 -I sms_m2_split.py <data_dir> <out_csv>
Reads sms2023_summary.out (N, BIO, M1, M2 per year/quarter/species/age) and
sms2023_partial_m2.out (Part.M2 by predator/pred.age/prey/prey.age per quarter).
Annual weighted rate = sum_q sum_a rate_qa * B_qa / mean_q(sum_a B_qa), B = BIO (start of quarter).
"""
import sys, csv, collections
d, out = sys.argv[1], sys.argv[2]
names = [l.strip().strip('_').strip() for l in open(d + '/sms2023_species_names.in')][:27]
# species 1..27
cls = {}
for i, n in enumerate(names, start=1):
    if n in ('Fulmar', 'Guillemot', 'Her.Gull', 'Kittiwake', 'GBB.Gull', 'Gannet', 'Puffin', 'Razorbill'):
        cls[i] = 'birds_nonmodel'
    elif n == 'Grey.seal':
        cls[i] = 'grey_seal_nonmodel'
    elif n in ('A.radiata', 'G.gurnards'):
        cls[i] = 'gurnard_ray_ambiguous'
    elif n == 'H.porpoise':
        cls[i] = 'porpoise_model'
    elif n in ('Cod', 'Haddock'):
        cls[i] = 'benthivorous_model'
    elif n in ('Whiting', 'Saithe', 'Mackerel', 'Hake', 'W.horse.mac', 'N.horse.mac'):
        cls[i] = 'predatory_within_group'
    else:
        cls[i] = 'planktivorous_model'
YEARS = range(2018, 2023)
PREY = {17: 'Whiting', 20: 'Mackerel', 18: 'Haddock', 16: 'Cod'}
B = {}; M1 = {}; M2 = {}
with open(d + '/sms2023_summary.out') as f:
    h = f.readline().split()
    for line in f:
        p = line.split()
        y, q, s, a = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        if s in PREY and y in YEARS:
            r = dict(zip(h, p))
            B[(y, q, s, a)] = float(r['BIO']); M1[(y, q, s, a)] = float(r['M1']); M2[(y, q, s, a)] = float(r['M2'])
part = collections.defaultdict(float)
with open(d + '/sms2023_partial_m2.out') as f:
    f.readline()
    for line in f:
        p = line.split()
        y = int(p[0])
        if y not in YEARS:
            continue
        prey = int(p[4])
        if prey not in PREY:
            continue
        part[(y, int(p[1]), prey, int(p[5]), int(p[2]))] += float(p[6])
rows = []
for s, sn in PREY.items():
    for amin in (0, 1, 3):
        acc = collections.defaultdict(float); meanB = 0.0
        for y in YEARS:
            qb = []
            for q in range(1, 5):
                bq = 0.0
                for (yy, qq, ss, a), b in B.items():
                    if yy == y and qq == q and ss == s and a >= amin:
                        bq += b
                        acc['M1'] += M1[(y, q, s, a)] * b
                        acc['M2_total'] += M2[(y, q, s, a)] * b
                        for pred in range(1, 28):
                            v = part.get((y, q, s, a, pred), 0.0)
                            if v:
                                acc[cls[pred]] += v * b
                                acc['pred_' + names[pred - 1]] += v * b
                qb.append(bq)
            meanB += sum(qb) / 4.0
        n = len(YEARS)
        for k in sorted(acc):
            rows.append((sn, amin, k, round(acc[k] / meanB, 4)))
        rows.append((sn, amin, 'mean_biomass_t', round(meanB / n, 0)))
with open(out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['prey', 'min_age', 'term', 'annual_rate_per_yr']); w.writerows(rows)
for r in rows:
    if not r[2].startswith('pred_') or r[3] >= 0.005:
        print(*r)
