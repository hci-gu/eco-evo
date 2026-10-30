"""EU landings + discards of mackerel (MAC) and horse mackerel (HOM, JAX) in 27.3.a.20
(Skagerrak) by year, quarter, country and gear, from STECF FDI Table A files already
downloaded by another note (untrusted data; read only).

Usage: python3 -I fdi_eu_3a20_quarters.py OUT_CSV fdi_catches_3a20_<year>.csv ...
'C' (confidential) or 'NA' weights count as 0 and are reported as a count.
"""
import csv, sys, collections

out = sys.argv[1]
agg = collections.defaultdict(float)
conf = collections.Counter()
for f in sys.argv[2:]:
    for r in csv.DictReader(open(f, newline='')):
        sp = r['species']
        if sp not in ('MAC', 'HOM', 'JAX'):
            continue
        sp = 'HOM' if sp == 'JAX' else sp
        def num(x):
            try:
                return float(x)
            except ValueError:
                conf[(sp, r['year'])] += 1
                return 0.0
        land = num(r['total_live_weight_landed'])
        disc = num(r['tot_discards_tonnes']) if r['tot_discards_tonnes'] not in ('NA', '') else 0.0
        q = r['quarter']
        agg[(sp, r['year'], q, r['country'], r['gear_type'])] += land + disc

with open(out, 'w', newline='') as fo:
    w = csv.writer(fo)
    w.writerow(['species', 'year', 'quarter', 'country', 'gear', 'catch_t'])
    for k in sorted(agg):
        w.writerow(list(k) + [f'{agg[k]:.3f}'])

# console summary
for sp in ('MAC', 'HOM'):
    print(f'== {sp}: EU catch (landings + discards) in 27.3.a.20, t ==')
    years = sorted({k[1] for k in agg if k[0] == sp})
    qtot = collections.defaultdict(float)
    print('year  Q1  Q2  Q3  Q4  total | by country')
    for y in years:
        qs = [sum(v for k, v in agg.items() if k[0] == sp and k[1] == y and k[2] == q) for q in '1234']
        cs = collections.defaultdict(float)
        for k, v in agg.items():
            if k[0] == sp and k[1] == y:
                cs[k[3]] += v
        for i, q in enumerate(qs):
            qtot[i] += q
        print(y, ' '.join(f'{q:7.1f}' for q in qs), f'{sum(qs):8.1f} |',
              ', '.join(f'{c} {v:.0f}' for c, v in sorted(cs.items(), key=lambda t: -t[1])))
    tot = sum(qtot.values())
    print('pooled quarter shares:', ' '.join(f'Q{i+1} {qtot[i]/tot:.3f}' for i in range(4)),
          f'mean/yr {tot/len(years):.1f} t over {len(years)} yr')
    gs = collections.defaultdict(float)
    for k, v in agg.items():
        if k[0] == sp:
            gs[k[4]] += v
    print('gear shares:', ', '.join(f'{g} {v/tot:.3f}' for g, v in sorted(gs.items(), key=lambda t: -t[1])[:8]))
    print('confidential/NA weight rows:', {k[1]: v for k, v in conf.items() if k[0] == sp})
