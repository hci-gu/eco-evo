"""Across-year summary of the box acoustic series (our derived table).
Usage: python3 -I summarise.py acoustic_box_by_year.csv OUT_CSV"""
import csv, math, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
cols = ['herring_t_km2', 'sprat_t_km2', 'norway_pout_t_km2', 'blue_whiting_t_km2', 'planktivores_t_km2', 'nasc_total']
periods = {'2010-2026': (2010, 2026), '2010-2014': (2010, 2014), '2015-2019': (2015, 2019),
           '2020-2026': (2020, 2026), '2021-2025': (2021, 2025), '2016-2025 (biotic own year)': (2016, 2025)}
out = csv.writer(open(sys.argv[2], 'w', newline=''))
out.writerow(['period', 'variable', 'n_years', 'mean', 'median', 'geomean', 'min', 'max'])
for name, (a, b) in periods.items():
    sel = [r for r in rows if a <= int(r['year']) <= b]
    for c in cols:
        v = [float(r[c]) for r in sel]
        pos = [x for x in v if x > 0]
        gm = math.exp(sum(math.log(x) for x in pos) / len(pos)) if pos else 0
        out.writerow([name, c, len(v), f'{st.mean(v):.2f}', f'{st.median(v):.2f}', f'{gm:.2f}', f'{min(v):.2f}', f'{max(v):.2f}'])
# trend: OLS of ln(herring) and ln(planktivores) on year
for c in ['herring_t_km2', 'planktivores_t_km2', 'nasc_total']:
    xs = [int(r['year']) for r in rows]; ys = [math.log(float(r[c])) for r in rows]
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    res = [y - my - b * (x - mx) for x, y in zip(xs, ys)]
    se = math.sqrt(sum(e * e for e in res) / (len(xs) - 2) / sum((x - mx) ** 2 for x in xs))
    out.writerow(['trend 2010-2026', c, len(xs), f'slope_ln_per_yr={b:.3f}', f'se={se:.3f}', f'pct_per_yr={100 * (math.exp(b) - 1):.1f}', '', ''])
