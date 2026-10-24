"""Catch rate and realised surplus production from ICES SAG summary JSON.
Usage: python3 -I sag_productivity.py <sag_summary.json> <stock_size_is_total:0|1> <out_csv>
u = catch / B (start of year); P = (B_{t+1} - B_t + C_t) / B_t; R/SSB medians by period.
"""
import sys, json, csv, statistics as st
d = json.load(open(sys.argv[1])); tot = sys.argv[2] == '1'; out = sys.argv[3]
L = sorted(d['Lines'], key=lambda r: r['Year'])
by = {r['Year']: r for r in L}
rows = []
for y in sorted(by):
    r = by[y]; B = r.get('StockSize') or r.get('SSB')
    C = r.get('Catches') or ((r.get('Landings') or 0) + (r.get('Discards') or 0)) or None
    nxt = by.get(y + 1, {}); B1 = nxt.get('StockSize') or nxt.get('SSB')
    u = C / B if (B and C) else None
    P = (B1 - B + C) / B if (B and B1 and C) else None
    rs = r['Recruitment'] / B if (r.get('Recruitment') and B) else None
    rows.append((y, B, C, r.get('F'), u, P, rs))
with open(out, 'w', newline='') as f:
    w = csv.writer(f); bs = 'Btot' if tot else 'SSB_only_not_production'
    w.writerow(['year', 'B_' + ('total' if tot else 'SSB'), 'catch', 'F_or_HR', 'catch_over_' + bs, 'surplus_prod_rate_' + bs, 'R_over_B'])
    for r in rows: w.writerow([x if not isinstance(x, float) else round(x, 5) for x in r])
def mean(a, b, k):
    v = [r[k] for r in rows if a <= r[0] <= b and r[k] is not None]; return round(st.mean(v), 4) if v else None
def med(a, b, k):
    v = [r[k] for r in rows if a <= r[0] <= b and r[k] is not None]; return round(st.median(v), 3) if v else None
print(d['FishStock'])
for a, b in ((2006, 2015), (2016, 2025), (2021, 2025), (2023, 2025)):
    print(f'  {a}-{b}: mean catch/B {mean(a,b,4)}  mean surplus prod {mean(a,b,5)}  median R/B {med(a,b,6)}')
for a, b in ((1949, 1990), (1990, 2005), (2006, 2025)):
    print(f'  median R/B {a}-{b}: {med(a,b,6)}')
