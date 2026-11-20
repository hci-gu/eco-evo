"""Q1: euphausiid partial stomach content (%BW) of eastern Bering Sea pollock, = euphausiid %W x mean
fullness %bw, from Buckley et al. 2016 Table 2 (stock-weighted annual means, daytime summer BTS).
Usage: python3 -I q1_buckley_pf.py <buckley2016.txt>"""
import re, sys
rows = []
for line in open(sys.argv[1], encoding='utf8', errors='replace'):
    m = re.match(r'\s*(19[89]\d|20[01]\d)\s+(.*)$', line)
    if not m:
        continue
    try:
        vals = [float(v) for v in m.group(2).split()]
    except ValueError:
        continue
    if len(vals) != 11:
        continue
    rows.append((int(m.group(1)), vals[4], vals[10]))
for y, eu, bw in rows:
    print(y, "euph %%W %.1f fullness %.3f -> euph content %.3f %%BW" % (eu, bw, eu / 100 * bw))
pf = [eu / 100 * bw for _, eu, bw in rows]
print("n %d min %.3f max %.3f mean %.3f %%BW" % (len(pf), min(pf), max(pf), sum(pf) / len(pf)))
print("max total fullness %.3f %%BW" % max(bw for _, _, bw in rows))
