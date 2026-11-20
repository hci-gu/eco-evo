"""Derived statistics from Everson, Tarling & Bergstrom 2007 (ICES JMS 64:39) Table 5, transcribed by hand
into tables/q1_everson2007_table5.csv: mean krill numerical density (ind/m3, dSv 6-12 dB, adult
M. norvegica) per transect, 8 transects x 8 surveys, Gullmarsfjord, 8 and 10 Sept 2003.
Support: one transect (about 8 min of steaming; length not stated). Survey = 8 transects in 60-74 min.
Outputs per survey: CV between transects, log-sd, max/mean, Lloyd 1+CV^2 (support = transect);
the same without transect 1 (the sill end, where the day layer cannot sit: a bed-depth gradient);
and the Pearson / Spearman correlation of the 8-transect pattern between survey pairs at different lags.
Usage: python3 -I q1_everson_table5.py <csv>"""
import csv, math, statistics, sys
from itertools import combinations
rows = list(csv.DictReader(open(sys.argv[1])))
S = {}
for r in rows:
    S.setdefault(int(r["survey"]), {})[int(r["transect"])] = (float(r["d120_6to12"]), float(r["d38_6to12"]))
meta = {int(r["survey"]): (r["date"], r["period"]) for r in rows}
start = {1: (8, 7.5), 2: (8, 13.5), 3: (8, 19.3), 4: (8, 23.0), 5: (10, 7.5), 6: (10, 12.0), 7: (10, 19.5), 8: (10, 23.6)}

def stats(v):
    m = statistics.mean(v); sd = statistics.stdev(v)
    lsd = statistics.stdev([math.log(x) for x in v])
    return m, sd / m, lsd, max(v) / m, 1 + (sd / m) ** 2

def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
    for k, i in enumerate(o): r[i] = k
    return r

def pear(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))

for f, name in ((0, "120 kHz"), (1, "38 kHz")):
    print(f"== {name}, dSv 6-12 dB")
    print("survey date period | all 8: mean CV logsd max/mean Lloyd | transects 2-8: mean CV logsd max/mean Lloyd")
    agg = {"day": [], "night": []}
    for s in sorted(S):
        v = [S[s][t][f] for t in range(1, 9)]
        a = stats(v); b = stats(v[1:])
        agg[meta[s][1]].append((a[1], b[1], a[2], b[2]))
        print(f"{s} {meta[s][0]} {meta[s][1]:5s} | {a[0]:.3f} {a[1]:.2f} {a[2]:.2f} {a[3]:.2f} {a[4]:.2f} | {b[0]:.3f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f} {b[4]:.2f}")
    for p in ("day", "night"):
        x = agg[p]
        print(f"  {p}: mean CV all {statistics.mean(i[0] for i in x):.2f}, mean CV 2-8 {statistics.mean(i[1] for i in x):.2f}, mean logsd all {statistics.mean(i[2] for i in x):.2f}, logsd 2-8 {statistics.mean(i[3] for i in x):.2f}")
    print("  survey totals (mean of 8): " + ", ".join(f"{s}:{statistics.mean(S[s][t][f] for t in range(1,9)):.3f}" for s in sorted(S)))
    print("  pattern correlation between surveys (transects 2-8; Pearson on log, Spearman), lag in h:")
    for a, b in combinations(sorted(S), 2):
        la = [math.log(S[a][t][f]) for t in range(2, 9)]; lb = [math.log(S[b][t][f]) for t in range(2, 9)]
        lag = (start[b][0] - start[a][0]) * 24 + start[b][1] - start[a][1]
        print(f"    {a}({meta[a][1][0]})-{b}({meta[b][1][0]}) lag {lag:5.1f} h  r_log {pear(la, lb):+.2f}  rho_s {pear(rank(la), rank(lb)):+.2f}")
