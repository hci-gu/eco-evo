"""Norway pout quarterly TSB from WGNSSK 2026 Table 10.3.6 (pdftotext -layout) + catch from SG CSV:
annual mean TSB (mean of 4 start-of-quarter values), catch/mean TSB, and SP/TSB = (TSB_Q1(y+1) - TSB_Q1(y) + C(y)) / TSB_Q1(y)
and the same on annual means. Usage: python3 -I nop_tsb.py wgnssk_nop.txt sg_nop.csv"""
import csv, re, sys
from collections import defaultdict
lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
i0 = next(i for i, l in enumerate(lines) if l.strip().startswith("Table 10.3.6. Norway pout"))
q = {}
for l in lines[i0:i0 + 500]:
    m = re.match(r"^\s*((?:19|20)\d\d)(\.25|\.5|\.75)?\s+(.*)$", l)
    if not m: continue
    toks = [t for t in m.group(3).split() if re.fullmatch(r"[\d.]+", t)]
    frac = m.group(2) or ""
    try:
        tsb = float(toks[6] if frac == ".5" else toks[3])
    except (IndexError, ValueError):
        continue
    q[(int(m.group(1)), {"": 1, ".25": 2, ".5": 3, ".75": 4}[frac])] = tsb
C = {int(r["Year"]): float(r["Catches"]) for r in csv.DictReader(open(sys.argv[2])) if r["Catches"].strip()}
years = sorted({y for y, _ in q})
mean = {y: sum(q[(y, k)] for k in range(1, 5)) / 4 for y in years if all((y, k) in q for k in range(1, 5))}
print("year TSB_Q1 meanTSB catch C/meanTSB SP/TSB_Q1 SP/meanTSB")
rows = []
for y in years:
    if y not in mean: 
        print(y, "TSB_Q1", q.get((y, 1))); continue
    c = C.get(y, float("nan"))
    spq = (q[(y + 1, 1)] - q[(y, 1)] + c) / q[(y, 1)] if (y + 1, 1) in q else float("nan")
    spm = (mean[y + 1] - mean[y] + c) / mean[y] if (y + 1) in mean else float("nan")
    rows.append((y, q[(y, 1)], mean[y], c, c / mean[y], spq, spm))
    print(f"{y} {q[(y,1)]:9.0f} {mean[y]:9.0f} {c:8.0f}  {c/mean[y]:.3f}  {spq:6.3f}  {spm:6.3f}")
def per(a, b):
    rr = [r for r in rows if a <= r[0] <= b]
    f = lambda k: [r[k] for r in rr if r[k] == r[k]]
    import statistics as st
    return {"meanTSB": st.mean(f(2)), "C/B": st.mean(f(4)), "SP/TSBq1": st.mean(f(5)) if f(5) else float('nan'), "SP/meanTSB": st.mean(f(6)) if f(6) else float('nan'), "median SP/meanTSB": st.median(f(6)) if f(6) else float('nan')}
for p in [(1984, 2000), (2001, 2010), (2011, 2024), (2016, 2024), (2016, 2025), (2021, 2025)]:
    print(p, {k: round(v, 3) for k, v in per(*p).items()})
