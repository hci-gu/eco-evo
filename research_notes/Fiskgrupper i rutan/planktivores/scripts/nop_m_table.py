"""Extract the Norway pout M-at-age-by-quarter table (WGNSSK 2026 Table 10.2.6) from pdftotext -layout output
and summarise annual sums by period. Usage: python3 -I nop_m_table.py wgnssk_nop.txt"""
import re, sys
lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
start = next(i for i, l in enumerate(lines) if "Table 10.2.6." in l and "Mean weight at age" in l)
rows = {}
for l in lines[start:start + 600]:
    m = re.match(r"^\s*(19[89]\d|20[0-2]\d)\s+([1-4])\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", l)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        rows[(y, q)] = [float(m.group(k)) for k in range(3, 8)]
years = sorted({y for y, _ in rows})
print("years", years[0], years[-1], "rows", len(rows))
print("year  annual sum M (Q1-Q4) ages 0(Q3+Q4 only) 1 2 3 4")
ann = {}
for y in years:
    if all((y, q) in rows for q in range(1, 5)):
        s = [sum(rows[(y, q)][a] for q in range(1, 5)) for a in range(5)]
        s[0] = rows[(y, 3)][0] + rows[(y, 4)][0]
        ann[y] = s
        print(y, " ".join(f"{v:.2f}" for v in s), " Q2:", " ".join(f"{rows[(y,2)][a]:.2f}" for a in range(1, 4)))
def mean(p):
    ys = [y for y in ann if p[0] <= y <= p[1]]
    return [sum(ann[y][a] for y in ys) / len(ys) for a in range(5)], len(ys)
for p in [(1984, 2000), (2001, 2022), (2011, 2022), (2016, 2026)]:
    m, n = mean(p)
    print("period", p, "n", n, "mean annual M ages 0(H2),1,2,3,4:", " ".join(f"{v:.2f}" for v in m))
# quarterly means 2011-2022
print("quarterly mean 2011-2022 by age 0..4:")
for q in range(1, 5):
    ys = [y for y in range(2011, 2023) if (y, q) in rows]
    print(" Q", q, " ".join(f"{sum(rows[(y,q)][a] for y in ys)/len(ys):.3f}" for a in range(5)))
