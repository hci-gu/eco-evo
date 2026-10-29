"""SMS key run 2023: partial M2 of saithe (predator 19) on haddock (prey 18) by prey age, summed over
quarters and saithe ages, per year; with haddock 0-group (Q3) and 1-group (Q1) numbers from summary.out.
Usage: python3 -I sms_saithe_partial_m2.py <partial_m2.out> <summary.out>
"""
import sys, collections, math
m2 = collections.defaultdict(float)
for l in open(sys.argv[1]).read().split("\n")[1:]:
    f = l.split()
    if len(f) < 7 or f[2] != "19" or f[4] != "18":
        continue
    m2[(int(f[0]), int(f[5]))] += float(f[6])
lines = open(sys.argv[2]).read().split("\n"); h = lines[0].split(); ix = {k: i for i, k in enumerate(h)}
N0 = {}; N1 = {}
for l in lines[1:]:
    f = l.split()
    if len(f) < len(h) or f[2] != "18":
        continue
    y, q, a = int(f[0]), int(f[1]), int(f[3])
    if a == 0 and q == 3: N0[y] = float(f[ix["N"]]) / 1e3
    if a == 1 and q == 1: N1[y] = float(f[ix["N"]]) / 1e3
print("year  M2_saithe_on_haddock_age0  age1  age2  | haddock N0 Q3 (1e6)  N1 Q1 (1e6)")
xs0, ys0, xs1, ys1 = [], [], [], []
for y in sorted({k[0] for k in m2}):
    print(y, "%.3f %.3f %.3f | %.0f %.0f" % (m2[(y, 0)], m2[(y, 1)], m2[(y, 2)], N0.get(y, 0), N1.get(y, 0)))
    if y in N0 and m2[(y, 0)] > 0: xs0.append(math.log(N0[y])); ys0.append(math.log(m2[(y, 0)]))
    if y in N1 and m2[(y, 1)] > 0: xs1.append(math.log(N1[y])); ys1.append(math.log(m2[(y, 1)]))
def slope(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)
print("log-log slope M2(age0) vs N0: %.2f (n %d); M2(age1) vs N1: %.2f (n %d)  [0 = density-independent rate,"
      " < 0 = per-capita risk falls when haddock are abundant]" % (slope(xs0, ys0), len(xs0), slope(xs1, ys1), len(xs1)))
for a, b in ((2018, 2022), (2013, 2022), (1974, 2022)):
    ys = [y for y in range(a, b + 1)]
    print(f"mean {a}-{b}: age0 {sum(m2[(y,0)] for y in ys)/len(ys):.3f}  age1 {sum(m2[(y,1)] for y in ys)/len(ys):.3f}")
