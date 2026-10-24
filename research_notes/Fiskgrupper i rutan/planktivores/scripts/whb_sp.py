"""Blue whiting: catch/TSB, SP/TSB = (TSB(y+1) - TSB(y) + C(y))/TSB(y), ln-TSB trend, from SG CSV.
Usage: python3 -I whb_sp.py sg_whb.csv"""
import csv, math, statistics as st, sys
d = {int(r["Year"]): r for r in csv.DictReader(open(sys.argv[1]))}
T = {y: float(r["TBiomass"]) for y, r in d.items() if r["TBiomass"]}
C = {y: float(r["Catches"]) for y, r in d.items() if r["Catches"]}
F = {y: float(r["F"]) for y, r in d.items() if r["F"]}
rows = []
for y in sorted(T):
    if y in C and (y + 1) in T and y <= 2025:
        rows.append((y, T[y], C[y], C[y] / T[y], (T[y + 1] - T[y] + C[y]) / T[y], F.get(y)))
for r in rows[-12:]:
    print(r[0], f"TSB {r[1]:.0f} C {r[2]:.0f} C/TSB {r[3]:.3f} SP/TSB {r[4]:.3f} Fbar3-7 {r[5]}")
def trend(a, b):
    ys = [y for y in range(a, b + 1) if y in T]; xs = [math.log(T[y]) for y in ys]
    mx, my = st.mean(ys), st.mean(xs)
    return sum((y - mx) * (x - my) for y, x in zip(ys, xs)) / sum((y - mx) ** 2 for y in ys)
for a, b in [(1985, 2000), (2001, 2010), (2011, 2024), (2016, 2024), (2016, 2025)]:
    rr = [r for r in rows if a <= r[0] <= b]
    print((a, b), f"C/TSB {st.mean(r[3] for r in rr):.3f} SP/TSB mean {st.mean(r[4] for r in rr):.3f} median {st.median(r[4] for r in rr):.3f} Fbar {st.mean(r[5] for r in rr):.3f}")
for a, b in [(2006, 2025), (2016, 2025)]:
    print("ln-TSB trend", a, b, f"{trend(a, b)*100:.1f} %/yr")
