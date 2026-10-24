"""Productivity decline d from an SG source-data CSV (as in Den pelagiska poolens reglering/productivity.md):
raw ratio of median R/SSB (early/recent) and Ricker ln(R/S) = a_period - b S with a shared slope; d = exp(a_e - a_r) +- 1 SE.
Usage: python3 -I ricker_d.py file.csv LAG e0 e1 r0 r1 [last_R_year]
LAG = years between SSB and the recruitment it produced (R(y) vs SSB(y-LAG))."""
import csv, math, sys
import statistics as st
fn, lag, e0, e1, r0, r1 = sys.argv[1], int(sys.argv[2]), *map(int, sys.argv[3:7])
last = int(sys.argv[7]) if len(sys.argv) > 7 else 9999
R, S = {}, {}
for r in csv.DictReader(open(fn)):
    y = int(r["Year"])
    if r.get("Recruitment"): R[y] = float(r["Recruitment"])
    if r.get("SSB"): S[y] = float(r["SSB"])
pts = []
for y in sorted(R):
    if y > last or (y - lag) not in S: continue
    p = "e" if e0 <= y <= e1 else ("r" if r0 <= y <= r1 else None)
    if p: pts.append((p, y, R[y], S[y - lag]))
E = [x for x in pts if x[0] == "e"]; Rr = [x for x in pts if x[0] == "r"]
me = st.median(x[2] / x[3] for x in E); mr = st.median(x[2] / x[3] for x in Rr)
# OLS: y = a_e*De + a_r*Dr - b*S
X = [[1.0 if p == "e" else 0.0, 1.0 if p == "r" else 0.0, -s] for p, _, rr, s in pts]
Y = [math.log(rr / s) for p, _, rr, s in pts]
n, k = len(Y), 3
XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
XtY = [sum(X[i][a] * Y[i] for i in range(n)) for a in range(k)]
def inv3(m):
    a, b, c = m[0]; d, e, f = m[1]; g, h, i = m[2]
    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    return [[(e*i-f*h)/det, (c*h-b*i)/det, (b*f-c*e)/det], [(f*g-d*i)/det, (a*i-c*g)/det, (c*d-a*f)/det], [(d*h-e*g)/det, (b*g-a*h)/det, (a*e-b*d)/det]]
Inv = inv3(XtX)
beta = [sum(Inv[a][b] * XtY[b] for b in range(k)) for a in range(k)]
res = [Y[i] - sum(X[i][a] * beta[a] for a in range(k)) for i in range(n)]
s2 = sum(x * x for x in res) / (n - k)
var_diff = s2 * (Inv[0][0] + Inv[1][1] - 2 * Inv[0][1])
diff = beta[0] - beta[1]; se = math.sqrt(var_diff)
print(f"{fn}: lag {lag}; early {e0}-{e1} (n={len(E)}), recent {r0}-{r1} (n={len(Rr)})")
print(f"  median R/SSB early {me:.4g}, recent {mr:.4g}; raw d = {me/mr:.2f}")
print(f"  median R early/recent = {st.median(x[2] for x in E)/st.median(x[2] for x in Rr):.2f}; median SSB early {st.median(x[3] for x in E):.0f}, recent {st.median(x[3] for x in Rr):.0f}")
print(f"  Ricker b = {beta[2]:.3g} per t (SSB at max R 1/b = {1/beta[2] if beta[2] > 0 else float('nan'):.0f} t); resid SD {math.sqrt(s2):.2f}")
print(f"  Ricker d = {math.exp(diff):.2f} (+-1 SE {math.exp(diff - se):.2f}-{math.exp(diff + se):.2f})")
