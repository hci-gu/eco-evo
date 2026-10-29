"""Benthivore share (haddock + cod + flatfish, % of stomach content weight) in North Sea saithe from
the ICES 'Year of the Stomach' database reports, transcribed by hand from
  CRR 164 (Daan 1989), Table 8-E, saithe 1980-82 pooled, half-years Q1+4 and Q2+3 (PDF p. 72, printed p. 68)
  CRR 219 (ICES 1997), Tables 8-E-1..4, saithe 1991 by age class and quarter (printed pp. 121-124)
and weighted by the model box's saithe length mix (40-59 cm 71 %, >= 60 cm 29 %; from
../Fiskgrupper i rutan / Torskfiskarnas bytesandelar). Quarter / half-year weights: equal, or the mean
stomach content weight x (1 - share empty) as a consumption proxy (CRR 219 gives % empty; CRR 164 not,
so there the proxy is the mean content weight). Unidentified Gadidae (CRR 219) are NOT redistributed in
the base case; a variant adds them pro-rata to haddock+cod among identified gadoids is not attempted
(blue whiting / Norway pout dominate identified gadoids) -- instead an upper bound adds all of them.
"""
# CRR 164 Table 8-E: size class (lower bound mm): (content g, haddock, cod, flatfish) per half-year
crr164 = {
    "Q1+4": {300: (4.212, 9.92, 0, 0), 400: (11.395, 2.84, 0, 0), 500: (21.916, 5.92, 0, 0),
             700: (42.032, 3.67, 0, 0.34), 1000: (57.696, 4.93, 0, 1.18)},
    "Q2+3": {300: (4.307, 35.16, 0.38, 0), 400: (9.562, 3.89, 0, 0), 500: (16.044, 5.36, 0.78, 0),
             700: (46.844, 12.75, 0.39, 0), 1000: (75.209, 31.82, 0, 0)},
}
print("CRR 164, 1980-82 northern North Sea, benthivore share by size class (equal / content-weighted half-years):")
res164 = {}
for sc in (300, 400, 500, 700, 1000):
    v = [crr164[h][sc] for h in ("Q1+4", "Q2+3")]
    eq = sum(x[1] + x[2] + x[3] for x in v) / 2
    cw = sum(x[0] * (x[1] + x[2] + x[3]) for x in v) / sum(x[0] for x in v)
    res164[sc] = (eq, cw)
    print(f"  {sc} mm: {eq:5.1f} / {cw:5.1f}")
# box mix: 40-59 cm ~ classes 400 (40-50) and 500 (50-70) ; >=60 ~ 500 and 700
for k in (0, 1):
    a = (res164[400][k] + res164[500][k]) / 2
    b = (res164[500][k] + res164[700][k]) / 2
    print(f"  box-weighted ({'equal' if k == 0 else 'content'}): 40-59 {a:.1f}, >=60 {b:.1f}, mix {0.71 * a + 0.29 * b:.1f} %")

# CRR 219 Tables 8-E-1..4, 1991. Per age: per quarter (mean length, n, %empty, content g, haddock, cod, flatfish, unid Gadidae)
ages = [3, 4, 5, 6, 7, 8, 9]
Q = {
 1: dict(L=[39.20, 45.12, 55.26, 61.59, 67.24, 73.90, 96.27], n=[371, 304, 46, 32, 18, 16, 10],
         e=[26.15, 26.32, 26.09, 25.00, 33.33, 37.50, 50.00], w=[3.26, 5.19, 12.91, 17.26, 21.35, 24.08, 165.92],
         had=[0]*7, cod=[0]*7, flat=[0]*7, ug=[4.77, 6.05, 3.46, 1.45, 0.45, 0, 0]),
 2: dict(L=[41.77, 46.65, 54.63, 67.10, 76.23, 82.43, 95.32], n=[451, 429, 105, 57, 40, 66, 35],
         e=[2.22, 2.33, 2.86, 1.75, 0, 1.52, 0], w=[14.11, 19.04, 44.05, 72.36, 100.00, 137.31, 177.02],
         had=[0.01, 0.06, 0.13, 0.03, 3.65, 8.64, 9.06], cod=[0]*7, flat=[0.03, 0.01, 0, 0, 0, 0, 0],
         ug=[2.46, 3.21, 3.28, 1.67, 0.70, 0.43, 2.37]),
 3: dict(L=[42.31, 51.60, 60.64, 67.19, 77.33, 81.26, 93.48], n=[242, 72, 28, 10, 13, 14, 12],
         e=[16.12, 13.89, 17.86, 20.00, 15.38, 14.29, 8.33], w=[9.31, 15.25, 16.88, 17.05, 138.32, 196.12, 300.34],
         had=[23.34, 10.73, 5.14, 3.36, 1.21, 1.13, 1.06], cod=[0]*7, flat=[0, 0, 0, 0, 1.04, 1.10, 1.20],
         ug=[5.87, 8.44, 7.49, 4.45, 1.33, 1.19, 1.04]),
 4: dict(L=[44.43, 53.61, 65.75, 73.94, 80.88, 86.98, 96.63], n=[628, 165, 27, 8, 5, 3, 13],
         e=[10.99, 9.70, 7.41, 12.50, 20.00, 33.33, 7.69], w=[13.57, 19.45, 40.02, 56.22, 88.33, 133.44, 154.43],
         had=[8.01, 5.69, 3.09, 4.85, 3.53, 1.30, 1.53], cod=[0]*7, flat=[0.03, 0, 0, 0, 0, 0, 1.94],
         ug=[6.23, 3.10, 0.95, 0.48, 0.15, 0.01, 0]),
}
print("\nCRR 219, 1991 (3012 of 3210 stomachs from roundfish area 1), benthivore share by age:")
print("  age | equal-quarter | consumption-proxy-weighted | upper (+ all unid. Gadidae, proxy-weighted)")
r219 = {}
for i, a in enumerate(ages):
    vals = [Q[q]["had"][i] + Q[q]["cod"][i] + Q[q]["flat"][i] for q in (1, 2, 3, 4)]
    ups = [vals[k] + Q[q]["ug"][i] for k, q in enumerate((1, 2, 3, 4))]
    wt = [Q[q]["w"][i] * (1 - Q[q]["e"][i] / 100) for q in (1, 2, 3, 4)]
    eq = sum(vals) / 4
    cw = sum(v * x for v, x in zip(vals, wt)) / sum(wt)
    up = sum(v * x for v, x in zip(ups, wt)) / sum(wt)
    r219[a] = (eq, cw, up)
    L = sum(Q[q]["L"][i] for q in (1, 2, 3, 4)) / 4
    print(f"  {a} ({L:.0f} cm) | {eq:5.1f} | {cw:5.1f} | {up:5.1f}")
for k, lab in ((0, "equal"), (1, "proxy"), (2, "upper")):
    a = (r219[3][k] + r219[4][k]) / 2           # ages 3-4: ~40-55 cm
    b = sum(r219[x][k] for x in (5, 6, 7, 8, 9)) / 5   # ages 5+: ~55-95 cm
    print(f"  box-weighted ({lab}): 40-59 {a:.1f}, >=60 {b:.1f}, mix {0.71 * a + 0.29 * b:.1f} %")

# Mehl 2005 Tables 13-14: Q4 consumption (t) by NEA saithe, haddock + cod over total, 1998-2003
north_tot = [290193, 141473, 310537, 177404, 102164, 101498]
north_hc = [41333 + 708, 8277 + 1805, 4367 + 51, 1694 + 0, 22576 + 664, 3986 + 0]
south_tot = [95999, 71325, 86298, 77880, 106918, 135159]
south_hc = [11732, 0, 75, 803, 586, 0]
print("\nMehl 2005 (NEA saithe, Norwegian coast 62-71 N, Oct-Nov), haddock+cod % of Q4 consumption:")
for y, nt, nh_, st, sh in zip(range(1998, 2004), north_tot, north_hc, south_tot, south_hc):
    print(f"  {y}: north {100 * nh_ / nt:5.1f}  south {100 * sh / st:5.1f}  all {100 * (nh_ + sh) / (nt + st):5.1f}")
print(f"  pooled: north {100 * sum(north_hc) / sum(north_tot):.1f}  south {100 * sum(south_hc) / sum(south_tot):.1f}  "
      f"all {100 * (sum(north_hc) + sum(south_hc)) / (sum(north_tot) + sum(south_tot)):.1f}")

# All gadoids (% weight): CRR 164 = Norway pout + haddock + cod + whiting (Table 8-E);
# CRR 219 = Norway pout + haddock + whiting + unidentified Gadidae + non-commercial Gadidae (Table 8-E).
g164 = {"Q1+4": {400: 11.47 + 2.84, 500: 67.44 + 5.92, 700: 75.49 + 3.67 + 1.01},
        "Q2+3": {400: 39.55 + 3.89 + 0.46, 500: 21.72 + 5.36 + 0.78 + 0.30, 700: 29.27 + 12.75 + 0.39}}
for sc in (400, 500, 700):
    print(f"CRR 164 gadoids {sc} mm: winter {g164['Q1+4'][sc]:.1f}, summer {g164['Q2+3'][sc]:.1f}, mean {(g164['Q1+4'][sc] + g164['Q2+3'][sc]) / 2:.1f}")
npout = {1: [50.38, 41.06, 25.20, 34.42, 33.79, 28.64, 1.32], 2: [14.04, 22.22, 32.29, 24.32, 17.37, 11.75, 7.15],
         3: [39.15, 59.69, 64.61, 59.89, 13.18, 10.61, 6.76], 4: [80.41, 77.79, 49.15, 26.67, 11.87, 7.99, 8.05]}
whg = {1: [0] * 7, 2: [0, 0, 0.01, 0.05, 2.81, 6.55, 15.59], 3: [4.40, 2.91, 3.03, 3.40, 0.18, 0.06, 0.03],
       4: [1.30, 1.16, 0.46, 0.95, 0.72, 0.20, 0.42]}
ncg = {1: [1.77, 2.05, 0.57, 0.21, 0.08, 0.03, 0.00], 2: [0.48, 0.66, 3.22, 11.09, 14.93, 15.28, 10.74],
       3: [0.52, 2.39, 2.33, 1.04, 0.38, 0.37, 0.40], 4: [0.47, 1.68, 1.56, 2.44, 1.47, 0.12, 0]}
gad = {}
for i, a in enumerate(ages):
    vals = [npout[q][i] + Q[q]["had"][i] + Q[q]["cod"][i] + whg[q][i] + Q[q]["ug"][i] + ncg[q][i] for q in (1, 2, 3, 4)]
    wt = [Q[q]["w"][i] * (1 - Q[q]["e"][i] / 100) for q in (1, 2, 3, 4)]
    np_ = sum(npout[q][i] * w for q, w in zip((1, 2, 3, 4), wt)) / sum(wt)
    gad[a] = (sum(v * w for v, w in zip(vals, wt)) / sum(wt), np_)
    print(f"CRR 219 gadoids age {a}: {gad[a][0]:.1f} (Norway pout {gad[a][1]:.1f}), by quarter " + " / ".join(f"{v:.0f}" for v in vals))
a = (gad[3][0] + gad[4][0]) / 2; b = sum(gad[x][0] for x in (5, 6, 7, 8, 9)) / 5
print(f"CRR 219 gadoids box-weighted: 40-59 {a:.1f}, >=60 {b:.1f}, mix {0.71 * a + 0.29 * b:.1f} %")
