"""Satiation growth vs temperature in gadoid proxies (no whiting lab data found).

Derived numbers only; every coefficient is typed in from a source read in full
text or abstract (see evidence.md). No downloaded file is read.

1. Bjornsson, Steinarsson & Oddgeirsson 2001 (ICES JMS 58:29), Table 3:
   G = a + bT - cT^2 (%/d), cod fed to satiation. The text layer lost minus
   signs; a is negative for A-E and positive for F-H (restored so that the
   printed Topt = b/2c and Gmax reproduce, as in ../Tillvaxt mot ranson).
2. Bjornsson & Steinarsson 2002 (CJFAS 59:494), abstract:
   G = 0.5735 T W^(-0.1934 - 0.02001 T) (%/d), 2-5000 g, 2-16 C.
3. Peck et al. 2003 (MEPS 251:233), Table 2: G at the observed Cmax from the
   fitted growth-feeding curves (T1 asymptotic, T2 linear), age-0 cod and haddock.
"""
import math

T_LIST = [7.0, 10.0, 13.0, 16.0]


def q10(g1, g2, t1, t2):
    if g1 <= 0 or g2 <= 0:
        return float("nan")
    return (g2 / g1) ** (10.0 / (t2 - t1))


print("== 1. Bjornsson et al. 2001, cod satiation growth G(T) %/d (Table 3 polynomials)")
B01 = [  # exp, W g, a (signed), b, c, Topt printed, Gmax printed
    ("A", 1.54, -3.36, 1.057, 0.0311, 17.0, 5.63),
    ("B", 11.6, -0.77, 0.509, 0.0156, 16.3, 3.38),
    ("C", 11.7, -1.23, 0.569, 0.0183, 15.5, 3.18),
    ("D", 29.4, -1.27, 0.529, 0.0200, 13.2, 2.23),
    ("E", 108.7, -0.58, 0.269, 0.0133, 10.1, 0.77),
    ("F", 447, 0.29, 0.094, 0.00533, 8.8, 0.70),
    ("G", 874, 0.18, 0.111, 0.00684, 8.1, 0.63),
    ("H", 2213, 0.08, 0.050, 0.00294, 8.6, 0.30),
]
print(f"{'exp':3} {'W g':>7} {'Topt':>5} {'Gmax':>5} | " + " ".join(f"G{t:.0f}" .rjust(6) for t in T_LIST)
      + " | G13/G7 G16/G7 Q10(7-13) Q10(7-16) | check Topt Gmax")
for e, w, a, b, c, topt, gmax in B01:
    g = [a + b * t - c * t * t for t in T_LIST]
    topt_c = b / (2 * c)
    gmax_c = a + b * topt_c - c * topt_c ** 2
    print(f"{e:3} {w:7.1f} {topt:5.1f} {gmax:5.2f} | " + " ".join(f"{x:6.2f}" for x in g)
          + f" | {g[2]/g[0]:6.2f} {g[3]/g[0]:6.2f} {q10(g[0], g[2], 7, 13):9.2f} {q10(g[0], g[3], 7, 16):9.2f}"
          + f" | {topt_c:5.1f} {gmax_c:5.2f}")

print()
print("== 2. Bjornsson & Steinarsson 2002, food-unlimited G = 0.5735 T W^(-0.1934-0.02001T) %/d")
print(f"{'W g':>6} {'Topt':>5} | " + " ".join(f"G{t:.0f}".rjust(6) for t in T_LIST) + " | G13/G7 G16/G7 Q10(7-13) Q10(7-16)")
for w in [5, 10, 25, 50, 100, 200, 500, 1000]:
    g = [0.5735 * t * w ** (-0.1934 - 0.02001 * t) for t in T_LIST]
    topt = 1.0 / (0.02001 * math.log(w))
    print(f"{w:6.0f} {topt:5.1f} | " + " ".join(f"{x:6.2f}" for x in g)
          + f" | {g[2]/g[0]:6.2f} {g[3]/g[0]:6.2f} {q10(g[0], g[2], 7, 13):9.2f} {q10(g[0], g[3], 7, 16):9.2f}")
print("   (Topt = 1/(0.02001 ln W) from dG/dT = 0; the abstract quotes 14.3 C for 50 g and 5.9 C for 5000 g)")

print()
print("== 3. Peck et al. 2003, G at observed Cmax (%/d) from the Table 2 fits")
P03 = [  # group, W g, T, Cmax, eq, phi, a, b
    ("codS", 0.92, 5, 5.8, "T1", 1.61, 2.98, 0.31),
    ("codS", 0.92, 8, 7.6, "T1", 1.90, 3.69, 0.47),
    ("codS", 0.92, 12, 13.0, "T2", None, -1.30, 0.39),
    ("codS", 0.92, 15, 9.3, "T1", 0.52, 2.06, 0.44),
    ("codL", 7.33, 5, 4.9, "T1", 2.19, 2.86, 0.25),
    ("codL", 7.33, 8, 6.3, "T1", 1.21, 2.32, 0.42),
    ("codL", 7.33, 12, 7.7, "T1", 0.76, 2.68, 0.38),
    ("codL", 7.33, 15, 6.7, "T1", 0.44, 1.80, 0.33),
    ("haddock", 3.80, 8, 9.6, "T2", None, -0.90, 0.37),
    ("haddock", 3.80, 12, 10.5, "T2", None, -1.08, 0.34),
]
for grp, w, t, cmax, eq, phi, a, b in P03:
    if eq == "T1":
        g = phi - a * math.exp(-cmax * b)
    else:
        g = a + b * cmax
    print(f"{grp:8} {w:5.2f} g {t:3d} C  Cmax {cmax:5.1f} %/d  G(Cmax) {g:5.2f} %/d  GGE at Cmax {g/cmax:5.2f}")
print("   (Peck et al. quote observed haddock Gmax 3.0-3.1 %/d at 8 and 12 C and codS Gmax 2.1 %/d at 8 C;")
print("    the fitted values above differ from the observed maxima - use them for the temperature pattern only)")

print()
print("== 4. Bjornsson et al. 2001 relative to the model's bottom-layer annual mean 7.4 C")
for e, w, a, b, c, topt, gmax in B01[1:6]:
    g = lambda t: a + b * t - c * t * t  # noqa: E731
    print(f"{e} {w:6.1f} g  G7.4 {g(7.4):5.2f}  G12/G7.4 {g(12)/g(7.4):5.2f}  G14/G7.4 {g(14)/g(7.4):5.2f}"
          f"  G16/G7.4 {g(16)/g(7.4):5.2f}")
