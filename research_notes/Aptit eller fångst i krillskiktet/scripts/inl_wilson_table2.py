"""Wilson, Jump & Buchheister 2009 (MEPS 392:239, doi:10.3354/meps08237), Table 2 transcribed by hand from the
rendered page (p. 248), plus the x-axis ranges of Fig. 2 (p. 244) converted to natural units. All output [D].

Run: python3 -I inl_wilson_table2.py > ../tables/inl_wilson_table2_stdout.txt

Table 2 columns used: year, hydrographic area, area (km2), krill standing stock (millions of krill),
krill mean body weight (g per krill, plankton, night), per capita consumption (krill per fish per day) of
Age-0 pollock, Age-1+ pollock, capelin, eulachon.

Caveat carried into every number: per capita consumption is PW x DR x KP / KW (Eq. 2) with ONE daily ration DR
per predator group (Age-0 pollock 1.60, Age-1+ pollock 1.67, capelin 0.68, eulachon 0.50 %BW/d). Its variation
between areas and years is the krill diet share KP, the predator weight PW and the stomach-krill size KW, not a
measured ration. KW is estimated from predator length (Eq. 4 analogue), not the plankton weight printed here.
"""

# year, area, km2, stock (millions), krill wt (g), pc Age0, pc Age1+, pc capelin, pc eulachon
T2 = [
    (2000, "NEin", 4871, 519517, 0.0244, 1.13, 7.11, 0.29, 0.68),
    (2000, "NEmid", 7799, 2644737, 0.0148, 0.94, 23.59, 0.10, 2.74),
    (2000, "SWin", 10520, 616419, 0.0208, 0.83, 8.42, 0.15, 1.68),
    (2000, "SWmid", 12068, 1778192, 0.0252, 0.15, 1.18, 0.21, 0.93),
    (2000, "Slope", 6127, 1603100, 0.0146, 7e-5, 0.0, 0.57, 0.0),
    (2001, "NEin", 4871, 689640, 0.0269, 1.60, 49.41, 0.77, 5.85),
    (2001, "NEmid", 7799, 7996945, 0.0134, 2.15, 58.84, 2.27, 4.62),
    (2001, "SWin", 10520, 1546807, 0.0186, 2.79, 63.86, 0.06, 1.31),
    (2001, "SWmid", 12068, 10416154, 0.0102, 0.93, 21.62, 0.81, 0.70),
    (2001, "Slope", 6127, 2611257, 0.0113, 0.0, 0.0, 0.0, 0.0),
    (2003, "NEin", 4871, 268348, 0.0367, 0.91, 12.92, 0.55, 1.53),
    (2003, "NEmid", 7799, 1104865, 0.0316, 0.13, 0.0, 0.04, 1.25),
    (2003, "SWin", 10520, 367043, 0.0387, 0.34, 0.0, 0.05, 0.45),
    (2003, "SWmid", 12068, 551274, 0.0269, 0.11, 7.80, 0.07, 0.92),
]

print("Area means: krill density from Table 2 [D]")
print("year area   ind/m2   g/m2 (x plankton krill wt)   Age-1+ pollock krill/fish/d   capelin   eulachon")
for y, a, km2, stock, kw, p0, p1, pc, pe in T2:
    ind_m2 = stock * 1e6 / (km2 * 1e6)
    g_m2 = ind_m2 * kw
    print("%d %-6s %7.1f   %6.2f                        %6.2f                        %5.2f     %5.2f"
          % (y, a, ind_m2, g_m2, p1, pc, pe))

# rank correlation between area krill density (g/m2) and Age-1+ per capita krill numbers, areas with pollock
def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r

def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / (sxx * syy) ** 0.5

rows = [r for r in T2 if r[1] != "Slope"]
dens = [r[3] / r[2] * r[4] for r in rows]
dens_n = [r[3] / r[2] for r in rows]
for name, idx in (("Age-1+ pollock", 6), ("Age-0 pollock", 5), ("capelin", 7), ("eulachon", 8)):
    pcs = [r[idx] for r in rows]
    print("Spearman rho, area krill g/m2 vs %s per capita krill numbers (n=%d areas x years, slope excluded): %.2f;"
          " vs ind/m2: %.2f" % (name, len(rows), spearman(dens, pcs), spearman(dens_n, pcs)))
print("  (per capita numbers carry the fixed ration and the stomach-krill size; see the docstring)")

print()
print("Fig. 2 x-axis spans, (ind/m2)^0.25 read off the plotted points [D, read off figure, +-0.3]:")
for grp, lo, hi in (("Age-1+ pollock", 2.5, 7.0), ("capelin", 0.0, 9.0), ("Age-0 pollock", 0.0, 9.5)):
    print("  %-15s x %.1f-%.1f -> %.0f-%.0f ind/m2; at the Table 2 krill weights 0.010-0.039 g: up to %.0f-%.0f g/m2"
          % (grp, lo, hi, lo ** 4, hi ** 4, hi ** 4 * 0.0102, hi ** 4 * 0.0387))
print("  For comparison: 5 g/m2 = 5 t/km2 (the box's krill outside density); 2401 ind/m2 over a 100-200 m column"
      " = %.0f-%.0f ind/m3" % (2401 / 200, 2401 / 100))
