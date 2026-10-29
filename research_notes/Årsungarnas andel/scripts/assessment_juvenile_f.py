"""F on age 0-1 tonnes vs F on all tonnes, from the latest ICES single-stock assessments (2015-2025).

Usage: python3 -I assessment_juvenile_f.py <data/selectivity dir> <haddock SAM run dir>

For each stock, year and age: mean biomass B = N * w * (1 - exp(-Z)) / Z, Z = F + M (annual models)
or per half-year season (sprat). The biomass-weighted F over a set of cells is sum F B / sum B,
which is the constant loss rate that takes the model catch from those tonnes. Ratios:
  rJ = F(ages 0-1) / F(all ages),  rA = F(ages 2+) / F(all ages),  pJ = J share of stock biomass.
Herring ages are winter rings (0-wr = the 0-group of the survey ages). Cod has no age 0 (F0 = 0).
Sprat (model year July-June, S1 = Jul-Dec, S2 = Jan-Jun): calendar age 0 = age 0 S1, calendar
age 1 = age 0 S2 + age 1 S1; J(A) = those three cells, J(B) = age 0 S1 + age 0 S2.
"""
import math
import os
import re
import sys

ddir, hdir = sys.argv[1], sys.argv[2]
YEARS = list(range(2015, 2026))


def table(path, ncol, drop_last=False):
    out = {}
    for line in open(path):
        p = line.split()
        if not p or not re.fullmatch(r"20\d\d", p[0]):
            continue
        v = [float(x) for x in p[1:]]
        if drop_last:
            v = v[:-1]
        out[int(p[0])] = v[:ncol]
    return out


def lowestoft(path):
    lines = open(path).read().splitlines()
    y0, y1 = (int(x) for x in lines[2].split()[:2])
    out = {}
    for i, y in enumerate(range(y0, y1 + 1)):
        out[y] = [float(x) for x in lines[5 + i].split()]
    return out


def html_table(path):
    t = re.sub(r"<[^>]+>", " ", open(path).read())
    out = {}
    for line in t.splitlines():
        p = line.split()
        if len(p) > 3 and re.fullmatch(r"(19|20)\d\d", p[0]):
            out[int(p[0])] = [float(x) for x in p[1:]]
    return out


def summarise(name, F, N, W, M, ages, jages, years=YEARS):
    acc = {"J": [0, 0], "A": [0, 0], "ALL": [0, 0]}
    fat = [0.0] * len(ages)
    for y in years:
        for i, a in enumerate(ages):
            f, n, w, m = F[y][i], N[y][i], W[y][i], M[y][i]
            z = f + m
            b = n * w * (1 - math.exp(-z)) / z
            fat[i] += f / len(years)
            for k in ("J" if a in jages else "A", "ALL"):
                acc[k][0] += f * b
                acc[k][1] += b
    fj, fa, fall = (acc[k][0] / acc[k][1] for k in ("J", "A", "ALL"))
    pj = acc["J"][1] / acc["ALL"][1]
    print(f"{name:22s} {years[0]}-{years[-1]} | F at age " + " ".join(f"{x:.3f}" for x in fat[:6]) +
          f" | F_J {fj:.3f} F_A {fa:.3f} F_all {fall:.3f} | rJ {fj / fall:.2f} rA {fa / fall:.2f} | pJ {pj:.2f}")


x = os.path.join(ddir, "hawg2026_extract")
for span in (YEARS, list(range(2021, 2026))):
    ages = list(range(9))
    summarise("NSAS herring", table(f"{x}/nsas_F.txt", 9), table(f"{x}/nsas_N.txt", 9), table(f"{x}/nsas_sw.txt", 9),
              table(f"{x}/nsas_M.txt", 9), ages, {0, 1}, span)
    summarise("WBSS herring", table(f"{x}/wbss_F.txt", 9), table(f"{x}/wbss_N.txt", 9), table(f"{x}/wbss_sw.txt", 9),
              table(f"{x}/wbss_M.txt", 9), ages, {0, 1}, span)
    c = os.path.join(ddir, "wgnssk2026_cod_extract")
    summarise("cod Viking (no age 0)", table(f"{c}/viking_F.txt", 7, True), table(f"{c}/viking_N.txt", 7, True),
              table(f"{c}/viking_sw.txt", 7), table(f"{c}/viking_M.txt", 7), list(range(1, 8)), {1}, span)
    hF = html_table(os.path.join(hdir, "res", "xxx-00-00.00.00_tab2.html"))
    hN = html_table(os.path.join(hdir, "res", "xxx-00-00.00.00_tab3.html"))
    hW = {y: v[:9] for y, v in lowestoft(os.path.join(hdir, "data", "sw.dat")).items()}
    hM = {y: v[:9] for y, v in lowestoft(os.path.join(hdir, "data", "nm.dat")).items()}
    summarise("haddock had.27.46a20", hF, hN, hW, hM, ages, {0, 1}, span)

# Sprat: seasons. Weights (g) and M from HAWG 2026 Tables 11.1.11 and 11.1.5 (recent, constant).
F = table(f"{x}/sprat_F.txt", 8)
N = table(f"{x}/sprat_N.txt", 8)
W = [5.336, 9.06, 11.574, 13.956, 6.404, 10.162, 13.23, 17.108]   # S1 west, S2 weca
M = [0.817, 0.78, 0.68, 0.68, 0.63, 0.60, 0.57, 0.57]
for span in (list(range(2015, 2025)), list(range(2020, 2025))):   # 2025 S2 has no F yet
    acc = {k: [0, 0] for k in ("JA", "JB", "ALL", "AA")}
    for y in span:
        for i in range(8):
            f, n = F[y][i], N[y][i]
            z = f + M[i]
            b = n * W[i] * (1 - math.exp(-z)) / z
            for k, inside in (("JA", i in (0, 4, 1)), ("AA", i not in (0, 4, 1)), ("JB", i in (0, 4)), ("ALL", True)):
                if inside:
                    acc[k][0] += f * b
                    acc[k][1] += b
    f = {k: 2 * acc[k][0] / acc[k][1] for k in acc}   # per half-year -> per year
    print(f"sprat spr.27.3a4 {span[0]}-{span[-1]} (model years) | F_J(A) {f['JA']:.3f} F_A(A) {f['AA']:.3f} F_all {f['ALL']:.3f}"
          f" | rJ(A) {f['JA'] / f['ALL']:.2f} rA(A) {f['AA'] / f['ALL']:.2f} | F_J(B) {f['JB']:.3f} rJ(B) {f['JB'] / f['ALL']:.2f}"
          f" | pJ(A) {acc['JA'][1] / acc['ALL'][1]:.2f}")
