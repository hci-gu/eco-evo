"""Seasonal (quarterly) rations of whiting, saithe and Norway pout as a fraction of the GER Cmax.

Usage (from scripts/):
    python3 -I sea_quarterly_rations.py <NS2023_consum_ab.in> <out_csv>

Inputs
- SMS North Sea key run 2023 ration parameters (ration per individual per quarter = a W^b, kg), read
  from the file given as the first argument (the copy in research_notes/"Fiskgrupper i rutan"/predators/
  data/diet/sms_consumption/).
- Everything else is transcribed below from pdftotext output of the source PDFs (see lit_seasonal.md for
  the file and the page of each table). Transcriptions are tagged [F]; values computed here [D];
  values taken from sibling notes [S]; judgement [J].

GER (ICES WGSAM 2024 Table 3-7, whiting = saithe parameters; Andersen & Beyer 2005):
    R (g/h) = rho exp(delta T) L^lambda E^-0.85 (N/F)^(alpha-1) S^alpha,  alpha 0.5,
S = mean stomach content over ALL stomachs (empties included), F/N = share of non-empty stomachs.
GER Cmax: the same with all fish fed (F/N = 1) at a sustained fill S_max = 2.5 %BW (central; the
convention of "Fiskgrupper i rutan" and "Krillintag och jaktsatt") or 5 %BW (low band).
Then ration / Cmax = sqrt( (F/N) (S/W) / s_max ). Temperature, prey energy density and the GER
constants cancel when ration and Cmax are taken at the same temperature and diet [D].
Because sqrt(mean S) >= mean(sqrt S), a pooled mean overstates the ration (upper-side) [J].
"""
import csv
import math
import sys

consum_path, out_csv = sys.argv[1], sys.argv[2]

RHO, LAM, DEL = 0.00171, 1.35, 0.081          # whiting = saithe, WGSAM 2024 Table 3-7 [F]
T_Q = {1: 6.0, 2: 7.0, 3: 10.0, 4: 8.0}        # North Sea, Greenstreet et al. 1997 after Daan 1989 [F]
E_CENTRAL, E_ALT = 5.0, 4.2                    # kJ/g wet, whiting/saithe mixed fish+crustacean diet [J]
SMAX = (0.025, 0.05)

rows = []


def add(source, doi, species, region, season, method, quantity, value, unit, tag):
    rows.append(dict(source=source, doi=doi, species=species, region=region, season=season, method=method,
                     quantity=quantity, value=value, unit=unit, tag=tag))


def ger_rate(L, W, T, E, S, fn):
    """g/h; S = mean content over all stomachs, fn = F/N."""
    return RHO * math.exp(DEL * T) * L ** LAM * E ** -0.85 * fn ** 0.5 * S ** 0.5


def cmax_pct(L, W, T, E, smax=0.025):
    return 24 * RHO * math.exp(DEL * T) * L ** LAM * E ** -0.85 * (smax * W) ** 0.5 / W * 100


def ratio(fn, s_over_w, smax):
    return math.sqrt(fn * s_over_w / smax)


def f(x, n=2):
    return f"{x:.{n}f}"


# =====================================================================================================
# 1. Whiting, North Sea 1991: Robb, Bell, MacMillan & Hislop, ICES CM 1994/G:42 [F]
# =====================================================================================================
SRC_R = "Robb et al. 1994 ICES CM 1994/G:42"
DOI_R = "url:vliz.be/imisdocs/publications/ocrd/275252.pdf"
# Table 5: size class (mm) -> {quarter: (L cm, W g, S g)}; mean content incl. empty stomachs.
T5 = {
    100: {1: (11.33, 14, 0.13), 2: (11.42, 13, 0.31), 3: (10.88, 12, 0.33), 4: (11.17, 13, 0.19)},
    120: {1: (13.72, 24, 0.20), 2: (14.16, 25, 0.26), 3: (12.86, 19, 0.63), 4: (13.49, 23, 0.29)},
    150: {1: (16.98, 44, 0.47), 2: (17.55, 46, 0.77), 3: (18.68, 58, 0.70), 4: (17.04, 46, 1.01)},
    # Q1 S printed "-0.83" (OCR of a dash/dot before the value); read as 0.83 [J]
    200: {1: (23.17, 111, 0.83), 2: (22.48, 95, 1.55), 3: (22.11, 95, 1.42), 4: (23.36, 118, 1.67)},
    250: {1: (26.87, 172, 1.27), 2: (26.87, 162, 2.80), 3: (27.15, 174, 2.99), 4: (26.76, 176, 3.54)},
    300: {1: (31.62, 278, 2.33), 2: (31.59, 260, 5.66), 3: (31.59, 272, 5.92), 4: (31.69, 289, 5.32)},
    350: {1: (36.62, 432, 3.43), 2: (36.64, 402, 8.71), 3: (36.76, 424, 8.79), 4: (36.81, 449, 5.81)},
    400: {1: (42.13, 647, 8.53), 2: (42.03, 602, 12.02), 3: (41.64, 613, 16.19), 4: (42.74, 696, 10.52)},
}
# Table 3, "Total" (whole North Sea) rows: % empty by size class and quarter (columns assigned from
# pdftotext -layout positions).
T3 = {
    1: {60: 0.0, 80: 16.3, 100: 23.3, 120: 30.8, 150: 35.8, 200: 54.1, 250: 57.1, 300: 54.1, 350: 53.2, 400: 32.4,
        "all": 45.3},
    2: {100: 10.9, 120: 16.8, 150: 12.9, 200: 11.6, 250: 11.6, 300: 11.5, 350: 9.9, 400: 12.4, "all": 12.0},
    3: {50: 47.4, 60: 25.0, 70: 12.2, 80: 16.3, 100: 10.2, 120: 4.5, 150: 13.5, 200: 13.1, 250: 11.0, 300: 14.6,
        350: 16.3, 400: 16.7, "all": 12.5},
    4: {50: 0.0, 60: 11.1, 70: 21.1, 80: 19.2, 100: 16.1, 120: 15.2, 150: 5.6, 200: 7.3, 250: 7.0, 300: 6.3,
        350: 4.6, 400: 2.9, "all": 7.9},
}
# Table 2 "Percent" rows: food / regurgitated / empty (skeletal "+" or 1)
T2 = {1: (38, 15, 45), 2: (53, 34, 12), 3: (42, 46, 12), 4: (50, 42, 8)}

print("=" * 100)
print("1. WHITING, whole North Sea 1991 (Robb et al. 1994 Tables 2, 3, 5) [F inputs, D outputs]")
print("   T by quarter (Greenstreet 1997 after Daan 1989):", T_Q, " E", E_CENTRAL, "kJ/g [J]")
print("=" * 100)
print("Table 2 percent food / regurgitated / empty:", T2)
for q, (fo, re, em) in T2.items():
    add(SRC_R, DOI_R, "whiting", "North Sea (RFA 1-8)", f"Q{q} 1991", "stomach classification at sea",
        "stomachs empty, all sizes", em, "%", "F")
hdr = (f"{'size':>5} {'Q':>2} {'L':>6} {'W':>5} {'S':>6} {'S/W%':>6} {'empty%':>7} {'GERr%BW/d':>9} "
       f"{'Cmax%':>6} {'r/Cmax2.5':>9} {'r/Cmax5':>8} {'r%BW E4.2':>9}")
print(hdr)
adult = {q: [] for q in T_Q}
juv = {q: [] for q in T_Q}
for size, qd in T5.items():
    for q, (L, W, S) in qd.items():
        emp = T3[q][size]
        fn = 1 - emp / 100
        sw = S / W
        r = 24 * ger_rate(L, W, T_Q[q], E_CENTRAL, S, fn) / W * 100
        r42 = 24 * ger_rate(L, W, T_Q[q], E_ALT, S, fn) / W * 100
        cm = cmax_pct(L, W, T_Q[q], E_CENTRAL)
        r25, r5 = ratio(fn, sw, 0.025), ratio(fn, sw, 0.05)
        print(f"{size:>5} {q:>2} {L:>6.2f} {W:>5} {S:>6.2f} {sw*100:>6.2f} {emp:>7.1f} {r:>9.2f} {cm:>6.2f} "
              f"{r25:>9.2f} {r5:>8.2f} {r42:>9.2f}")
        (adult if size >= 200 else juv)[q].append((r25, r5, r, sw, emp))
        reg = "North Sea (RFA 1-8)"
        sea = f"Q{q} 1991"
        add(SRC_R, DOI_R, "whiting", reg, sea, f"Table 5, size class {size} mm", "mean stomach content incl. empties", S, "g", "F")
        add(SRC_R, DOI_R, "whiting", reg, sea, f"Table 5, size class {size} mm", "live weight", W, "g", "F")
        add(SRC_R, DOI_R, "whiting", reg, sea, f"Table 3, size class {size} mm", "stomachs empty", emp, "%", "F")
        add(SRC_R, DOI_R, "whiting", reg, sea, f"size {size} mm; GER at {T_Q[q]} C, E {E_CENTRAL}", "ration (upper-side)", round(r, 3), "%BW/d", "D")
        add(SRC_R, DOI_R, "whiting", reg, sea, f"size {size} mm; GER, S_max 2.5 %BW", "ration / GER Cmax (T-free)", round(r25, 2), "-", "D")
        add(SRC_R, DOI_R, "whiting", reg, sea, f"size {size} mm; GER, S_max 5 %BW", "ration / GER Cmax (T-free)", round(r5, 2), "-", "D")
print()
print("Means over size classes (unweighted):")
for name, grp in (("200-400 mm (adults, mostly mature)", adult), ("100-150 mm (juveniles)", juv)):
    print(" ", name)
    for q in T_Q:
        v = grp[q]
        m = [sum(x[i] for x in v) / len(v) for i in range(5)]
        print(f"    Q{q}: ration/Cmax {m[0]:.2f} (S_max 5 %: {m[1]:.2f}); ration {m[2]:.2f} %BW/d at {T_Q[q]} C; "
              f"S/W {m[3]*100:.2f} %; empty {m[4]:.1f} %")
        add(SRC_R, DOI_R, "whiting", "North Sea (RFA 1-8)", f"Q{q} 1991", f"mean of size classes {name}; S_max 2.5 %BW",
            "ration / GER Cmax (T-free)", round(m[0], 2), "-", "D")
        add(SRC_R, DOI_R, "whiting", "North Sea (RFA 1-8)", f"Q{q} 1991", f"mean of size classes {name}; S_max 5 %BW",
            "ration / GER Cmax (T-free)", round(m[1], 2), "-", "D")
        add(SRC_R, DOI_R, "whiting", "North Sea (RFA 1-8)", f"Q{q} 1991", f"mean of size classes {name}; GER at {T_Q[q]} C",
            "ration (upper-side)", round(m[2], 2), "%BW/d", "D")
a = {q: sum(x[0] for x in adult[q]) / len(adult[q]) for q in T_Q}
print(f"  adults: Q1 / mean(Q2-Q4) of ration/Cmax = {a[1] / ((a[2] + a[3] + a[4]) / 3):.2f}")
j = {q: sum(x[0] for x in juv[q]) / len(juv[q]) for q in T_Q}
print(f"  juveniles: Q1 / mean(Q2-Q4) of ration/Cmax = {j[1] / ((j[2] + j[3] + j[4]) / 3):.2f}")

# Sensitivity to the stomach classification (Robb: empty vs regurgitated split varied by vessel).
# Table 2 counts: food, regurgitated, empty, skeletal (all sizes, whole North Sea).
T2N = {1: (2365, 945, 2783, 52), 2: (6043, 3884, 1354, 16), 3: (4795, 5298, 1435, 2), 4: (5170, 4381, 819, 38)}
print()
print("Classification sensitivity, adults 200-400 mm (ratio to GER Cmax, S_max 2.5 %):")
print("  A = Robb's convention (regurgitated fish carry the valid mean; F/N = 1 - empty)")
print("  B = regurgitated counted as empty (F/N x food/(food+regurgitated) of the quarter)")
print("  C = empty/regurgitated split treated as artefact: same fed share in every quarter, only the")
print("      content per fed fish differs (relative to the Q2-Q4 mean)")
resA, resB, resC = {}, {}, {}
for q in T_Q:
    fo, re_, em, sk = T2N[q]
    share = (fo + sk) / (fo + sk + re_)
    va, vb, sfed = [], [], []
    for size in (200, 250, 300, 350, 400):
        L, W, S = T5[size][q]
        fn = 1 - T3[q][size] / 100
        va.append(ratio(fn, S / W, 0.025))
        vb.append(ratio(fn * share, S / W, 0.025))
        sfed.append(S / W / fn)
    resA[q], resB[q], resC[q] = sum(va) / 5, sum(vb) / 5, sum(sfed) / 5
for name, res in (("A", resA), ("B", resB)):
    oth = (res[2] + res[3] + res[4]) / 3
    print(f"  {name}: " + ", ".join(f"Q{q} {res[q]:.2f}" for q in T_Q) + f"; Q1 / mean(Q2-Q4) {res[1] / oth:.2f}")
    add(SRC_R, DOI_R, "whiting", "North Sea (RFA 1-8)", "Q1 vs Q2-Q4 1991", f"adults 200-400 mm, classification {name}",
        "Q1 / mean(Q2-Q4) of ration / GER Cmax", round(res[1] / oth, 2), "-", "D")
oth = (resC[2] + resC[3] + resC[4]) / 3
print("  C: content per fed fish S_fed/W " + ", ".join(f"Q{q} {resC[q]*100:.2f} %" for q in T_Q)
      + f"; Q1 / mean(Q2-Q4) of S_fed {resC[1] / oth:.2f}, of ration (sqrt) {math.sqrt(resC[1] / oth):.2f}")
add(SRC_R, DOI_R, "whiting", "North Sea (RFA 1-8)", "Q1 vs Q2-Q4 1991", "adults 200-400 mm, classification C",
    "Q1 / mean(Q2-Q4) of ration / GER Cmax", round(math.sqrt(resC[1] / oth), 2), "-", "D")
print("  empty + regurgitated, all sizes:", {q: T2N[q][1] + T2N[q][2] for q in T_Q},
      "as %:", {q: round((T2N[q][1] + T2N[q][2]) / sum(T2N[q]) * 100) for q in T_Q})

# 1981 empties by quarter: Hislop et al. 1983 (ICES CM 1983/G:59) Table 1, TOTAL row of each Roundfish
# Area (all size groups): (total, full, regurgitated, empty) per quarter. Area 1 Q1 "full" is printed
# "5213" (OCR); it is not used (empty and total are legible and each row sums).
SRC_H = "Hislop et al. 1983 ICES CM 1983/G:59"
DOI_H = "url:ices.dk/sites/pub/CM Doccuments/1983/G/1983_G59.pdf"
# Columns as printed: Total, "Full", "Reg", "Empty" (page image checked). Area 1 Q1 "Full" = 528 on the
# page image (OCR "5213"). area: [Q1..Q4] as (total, col2 "Full", col3 "Reg", col4 "Empty")
H1 = {
    1: [(1615, 528, 153, 934), (1334, 140, 881, 313), (954, 173, 183, 598), (539, 45, 286, 208)],
    2: [(940, 319, 111, 510), (214, 77, 2, 135), (393, 98, 84, 211), (555, 187, 37, 331)],
    3: [(1867, 372, 820, 675), (1104, 155, 488, 461), (608, 166, 227, 215), (432, 50, 183, 199)],
    4: [(713, 81, 386, 246), (277, 34, 6, 237), (363, 142, 54, 167), (243, 102, 24, 117)],
    5: [(108, 20, 24, 64), (184, 69, 0, 115), (258, 112, 0, 146), (38, 9, 10, 19)],
    6: [(2035, 517, 509, 1009), (1098, 435, 21, 642), (1066, 255, 28, 783), (1213, 248, 98, 867)],
    7: [(554, 181, 102, 271), (0, 0, 0, 0), (85, 38, 6, 41), (427, 85, 39, 303)],
}
print()
print("1981 whiting stomach classes by quarter (Hislop et al. 1983 Table 1, areas 1-7 summed) [D from F]:")
yr = [0, 0, 0, 0]
for q in range(4):
    for a in H1:
        for i in range(4):
            yr[i] += H1[a][q][i]
print(f"   year sums: total {yr[0]}, col 'Full' {yr[1]}, col 'Reg' {yr[2]}, col 'Empty' {yr[3]}")
print("   Hislop et al. 1991 abstract: 'Almost 50% were either empty or ... regurgitation. The contents of the")
print("   remaining stomachs (approximately 9800) were analysed.' -> the column printed 'Empty' matches the")
print(f"   analysed (fed) count ({yr[3]}); 'Full' + 'Reg' = {yr[1] + yr[2]} ({(yr[1] + yr[2]) / yr[0] * 100:.0f} %) matches")
print("   'almost 50%'. Reading B swaps the printed 'Full' and 'Empty' headers [J]; it also fits Robb 1994's")
print("   statement that Q1 1991 empties were 'considerably higher' and Q2-Q4 'similar to or lower' than 1981.")
for q in range(4):
    tot = sum(H1[a][q][0] for a in H1)
    c2 = sum(H1[a][q][1] for a in H1)
    c4 = sum(H1[a][q][3] for a in H1)
    print(f"   Q{q+1}: N {tot}; empty as printed (col 4) {c4 / tot * 100:.1f} %; "
          f"empty under reading B (col 2) {c2 / tot * 100:.1f} %")
    add(SRC_H, DOI_H, "whiting", "North Sea (RFA 1-7)", f"Q{q+1} 1981", "Table 1 TOTAL rows summed; headers as printed",
        "stomachs in column printed 'Empty', all sizes", round(c4 / tot * 100, 1), "%", "D")
    add(SRC_H, DOI_H, "whiting", "North Sea (RFA 1-7)", f"Q{q+1} 1981",
        "Table 1 TOTAL rows summed; 'Full'/'Empty' headers read as swapped (reading B, J)",
        "stomachs empty, all sizes", round(c2 / tot * 100, 1), "%", "D")
print("1991 for comparison (Robb Table 2): Q1-Q4", [T2[q][2] for q in T_Q], "% empty")

# 1981 comparison (Hislop et al. 1991 abstract; Robb Table 6): S = 0.009 W^1.057, no quarter split
print()
print("1981 (Hislop et al. 1991 [A]: S = 0.009 W^1.057, all quarters; Hislop et al. 1983 [F]: 'no obvious")
print("between-quarter differences in the weights of food in the stomachs'):")
for W in (95, 172, 278):
    s81 = 0.009 * W ** 1.057
    s91 = 0.009 * W ** 1.158
    print(f"   W {W} g: S/W 1981 {s81 / W * 100:.2f} %, 1991 annual fit {s91 / W * 100:.2f} %")

# =====================================================================================================
# 2. SMS key run 2023 quarterly rations (a W^b kg/quarter) [F data], whiting 0.2 kg and saithe 1.066 kg
# =====================================================================================================
print()
print("=" * 100)
print("2. SMS KEY RUN 2023 quarterly rations (WGSAM 2024 Table 3-8 = NS2023_consum_ab.in) [F data, D]")
print("=" * 100)
par = {}
cur = None
for line in open(consum_path, encoding="utf-8"):
    s = line.strip()
    if s.startswith("#") and "quarter" not in s and "paramter" not in s and "check" not in s:
        cur = s.lstrip("#").strip()
        par[cur] = []
    elif "# quarter" in s and cur is not None:
        a_, b_ = s.split("#")[0].split()
        par[cur].append((float(a_), float(b_)))
SRC_S = "ICES WGSAM 2024 (SMS North Sea key run 2023), Table 3-8 / consum_ab.in"
DOI_S = "10.17895/ices.pub.25020968"
cases = [("Whiting", "whiting", 0.200, 29.5), ("Saithe", "saithe", 1.066, 50.0)]
for key, sp, Wkg, L in cases:
    W = Wkg * 1000
    print(f"{sp} {Wkg} kg, L {L} cm (whiting W = 0.00603 L^3.077 [S]; saithe 50 cm = 1066 g [S, calc_ger_cmax.py])")
    vals = {}
    for q, (aa, bb) in enumerate(par[key], start=1):
        rq = aa * Wkg ** bb / 91.25 / Wkg * 100       # %BW/d
        vals[q] = rq
        cm = cmax_pct(L, W, T_Q[q], E_CENTRAL)
        cm42 = cmax_pct(L, W, T_Q[q], E_ALT)
        cm5 = cmax_pct(L, W, T_Q[q], E_CENTRAL, 0.05)
        print(f"   Q{q}: a {aa:.5f} b {bb:.5f} -> ration {rq:.2f} %BW/d; GER Cmax at {T_Q[q]} C "
              f"{cm:.2f} (E 4.2: {cm42:.2f}; S_max 5 %: {cm5:.2f}) %BW/d; ration/Cmax {rq / cm:.2f} "
              f"(E 4.2: {rq / cm42:.2f}; S_max 5 %: {rq / cm5:.2f})")
        add(SRC_S, DOI_S, sp, "North Sea (SMS area IV)", f"Q{q}", "SMS a W^b (GER on stomach data, 2020 key run)",
            f"ration at {Wkg} kg", round(rq, 3), "%BW/d", "D")
        add(SRC_S, DOI_S, sp, "North Sea (SMS area IV)", f"Q{q}", f"vs GER Cmax at {T_Q[q]} C (Daan temps), E {E_CENTRAL}, S_max 2.5 %BW",
            "ration / GER Cmax", round(rq / cm, 2), "-", "D")
        add(SRC_S, DOI_S, sp, "North Sea (SMS area IV)", f"Q{q}", f"vs GER Cmax at {T_Q[q]} C (Daan temps), E {E_CENTRAL}, S_max 5 %BW",
            "ration / GER Cmax", round(rq / cm5, 2), "-", "D")
    tcorr = {q: vals[q] * math.exp(-DEL * (T_Q[q] - T_Q[1])) for q in T_Q}
    print("   temperature-normalised to Q1 (x exp(-0.081 (T - 6))):", {q: round(v, 2) for q, v in tcorr.items()})

# =====================================================================================================
# 3. Greenstreet et al. 1997 guild-level quarterly consumption and biomass [F], proxies
# =====================================================================================================
print()
print("=" * 100)
print("3. GREENSTREET et al. 1997 Tables 1 and 3 (guild level, North Sea) [F inputs, D ratios] - PROXIES")
print("=" * 100)
SRC_G = "Greenstreet et al. 1997 ICES JMS 54:243"
DOI_G = "url:academic.oup.com/icesjms/article-pdf/54/2/243"
B = {"Demersal Piscivore": (1961013, 2143494, 2326758, 2510878),
     "Pelagic Planktivore": (3645992, 4285440, 4939087, 5606931)}
C_tot = {"Demersal Piscivore": (21829.0, 28433.0, 41970.7, 34368.9),
         "Pelagic Planktivore": (39650.0, 71137.3, 82713.5, 65891.9)}
C_mzc = {"Demersal Piscivore": (3627.7, 6225.2, 6803.7, 3971.8),
         "Pelagic Planktivore": (17126.2, 25278.5, 28803.3, 22535.4)}
C_pp = {"Demersal Piscivore": (9116.8, 9972.0, 14779.6, 13946.7)}   # pelagic planktivores eaten
CPR_mzc = (5.44, 9.48, 3.31, 4.75)   # g WW/m2, meso-zooplankton carnivores (CPR euphausiids + hyperiids)
for g in B:
    sp = "demersal piscivore guild (cod, haddock, whiting, saithe = 75-80 %)" if g.startswith("Dem") else \
        "pelagic planktivore guild (Norway pout, herring, sprat, sandeel)"
    print(g)
    for q in range(4):
        r = C_tot[g][q] / B[g][q] * 100
        sh = C_mzc[g][q] / C_tot[g][q]
        rm = C_mzc[g][q] / B[g][q] * 100
        cm = None
        line = (f"   Q{q+1} (T {T_Q[q+1]} C): ration {r:.2f} %BW/d; meso-zoo carnivore share {sh:.3f} "
                f"({rm:.3f} %BW/d)")
        if g in C_pp:
            line += f"; pelagic planktivores share {C_pp[g][q] / C_tot[g][q]:.3f}"
        print(line)
        add(SRC_G, DOI_G, sp, "North Sea", f"Q{q+1}", "guild consumption (Table 3) / biomass (Table 1)",
            "ration", round(r, 3), "%BW/d", "D")
        add(SRC_G, DOI_G, sp, "North Sea", f"Q{q+1}", "Table 3", "meso-zooplankton carnivores (euphausiids+) share of food",
            round(sh, 3), "-", "D")
        add(SRC_G, DOI_G, sp, "North Sea", f"Q{q+1}", "Table 3", "total consumption", C_tot[g][q], "t WW/d", "F")
        add(SRC_G, DOI_G, sp, "North Sea", f"Q{q+1}", "Table 1 (Q2-Q3 interpolated linearly by the authors)",
            "biomass", B[g][q], "t WW", "F")
    t = [C_tot[g][q] / B[g][q] * 100 * math.exp(-DEL * (T_Q[q + 1] - 6.0)) for q in range(4)]
    print("   ration normalised to 6 C by the GER temperature term:", [round(x, 2) for x in t])
mean_cpr = sum(CPR_mzc) / 4
print("CPR meso-zooplankton carnivores (euphausiids + hyperiids), g WW/m2:", CPR_mzc,
      " relative to annual mean:", [round(x / mean_cpr, 2) for x in CPR_mzc])
for q in range(4):
    add(SRC_G, DOI_G, "meso-zooplankton carnivores (CPR euphausiids + hyperiids)", "North Sea", f"Q{q+1}",
        "CPR, smoothed monthly counts (Table 1)", "biomass density", CPR_mzc[q], "g WW/m2", "F")
    add(SRC_G, DOI_G, "-", "North Sea", f"Q{q+1}", "assumed (after Daan 1989)", "sea temperature", T_Q[q + 1], "C", "F")

# =====================================================================================================
# 4. Cod 1991 feeding coefficient by quarter (Kikkert 1993) - PROXY
# =====================================================================================================
print()
print("4. COD (proxy), North Sea 1991, S = phi L^3 (Kikkert ICES CM 1993/G:13 Table 7, b fixed 3) [F]")
phi = {1: 15.41, 2: 22.69, 3: 19.02, 4: 16.42}
m = (phi[2] + phi[3] + phi[4]) / 3
print("   phi (x 1e-5):", phi, f" Q1 / mean(Q2-Q4) = {phi[1] / m:.2f}")
for q, v in phi.items():
    add("Kikkert 1993 ICES CM 1993/G:13", "url:vliz.be/imisdocs/publications/ocrd/275167.pdf", "cod (proxy)",
        "North Sea (RFA 1-7)", f"Q{q} 1991", "Table 7, S = phi L^3", "feeding coefficient phi", v, "1e-5 g/cm^3", "F")

# =====================================================================================================
# 5. Norway pout (proxies on the whiting GER)
# =====================================================================================================
print()
print("5. NORWAY POUT on the whiting GER (proxy) [D]")
# Albert 1995 [F]: smaller pout (11.0-14.9 cm), July, 140 m, coastal bank SW Norway, ration 1.1 %BW/d,
# Jones' method at 7 C. W = 0.0065 L^3.17 [J, sibling note]. Diet mostly copepods / shrimps: E 4.0 [J].
for L in (13.0,):
    W = 0.0065 * L ** 3.17
    cm = cmax_pct(L, W, 7.0, 4.0)
    cm5 = cmax_pct(L, W, 7.0, 4.0, 0.05)
    print(f"   Albert 1995 July pout L {L} cm W {W:.1f} g: GER Cmax at 7 C, E 4.0 = {cm:.2f} %BW/d "
          f"(S_max 5 %: {cm5:.2f}); ration 1.1 -> ration/Cmax {1.1 / cm:.2f} ({1.1 / cm5:.2f})")
    add("Albert 1995 ICES JMS 52:873", "url:academic.oup.com/icesjms/article/52/5/873/783892", "Norway pout",
        "coastal bank SW Norway, 140 m", "July", "MAXIMS / Jones 1974 at 7 C", "ration (smaller pout 11-14.9 cm)",
        1.1, "%BW/d", "F")
    add("Albert 1995 ICES JMS 52:873", "url:academic.oup.com/icesjms/article/52/5/873/783892", "Norway pout",
        "coastal bank SW Norway, 140 m", "July", f"vs whiting-GER Cmax at 7 C, E 4.0, L {L} cm (proxy), S_max 2.5 %BW",
        "ration / GER Cmax", round(1.1 / cm, 2), "-", "D")
    add("Albert 1995 ICES JMS 52:873", "url:academic.oup.com/icesjms/article/52/5/873/783892", "Norway pout",
        "coastal bank SW Norway, 140 m", "July", f"vs whiting-GER Cmax at 7 C, E 4.0, L {L} cm (proxy), S_max 5 %BW",
        "ration / GER Cmax", round(1.1 / cm5, 2), "-", "D")
# Albert 1994 content (sibling [D]): 0.36-1.3 %BW wet incl. empties, the Fig. 10 range. Fig. 10 compares
# winter and summer, so the range is NOT winter-only (season not separable). Empties 18.6-33.6 % over
# both seasons and sizes.
lo, hi = ratio(1 - 0.336, 0.0036, 0.025), ratio(1 - 0.186, 0.013, 0.025)
print(f"   Albert 1994 content 0.36-1.30 %BW (winter and summer together), empty 18.6-33.6 %: ratio/Cmax "
      f"{lo:.2f}-{hi:.2f} (S_max 5 %: {lo / math.sqrt(2):.2f}-{hi / math.sqrt(2):.2f})")
add("Albert 1994 ICES JMS 51:45 (via Planktivorernas vinterkostnad)", "10.1006/jmsc.1994.1005", "Norway pout",
    "Norwegian Deep incl. Skagerrak", "winter and summer (Fig. 10 range; season not separable)",
    "whiting-GER proxy on the content range, S_max 2.5 %BW", "ration / GER Cmax, range", f"{lo:.2f}-{hi:.2f}", "-", "D")

# =====================================================================================================
# 6. Transcribed and secondary rows (see lit_seasonal.md for quotes)
# =====================================================================================================
H83 = ("Hislop et al. 1983 ICES CM 1983/G:59", "url:ices.dk/sites/pub/CM Doccuments/1983/G/1983_G59.pdf")
H91 = ("Hislop et al. 1991 ICES JMS 48:139", "10.1093/icesjms/48.2.139")
P00 = ("Pedersen 2000 J Fish Biol 57:402", "10.1111/j.1095-8649.2000.tb02180.x")
ONS = ("Onsrud et al. 2004 via Krillparen mot vintermagar E1", "10.1016/j.icesjms.2004.09.005")
DB = ("ICES stomach DB via Krillintag och jaktsatt (krill_ration_population_no2022)", "-")
DOR = ("Dorman, Johnsson & Larsson 1992 ICES CM 1992/G:70", "url:vliz.be/imisdocs/publications/ocrd/274761.pdf")
KIK = ("Kikkert 1993 ICES CM 1993/G:13", "url:vliz.be/imisdocs/publications/ocrd/275167.pdf")
A94 = ("Albert 1994 ICES JMS 51:45 (via Planktivorernas vinterkostnad)", "10.1006/jmsc.1994.1005")
A95 = ("Albert 1995 ICES JMS 52:873", "url:academic.oup.com/icesjms/article/52/5/873/783892")
RAI = ("Raitt 1968 FAO Fish. Synop. 33 Rev.1", "url:fao.org/4/69606e/69606e.pdf")
MAT = ("Mathers et al. 1992 MEPS 86:31 (via Mattnad hos rovfisk)", "10.3354/meps086031")
KS = ("Krill i Skagerrak summary (J on F inputs)", "-")
KG = ("Krill over grunda bottnar summary (RV Dana HERAS tows)", "-")
lit = [
    # Robb 1994 text and the Skagerrak (Roundfish Area 8, Figure 1) row
    (SRC_R, DOI_R, "whiting", "North Sea", "Q2 1991", "Fig. 5 text", "sandeels relatively more important", "yes", "-", "F"),
    (SRC_R, DOI_R, "whiting", "North Sea", "Q4 1991", "Fig. 5 text", "dominant fish prey", "Norway pout", "-", "F"),
    (SRC_R, DOI_R, "whiting", "North Sea", "Q3-Q4 1991", "Fig. 5 text", "haddock (0-group) prominent", "yes", "-", "F"),
    (SRC_R, DOI_R, "whiting", "North Sea", "Q3 1991", "Fig. 5 text", "herring most noticeable", "yes", "-", "F"),
    (SRC_R, DOI_R, "cod, haddock, saithe, mackerel", "North Sea", "Q1 1991", "text citing Anon. 1993",
     "empty stomachs far more frequent than other quarters", "yes", "-", "S"),
    (SRC_R, DOI_R, "whiting", "Skagerrak (Roundfish Area 8, Fig. 1)", "Q4 1991", "Table 2 / Table 3",
     "stomachs empty, all sizes", 18.3, "%", "F"),
    (SRC_R, DOI_R, "whiting", "Skagerrak (Roundfish Area 8, Fig. 1)", "Q4 1991", "Table 2", "stomachs examined", 1045, "n", "F"),
    (SRC_R, DOI_R, "whiting", "Skagerrak (Roundfish Area 8, Fig. 1)", "Q1-Q3 1991", "Table 1", "stomachs examined", 0, "n", "F"),
    # Hislop 1983 / 1991
    H83 + ("whiting", "North Sea", "Q1-Q4 1981", "text", "between-quarter differences in stomach content weight", "none obvious", "-", "F"),
    H83 + ("whiting", "North Sea", "1981", "Daan 1973 rectilinear model", "digestion time applied", 2.5, "d", "F"),
    H83 + ("whiting", "northern North Sea (RRA 1-3)", "Q1 1981", "text", "molluscs (mainly cephalopods) relatively important", "yes", "-", "F"),
    H91 + ("whiting", "North Sea", "1981 (all quarters)", "stomach weight regression", "S = 0.009 W^1.057 (incl. empties)", "0.009 W^1.057", "g", "A"),
    H91 + ("whiting", "North Sea", "1981 (all quarters)", "stomach classification", "empty or regurgitated", "almost 50", "%", "A"),
    H91 + ("whiting", "North Sea", "1981 (all quarters)", "stomach classification", "stomachs analysed", "approximately 9800", "n", "A"),
    # Pedersen 2000
    P00 + ("whiting", "North Sea", "1992 (season not in abstract)", "Andersen evacuation model", "specific daily ration range", "0.41-1.66", "%BW/d", "A"),
    P00 + ("whiting", "North Sea", "1993 (season not in abstract)", "Andersen evacuation model", "specific daily ration range", "0.78-1.75", "%BW/d", "A"),
    # Onsrud 2004 via Krillparen mot vintermagar
    ONS + ("whiting", "Oslofjord, 120 m", "Nov 2000", "GER at 7 C on counted krill (sibling D)", "krill ration", "0.23-0.54", "%BW/d", "S"),
    ONS + ("whiting", "Oslofjord, 120 m", "Mar 2001", "GER at 7 C on counted krill (sibling D)", "krill ration", "0.65-1.51", "%BW/d", "S"),
    ONS + ("whiting", "Oslofjord, 120 m", "Nov 2000", "Table 3, day", "stomachs empty", 25.0, "%", "S"),
    ONS + ("whiting", "Oslofjord, 120 m", "Nov 2000", "Table 3, night", "stomachs empty", 14.81, "%", "S"),
    ONS + ("whiting", "Oslofjord, 120 m", "Mar 2001", "Table 3, day", "stomachs empty", 8.16, "%", "S"),
    ONS + ("whiting", "Oslofjord, 120 m", "Mar 2001", "Table 3, night", "stomachs empty", 6.25, "%", "S"),
    ONS + ("whiting", "Oslofjord, 120 m", "Nov / Mar", "stomach weight", "krill share of stomach content", "~82", "%W", "S"),
    ONS + ("Norway pout", "Oslofjord, 120 m", "Nov 2000", "GER (whiting proxy) at 7 C (sibling D)", "krill ration", "0.36-0.82", "%BW/d", "S"),
    ONS + ("Norway pout", "Oslofjord, 120 m", "Mar 2001", "GER (whiting proxy) at 7 C (sibling D)", "krill ration", "0.51-1.14", "%BW/d", "S"),
    # ICES stomach DB krill ration / GER Cmax (main agent; brief)
    DB + ("whiting", "NE North Sea", "Q1", "pooled 1980-91 records, GER", "krill ration / GER Cmax", 0.31, "-", "S"),
    DB + ("whiting", "NE North Sea", "Q2", "pooled 1980-91 records, GER", "krill ration / GER Cmax", 0.00, "-", "S"),
    DB + ("whiting", "NE North Sea", "Q4", "pooled 1980-91 records, GER", "krill ration / GER Cmax", 0.007, "-", "S"),
    DB + ("whiting", "NW North Sea", "Q1", "pooled 1980-91 records, GER", "krill ration / GER Cmax", 0.28, "-", "S"),
    DB + ("whiting", "NW North Sea", "Q2", "pooled 1980-91 records, GER", "krill ration / GER Cmax", 0.0, "-", "S"),
    DB + ("whiting", "NW North Sea", "Q3", "pooled 1980-91 records, GER", "krill ration / GER Cmax", 0.012, "-", "S"),
    DB + ("whiting", "NW North Sea", "Q4", "pooled 1980-91 records, GER", "krill ration / GER Cmax", 0.08, "-", "S"),
    DB + ("whiting", "Skagerrak", "Q1", "pooled 1980-91 records, GER", "krill ration / GER Cmax", 0.06, "-", "S"),
    DB + ("whiting", "Skagerrak", "Q2", "pooled 1980-91 records, GER", "krill ration / GER Cmax", 0.37, "-", "S"),
    DB + ("whiting", "Kattegat", "Q2", "pooled 1980-91 records, GER", "krill ration / GER Cmax", 0.05, "-", "S"),
    ("Vitlingens ranson (ICES DB, 106 whiting)", "-", "whiting", "Skagerrak", "April 1991", "stomach weight",
     "euphausiid share of diet", 0.537, "-", "S"),
    # Krill presence
    KS + ("krill (M. norvegica dominant)", "Skagerrak box", "Q1", "judgement on Kattegat dry weight, spawning, IKMT", "stock factor relative to annual mean", 1.0, "-", "S"),
    KS + ("krill (M. norvegica dominant)", "Skagerrak box", "Q2", "judgement on Kattegat dry weight, spawning, IKMT", "stock factor relative to annual mean", 0.6, "-", "S"),
    KS + ("krill (M. norvegica dominant)", "Skagerrak box", "Q3", "judgement on Kattegat dry weight, spawning, IKMT", "stock factor relative to annual mean", 1.1, "-", "S"),
    KS + ("krill (M. norvegica dominant)", "Skagerrak box", "Q4", "judgement on Kattegat dry weight, spawning, IKMT", "stock factor relative to annual mean", 1.3, "-", "S"),
    KG + ("krill", "N North Sea / Skagerrak, 100-170 m beds", "June-July 2021-25", "daytime near-bed trawl", "tows with krill (of 4)", 3, "n", "S"),
    KG + ("krill", "beds < 100 m", "June-July 2021-25", "daytime near-bed trawl", "tows with krill (of 14)", 0, "n", "S"),
    (SRC_G, DOI_G, "euphausiids", "North Sea", "late winter / Q2", "text (after Lindley 1978, 1982a; Mauchline 1980, 1984)",
     "adults bulk of population in late winter; most die after spawning; young dominate Q2", "yes", "-", "F"),
    DOR + ("cod (proxy)", "Skagerrak", "Q1 1991", "stomach classification", "stomachs with content", 87.5, "%", "F"),
    DOR + ("cod (proxy)", "Kattegat", "Q1 1991", "stomach classification", "stomachs with content", 89.7, "%", "F"),
    DOR + ("cod (proxy)", "E North Sea", "Q1 1991", "stomach classification", "stomachs with content", 91, "%", "F"),
    DOR + ("cod (proxy)", "Kattegat", "Q1 1991", "stomach weight", "krill share of crustacean prey", 63, "%W", "F"),
    DOR + ("cod (proxy)", "Kattegat", "Feb 1981", "stomach weight (Borje et al. 1987, cited)", "krill share of crustacean prey", "~2", "%W", "S"),
    # Kikkert text
    KIK + ("cod (proxy)", "North Sea", "Q1 1991", "text", "empty stomachs in larger cod", "circa 15", "%", "F"),
    # Norway pout
    A94 + ("Norway pout < 14 cm", "Norwegian Deep incl. Skagerrak", "winter", "Table 8", "stomachs empty", 33.6, "%", "S"),
    A94 + ("Norway pout >= 14 cm", "Norwegian Deep incl. Skagerrak", "winter", "Table 8", "stomachs empty", 24.5, "%", "S"),
    A94 + ("Norway pout < 14 cm", "Norwegian Deep incl. Skagerrak", "summer", "Table 8", "stomachs empty", 27.8, "%", "S"),
    A94 + ("Norway pout >= 14 cm", "Norwegian Deep incl. Skagerrak", "summer", "Table 8", "stomachs empty", 18.6, "%", "S"),
    A94 + ("Norway pout < 14 cm", "Norwegian Deep incl. Skagerrak", "winter", "Table 8", "euphausiid share of diet", 78.6, "%W", "S"),
    A94 + ("Norway pout >= 14 cm", "Norwegian Deep incl. Skagerrak", "winter", "Table 8", "euphausiid share of diet", 81.4, "%W", "S"),
    A94 + ("Norway pout", "Norwegian Deep incl. Skagerrak", "winter vs summer", "Fig. 10", "content relative to fish weight", "consistently higher in winter", "-", "S"),
    A94 + ("Norway pout", "Norwegian Deep incl. Skagerrak", "winter and summer (Fig. 10 range)", "Fig. 10 (sibling D)", "content incl. empties", "0.36-1.3", "%BW wet", "S"),
    A95 + ("Norway pout", "coastal bank SW Norway, 140 m", "July", "Table 4", "euphausiid share of diet", 3.0, "%W", "F"),
    RAI + ("Norway pout adults", "northern North Sea", "January", "text (inconclusive per author)", "low feeding intensity", "yes", "-", "F"),
    RAI + ("Norway pout adults", "northern North Sea", "March-April (main spawning)", "text (inconclusive per author)", "low feeding intensity", "yes", "-", "F"),
    RAI + ("Norway pout adults", "northern North Sea", "August", "text (inconclusive per author)", "low feeding intensity", "yes", "-", "F"),
    # Saithe
    MAT + ("saithe 2+-5+ (~42 cm)", "Beryl oilfield, N North Sea", "Apr 1989", "gall-bladder index", "GLI", 16.5, "-", "S"),
    MAT + ("saithe 2+-5+ (~42 cm)", "Beryl oilfield, N North Sea", "Aug and Oct 1988", "gall-bladder index", "GLI", "~0.3", "-", "S"),
    MAT + ("saithe 2+-5+ (~42 cm)", "Beryl oilfield, N North Sea", "Apr 1989", "Fulton K", "condition factor", 0.87, "-", "S"),
    MAT + ("saithe 2+-5+ (~42 cm)", "Beryl oilfield, N North Sea", "Aug 1988", "Fulton K", "condition factor", 1.13, "-", "S"),
    ("Krillintag och jaktsatt summary (ICES DB)", "-", "saithe", "North Sea areas", "by area and quarter",
     "all stomachs, GER", "krill ration / GER Cmax range", "0.04-0.55", "-", "S"),
    ("Krill over grunda bottnar summary (ICES DB)", "-", "saithe", "NE North Sea", "Q3", "stomach mass, spread",
     "krill mass share", "0.31-0.58", "-", "S"),
    ("ICES 1992 ICES CM 1992/G:12 (1991 stomach project progress report)", "url:vliz.be/imisdocs/publications/ocrd/278195.pdf",
     "saithe", "North Sea", "1991 (no season)", "text", "principal fish prey", "Norway pout and herring; large amounts of mysids", "-", "F"),
]
for r in lit:
    add(*r)

with open(out_csv, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["source", "doi", "species", "region", "season", "method", "quantity", "value",
                                       "unit", "tag"])
    w.writeheader()
    w.writerows(rows)
print()
print(f"wrote {len(rows)} rows to {out_csv}")
