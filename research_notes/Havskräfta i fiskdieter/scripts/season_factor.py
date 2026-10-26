"""Season lever: annual-mean Nephrops share of cod (and haddock) diet relative to the Q1 share.

Usage: python3 -I season_factor.py <lit_rows_A.csv> <out_csv>

All Skagerrak / northern North Sea stomach data used for the cod share are Q1 (Jan-Feb). The North Sea
1981 and 1991 stomach projects (Daan 1989 ICES CRR 164 Table 8; Hislop et al. 1997 ICES CRR 219 Table 8;
rows in lit_rows_A.csv, % by weight per age class and quarter, ICES IV without IIIa) give the quarterly
pattern. For each year and age class with all four quarters:
  unweighted  F = mean(Q1..Q4) / Q1
  ration-weighted F_r = sum(w_q s_q) / sum(w_q) / Q1 with w_Q1 = 0.5 and w_Q2..Q4 = 1 (J: Armstrong et al.
              1991, Irish Sea cod, daily intake 0.8-0.9 % of body mass in spring vs 1.6-1.8 % the rest of
              the year [F via ../../Refugium och utbyte i bentos/tables/part_D_predation.txt])
Age classes with Q1 share < 0.5 % are listed but left out of the summary (ratio unstable).
"""
import csv, re, sys, statistics, collections

src, out = sys.argv[1], sys.argv[2]
D = collections.defaultdict(dict)
for r in csv.DictReader(open(src)):
    if r["predator"] not in ("cod", "haddock"):
        continue
    if "CRR 164" not in r["source"] and "CRR 219" not in r["source"]:
        continue
    if "Table 8" not in r["source"]:
        continue
    m = re.search(r"age (\d\+?)", r["predator_length"])
    if not m:
        continue
    try:
        v = float(r["nephrops_pct_mass"])
    except ValueError:
        continue
    D[(r["predator"], r["year"], m.group(1))][r["season"]] = v

rows = []
for (sp, yr, age), q in sorted(D.items()):
    if not all(k in q for k in ("Q1", "Q2", "Q3", "Q4")):
        continue
    s = [q["Q1"], q["Q2"], q["Q3"], q["Q4"]]
    w = [0.5, 1, 1, 1]
    ann = sum(s) / 4
    annw = sum(a * b for a, b in zip(s, w)) / sum(w)
    rows.append([sp, yr, age] + s + [round(ann, 3), round(annw, 3),
                                      round(ann / s[0], 2) if s[0] > 0 else "", round(annw / s[0], 2) if s[0] > 0 else ""])
with open(out, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["predator", "year", "age", "Q1", "Q2", "Q3", "Q4", "annual_mean", "annual_ration_weighted",
                 "F_unweighted", "F_ration_weighted"])
    wr.writerows(rows)
for r in rows:
    print(*r)
for sp in ("cod", "haddock"):
    for yr in ("1981", "1991", "both"):
        sel = [r for r in rows if r[0] == sp and (yr == "both" or r[1] == yr) and r[3] >= 0.5 and r[2] not in ("0", "1")]
        if sel:
            fu = [r[9] for r in sel]; fw = [r[10] for r in sel]
            print(f"{sp} {yr} ages 2+ with Q1 >= 0.5 %: n {len(sel)}  F unweighted median {statistics.median(fu):.2f} "
                  f"(range {min(fu):.2f}-{max(fu):.2f});  ration-weighted median {statistics.median(fw):.2f} "
                  f"(range {min(fw):.2f}-{max(fw):.2f})")
