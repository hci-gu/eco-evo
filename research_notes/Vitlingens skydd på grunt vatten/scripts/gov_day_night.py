"""Whiting (10-30 cm) GOV catch rate by day and night, matched within statistical rectangle x year x quarter, by bed
class, NS-IBTS north of 56 N (Q1 + Q3 2000-2025 and Q2 + Q4 1991-97). A night/day ratio < 1 means fewer whiting in the
bottom trawl's ~5 m opening at night (rising off the bed or dispersing); > 1 the reverse (or day avoidance of the gear).

Usage: python3 -I gov_day_night.py <gov dir> <gov q2q4 dir> > ../tables/gov_day_night_stdout.txt
Inputs read in place from research_notes/"Sejens djupläge"/data (DATRAS, ICES data policy, CC BY 4.0).
Statistic: per matched stratum the mean kg/h of night and of day hauls; ratio of sums over strata (each stratum
weighted equally after normalising by its day mean) and the median per-stratum log ratio; bootstrap over strata.
"""
import csv
import gzip
import sys
from collections import defaultdict

import numpy as np

rng = np.random.default_rng(2)


def load(d):
    H = {}
    for r in csv.DictReader(open(f"{d}/hauls.csv")):
        if float(r["ShootLat"]) < 56.0 or r["DayNight"] not in ("D", "N"):
            continue
        dep = float(r["Depth"])
        cl = "<100" if dep < 100 else "100-150" if dep < 150 else "150-250" if dep < 250 else ">=250"
        H[(r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip())] = dict(
            s=(r["StatRec"], r["Year"], r["Quarter"]), dn=r["DayNight"], cl=cl, kg=0.0)
    for r in csv.DictReader(gzip.open(f"{d}/catch.csv.gz", "rt")):
        if r["aphia"] != "126438":
            continue
        h = H.get((r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip()))
        L, n = float(r["length_cm"]), float(r["n_per_h"])
        if h is not None and 10 <= L < 30 and n > 0:
            h["kg"] += n * 0.0058 * L ** 3.10 / 1000
    return H


for d, lab in ((sys.argv[1], "Q1+Q3 2000-2025"), (sys.argv[2], "Q2+Q4 1991-97")):
    H = load(d)
    print(f"=== {lab}, N of 56 N, whiting 10-30 cm")
    for cl in ("<100", "100-150", "150-250", "all"):
        st = defaultdict(lambda: {"D": [], "N": []})
        for h in H.values():
            if cl == "all" or h["cl"] == cl:
                st[h["s"]][h["dn"]].append(h["kg"])
        pairs = [(np.mean(v["N"]), np.mean(v["D"])) for v in st.values() if v["N"] and v["D"]]
        pairs = [(n, dd) for n, dd in pairs if dd + n > 0]
        if len(pairs) < 5:
            print(f"  {cl:8s} matched strata {len(pairs)}")
            continue
        P = np.array(pairs)
        rs = P[:, 0].sum() / P[:, 1].sum()
        lr = np.median(np.log((P[:, 0] + 1) / (P[:, 1] + 1)))
        boots = []
        for _ in range(1000):
            Q = P[rng.integers(0, len(P), len(P))]
            boots.append(Q[:, 0].sum() / max(Q[:, 1].sum(), 1e-9))
        print(f"  {cl:8s} matched strata {len(P):4d}: night/day (ratio of sums) {rs:.2f} [{np.percentile(boots, 5):.2f}-"
              f"{np.percentile(boots, 95):.2f}]; median log((N+1)/(D+1)) {lr:+.2f}")
