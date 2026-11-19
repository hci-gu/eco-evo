"""Does SMS weight-in-stock (west) resolve a spawning loss? Q-to-Q change of mean weight at age,
cohort-followed, 2013-22, ages 2-6 (adult), SMS 2023 key run.

Usage: python3 -I sms_spawning_dip.py <sms summary.out>
(the file is research_notes/"Fiskgrupper i rutan"/benthivores/data/sms2023/summary.out)
Also prints the annual P as the sum of the four quarterly log increments vs the Q1-to-Q1 log
increment (telescoping check: if equal, the annual P is net of any within-year loss).
"""
import sys

import numpy as np
import pandas as pd

S = pd.read_csv(sys.argv[1], sep=r"\s+")
S = S[(S.Year >= 2013) & (S.Year <= 2022)]
S["west"] = pd.to_numeric(S.west, errors="coerce")
names = {16: "cod", 18: "haddock", 17: "whiting", 19: "saithe", 21: "herring", 24: "Norway pout",
         25: "sprat", 20: "mackerel"}
for sp, n in names.items():
    d = S[S["Species.n"] == sp].groupby(["Age", "Quarter"]).west.mean().unstack()
    rows = []
    for a in range(2, 7):
        if a not in d.index or a + 1 not in d.index:
            continue
        w = d.loc[a]
        inc = [np.log(w[2] / w[1]), np.log(w[3] / w[2]), np.log(w[4] / w[3]), np.log(d.loc[a + 1][1] / w[4])]
        rows.append([a] + [f"{x:+.3f}" for x in inc] + [f"{sum(inc):+.3f}", f"{np.log(d.loc[a + 1][1] / w[1]):+.3f}"])
    print(f"--- {n}: ln w increments Q1->Q2, Q2->Q3, Q3->Q4, Q4->Q1(next age); sum; Q1->Q1")
    for r in rows:
        print("  age", *r)
