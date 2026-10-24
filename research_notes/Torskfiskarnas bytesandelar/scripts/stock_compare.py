"""Stock-level vs box-survey biomass shares in the model's stage classes.

Usage: python3 -I stock_compare.py <tables_dir>
Reads length_class_pct.csv (box survey; raw and Fraser-corrected A_floor) and stock_length_class_share.csv
(stock_age_shares.py; primary runs, survey Q3 weights). Writes stock_vs_box_comparison.csv.
Stage classes: <10 cm, 10-20 cm, <20 cm, 20-30 cm, >= 30 cm (percent of species biomass).
"""
import os
import sys

import pandas as pd

tab = sys.argv[1]
box = pd.read_csv(os.path.join(tab, "length_class_pct.csv"))
stk = pd.read_csv(os.path.join(tab, "stock_length_class_share.csv"))
stk = stk[(stk.q3_weights == "survey") & (stk.role == "primary")]
G = {"lt10": ["[0, 10)"], "10_20": ["[10, 15)", "[15, 20)"], "lt20": ["[0, 10)", "[10, 15)", "[15, 20)"],
     "20_30": ["[20, 25)", "[25, 30)"], "ge30": ["[30, 40)", "[40, 50)", "[50, 60)", "[60, 70)", "[70, 200)"]}
rows = []
for _, r in box[box.variant.isin(["raw", "A_floor"]) & box.quarter.isin(["1", "3"])].iterrows():
    rows.append(dict(source="box_survey_" + r.variant, species=r.species, quarter=int(r.quarter), years="2021-2025",
                     **{k: round(sum(r[c] for c in v), 1) for k, v in G.items()}))
for _, r in stk.iterrows():
    rows.append(dict(source=f"stock_{r.stock}_alk_{r.alk_variant}", species=r.species, quarter=int(r.quarter),
                     years=r.years, **{k: round(sum(r[c] for c in v), 1) for k, v in G.items()}))
C = pd.DataFrame(rows).sort_values(["species", "quarter", "years", "source"])
C.to_csv(os.path.join(tab, "stock_vs_box_comparison.csv"), index=False)
pd.set_option("display.width", 200, "display.max_rows", 200)
print(C.to_string(index=False))
