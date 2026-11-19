"""Condition (Fulton K = 100 W / L^3, g and cm) of adult saithe (>= 35 cm) by bed class in the ICES stomach database,
records with an individual weight; and the same for GOV-length saithe if present.

Usage: python3 -I condition_bed.py <records_linked.csv.gz> <ices_stomach_all_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

R = pd.read_csv(sys.argv[1], low_memory=False)
P = pd.read_csv(os.path.join(sys.argv[2], "PredatorInformation.csv"), dtype=str, on_bad_lines="skip")
R = R[(R.sp == "saithe") & (R.L_cm >= 35) & (R.lat >= 57.0)].copy()
R["tblPredatorInformationID"] = R.tblPredatorInformationID.astype(str)
P["IndWgt"] = pd.to_numeric(P.IndWgt, errors="coerce")
R = R.merge(P[["tblPredatorInformationID", "IndWgt"]], on="tblPredatorInformationID", how="left")
R = R[(R.IndWgt > 0) & (R.Number.fillna(1) == 1)]
R["K"] = 100 * R.IndWgt / R.L_cm ** 3
R = R[(R.K > 0.4) & (R.K < 2.0)]
print(f"records with individual weight: {len(R)}; positions {R.pos.value_counts().to_dict()}; years "
      f"{sorted(R.year.dropna().astype(int).unique().tolist())}; quarters {R.quarter.value_counts().to_dict()}")
for keys in (["bed"], ["quarter", "bed"]):
    g = R.groupby(keys).agg(n=("K", "size"), stations=("station", "nunique"), K_mean=("K", "mean"), K_sd=("K", "std"),
                            L_mean=("L_cm", "mean"))
    print(g.round(3).to_string())
# length-adjusted: residual of log W on log L
b = np.polyfit(np.log(R.L_cm), np.log(R.IndWgt), 1)
R["res"] = np.log(R.IndWgt) - np.polyval(b, np.log(R.L_cm))
print(f"\nlog W = {b[1]:.3f} + {b[0]:.3f} log L; mean residual by bed class (x100 = % weight at length):")
print((R.groupby("bed").res.agg(["size", "mean", "sem"]) * [1, 100, 100]).round(2).to_string())
