"""Checks for the haul statistics: position source, uncounted krill, whiting single-stomach mix.
Usage: python3 -I q2_records_checks.py <records_linked.csv.gz>"""
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1], low_memory=False)
df = df[df["area"] != "other"]
print("pos values:", df["pos"].value_counts(dropna=False).to_dict())
for sp in ["whiting", "saithe", "Norway pout", "cod"]:
    d = df[(df["sp"] == sp) & (df["occ_krill"] > 0)]
    print(sp, "krill records", len(d), "krill_count NaN", int(d["krill_count"].isna().sum()),
          "krill_uncounted>0", int((d["krill_uncounted"].fillna(0) > 0).sum()),
          "pos:", d["pos"].value_counts(dropna=False).to_dict(), "day NaN", int(d["day"].isna().sum()))
w = df[(df["sp"] == "whiting") & (df["Number"] == 1) & (df["fed"] > 0)]
w = w.assign(kr=(w["occ_krill"] > 0).astype(int), cu=(w["occ_crustu"] > 0).astype(int))
print(w.groupby(["area", "year"]).agg(n=("kr", "size"), krill=("kr", "sum"), crustu=("cu", "sum"), L=("L_cm", "median"),
                                      hauls=("tblHaulID", "nunique")).to_string())
