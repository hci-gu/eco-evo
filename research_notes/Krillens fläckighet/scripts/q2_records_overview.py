"""Overview of the ICES stomach records table: single-stomach records per species,
hauls with >= k single stomachs, by period. Usage: python3 -I q2_records_overview.py <records_linked.csv.gz>"""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
print("columns:", len(df.columns), "rows:", len(df))
print(df["Number"].describe())
df["single"] = df["Number"] == 1
df["period"] = pd.cut(df["year"], [0, 1995, 2010, 2100], labels=["<=1995", "1996-2010", ">2010"])
g = df.groupby(["sp", "period", "single"], observed=True).size().unstack(fill_value=0)
print(g.to_string())
s = df[df["single"]]
h = s.groupby(["sp", "tblHaulID"]).size()
for k in (3, 5, 10):
    hk = h[h >= k].groupby(level=0).size()
    print(f"hauls with >= {k} single stomachs:\n", hk.to_string())
print(df["gear"].value_counts().head(10).to_string())
print(df["area"].value_counts().head(10).to_string())
