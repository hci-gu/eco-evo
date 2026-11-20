"""Cross-tab of content flags for single-stomach records of selected species outside the 'other' area.
Usage: python3 -I q2_records_flags.py <records_linked.csv.gz>"""
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1], low_memory=False)
s = df[(df["Number"] == 1) & (df["area"] != "other")]
for sp in ["whiting", "cod", "haddock", "hake", "mackerel", "saithe", "horse mackerel", "plaice"]:
    d = s[s["sp"] == sp]
    if d.empty:
        continue
    print("==", sp, len(d), "years", sorted(d["year"].unique())[:3], "...", sorted(d["year"].unique())[-3:])
    print(pd.crosstab([d["StomachEmpty"], d["anyprey"]], d["fed"].fillna(-1)).to_string())
    print("occ_krill>0:", int((d["occ_krill"] > 0).sum()), " areas:", d["area"].value_counts().to_dict())
    print("quarters:", d["quarter"].value_counts().to_dict(), " gear:", d["gear"].value_counts().head(4).to_dict())
