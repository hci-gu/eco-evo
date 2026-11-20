"""Check whether occ_krill in pooled records is a count of stomachs (0..Number) or binary.
Usage: python3 -I q2_records_pooled.py <records_linked.csv.gz>"""
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1], low_memory=False)
p = df[(df["Number"] > 1)]
for sp in ["whiting", "saithe", "Norway pout", "cod", "horse mackerel", "haddock"]:
    d = p[p["sp"] == sp]
    print("==", sp, len(d), "records,", int(d["Number"].sum()), "stomachs; years", sorted(d["year"].unique()))
    k = d[d["occ_krill"] > 0]
    print(" records with krill:", len(k))
    print(k[["Number", "occ_krill", "krill_count", "g_krill", "fed", "anyprey", "StomachEmpty", "station", "ship", "year", "month", "L_cm"]].head(12).to_string())
    print(" occ_krill > Number:", int((k["occ_krill"] > k["Number"]).sum()), " occ_krill==Number:", int((k["occ_krill"] == k["Number"]).sum()))
    print(" occ_krill value counts:", d["occ_krill"].value_counts().head(8).to_dict())
