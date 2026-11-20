"""Merge tables/main_rows.csv, q1_raw.csv and q2_raw.csv into tables/evidence.csv with one schema:
question, source, doi, region, species, proxy, method, support, quantity, value, unit, tag, note.
Usage: python3 -I merge_rows.py <tables dir>"""
import csv, os, sys
d = sys.argv[1]
cols = ["question", "source", "doi", "region", "species", "proxy", "method", "support", "quantity", "value", "unit", "tag", "note"]
out = []
for q, fn in (("Q3/stomach", "main_rows.csv"), ("Q1", "q1_raw.csv"), ("Q2", "q2_raw.csv")):
    for r in csv.DictReader(open(os.path.join(d, fn), newline="")):
        row = {c: (r.get(c) or "").strip() for c in cols}
        row["question"] = q
        if not row["species"]:
            row["species"] = "krill" if q == "Q1" else ""
        out.append(row)
with open(os.path.join(d, "evidence.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(out)
print(f"{len(out)} rows written")
