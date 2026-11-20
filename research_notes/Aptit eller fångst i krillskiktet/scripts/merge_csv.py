"""Merge the per-trail CSVs into tables/evidence.csv (one row per sourced number) and check the columns.

Usage: python3 -I merge_csv.py <tables_dir>
Inputs: main_rows.csv (main agent), appetite_raw.csv, inlayer_raw.csv, seasonal_raw.csv (subagent trails).
Output columns: trail + source,doi,species,region,season,method,quantity,value,unit,tag
"""
import csv, os, sys

COLS = ["source", "doi", "species", "region", "season", "method", "quantity", "value", "unit", "tag"]
d = sys.argv[1]
out = []
for trail, name in (("main", "main_rows.csv"), ("appetite", "appetite_raw.csv"),
                    ("inlayer", "inlayer_raw.csv"), ("seasonal", "seasonal_raw.csv")):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        print("missing", name)
        continue
    with open(p, newline="") as fh:
        r = csv.DictReader(fh)
        if r.fieldnames != COLS:
            print(name, "header differs:", r.fieldnames)
        n = bad = 0
        for row in r:
            n += 1
            if None in row or any(row.get(c) is None for c in COLS):
                bad += 1
                continue
            out.append([trail] + [row[c].strip() for c in COLS])
        print(f"{name}: {n} rows, {bad} malformed")
tags = {}
for row in out:
    tags[row[-1]] = tags.get(row[-1], 0) + 1
print("tags:", dict(sorted(tags.items(), key=lambda x: -x[1])))
with open(os.path.join(d, "evidence.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["trail"] + COLS)
    w.writerows(out)
print("evidence.csv rows:", len(out))
