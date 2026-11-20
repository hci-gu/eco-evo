"""Append rows to tables/q1_raw.csv. Input on stdin: one row per line, 11 fields separated by ' | '.
Fields: source | doi | region | proxy | method | support | quantity | value | unit | tag | note
Usage: python3 -I q1_addrows.py <csv path> < rows.txt"""
import csv, os, sys
path = sys.argv[1]
cols = ["source", "doi", "region", "proxy", "method", "support", "quantity", "value", "unit", "tag", "note"]
new = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(cols)
    n = 0
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) != len(cols):
            sys.exit(f"bad field count {len(parts)}: {line[:80]}")
        w.writerow(parts); n += 1
print("appended", n)
