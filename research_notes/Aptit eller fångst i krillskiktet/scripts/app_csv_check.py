"""Clean and check tables/appetite_raw.csv in place: re-tag WebFetch rows, split range values, validate.

Usage: python3 -I app_csv_check.py <appetite_raw.csv>
"""
import csv, re, sys

COLS = ["source", "doi", "species", "region", "season", "method", "quantity", "value", "unit", "tag"]
TAGS = {"F", "A", "S", "D", "J"}

def main():
    p = sys.argv[1]
    with open(p, newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        assert head == COLS, head
        raw = list(rd)
    out, bad = [], 0
    for i, r in enumerate(raw, start=2):
        if len(r) != 10:
            print("line", i, "has", len(r), "fields:", r); bad += 1; continue
        d = dict(zip(COLS, r))
        if d["tag"] == "F-w":
            d["tag"] = "S"
            d["method"] = d["method"] + "; full text via WebFetch tool extraction"
        if d["season"].startswith("season pooled (mostly summer-autumn)"):
            d["season"] = "season pooled (not stated)"
        m = re.fullmatch(r"([0-9.]+)-([0-9.]+)", d["value"])
        if m:
            for lab, v in (("min", m.group(1)), ("max", m.group(2))):
                e = dict(d); e["quantity"] = d["quantity"] + f" ({lab} of range)"; e["value"] = v
                out.append(e)
            continue
        try:
            float(d["value"])
        except ValueError:
            print("line", i, "non-numeric value:", d["value"]); bad += 1
        if d["tag"] not in TAGS:
            print("line", i, "bad tag:", d["tag"]); bad += 1
        out.append(d)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)
    print(len(out), "rows written;", bad, "problems")
    from collections import Counter
    print("tags:", dict(Counter(r["tag"] for r in out)))
    print("sources:", dict(Counter(r["source"] for r in out)))

if __name__ == "__main__":
    main()
