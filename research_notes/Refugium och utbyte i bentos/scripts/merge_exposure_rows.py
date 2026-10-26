"""Merge tables/part_[A-D]_rows.csv into tables/epifauna_exposure_rates.csv.

Checks the header and the field count of every row; keeps the eight columns
taxon,quantity,low,central,high,unit,source,tag; sorts by question block (A..D order kept).
Usage: python3 -I merge_exposure_rows.py <tables_dir>
"""
import csv
import sys
from pathlib import Path

HEADER = ["taxon", "quantity", "low", "central", "high", "unit", "source", "tag"]


def main(tables: Path) -> None:
    rows = []
    for part in "ABCD":
        path = tables / f"part_{part}_rows.csv"
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            if header != HEADER:
                raise SystemExit(f"{path.name}: bad header {header}")
            for i, row in enumerate(reader, start=2):
                if not row:
                    continue
                if len(row) != len(HEADER):
                    raise SystemExit(f"{path.name} line {i}: {len(row)} fields")
                if not any(row[2:5]):
                    raise SystemExit(f"{path.name} line {i}: no value")
                rows.append(row)
    extra = tables / "synthesis_rows.csv"
    if extra.exists():
        with extra.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            if next(reader) != HEADER:
                raise SystemExit("synthesis_rows.csv: bad header")
            rows.extend(r for r in reader if r)
    out = tables / "epifauna_exposure_rates.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    tags = {}
    for r in rows:
        tags[r[7]] = tags.get(r[7], 0) + 1
    print(f"{len(rows)} rows -> {out.name}; tags {tags}")
    for r in rows:
        print(f"{r[0][:28]:28s} {r[1][:52]:52s} {r[2]:>7s} {r[3]:>7s} {r[4]:>7s} {r[5][:18]:18s} {r[7]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
