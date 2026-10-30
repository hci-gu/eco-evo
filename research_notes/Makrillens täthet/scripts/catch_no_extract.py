"""Extract mackerel and horse mackerel rows from Norwegian landing-note open data.

Input: one or more Fiskeridirektoratet open-data zips (fangstdata_<year>.csv.zip,
https://register.fiskeridir.no/uttrekk/), each holding one ';'-separated UTF-8 CSV
with decimal commas. The data are treated as untrusted text: only parsed, never run.

Output (argument --out): a CSV aggregated over landing-note lines with the key
  year, species_fao, area_group (Omraadegruppering code), area_group_name,
  main_area (Hovedomraade code), main_area_name, location (Lokasjon code),
  catch_month (month of 'Siste fangstdato'), landing_month,
  gear_main_group, gear_group, gear, vessel_nationality, length_group
and the summed round weight (kg) and the number of note lines.

Usage: python3 -I catch_no_extract.py --out OUT.csv ZIP [ZIP ...]
"""
import csv
import io
import sys
import zipfile
from collections import defaultdict

SPECIES = {"MAC", "HOM", "JAX"}


def num(s):
    s = (s or "").strip().replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def month_of(date_str):
    # dd.mm.yyyy
    parts = (date_str or "").strip().split(".")
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return 0


def main(argv):
    if "--out" not in argv:
        sys.exit(__doc__)
    i = argv.index("--out")
    out = argv[i + 1]
    zips = argv[:i] + argv[i + 2:]
    csv.field_size_limit(10_000_000)
    agg = defaultdict(lambda: [0.0, 0])
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(name) as fh:
                rd = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"),
                                delimiter=";", quotechar='"')
                hdr = next(rd)
                ix = {h: k for k, h in enumerate(hdr)}

                def col(row, h):
                    k = ix[h]
                    return row[k].strip() if k < len(row) else ""

                c_sp = ix["Art FAO (kode)"]
                n_rows = 0
                for row in rd:
                    if len(row) <= c_sp or row[c_sp].strip() not in SPECIES:
                        continue
                    n_rows += 1
                    key = (
                        col(row, "Fangstår"),
                        row[c_sp].strip(),
                        col(row, "Områdegruppering (kode)"),
                        col(row, "Områdegruppering"),
                        col(row, "Hovedområde (kode)"),
                        col(row, "Hovedområde"),
                        col(row, "Lokasjon (kode)"),
                        month_of(col(row, "Siste fangstdato")),
                        int(col(row, "Landingsmåned (kode)") or 0),
                        col(row, "Redskap - hovedgruppe"),
                        col(row, "Redskap - gruppe"),
                        col(row, "Redskap"),
                        col(row, "Fartøynasjonalitet (kode)"),
                        col(row, "Lengdegruppe").strip(),
                        col(row, "Dokumenttype"),
                    )
                    a = agg[key]
                    a[0] += num(col(row, "Rundvekt"))
                    a[1] += 1
                print(f"{zp}: {n_rows} MAC/HOM/JAX lines", file=sys.stderr)
    cols = ["year", "species_fao", "area_group", "area_group_name", "main_area",
            "main_area_name", "location", "catch_month", "landing_month",
            "gear_main_group", "gear_group", "gear", "vessel_nationality",
            "length_group", "document_type", "round_weight_kg", "n_lines"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in sorted(agg, key=lambda t: tuple(str(x) for x in t)):
            w.writerow(list(k) + [round(agg[k][0], 1), agg[k][1]])


if __name__ == "__main__":
    main(sys.argv[1:])
