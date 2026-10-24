"""Stream-parse DATRAS XML downloads (untrusted data; parsed only, never executed)
into compact gzipped CSVs.

Usage: python3 -I extract_datras.py <datras_dir> <out_dir>   (reads *.xml or *.xml.gz)
- HH: all hauls, selected columns.
- CPUEL: rows in the wider Skagerrak region (56.5-59.5 N, 6.5-12.5 E), CPUE > 0 only
  (zeros are implied by the HH haul list).
- CA: all rows with IndWgt > 0 (for length-weight fits), selected columns.
"""
import csv, gzip, os, sys
import xml.etree.ElementTree as ET

NS = "{ices.dk.local/DATRAS}"
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

HH_COLS = ["Survey", "Quarter", "Country", "Ship", "Gear", "StNo", "HaulNo", "Year", "Month", "Day",
           "TimeShot", "HaulDur", "DayNight", "ShootLat", "ShootLong", "HaulLat", "HaulLong", "StatRec",
           "Depth", "HaulVal", "StdSpecRecCode", "BySpecRecCode", "DataType", "Distance", "GroundSpeed",
           "WingSpread", "DoorSpread", "Netopening"]
CP_COLS = ["Year", "Quarter", "Ship", "Gear", "HaulNo", "HaulDur", "ShootLat", "ShootLon", "Depth", "SubArea",
           "AphiaID", "Species", "Sex", "LngtClas", "CPUE_number_per_hour"]
CA_COLS = ["Quarter", "Country", "Ship", "HaulNo", "Year", "SpecCode", "Valid_Aphia", "AreaCode", "LngtCode",
           "LngtClass", "Sex", "Age", "CANoAtLngt", "IndWgt"]


def rows(path, tag):
    fh = gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")
    for ev, el in ET.iterparse(fh, events=("end",)):
        if el.tag == NS + tag:
            yield {c.tag[len(NS):]: (c.text or "").strip() for c in el}
            el.clear()


def run(kind, tag, cols, keep):
    fn = os.path.join(out, f"{kind}.csv.gz")
    with gzip.open(fn, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        n = 0
        for name in sorted(os.listdir(src)):
            if not name.startswith(kind + "_"):
                continue
            for r in rows(os.path.join(src, name), tag):
                if keep(r):
                    w.writerow([r.get(c, "") for c in cols])
                    n += 1
    print(kind, n)


def f(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


run("HH", "Cls_DatrasExchange_HH", HH_COLS, lambda r: True)
run("CPUEL", "Cls_Datras_CPUE_Length", CP_COLS,
    lambda r: 56.5 <= f(r["ShootLat"]) <= 59.5 and 6.5 <= f(r["ShootLon"]) <= 12.5
    and f(r["CPUE_number_per_hour"]) > 0)
run("CA", "Cls_DatrasExchange_CA", CA_COLS, lambda r: f(r["IndWgt"]) > 0)
