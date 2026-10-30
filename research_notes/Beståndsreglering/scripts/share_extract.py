"""Stream-parse DATRAS NS-IBTS CPUE-per-length XML (untrusted; parsed only) for the member species,
whole survey area, Q1 and Q3 (WP3, box_share.md).

Usage: python3 -I share_extract.py <out_csv_gz> <datras_dir> [<datras_dir> ...]
Reads CPUEL_<year>_Q<q>.xml.gz in each directory; keeps rows with CPUE > 0 for the AphiaIDs below.
"""
import csv
import gzip
import os
import sys
import xml.etree.ElementTree as ET

NS = "{ices.dk.local/DATRAS}"
APHIA = {126417, 126425, 126444, 126439, 126437, 126436, 127139, 127137, 127143, 127136,
         126438, 126441, 126484, 127023, 126822, 105923}
COLS = ["Year", "Quarter", "Ship", "Gear", "HaulNo", "HaulDur", "ShootLat", "ShootLon", "Depth", "SubArea",
        "AphiaID", "LngtClas", "CPUE_number_per_hour"]
out = sys.argv[1]
n = 0
with gzip.open(out, "wt", newline="") as f:
    w = csv.writer(f)
    w.writerow(COLS)
    for d in sys.argv[2:]:
        for name in sorted(os.listdir(d)):
            if not (name.startswith("CPUEL_") and name.endswith(".xml.gz")):
                continue
            fh = gzip.open(os.path.join(d, name), "rb")
            for ev, el in ET.iterparse(fh, events=("end",)):
                if el.tag == NS + "Cls_Datras_CPUE_Length":
                    r = {c.tag[len(NS):]: (c.text or "").strip() for c in el}
                    el.clear()
                    try:
                        if int(r["AphiaID"]) in APHIA and float(r["CPUE_number_per_hour"]) > 0:
                            w.writerow([r.get(c, "") for c in COLS])
                            n += 1
                    except (KeyError, ValueError):
                        pass
            print(name, n, flush=True)
