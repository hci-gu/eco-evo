"""Extract NS-IBTS CPUE-at-age per haul (DATRAS getCPUEAge) into one compact CSV.

Usage: python3 -I mig_extract_cpue_age.py <cpue_age_dir> <out_csv_gz>

Input files: CPUEA_<year>_Q<q>.xml.gz, downloaded from
https://datras.ices.dk/WebServices/DATRASWebService.asmx/getCPUEAge?survey=NS-IBTS&year=<y>&quarter=<q>
(untrusted data: parsed with ElementTree only, never executed). DATRAS computes these per haul
from the HL length frequencies and the area ALK; they cover nine species only (herring, sprat,
cod, haddock, whiting, saithe, Norway pout, mackerel, plaice).
"""
import csv
import glob
import gzip
import os
import sys
import xml.etree.ElementTree as ET

src, out = sys.argv[1:3]
NS = "{ices.dk.local/DATRAS}"
FIELDS = ["Year", "Quarter", "Ship", "Gear", "HaulNo", "ShootLat", "ShootLon", "DateTime", "Depth",
          "Area", "SubArea", "DayNight", "AphiaID", "Species"]
AGES = [f"Age_{a}" for a in range(16)]
n = 0
with gzip.open(out, "wt", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(FIELDS + AGES)
    for f in sorted(glob.glob(os.path.join(src, "CPUEA_*_Q*.xml.gz"))):
        with gzip.open(f) as fi:
            for _, el in ET.iterparse(fi):
                if el.tag != NS + "Cls_Datras_CPUE_Age":
                    continue
                row = []
                for k in FIELDS + AGES:
                    c = el.find(NS + k)
                    v = "" if c is None or c.text is None else c.text.strip()
                    row.append(v)
                w.writerow(row)
                n += 1
                el.clear()
print(f"{n} rows written to {out}")
