"""Fetch NS-IBTS HH, HL and CA from DATRAS and keep the eastern Skagerrak.

Usage: python3 -I datras_fetch.py RAW_DIR OUT_DIR Y0 Y1 QUARTERS   (QUARTERS e.g. 13)

RAW_DIR holds the downloaded XML (untrusted data, outside the repo; each file is deleted after
parsing). OUT_DIR gets hh.csv (hauls with StatRec in REGION), hl.csv (length records of those
hauls) and ca.csv (individual records with sex, maturity, maturity scale, age and weight, whose
AreaCode is in REGION or whose haul is one of the kept hauls - some years code CA by roundfish
area, not rectangle). Appends; rerun on an empty OUT_DIR to rebuild.

REGION = the four box rectangles 44G0, 44G1, 45G0, 45G1 and the Skagerrak / Kattegat rectangles
around them (43F8-43G2, 44F8-44F9, 45F8-45F9, 46F8-46G1, 47F8-47G0).
"""
import csv
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

BASE = "https://datras.ices.dk/WebServices/DATRASWebService.asmx/get{rt}data?survey=NS-IBTS&year={y}&quarter={q}"
NS = "{ices.dk.local/DATRAS}"
BOX = {"44G0", "44G1", "45G0", "45G1"}
REGION = BOX | {"43F8", "43F9", "43G0", "43G1", "43G2", "44F8", "44F9", "45F8", "45F9",
                "46F8", "46F9", "46G0", "46G1", "47F8", "47F9", "47G0"}
HH_COLS = ["Year", "Quarter", "Month", "Day", "Country", "Ship", "Gear", "HaulNo", "StNo", "HaulDur",
           "DayNight", "ShootLat", "ShootLong", "StatRec", "Depth", "HaulVal", "DataType"]
HL_COLS = ["Year", "Quarter", "Country", "Ship", "Gear", "HaulNo", "StNo", "Valid_Aphia", "CatIdentifier",
           "SubFactor", "LngtCode", "LngtClass", "HLNoAtLngt", "Sex", "SpecVal"]
CA_COLS = ["Year", "Quarter", "Country", "Ship", "Gear", "HaulNo", "StNo", "AreaCode", "Valid_Aphia",
           "LngtCode", "LngtClass", "Sex", "Maturity", "MaturityScale", "Age", "CANoAtLngt", "IndWgt"]


def records(path, tag):
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == NS + tag:
            yield {c.tag.replace(NS, ""): (c.text or "").strip() for c in el}
            el.clear()


def key(r):
    return (r["Year"], r["Quarter"], r["Country"], r["Ship"], r["Gear"], r["HaulNo"], r["StNo"])


def main():
    raw, out, y0, y1, qs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    os.makedirs(raw, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    files = {n: os.path.join(out, n + ".csv") for n in ("hh", "hl", "ca")}
    cols = {"hh": HH_COLS, "hl": HL_COLS, "ca": CA_COLS}
    writers = {}
    handles = {}
    for n, p in files.items():
        new = not os.path.exists(p)
        handles[n] = open(p, "a", newline="")
        writers[n] = csv.writer(handles[n])
        if new:
            writers[n].writerow(cols[n])
    for y in range(y0, y1 + 1):
        for q in (int(c) for c in qs):
            counts = {}
            hauls = set()
            for rt, tag in (("HH", "Cls_DatrasExchange_HH"), ("HL", "Cls_DatrasExchange_HL"),
                            ("CA", "Cls_DatrasExchange_CA")):
                path = os.path.join(raw, f"{rt}_{y}_{q}.xml")
                try:
                    urllib.request.urlretrieve(BASE.format(rt=rt, y=y, q=q), path)
                except Exception as e:  # noqa: BLE001 - a missing year-quarter is reported
                    print(f"{y} Q{q} {rt}: download failed {e}")
                    continue
                n = 0
                try:
                    for r in records(path, tag):
                        if rt == "HH":
                            if r.get("StatRec") in REGION:
                                hauls.add(key(r))
                                writers["hh"].writerow([r.get(c, "") for c in HH_COLS])
                                n += 1
                        elif rt == "HL":
                            if key(r) in hauls:
                                writers["hl"].writerow([r.get(c, "") for c in HL_COLS])
                                n += 1
                        else:
                            if r.get("AreaCode") in REGION or key(r) in hauls:
                                writers["ca"].writerow([r.get(c, "") for c in CA_COLS])
                                n += 1
                except ET.ParseError as e:
                    print(f"{y} Q{q} {rt}: parse error {e}")
                os.remove(path)
                counts[rt] = n
            print(f"{y} Q{q} kept {counts}", flush=True)
    for h in handles.values():
        h.close()


main()
