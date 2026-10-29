"""Fetch NS-IBTS Q2 and Q4 (1991-1997) HH, HL and CA from DATRAS; keep the Mareld box.

Usage: python3 -I datras_q2q4_fetch.py RAW_DIR OUT_DIR [Y0 Y1 [QUARTERS]]   (QUARTERS e.g. 13; default 24)

The quarterly survey ran in Q2 and Q4 only in 1991-1997. RAW_DIR holds the downloaded XML
(untrusted data, outside the repo); each file is deleted after parsing. OUT_DIR gets three CSVs
(hh.csv, hl.csv, ca.csv) with the hauls inside the box (the polygon of
../../Sejens diet pa unga torskfiskar/scripts/box_gape_by_quarter.py plus rectangles 44G0, 44G1,
45G0, 45G1) and, for HL and CA, only the species below. The CA rows are kept for the wider
Skagerrak (lat 57-59.5, lon 7-12) so the age-length keys have enough fish.
"""
import csv
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

BASE = "https://datras.ices.dk/WebServices/DATRASWebService.asmx/get{rt}data?survey=NS-IBTS&year={y}&quarter={q}"
NS = "{ices.dk.local/DATRAS}"
SPECIES = {"126417", "126425", "126444", "126439", "126437", "126436", "126438", "127143", "127139",
           "127137", "127140", "127136", "150637", "105865"}
raw, out = sys.argv[1:3]
y0, y1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1991, 1997)
QUARTERS = [int(c) for c in sys.argv[5]] if len(sys.argv) > 5 else [2, 4]
os.makedirs(raw, exist_ok=True)
os.makedirs(out, exist_ok=True)


def in_box(lat, lon, rect):
    if rect in ("44G0", "44G1", "45G0", "45G1"):
        return True
    x1, y1_, x2, y2 = 8.60, 57.12, 7.05, 57.98
    east = ((x2 - x1) * (lat - y1_) - (y2 - y1_) * (lon - x1)) < 0
    return east and lat < 59.2 and not (lon > 10.6 and lat < 57.75) and lon < 12.5 and lat > 57.0


def records(path):
    for _, el in ET.iterparse(path):
        if el.tag in (NS + "Cls_DatrasExchange_HH", NS + "Cls_DatrasExchange_HL",
                      NS + "Cls_DatrasExchange_CA"):
            yield {c.tag.replace(NS, ""): (c.text or "").strip() for c in el}
            el.clear()


def fetch(rt, y, q):
    path = os.path.join(raw, f"{rt}_{y}_{q}.xml")
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE.format(rt=rt, y=y, q=q), path)
    return path


HH_COLS = ["Year", "Quarter", "Month", "Country", "Ship", "Gear", "HaulNo", "StNo", "HaulDur", "DayNight",
           "ShootLat", "ShootLong", "StatRec", "Depth", "HaulVal", "DataType", "BySpecRecCode"]
HL_COLS = ["Year", "Quarter", "Country", "Ship", "Gear", "HaulNo", "StNo", "Valid_Aphia", "CatIdentifier",
           "SubFactor", "LngtCode", "LngtClass", "HLNoAtLngt", "Sex"]
CA_COLS = ["Year", "Quarter", "Country", "Ship", "Gear", "HaulNo", "StNo", "AreaCode", "Valid_Aphia",
           "LngtCode", "LngtClass", "Age", "CANoAtLngt", "IndWgt"]
files = {k: open(os.path.join(out, f"{k}.csv"), "w", newline="") for k in ("hh", "hl", "ca")}
w = {k: csv.writer(files[k]) for k in files}
w["hh"].writerow(HH_COLS)
w["hl"].writerow(HL_COLS)
w["ca"].writerow(CA_COLS)
for y in range(y0, y1 + 1):
    for q in QUARTERS:
        try:
            hh_path = fetch("HH", y, q)
        except Exception as e:  # noqa: BLE001 - a missing year-quarter is reported, not fatal
            print(f"{y} Q{q}: HH not available ({e})")
            continue
        keys, skag = set(), set()
        n = 0
        for r in records(hh_path):
            try:
                lat, lon = float(r["ShootLat"]), float(r["ShootLong"])
            except (KeyError, ValueError):
                continue
            k = (r["Country"], r["Ship"], r["StNo"], r["HaulNo"])
            if 57.0 <= lat <= 59.5 and 7.0 <= lon <= 12.0:
                skag.add(k)
            if in_box(lat, lon, r.get("StatRec", "")):
                keys.add(k)
                w["hh"].writerow([r.get(c, "") for c in HH_COLS])
                n += 1
        os.remove(hh_path)
        if not skag:
            print(f"{y} Q{q}: no Skagerrak hauls")
            continue
        nl = nc = 0
        for rt, key in (("HL", "hl"), ("CA", "ca")):
            try:
                p = fetch(rt, y, q)
            except Exception as e:  # noqa: BLE001
                print(f"{y} Q{q}: {rt} not available ({e})")
                continue
            for r in records(p):
                k = (r["Country"], r["Ship"], r["StNo"], r["HaulNo"])
                if r.get("Valid_Aphia") not in SPECIES:
                    continue
                if key == "hl" and k in keys:
                    w["hl"].writerow([r.get(c, "") for c in HL_COLS])
                    nl += 1
                elif key == "ca" and k in skag:
                    w["ca"].writerow([r.get(c, "") for c in CA_COLS])
                    nc += 1
            os.remove(p)
        print(f"{y} Q{q}: box hauls {n}, Skagerrak hauls {len(skag)}, HL rows {nl}, CA rows {nc}", flush=True)
for f in files.values():
    f.close()
