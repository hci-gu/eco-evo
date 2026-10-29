"""Juvenile haddock length at age, month and depth from DATRAS NS-IBTS.

Downloads (cached) HH, CA and HL exchange data for NS-IBTS Q1 and Q3 from
the public DATRAS web service, keeps haddock (AphiaID 126437) only, and
writes small CSVs plus a summary:

  out/haddock_ca.csv   one row per CA record (age-length) with haul date,
                       position, depth and bottom temperature
  out/haddock_hl.csv   Q3 length frequencies (numbers per hour) with haul
                       date, position, depth
  out/hauls.csv        all valid hauls (for zero catches in depth bands)
  out/summary.txt      length at age by quarter, month and region; Q3
                       0-group CPUE by depth band

Usage: python3 -I datras_haddock_juveniles.py RAW_DIR OUT_DIR [Y0 Y1]
RAW_DIR holds the downloaded XML (untrusted data, kept outside the repo);
DATRAS_DELETE_RAW=1 removes each year's XML after parsing (disk space).
"""
import csv
import os
import statistics as st
import sys
import urllib.request
import xml.etree.ElementTree as ET

HADDOCK = "126437"
BASE = "https://datras.ices.dk/WebServices/DATRASWebService.asmx/get{rt}data?survey=NS-IBTS&year={y}&quarter={q}"


def fetch(raw, rt, y, q):
    path = os.path.join(raw, f"{rt}_{y}_{q}.xml")
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        url = BASE.format(rt=rt, y=y, q=q)
        with urllib.request.urlopen(url, timeout=600) as r, open(path, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
    return path


def records(path, tag):
    ns = "{ices.dk.local/DATRAS}"
    for ev, el in ET.iterparse(path, events=("end",)):
        if el.tag == ns + tag:
            yield {c.tag[len(ns):]: (c.text or "").strip() for c in el}
            el.clear()


def hkey(r):
    return (r["Country"], r["Ship"], r["Gear"], r["StNo"], r["HaulNo"], r["Year"], r["Quarter"])


def length_mm(code, cls):
    v = float(cls)
    if code == "." or code == "0":  # mm (0 = 0.5 cm classes reported in mm)
        return v
    return v * 10.0  # "1" = cm, "5" = 5 cm classes reported in cm


def region(lat, lon):
    """Coarse regions: NE = Norwegian Trench edge / Skagerrak (lat>=57, lon>=3 E);
    NW = northern North Sea west of 3 E (lat>=57); S = rest."""
    if lat >= 57.0 and lon >= 3.0:
        return "NE_trench_skagerrak"
    if lat >= 57.0:
        return "N_west"
    return "south"


def main():
    raw, out = sys.argv[1], sys.argv[2]
    y0, y1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (2010, 2023)
    os.makedirs(out, exist_ok=True)
    hauls = {}
    ca_rows, hl_rows = [], []
    for y in range(y0, y1 + 1):
        for q in (1, 3):
            for r in records(fetch(raw, "HH", y, q), "Cls_DatrasExchange_HH"):
                if r.get("HaulVal") != "V":
                    continue
                try:
                    hauls[hkey(r)] = dict(
                        year=y, quarter=q, month=int(r["Month"]), day=int(r["Day"]),
                        lat=float(r["ShootLat"]), lon=float(r["ShootLong"]),
                        depth=float(r["Depth"]), bottemp=float(r["BotTemp"]),
                        dur=float(r["HaulDur"]), daynight=r["DayNight"],
                        datatype=r["DataType"], statrec=r["StatRec"])
                except (ValueError, KeyError):
                    continue
            for r in records(fetch(raw, "CA", y, q), "Cls_DatrasExchange_CA"):
                if HADDOCK not in (r.get("Valid_Aphia"), r.get("SpecCode")):
                    continue
                h = hauls.get(hkey(r))
                if h is None or r["Age"] in ("-9", ""):
                    continue
                ca_rows.append(dict(h, age=int(r["Age"]),
                                    length_mm=length_mm(r["LngtCode"], r["LngtClass"]),
                                    n=int(float(r["CANoAtLngt"]))))
            if q == 3:
                for r in records(fetch(raw, "HL", y, q), "Cls_DatrasExchange_HL"):
                    if HADDOCK not in (r.get("Valid_Aphia"), r.get("SpecCode")):
                        continue
                    h = hauls.get(hkey(r))
                    if h is None or r["LngtClass"] in ("-9", ""):
                        continue
                    n = float(r["HLNoAtLngt"])
                    if h["datatype"] == "R":
                        sub = float(r["SubFactor"]) if r["SubFactor"] not in ("-9", "") else 1.0
                        n = n * sub * 60.0 / h["dur"]
                    hl_rows.append(dict(h, length_mm=length_mm(r["LngtCode"], r["LngtClass"]),
                                        n_per_h=n))
            print(y, q, len(hauls), len(ca_rows), len(hl_rows), flush=True)
            if os.environ.get("DATRAS_DELETE_RAW") == "1":
                for rt in ("HH", "CA", "HL"):
                    fp = os.path.join(raw, f"{rt}_{y}_{q}.xml")
                    if os.path.exists(fp):
                        os.remove(fp)

    fields = ["year", "quarter", "month", "day", "lat", "lon", "depth", "bottemp", "dur",
              "daynight", "datatype", "statrec"]
    with open(os.path.join(out, "haddock_ca.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fields + ["age", "length_mm", "n"])
        w.writeheader()
        w.writerows(ca_rows)
    with open(os.path.join(out, "haddock_hl.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fields + ["length_mm", "n_per_h"])
        w.writeheader()
        w.writerows(hl_rows)
    with open(os.path.join(out, "hauls.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(hauls.values())

    lines = [f"DATRAS NS-IBTS {y0}-{y1}, haddock (AphiaID {HADDOCK}); CA = aged fish, HL = length freq.",
             "Regions: NE_trench_skagerrak lat>=57 & lon>=3E; N_west lat>=57 & lon<3E; south lat<57.",
             "", "Length at age (mm), CA records weighted by CANoAtLngt:",
             "quarter age region month  n   mean  p10  p50  p90  share>=190mm"]
    groups = {}
    for r in ca_rows:
        if r["age"] > 2:
            continue
        for reg in (region(r["lat"], r["lon"]), "all"):
            groups.setdefault((r["quarter"], r["age"], reg, r["month"]), []).extend([r["length_mm"]] * r["n"])
    for k in sorted(groups):
        v = sorted(groups[k])
        if len(v) < 20:
            continue
        p = lambda f: v[min(len(v) - 1, int(f * len(v)))]
        lines.append(f"Q{k[0]} age{k[1]} {k[2]:20s} m{k[3]:<2d} {len(v):6d} {st.mean(v):6.1f} {p(.1):5.0f}"
                     f" {p(.5):5.0f} {p(.9):5.0f} {sum(x >= 190 for x in v) / len(v):6.3f}")

    # Q3 0-group (length < 150 mm, the Q3 0-group mode; see CA table for the cut) CPUE by depth band
    lines += ["", "Q3 haddock < 150 mm (0-group) numbers per hour by depth band and region",
              "(mean over all valid Q3 hauls in band, zeros included):",
              "region depth_band n_hauls mean_n_per_h share_of_hauls_with_0group"]
    bands = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 700)]
    catch = {}
    for r in hl_rows:
        if r["length_mm"] < 150:
            k = (r["year"], r["lat"], r["lon"], r["month"], r["day"], r["depth"])
            catch[k] = catch.get(k, 0.0) + r["n_per_h"]
    for reg in ("NE_trench_skagerrak", "N_west", "south"):
        for lo, hi in bands:
            hs = [h for h in hauls.values() if h["quarter"] == 3 and region(h["lat"], h["lon"]) == reg
                  and lo <= h["depth"] < hi]
            if not hs:
                continue
            vals = [catch.get((h["year"], h["lat"], h["lon"], h["month"], h["day"], h["depth"]), 0.0)
                    for h in hs]
            lines.append(f"{reg:20s} {lo:3d}-{hi:<3d} {len(hs):5d} {st.mean(vals):9.1f} "
                         f"{sum(v > 0 for v in vals) / len(vals):5.2f}")
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
