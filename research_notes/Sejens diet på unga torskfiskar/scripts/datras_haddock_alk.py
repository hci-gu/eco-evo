"""Population (ALK-weighted) length of juvenile haddock by quarter, month and region.

The CA (age) records of NS-IBTS are length-stratified, so their raw mean is not
the population mean. This script applies an age-length key (year x quarter x
region, 1 cm classes, from out/haddock_ca.csv of datras_haddock_juveniles.py)
to the length frequencies (numbers per hour): Q3 from out/haddock_hl.csv, Q1
downloaded here (HL, haddock only kept). Output: mean length, percentiles and
the share above 190 mm for age 0 (Q3) and age 1 (Q1, Q3), by month and region.

Usage: python3 -I datras_haddock_alk.py RAW_DIR OUT_DIR Y0 Y1
"""
import csv
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

HAD = "126437"
NS = "{ices.dk.local/DATRAS}"
BASE = "https://datras.ices.dk/WebServices/DATRASWebService.asmx/get{rt}data?survey=NS-IBTS&year={y}&quarter={q}"


def region(lat, lon):
    if lat >= 57.0 and lon >= 3.0:
        return "NE_trench_skagerrak"
    if lat >= 57.0:
        return "N_west"
    return "south"


def fetch(raw, rt, y, q):
    path = os.path.join(raw, f"{rt}_{y}_{q}.xml")
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        with urllib.request.urlopen(BASE.format(rt=rt, y=y, q=q), timeout=600) as r, open(path, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
    return path


def records(path, tag):
    for ev, el in ET.iterparse(path, events=("end",)):
        if el.tag == NS + tag:
            yield {c.tag[len(NS):]: (c.text or "").strip() for c in el}
            el.clear()


def hkey(r):
    return (r["Country"], r["Ship"], r["Gear"], r["StNo"], r["HaulNo"], r["Year"], r["Quarter"])


def q1_hl(raw, y0, y1):
    rows = []
    for y in range(y0, y1 + 1):
        hh = {}
        for r in records(fetch(raw, "HH", y, 1), "Cls_DatrasExchange_HH"):
            if r.get("HaulVal") != "V":
                continue
            try:
                hh[hkey(r)] = dict(year=y, quarter=1, month=int(r["Month"]), lat=float(r["ShootLat"]),
                                   lon=float(r["ShootLong"]), dur=float(r["HaulDur"]), datatype=r["DataType"])
            except (ValueError, KeyError):
                continue
        for r in records(fetch(raw, "HL", y, 1), "Cls_DatrasExchange_HL"):
            if HAD not in (r.get("Valid_Aphia"), r.get("SpecCode")):
                continue
            h = hh.get(hkey(r))
            if h is None or r["LngtClass"] in ("-9", ""):
                continue
            n = float(r["HLNoAtLngt"])
            if h["datatype"] == "R":
                sub = float(r["SubFactor"]) if r["SubFactor"] not in ("-9", "") else 1.0
                n = n * sub * 60.0 / h["dur"]
            v = float(r["LngtClass"])
            L = v if r["LngtCode"] in (".", "0") else v * 10.0
            rows.append(dict(h, length_mm=L, n_per_h=n))
        for rt in ("HH", "HL"):
            fp = os.path.join(raw, f"{rt}_{y}_1.xml")
            if os.environ.get("DATRAS_DELETE_RAW") == "1" and os.path.exists(fp):
                os.remove(fp)
        print("Q1 HL", y, len(rows), flush=True)
    return rows


def main():
    raw, out, y0, y1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    # age-length key: (year, quarter, region, cm) -> {age: n}
    alk = {}
    with open(os.path.join(out, "haddock_ca.csv")) as f:
        for r in csv.DictReader(f):
            y, q = int(r["year"]), int(r["quarter"])
            if not (y0 <= y <= y1):
                continue
            k = (y, q, region(float(r["lat"]), float(r["lon"])), int(float(r["length_mm"]) // 10))
            a = min(int(r["age"]), 3)
            alk.setdefault(k, {}).setdefault(a, 0)
            alk[k][a] += int(r["n"])
    hl = []
    with open(os.path.join(out, "haddock_hl.csv")) as f:
        for r in csv.DictReader(f):
            if y0 <= int(r["year"]) <= y1:
                hl.append(dict(year=int(r["year"]), quarter=3, month=int(r["month"]), lat=float(r["lat"]),
                               lon=float(r["lon"]), length_mm=float(r["length_mm"]),
                               n_per_h=float(r["n_per_h"])))
    hl += q1_hl(raw, y0, y1)

    acc = {}  # (quarter, age, region, month) -> {length_mm: n}
    unkeyed = 0.0
    for r in hl:
        reg = region(r["lat"], r["lon"])
        cm = int(r["length_mm"] // 10)
        key = alk.get((r["year"], r["quarter"], reg, cm))
        if key is None:
            if r["length_mm"] < 100 and r["quarter"] == 3:
                key = {0: 1}  # below every Q3 age-1 length: 0-group
            elif r["length_mm"] < 100 and r["quarter"] == 1:
                key = {1: 1}
            else:
                unkeyed += r["n_per_h"]
                continue
        tot = sum(key.values())
        for a, n in key.items():
            if a > 1:
                continue
            p = n / tot
            for reg2 in (reg, "all"):
                d = acc.setdefault((r["quarter"], a, reg2, r["month"]), {})
                d[r["length_mm"]] = d.get(r["length_mm"], 0.0) + p * r["n_per_h"]
                d2 = acc.setdefault((r["quarter"], a, reg2, 0), {})
                d2[r["length_mm"]] = d2.get(r["length_mm"], 0.0) + p * r["n_per_h"]

    L = [f"NS-IBTS {y0}-{y1}, haddock, ALK-weighted (year x quarter x region, 1 cm) population lengths.",
         "month 0 = all months of the quarter pooled. Weights = numbers per hour summed over hauls.",
         f"(length classes without an ALK entry and >= 100 mm dropped: {unkeyed:.0f} n/h in total)",
         "quarter age region               month   sum_n/h   mean  p10  p25  p50  p75  p90  share>=190mm"]
    for k in sorted(acc):
        d = acc[k]
        tot = sum(d.values())
        if tot < 50:
            continue
        xs = sorted(d)

        def pct(f):
            c = 0.0
            for x in xs:
                c += d[x]
                if c >= f * tot:
                    return x
            return xs[-1]
        mean = sum(x * d[x] for x in xs) / tot
        over = sum(d[x] for x in xs if x >= 190) / tot
        L.append(f"Q{k[0]}  age{k[1]} {k[2]:20s} m{k[3]:<2d} {tot:10.0f} {mean:6.1f} {pct(.1):4.0f} {pct(.25):4.0f}"
                 f" {pct(.5):4.0f} {pct(.75):4.0f} {pct(.9):4.0f}  {over:6.3f}")
    with open(os.path.join(out, "haddock_length_alk_summary.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
