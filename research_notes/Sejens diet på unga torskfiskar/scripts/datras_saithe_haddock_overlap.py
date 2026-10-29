"""Saithe (35-70 cm) vs juvenile haddock overlap in NS-IBTS Q1 and Q3 hauls.

Per valid haul: numbers per hour of saithe 35-70 cm (AphiaID 126441) and of
juvenile haddock (126437): Q1 1-group = haddock < 220 mm, Q3 0-group =
haddock < 150 mm (cuts from the CA length-at-age table, datras_haddock_juveniles.py).
Reports, by quarter and region: mean saithe CPUE by depth band, number-weighted
mean depth of each, the share of saithe caught in hauls that also caught
juvenile haddock, Williamson's overlap index and the juvenile-haddock CPUE
"seen" by the average saithe (sum s*h / sum s).

Usage: python3 -I datras_saithe_haddock_overlap.py RAW_DIR OUT_DIR Y0 Y1
DATRAS_DELETE_RAW=1 removes each year's XML after parsing.
"""
import csv
import os
import statistics as st
import sys
import urllib.request
import xml.etree.ElementTree as ET

HAD, SAI = "126437", "126441"
BASE = "https://datras.ices.dk/WebServices/DATRASWebService.asmx/get{rt}data?survey=NS-IBTS&year={y}&quarter={q}"
NS = "{ices.dk.local/DATRAS}"


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


def length_mm(code, cls):
    v = float(cls)
    return v if code in (".", "0") else v * 10.0


def region(lat, lon):
    if lat >= 57.0 and lon >= 3.0:
        return "NE_trench_skagerrak"
    if lat >= 57.0:
        return "N_west"
    return "south"


def main():
    raw, out, y0, y1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    os.makedirs(out, exist_ok=True)
    hauls = {}
    for y in range(y0, y1 + 1):
        for q in (1, 3):
            hh = {}
            for r in records(fetch(raw, "HH", y, q), "Cls_DatrasExchange_HH"):
                if r.get("HaulVal") != "V" or r.get("Gear") != "GOV":
                    continue
                try:
                    hh[hkey(r)] = dict(year=y, quarter=q, month=int(r["Month"]), lat=float(r["ShootLat"]),
                                       lon=float(r["ShootLong"]), depth=float(r["Depth"]),
                                       dur=float(r["HaulDur"]), datatype=r["DataType"],
                                       daynight=r["DayNight"], sai=0.0, had=0.0)
                except (ValueError, KeyError):
                    continue
            cut = 220.0 if q == 1 else 150.0
            for r in records(fetch(raw, "HL", y, q), "Cls_DatrasExchange_HL"):
                sp = r.get("Valid_Aphia") or r.get("SpecCode")
                if sp not in (HAD, SAI):
                    continue
                h = hh.get(hkey(r))
                if h is None or r["LngtClass"] in ("-9", ""):
                    continue
                n = float(r["HLNoAtLngt"])
                if h["datatype"] == "R":
                    sub = float(r["SubFactor"]) if r["SubFactor"] not in ("-9", "") else 1.0
                    n = n * sub * 60.0 / h["dur"]
                L = length_mm(r["LngtCode"], r["LngtClass"])
                if sp == SAI and 350 <= L < 700:
                    h["sai"] += n
                elif sp == HAD and L < cut:
                    h["had"] += n
            hauls.update(hh)
            print(y, q, len(hh), flush=True)
            if os.environ.get("DATRAS_DELETE_RAW") == "1":
                for rt in ("HH", "HL"):
                    fp = os.path.join(raw, f"{rt}_{y}_{q}.xml")
                    if os.path.exists(fp):
                        os.remove(fp)

    with open(os.path.join(out, "saithe_haddock_hauls.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(next(iter(hauls.values())).keys()))
        w.writeheader()
        w.writerows(hauls.values())

    L = [f"NS-IBTS GOV {y0}-{y1}. saithe = 35-70 cm; juvenile haddock = Q1 < 220 mm (1-group), "
         "Q3 < 150 mm (0-group). n/h per haul.", ""]
    bands = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 800)]
    for reg in ("NE_trench_skagerrak", "N_west", "north_all", "all"):
        L.append(f"== {reg}")
        for q in (1, 3):
            hs = [h for h in hauls.values() if h["quarter"] == q and (
                reg == "all" or (reg == "north_all" and h["lat"] >= 57) or region(h["lat"], h["lon"]) == reg)]
            if not hs:
                continue
            S = sum(h["sai"] for h in hs)
            H = sum(h["had"] for h in hs)
            SH = sum(h["sai"] * h["had"] for h in hs)
            n = len(hs)
            share = sum(h["sai"] for h in hs if h["had"] > 0) / S if S else float("nan")
            wil = n * SH / (S * H) if S and H else float("nan")
            seen = SH / S if S else float("nan")
            dS = sum(h["sai"] * h["depth"] for h in hs) / S if S else float("nan")
            dH = sum(h["had"] * h["depth"] for h in hs) / H if H else float("nan")
            L.append(f"Q{q}: hauls {n}, mean saithe n/h {S / n:.2f}, mean juv. haddock n/h {H / n:.1f}, "
                     f"saithe-weighted depth {dS:.0f} m, haddock-weighted depth {dH:.0f} m")
            L.append(f"    share of saithe in hauls with juv. haddock {share:.2f}; Williamson overlap {wil:.2f}; "
                     f"juv. haddock n/h seen by mean saithe {seen:.1f} (vs haul mean {H / n:.1f})")
            row = "    saithe n/h by depth band: " + ", ".join(
                f"{lo}-{hi}: {st.mean([h['sai'] for h in hs if lo <= h['depth'] < hi]):.2f} "
                f"(n={sum(lo <= h['depth'] < hi for h in hs)})"
                for lo, hi in bands if any(lo <= h['depth'] < hi for h in hs))
            L.append(row)
            row = "    juv. haddock n/h by depth band: " + ", ".join(
                f"{lo}-{hi}: {st.mean([h['had'] for h in hs if lo <= h['depth'] < hi]):.1f}"
                for lo, hi in bands if any(lo <= h['depth'] < hi for h in hs))
            L.append(row)
            dn = {dn: [h["sai"] for h in hs if h["daynight"] == dn] for dn in ("D", "N")}
            L.append("    saithe n/h day %.2f (n=%d), night %.2f (n=%d)" % (
                st.mean(dn["D"]) if dn["D"] else float("nan"), len(dn["D"]),
                st.mean(dn["N"]) if dn["N"] else float("nan"), len(dn["N"])))
        L.append("")
    with open(os.path.join(out, "saithe_haddock_overlap_summary.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
