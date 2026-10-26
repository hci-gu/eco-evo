"""Herring (and sprat) vertical distribution in the Mareld region from Danish HERAS logs
(RV Dana, ICES acoustic-trawl database zips downloaded by the sibling study
research_notes/Ekolodsdata planktivorer/data/ices_zip; untrusted data, only parsed).

Usage: python3 -I heras_herring_depth.py <zip> [<zip> ...]

Box: 57.7-58.4 N, 9.9-11.3 E (the sibling study's box) and a wider region 57.5-58.8 N,
9.0-11.5 E. NASC by SaCategory (herring = 'HER', sprat = 'SPR' if present) by channel upper
depth; logs classed by local solar sun elevation at LogTime (UTC + lon/15 h; equation of time
ignored): day (> 6 deg), low sun (0..6), twilight (-12..0), night (< -12).
Reports share of NASC in 0-10 (from the integration start), 10-25, 25-50, 50-100, >100 m and the
NASC-weighted mean depth, plus the share of herring NASC within the bottom 10 m of each log.
"""
import collections
import csv
import io
import math
import sys
import zipfile
from datetime import datetime

REGIONS = {"box": (57.7, 58.4, 9.9, 11.3), "region": (57.5, 58.8, 9.0, 11.5)}
BINS = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 1e4)]


def elev(lat, lon, t):
    n = t.timetuple().tm_yday
    hrs = t.hour + t.minute / 60 + lon / 15.0
    dec = math.radians(23.44) * math.sin(2 * math.pi * (284 + n) / 365.0)
    ha = math.radians(15 * (hrs - 12))
    p = math.radians(lat)
    s = math.sin(p) * math.sin(dec) + math.cos(p) * math.cos(dec) * math.cos(ha)
    return math.degrees(math.asin(max(-1, min(1, s))))


def cls(e):
    return "day>6" if e > 6 else "low0-6" if e > 0 else "twil-12-0" if e > -12 else "night<-12"


acc = collections.defaultdict(lambda: collections.Counter())
nlogs = collections.Counter()
for zp in sys.argv[1:]:
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if not names:
            continue
        t = z.read(names[0]).decode("utf-8-sig", errors="replace")
    hdr = {}
    cruise = None
    logs = collections.defaultdict(list)
    for row in csv.reader(io.StringIO(t)):
        if len(row) < 2:
            continue
        if row[1] == "Header":
            hdr[row[0]] = row[2:]
            continue
        if row[0] == "Cruise":
            cruise = dict(zip(hdr["Cruise"], row[2:]))
        if row[0] != "Data":
            continue
        r = dict(zip(hdr["Data"], row[2:]))
        try:
            lat, lon = float(r["LogLatitude"]), float(r["LogLongitude"])
            up, lo = float(r["SampleChannelDepthUpper"]), float(r["SampleChannelDepthLower"])
            v = float(r["DataValue"] or 0)
        except (ValueError, KeyError):
            continue
        if r.get("LogValidity") not in ("V", ""):
            continue
        logs[(r["LogTime"], lat, lon)].append((up, lo, r.get("DataSaCategory", ""), v))
    if cruise is None or cruise.get("CruiseSurvey") != "HERAS":
        continue
    for (lt, lat, lon), chans in logs.items():
        # skip logs integrated as one layer (no vertical resolution)
        if max(c[1] - c[0] for c in chans) > 60:
            continue
        try:
            tt = datetime.strptime(lt[:16].replace("T", " "), "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        e = elev(lat, lon, tt)
        deepest = max(c[1] for c in chans)
        for reg, (a, b, c_, d_) in REGIONS.items():
            if not (a <= lat <= b and c_ <= lon <= d_):
                continue
            for sp in ("MIX", "HER", "CLU"):
                key = (reg, sp, cls(e))
                had = False
                for up, lo, cat, v in chans:
                    if cat != sp or v <= 0:
                        continue
                    had = True
                    mid = (up + lo) / 2
                    for i, (z0, z1) in enumerate(BINS):
                        if z0 <= up < z1:
                            acc[key][i] += v
                    acc[key]["tot"] += v
                    acc[key]["zsum"] += v * mid
                    if lo >= deepest - 10:
                        acc[key]["near"] += v
                    acc[key]["bottom_sum"] += v * deepest
                nlogs[(reg, cls(e))] += 0
            nlogs[(reg, cls(e))] += 1

print("region species class logs | NASC share by channel upper depth 0-10 10-25 25-50 50-100 >100 | mean depth | "
      "share in bottom 10 m | NASC-weighted bottom depth")
for key in sorted(acc):
    a = acc[key]
    tot = a["tot"]
    if tot <= 0:
        continue
    sh = " ".join(f"{a[i] / tot:5.2f}" for i in range(len(BINS)))
    print(f"{key[0]:6s} {key[1]} {key[2]:10s} {nlogs[(key[0], key[2])]:5d} | {sh} | {a['zsum'] / tot:6.1f} | "
          f"{a['near'] / tot:4.2f} | {a['bottom_sum'] / tot:6.1f}")
