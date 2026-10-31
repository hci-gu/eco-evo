"""Share of the planktivore (HERAS category MIX, herring ~87 %) backscatter near the bed, by the
log's bottom depth and sun elevation (Danish HERAS logs, RV Dana, July 2010-2020).

Usage: python3 -I step2_heras_bottom_share.py <zip> [<zip> ...]
  zips: research_notes/Ekolodsdata planktivorer/data/ices_zip/*.zip (ICES acoustic-trawl database,
  untrusted, parsed only). Same parsing and box/region as
  ../../Attackrater ur motesteori/scripts/heras_herring_depth.py (logs integrated as one layer skipped).

Bottom depth proxy = the deepest channel's lower bound. The Danish channels stop at ~154 m, so only
logs with deepest < 150 m have their true bed inside the integration; for those the bottom 10 / 20 m
share is the herring share inside a whiting band of 10 / 20 m above the bed (the acoustic dead zone,
~1-3 m above the bed, is unseen, so these are lower bounds) [D]. Deeper logs are reported separately
(their 'near-bottom' is just the deepest channels, not the bed).
"""
import collections
import csv
import io
import math
import sys
import zipfile
from datetime import datetime

REGIONS = {"box": (57.7, 58.4, 9.9, 11.3), "region": (57.5, 58.8, 9.0, 11.5)}
DBINS = [(0, 80), (80, 110), (110, 130), (130, 150), (150, 1e4)]


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


acc = collections.defaultdict(collections.Counter)
for zp in sys.argv[1:]:
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if not names:
            continue
        t = z.read(names[0]).decode("utf-8-sig", errors="replace")
    hdr, cruise = {}, None
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
        if max(c[1] - c[0] for c in chans) > 60:
            continue
        try:
            tt = datetime.strptime(lt[:16].replace("T", " "), "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        e = elev(lat, lon, tt)
        deepest = max(c[1] for c in chans)
        dbin = next(i for i, (a, b) in enumerate(DBINS) if a <= deepest < b)
        for reg, (a, b, c_, d_) in REGIONS.items():
            if not (a <= lat <= b and c_ <= lon <= d_):
                continue
            key = (reg, cls(e), dbin)
            acc[key]["logs"] += 1
            for up, lo, cat, v in chans:
                if cat != "MIX" or v <= 0:
                    continue
                acc[key]["tot"] += v
                if lo >= deepest - 10:
                    acc[key]["b10"] += v
                if lo >= deepest - 20:
                    acc[key]["b20"] += v
                if up >= 50:
                    acc[key]["below50"] += v

print("MIX backscatter share near the bed, by log bottom depth (deepest channel) and sun class")
print("region class       bottom bin   logs  NASC-share in bottom 10 m / 20 m / below 50 m   (mean NASC per log)")
for key in sorted(acc):
    a = acc[key]
    reg, c, d = key
    lab = f"{DBINS[d][0]:.0f}-{DBINS[d][1]:.0f}" if DBINS[d][1] < 1e4 else ">=150 (bed not seen)"
    if a["tot"] <= 0:
        print(f"{reg:6s} {c:11s} {lab:20s} {a['logs']:5d}   no MIX backscatter")
        continue
    print(f"{reg:6s} {c:11s} {lab:20s} {a['logs']:5d}   {a['b10'] / a['tot']:.2f} / {a['b20'] / a['tot']:.2f} / "
          f"{a['below50'] / a['tot']:.2f}   ({a['tot'] / a['logs']:.0f})")
