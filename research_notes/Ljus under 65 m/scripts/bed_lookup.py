"""Bed depth under every glider cast position, from the EMODnet Bathymetry REST point service.

Usage: python3 -I bed_lookup.py   (reads ../data/voto/*_bin5.csv.gz, writes ../data/emodnet_bed.json)
Positions rounded to 0.01 deg (~0.6 x 1.1 km); one request per unique rounded point:
GET https://rest.emodnet-bathymetry.eu/depth_sample?geom=POINT(lon lat) -> 'avg' (negative = below sea level).
EMODnet Bathymetry DTM: free use with citation (EMODnet Bathymetry Consortium).
"""
import csv
import gzip
import glob
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "emodnet_bed.json")


def main():
    pts = set()
    for f in glob.glob(os.path.join(HERE, "..", "data", "voto", "*_bin5.csv.gz")):
        rows = list(csv.reader(gzip.open(f, "rt")))
        h = rows[0]
        il, io = h.index("latitude"), h.index("longitude")
        for r in rows[2:]:
            if r[il] != "NaN" and r[io] != "NaN":
                pts.add((round(float(r[il]), 2), round(float(r[io]), 2)))
    cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [p for p in sorted(pts) if cache.get(f"{p[0]:.2f},{p[1]:.2f}") is None]
    print(len(pts), "points,", len(todo), "to fetch", flush=True)
    for i, (la, lo) in enumerate(todo):
        url = f"https://rest.emodnet-bathymetry.eu/depth_sample?geom=POINT({lo:.2f}%20{la:.2f})"
        cache[f"{la:.2f},{lo:.2f}"] = None
        for attempt in range(4):  # the service drops requests intermittently: retry with a pause
            try:
                d = json.loads(urllib.request.urlopen(url, timeout=30).read())
                cache[f"{la:.2f},{lo:.2f}"] = -float(d["avg"]) if d.get("avg") is not None else None
                break
            except Exception:  # noqa: BLE001
                time.sleep(2 + 3 * attempt)
        if i % 100 == 0:
            json.dump(cache, open(OUT, "w"))
            print(i, flush=True)
    json.dump(cache, open(OUT, "w"))


if __name__ == "__main__":
    main()
