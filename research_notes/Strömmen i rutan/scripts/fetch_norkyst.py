"""Fetch NorKyst-800 (MET Norway, thredds.met.no, open OPeNDAP, CC BY 4.0)
near-surface currents for the Mareld box, one daily analysis file at a time.

Two subsets per day, both of the geographic components u_eastward / v_northward
(m/s, at rho points, already rotated to east/north):
  1. points: 7 grid points, all 24 hourly fields, depths 0, 10, 25, 50 m
  2. field:  the whole box at stride 4 (~3.2 km), depths 0 and 25 m,
             at 00, 06, 12, 18 UTC (the model's 6-h tick)

Output: data/norkyst_<year>/<yyyymmdd>.npz (parsed arrays; the raw ascii is
not kept). Re-running skips days already fetched.

Usage: python3 -I fetch_norkyst.py <outdir> <year> [workers] [--reverse]
"""
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dods_ascii import parse_grid  # noqa: E402

BASE = "https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/"
# name: (Y row, X col) in the NorKyst-800 grid (found from lat/lon, see summary)
POINTS = {
    "NW": (97, 326), "N": (65, 344), "NE": (43, 358), "C": (58, 318),
    "SW": (70, 279), "S": (41, 296), "SE": (15, 312),
}
FIELD = "%5B0:6:23%5D%5B0:4:4%5D%5B5:4:109%5D%5B268:4:368%5D"  # [t][0,25 m][Y][X]


def enc(s):
    return s.replace("[", "%5B").replace("]", "%5D")


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read().decode("ascii", "replace")
        except Exception as exc:  # network hiccup: back off and retry
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))
    return None


def fetch_day(fname, outdir):
    day = fname.split(".")[-2][:8]
    out = os.path.join(outdir, day + ".npz")
    if os.path.exists(out):
        return day, "skip"
    url = BASE + fname + ".ascii?"
    res = {}
    # points: depths 0,10,25,50 m are indices 0,2,4,5
    for var in ("u_eastward", "v_northward"):
        parts = []
        for name, (iy, ix) in POINTS.items():
            parts.append(enc("%s[0:1:23][0:1:5][%d][%d]" % (var, iy, ix)))
        txt_all = []
        for p in parts:  # one projection per request (same var twice is not allowed)
            txt_all.append(get(url + p))
        for (name, _), txt in zip(POINTS.items(), txt_all):
            a = parse_grid(txt, var)[:, [0, 2, 4, 5], 0, 0]  # [24, 4]
            res["pt_%s_%s" % (name, var)] = a.astype(np.float32)
    for var in ("u_eastward", "v_northward"):
        txt = get(url + var + FIELD)
        res["field_" + var] = parse_grid(txt, var).astype(np.float32)  # [4,2,27,26]
    txt = get(url + "time")
    res["time"] = parse_grid(txt, "time")
    np.savez_compressed(out, **res)
    return day, "ok"


def main():
    outdir, year = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 3
    os.makedirs(outdir, exist_ok=True)
    listing = os.path.join(os.path.dirname(outdir.rstrip("/")), "norkyst_an_files.txt")
    files = [f.strip() for f in open(listing) if (".an." + year) in f]
    if "--reverse" in sys.argv:  # a second process can work from the other end
        files = files[::-1]
    print("files:", len(files), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for k, (day, st) in enumerate(ex.map(lambda f: _safe(f, outdir), files)):
            if k % 20 == 0:
                print(k, day, st, "%.0f s" % (time.time() - t0), flush=True)


def _safe(f, outdir):
    try:
        return fetch_day(f, outdir)
    except Exception as exc:
        return f, "FAIL %r" % (exc,)


if __name__ == "__main__":
    main()
