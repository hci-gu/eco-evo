"""Fetch hourly NorKyst-800 surface (0 m) u_eastward / v_northward at one grid
point for one year (used for the HF-radar cross-check NW of the box).

Usage: python3 -I fetch_point.py <datadir> <year> <Yrow> <Xcol> <name> [workers]
Output: <datadir>/point_<name>_<year>.npz (time, u, v; m/s)
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dods_ascii import parse_grid  # noqa: E402
from fetch_norkyst import BASE, get  # noqa: E402


def one(fname, iy, ix):
    url = BASE + fname + ".ascii?"
    out = []
    for var in ("u_eastward", "v_northward"):
        q = "%s%%5B0:1:23%%5D%%5B0%%5D%%5B%d%%5D%%5B%d%%5D" % (var, iy, ix)
        out.append(parse_grid(get(url + q), var)[:, 0, 0, 0])
    t = parse_grid(get(url + "time"), "time")
    return t, out[0], out[1]


def main():
    datadir, year, iy, ix, name = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    files = [f.strip() for f in open(os.path.join(datadir, "norkyst_an_files.txt")) if (".an." + year) in f]

    def safe(f):
        try:
            return one(f, iy, ix)
        except Exception as exc:
            print("FAIL", f, exc, flush=True)
            return None
    with ThreadPoolExecutor(workers) as ex:
        res = [r for r in ex.map(safe, files) if r is not None]
    t = np.concatenate([r[0] for r in res])
    u = np.concatenate([r[1] for r in res])
    v = np.concatenate([r[2] for r in res])
    o = np.argsort(t)
    np.savez_compressed(os.path.join(datadir, "point_%s_%s.npz" % (name, year)), time=t[o], u=u[o], v=v[o])
    print("done", len(t))


if __name__ == "__main__":
    main()
