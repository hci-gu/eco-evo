"""Fetch the NorKyst-800 surface wind forcing (Uwind, Vwind, m/s, 10 m, as stored
in the ocean model's own files) at the box-centre point C, hourly, for one year.

Output: data/wind_<year>.npz with time (s since 1970) and u10, v10.
Usage: python3 -I fetch_wind.py <datadir> <year> [workers]
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dods_ascii import parse_grid  # noqa: E402
from fetch_norkyst import BASE, POINTS, get  # noqa: E402


def one(fname):
    iy, ix = POINTS["C"]
    url = BASE + fname + ".ascii?"
    out = []
    for var in ("Uwind", "Vwind"):
        q = "%s%%5B0:1:23%%5D%%5B%d%%5D%%5B%d%%5D" % (var, iy, ix)
        out.append(parse_grid(get(url + q), var)[:, 0, 0])
    t = parse_grid(get(url + "time"), "time")
    return t, out[0], out[1]


def main():
    datadir, year = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    files = [f.strip() for f in open(os.path.join(datadir, "norkyst_an_files.txt"))
             if (".an." + year) in f]
    res = []
    with ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(lambda f: _safe(f), files)):
            if r is not None:
                res.append(r)
            if k % 50 == 0:
                print(k, flush=True)
    t = np.concatenate([r[0] for r in res])
    u = np.concatenate([r[1] for r in res])
    v = np.concatenate([r[2] for r in res])
    o = np.argsort(t)
    np.savez_compressed(os.path.join(datadir, "wind_%s.npz" % year), time=t[o], u10=u[o], v10=v[o])
    print("done", len(t))


def _safe(f):
    try:
        return one(f)
    except Exception as exc:
        print("FAIL", f, exc, flush=True)
        return None


if __name__ == "__main__":
    main()
