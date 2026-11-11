"""Cross-check NorKyst-800 surface currents against the HF radar (HFR-Skagerrak,
MET Norway) at the nearest well-covered radar cell NW of the box.

Radar: data/hfr/hfr_2023_northedge.nc (ERDDAP subset, hourly, 58.28-58.50 N,
9.85-10.45 E). Model: data/point_HFRcell_2023.npz (NorKyst 0 m at 58.388 N
10.002 E). The radar cell used is the one nearest that point. Statistics are
computed on hours where both exist and the radar QCflag is 1 (good), and on daily means of days with >= 12 radar
hours.

Note: HF radar measures the upper ~0.3-2.5 m; NorKyst's 0 m level is its top
sigma layer interpolated to 0 m. Wind drift of the top metre makes the radar
expected to be faster and more wind-following than the model's 0 m.

Usage: python3 -I hfr_vs_norkyst.py <research-note dir>
"""
import csv
import os
import sys

import numpy as np
from scipy.io import netcdf_file


def stats(u, v):
    m = np.isfinite(u) & np.isfinite(v)
    u, v = u[m], v[m]
    return dict(n=int(m.sum()), mean_u_cm_s=100 * u.mean(), mean_v_cm_s=100 * v.mean(),
                mean_speed_vector_cm_s=100 * np.hypot(u.mean(), v.mean()),
                dir_to_deg=(np.degrees(np.arctan2(u.mean(), v.mean())) + 360) % 360,
                std_u_cm_s=100 * u.std(), std_v_cm_s=100 * v.std(),
                mean_scalar_speed_cm_s=100 * np.hypot(u, v).mean())


def main():
    root = sys.argv[1]
    f = netcdf_file(os.path.join(root, "data", "hfr", "hfr_2023_northedge.nc"), "r", mmap=False)
    t = f.variables["time"][:].astype(float)
    lat = f.variables["latitude"][:]
    lon = f.variables["longitude"][:]
    U = f.variables["EWCT"][:].astype(float)[:, 0]
    V = f.variables["NSCT"][:].astype(float)[:, 0]
    U[U < -30] = np.nan
    V[V < -30] = np.nan
    m = np.load(os.path.join(root, "data", "point_HFRcell_2023.npz"))
    tm, um, vm = m["time"], m["u"].astype(float), m["v"].astype(float)
    um[np.abs(um) > 100] = np.nan
    vm[np.abs(vm) > 100] = np.nan
    i = int(np.argmin(np.abs(lat - 58.388)))
    j = int(np.argmin(np.abs(lon - 10.002)))
    ur, vr = U[:, i, j].copy(), V[:, i, j].copy()
    # keep only QCflag == 1 ("good_data"); flags from data/hfr/hfr_2023_cell_qc.nc
    q = netcdf_file(os.path.join(root, "data", "hfr", "hfr_2023_cell_qc.nc"), "r", mmap=False)
    assert np.allclose(q.variables["time"][:], t)
    good = q.variables["QCflag"][:][:, 0, 0, 0] == 1
    ur[~good] = np.nan
    vr[~good] = np.nan
    # hourly matching
    idx = {int(round(x)): k for k, x in enumerate(tm)}
    pair = [(k, idx.get(int(round(x)))) for k, x in enumerate(t)]
    pair = [(a, b) for a, b in pair if b is not None]
    a = np.array([p[0] for p in pair])
    b = np.array([p[1] for p in pair])
    ok = np.isfinite(ur[a]) & np.isfinite(um[b])
    a, b = a[ok], b[ok]
    rows = []
    rows.append(dict(series="radar_hourly_matched", lat=float(lat[i]), lon=float(lon[j]), **stats(ur[a], vr[a])))
    rows.append(dict(series="norkyst0m_hourly_matched", lat=58.388, lon=10.002, **stats(um[b], vm[b])))
    rows.append(dict(series="norkyst0m_hourly_all", lat=58.388, lon=10.002, **stats(um, vm)))
    cu = np.corrcoef(ur[a], um[b])[0, 1]
    cv = np.corrcoef(vr[a], vm[b])[0, 1]
    # daily means on days with >= 12 matched hours
    day = (t[a] // 86400).astype(int)
    du, dv, mu, mv = [], [], [], []
    for d in np.unique(day):
        s = day == d
        if s.sum() >= 12:
            du.append(ur[a][s].mean())
            dv.append(vr[a][s].mean())
            mu.append(um[b][s].mean())
            mv.append(vm[b][s].mean())
    du, dv, mu, mv = map(np.array, (du, dv, mu, mv))
    rows.append(dict(series="radar_daily(>=12h)", lat=float(lat[i]), lon=float(lon[j]), **stats(du, dv)))
    rows.append(dict(series="norkyst0m_daily_same_hours", lat=58.388, lon=10.002, **stats(mu, mv)))
    with open(os.path.join(root, "tables", "hfr_vs_norkyst_2023.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.3f" % v if isinstance(v, float) else v) for k, v in r.items()})
        fh.write("# hourly correlation radar vs model: r_u=%.3f r_v=%.3f (n=%d hours)\n" % (cu, cv, len(a)))
        fh.write("# daily correlation (days with >=12 matched hours, n=%d): r_u=%.3f r_v=%.3f\n"
                 % (len(du), np.corrcoef(du, mu)[0, 1], np.corrcoef(dv, mv)[0, 1]))
    print(open(os.path.join(root, "tables", "hfr_vs_norkyst_2023.csv")).read())


if __name__ == "__main__":
    main()
