"""Statistics of NorKyst-800 near-surface currents in the Mareld box.

Reads data/norkyst_<year>/*.npz (fetch_norkyst.py) and data/wind_<year>.npz
(fetch_wind.py, optional) and writes CSV tables to tables/.

Point statistics (7 points, 0/10/25/50 m) are computed on DAILY MEANS of the
hourly fields (removes tides, ~1 cm/s here, and most of the 14.1-h inertial
motion), annually and per season:
  mean vector, std per component, principal axis, reversal fractions,
  e-folding and integral decorrelation times of daily anomalies about the
  calendar-month mean.
Field statistics (box at stride 4 = ~3.2 km, 0 and 25 m, 6-hourly):
  mean normal flow per box edge, box-mean current per month, the Eulerian
  exchange time A / (integrated inflow), and Lagrangian first-exit times from
  2-D particle tracking in the 6-hourly fields.

Usage: python3 -I analyse_norkyst.py <research-note dir> <year>
"""
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo  # noqa: E402
from dods_ascii import parse_grid  # noqa: E402

DEPTHS = [0, 10, 25, 50]
FIELD_DEPTHS = [0, 25]
POINTS = ["NW", "N", "NE", "C", "SW", "S", "SE"]
PT_IDX = {"NW": (97, 326), "N": (65, 344), "NE": (43, 358), "C": (58, 318),
          "SW": (70, 279), "S": (41, 296), "SE": (15, 312)}
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def load(root, year):
    files = sorted(glob.glob(os.path.join(root, "data", "norkyst_%s" % year, "*.npz")))
    days, pts, fu, fv, ft = [], {}, [], [], []
    for f in files:
        d = np.load(f)
        days.append(os.path.basename(f)[:8])
        for p in POINTS:
            for c in ("u_eastward", "v_northward"):
                a = d["pt_%s_%s" % (p, c)].astype(float)
                a[np.abs(a) > 100] = np.nan
                pts.setdefault((p, c), []).append(a)
        u = d["field_u_eastward"].astype(float)
        v = d["field_v_northward"].astype(float)
        u[np.abs(u) > 100] = np.nan
        v[np.abs(v) > 100] = np.nan
        fu.append(u)
        fv.append(v)
        ft.append(d["time"][::6][:4])
    for k in pts:
        pts[k] = np.stack(pts[k])  # [day, 24, 4]
    return days, pts, np.stack(fu), np.stack(fv), np.stack(ft)


def months_of(days):
    return np.array([int(d[4:6]) for d in days])


def acf(x, maxlag):
    x = x - np.nanmean(x)
    ok = np.isfinite(x)
    x = np.where(ok, x, 0.0)
    out = []
    for L in range(maxlag + 1):
        a, b = x[:len(x) - L], x[L:]
        m = ok[:len(x) - L] & ok[L:]
        out.append(np.sum(a[m] * b[m]) / np.sqrt(np.sum(a[m] ** 2) * np.sum(b[m] ** 2)))
    return np.array(out)


def efold(r):
    """First lag (days, linear interpolation) where r drops below 1/e."""
    e = 1 / np.e
    for L in range(1, len(r)):
        if r[L] < e:
            return L - 1 + (r[L - 1] - e) / (r[L - 1] - r[L])
    return np.nan


def integral_scale(r):
    """Sum of the ACF (trapezoid, days) up to its first zero crossing."""
    s = 0.0
    for L in range(1, len(r)):
        if r[L] <= 0:
            break
        s += 0.5 * (r[L - 1] + r[L])
    return s


def vec_stats(u, v):
    """u, v in m/s (daily means). Returns dict in cm/s and degrees."""
    m = np.isfinite(u) & np.isfinite(v)
    u, v = u[m], v[m]
    mu, mv = u.mean(), v.mean()
    cov = np.cov(np.vstack([u, v]))
    w, vecs = np.linalg.eigh(cov)
    major = vecs[:, 1]
    ang = (np.degrees(np.arctan2(major[0], major[1])) + 360) % 180  # axis bearing, 0-180
    spd = np.hypot(mu, mv)
    dirto = (np.degrees(np.arctan2(mu, mv)) + 360) % 360
    # projection on the mean direction
    if spd > 0:
        along_mean = (u * mu + v * mv) / spd
        rev_mean = np.mean(along_mean < 0)
    else:
        rev_mean = np.nan
    # along principal axis, sign chosen so the mean is positive
    pa = u * major[0] + v * major[1]
    if pa.mean() < 0:
        pa = -pa
    rev_axis = np.mean(pa < 0)
    return dict(n=len(u), mean_u=100 * mu, mean_v=100 * mv, mean_speed=100 * spd, dir_to=dirto,
                std_u=100 * u.std(), std_v=100 * v.std(), axis_bearing=ang,
                std_major=100 * np.sqrt(w[1]), std_minor=100 * np.sqrt(max(w[0], 0)),
                mean_scalar_speed=100 * np.hypot(u, v).mean(),
                frac_against_mean=rev_mean, frac_axis_reversed=rev_axis,
                steadiness=spd / np.hypot(u, v).mean())


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (("%.3f" % v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


def point_tables(days, pts, outdir, year):
    mon = months_of(days)
    rows, srows, drows, mrows = [], [], [], []
    for p in POINTS:
        for k, dep in enumerate(DEPTHS):
            uh = pts[(p, "u_eastward")][:, :, k]
            vh = pts[(p, "v_northward")][:, :, k]
            ud, vd = np.nanmean(uh, 1), np.nanmean(vh, 1)
            st = vec_stats(ud, vd)
            # hourly std for comparison (includes inertial + tides)
            st["std_u_hourly"] = 100 * np.nanstd(uh)
            st["std_v_hourly"] = 100 * np.nanstd(vh)
            rows.append(dict(year=year, point=p, depth_m=dep, **st))
            for s, ms in SEASONS.items():
                sel = np.isin(mon, ms)
                srows.append(dict(year=year, point=p, depth_m=dep, season=s, **vec_stats(ud[sel], vd[sel])))
            for m in range(1, 13):
                sel = mon == m
                if sel.sum() > 5:
                    mrows.append(dict(year=year, point=p, depth_m=dep, month=m, **vec_stats(ud[sel], vd[sel])))
            # decorrelation of anomalies about the calendar-month mean
            ua, va = ud.copy(), vd.copy()
            for m in range(1, 13):
                sel = mon == m
                ua[sel] -= np.nanmean(ud[sel])
                va[sel] -= np.nanmean(vd[sel])
            ru, rv = acf(ua, 30), acf(va, 30)
            # along the principal axis (raw daily, not anomalies) and its anomaly
            cov = np.cov(np.vstack([ud[np.isfinite(ud)], vd[np.isfinite(vd)]]))
            major = np.linalg.eigh(cov)[1][:, 1]
            pa = ua * major[0] + va * major[1]
            rp = acf(pa, 30)
            # sign changes per month of the along-axis daily velocity (raw, not anomaly)
            pr = ud * major[0] + vd * major[1]
            if np.nanmean(pr) < 0:
                pr = -pr
            sc = np.sum(np.diff(np.sign(pr[np.isfinite(pr)])) != 0)
            # hourly ACF of along-axis component (shows inertial ringing)
            ph = (uh * major[0] + vh * major[1]).ravel()
            rh = acf(ph, 72)
            drows.append(dict(year=year, point=p, depth_m=dep,
                              efold_u_days=efold(ru), efold_v_days=efold(rv), efold_axis_days=efold(rp),
                              integral_u_days=integral_scale(ru), integral_v_days=integral_scale(rv),
                              integral_axis_days=integral_scale(rp),
                              acf_axis_lag1=rp[1], acf_axis_lag2=rp[2], acf_axis_lag5=rp[5],
                              acf_hourly_axis_lag6h=rh[6], acf_hourly_axis_lag12h=rh[12],
                              acf_hourly_axis_lag14h=rh[14], acf_hourly_axis_lag24h=rh[24],
                              axis_sign_changes_per_month=sc / (np.isfinite(pr).sum() / 30.4)))
    write_csv(os.path.join(outdir, "points_annual_%s.csv" % year), rows)
    write_csv(os.path.join(outdir, "points_seasonal_%s.csv" % year), srows)
    write_csv(os.path.join(outdir, "points_monthly_%s.csv" % year), mrows)
    write_csv(os.path.join(outdir, "points_decorrelation_%s.csv" % year), drows)
    return rows, drows


def spatial_correlation(days, pts, outdir, year, lat, lon):
    """Correlation of daily anomalies between point pairs vs distance."""
    mon = months_of(days)
    rows = []
    pos = {}
    for p in POINTS:
        r, c = PT_IDX[p]
        pos[p] = geo.to_sweref(lat[r, c - 240], lon[r, c - 240])
    for k, dep in enumerate(DEPTHS):
        an = {}
        for p in POINTS:
            for comp in ("u_eastward", "v_northward"):
                x = np.nanmean(pts[(p, comp)][:, :, k], 1)
                for m in range(1, 13):
                    sel = mon == m
                    x[sel] -= np.nanmean(x[sel])
                an[(p, comp)] = x
        for i, a in enumerate(POINTS):
            for b in POINTS[i + 1:]:
                d = np.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]) / 1000
                ru = np.corrcoef(an[(a, "u_eastward")], an[(b, "u_eastward")])[0, 1]
                rv = np.corrcoef(an[(a, "v_northward")], an[(b, "v_northward")])[0, 1]
                rows.append(dict(year=year, depth_m=dep, a=a, b=b, dist_km=d, r_u=ru, r_v=rv))
    write_csv(os.path.join(outdir, "points_spatial_corr_%s.csv" % year), rows)
    return rows


def field_geometry(lat, lon):
    fl = lat[5:110:4, 28:129:4]
    fo = lon[5:110:4, 28:129:4]
    E, N = geo.to_sweref(fl, fo)
    gam = geo.grid_convergence(fl, fo)
    # unit vectors of the NorKyst index axes (X: columns, Y: rows) in SWEREF metres
    eX = np.gradient(E, axis=1), np.gradient(N, axis=1)
    eY = np.gradient(E, axis=0), np.gradient(N, axis=0)
    return fl, fo, E, N, gam, eX, eY


def edge_flux(fu, fv, ft, days, E, N, gam, outdir, year):
    """Mean inward normal velocity per edge, box-mean current per month and the
    Eulerian exchange time A / integral of inflow (per unit depth)."""
    ib = geo.in_box(E, N)
    # edge bands: field points inside the box within 2.5 km of an edge
    b = geo.BOX
    bands = {"W": ib & (E - b["E0"] < 2500), "E": ib & (b["E1"] - E < 2500),
             "N": ib & (b["N1"] - N < 2500), "S": ib & (N - b["N0"] < 2500)}
    inward = {"W": (1, 0), "E": (-1, 0), "N": (0, -1), "S": (0, 1)}
    L = 60000.0
    A = 60000.0 * 60000.0
    mon = months_of(days)
    rows, mrows = [], []
    for k, dep in enumerate(FIELD_DEPTHS):
        uE, vN = geo.to_grid_components(fu[:, :, k], fv[:, :, k], gam)  # [day, 4, Y, X]
        series = {}
        qin = np.zeros(uE.shape[:2])
        for e, m in bands.items():
            nx, ny = inward[e]
            un = (uE * nx + vN * ny)[:, :, m]  # [day, 4, npts]
            series[e] = np.nanmean(un, 2)
            # inflow per unit depth through this edge (m2/s): mean of max(un,0) times L
            qin += np.nanmean(np.clip(un, 0, None), 2) * L
        texch = A / qin / 86400.0  # days, per snapshot
        ubox = np.nanmean(uE[:, :, ib], 2)
        vbox = np.nanmean(vN[:, :, ib], 2)
        for e in bands:
            rows.append(dict(year=year, depth_m=dep, edge=e, mean_inward_cm_s=100 * np.nanmean(series[e]),
                             std_inward_cm_s=100 * np.nanstd(series[e]),
                             frac_time_inward=np.nanmean(series[e] > 0)))
        rows.append(dict(year=year, depth_m=dep, edge="exchange_time_days_median",
                         mean_inward_cm_s=np.nanmedian(texch), std_inward_cm_s=np.nanpercentile(texch, 10),
                         frac_time_inward=np.nanpercentile(texch, 90)))
        # box-mean current per month
        for m in range(1, 13):
            sel = mon == m
            if sel.sum() == 0:
                continue
            ub, vb = ubox[sel].ravel(), vbox[sel].ravel()
            mrows.append(dict(year=year, depth_m=dep, month=m,
                              box_mean_uE_cm_s=100 * np.nanmean(ub), box_mean_vN_cm_s=100 * np.nanmean(vb),
                              box_mean_speed_cm_s=100 * np.hypot(np.nanmean(ub), np.nanmean(vb)),
                              mean_abs_box_vector_cm_s=100 * np.nanmean(np.hypot(ub, vb)),
                              std_box_uE_cm_s=100 * np.nanstd(ub), std_box_vN_cm_s=100 * np.nanstd(vb),
                              mean_point_speed_cm_s=100 * np.nanmean(np.hypot(uE[sel][:, :, ib], vN[sel][:, :, ib])),
                              exchange_time_days_median=np.nanmedian(texch[sel]),
                              W_in=100 * np.nanmean(series["W"][sel]), E_in=100 * np.nanmean(series["E"][sel]),
                              N_in=100 * np.nanmean(series["N"][sel]), S_in=100 * np.nanmean(series["S"][sel])))
        # annual box mean
        ub, vb = ubox.ravel(), vbox.ravel()
        mrows.append(dict(year=year, depth_m=dep, month=0,
                          box_mean_uE_cm_s=100 * np.nanmean(ub), box_mean_vN_cm_s=100 * np.nanmean(vb),
                          box_mean_speed_cm_s=100 * np.hypot(np.nanmean(ub), np.nanmean(vb)),
                          mean_abs_box_vector_cm_s=100 * np.nanmean(np.hypot(ub, vb)),
                          std_box_uE_cm_s=100 * np.nanstd(ub), std_box_vN_cm_s=100 * np.nanstd(vb),
                          mean_point_speed_cm_s=100 * np.nanmean(np.hypot(uE[:, :, ib], vN[:, :, ib])),
                          exchange_time_days_median=np.nanmedian(texch),
                          W_in=100 * np.nanmean(series["W"]), E_in=100 * np.nanmean(series["E"]),
                          N_in=100 * np.nanmean(series["N"]), S_in=100 * np.nanmean(series["S"])))
    write_csv(os.path.join(outdir, "box_edges_%s.csv" % year), rows)
    write_csv(os.path.join(outdir, "box_monthly_%s.csv" % year), mrows)
    return rows, mrows


def mean_field_map(fu, fv, E, N, gam, outdir, year):
    """Annual-mean grid-frame current on the stride-4 field, coarsened to 10 km cells."""
    rows = []
    for k, dep in enumerate(FIELD_DEPTHS):
        uE, vN = geo.to_grid_components(fu[:, :, k], fv[:, :, k], gam)
        mu, mv = np.nanmean(uE, (0, 1)), np.nanmean(vN, (0, 1))
        su, sv = np.nanstd(uE, (0, 1)), np.nanstd(vN, (0, 1))
        for i in range(6):
            for j in range(6):
                e0 = geo.BOX["E0"] + 10000 * j
                n1 = geo.BOX["N1"] - 10000 * i
                m = (E >= e0) & (E < e0 + 10000) & (N <= n1) & (N > n1 - 10000)
                if m.sum() == 0:
                    continue
                rows.append(dict(year=year, depth_m=dep, row_from_north=i, col_from_west=j, n_pts=int(m.sum()),
                                 mean_uE_cm_s=100 * mu[m].mean(), mean_vN_cm_s=100 * mv[m].mean(),
                                 std_uE_cm_s=100 * su[m].mean(), std_vN_cm_s=100 * sv[m].mean()))
    write_csv(os.path.join(outdir, "box_mean_map_10km_%s.csv" % year), rows)
    return rows


def track(fu, fv, E, N, gam, eX, eY, outdir, year, release_every=5, max_days=120, daily=False):
    """2-D first-exit times of particles released on a 2-km lattice in the box.

    Velocities are rotated to the NorKyst index axes (stride-4 field units),
    interpolated bilinearly in space and linearly in time between the 6-hourly
    snapshots, RK2 with dt = 1 h. A particle leaving the box (or the field)
    stops. Re-entry is not allowed (first-exit residence time).
    """
    nd = fu.shape[0]
    # metres per index step along X and Y and the unit vectors
    dXm = np.hypot(*eX)
    dYm = np.hypot(*eY)
    exE, exN = eX[0] / dXm, eX[1] / dXm
    eyE, eyN = eY[0] / dYm, eY[1] / dYm
    res = []
    for k, dep in enumerate(FIELD_DEPTHS):
        uE, vN = geo.to_grid_components(fu[:, :, k], fv[:, :, k], gam)
        uE = np.nan_to_num(uE)
        vN = np.nan_to_num(vN)
        if daily:
            uE = np.repeat(uE.mean(1, keepdims=True), 4, 1)
            vN = np.repeat(vN.mean(1, keepdims=True), 4, 1)
        # index-space velocity (index steps per second), flattened in time
        uX = ((uE * exE + vN * exN) / dXm).reshape(nd * 4, *E.shape)
        uY = ((uE * eyE + vN * eyN) / dYm).reshape(nd * 4, *E.shape)
        # release lattice: field-index coordinates of a 2 km SWEREF lattice
        ee, nn = np.meshgrid(np.arange(geo.BOX["E0"] + 1000, geo.BOX["E1"], 2000),
                             np.arange(geo.BOX["N0"] + 1000, geo.BOX["N1"], 2000))
        Y0, X0 = sweref_to_index(ee.ravel(), nn.ravel(), E, N)
        for d0 in range(0, nd - 10, release_every):
            t0 = d0 * 4
            X, Y = X0.copy(), Y0.copy()
            alive = np.ones(X.size, bool)
            texit = np.full(X.size, np.nan)
            dt = 3600.0
            nsteps = int(min(max_days * 24, (nd * 4 - 1 - t0) * 6))
            for s in range(nsteps):
                tt = t0 + s / 6.0
                k1x, k1y = interp(uX, uY, tt, X, Y)
                k2x, k2y = interp(uX, uY, tt + 1 / 6.0, X + dt * k1x, Y + dt * k1y)
                X = np.where(alive, X + 0.5 * dt * (k1x + k2x), X)
                Y = np.where(alive, Y + 0.5 * dt * (k1y + k2y), Y)
                Ep, Np = index_to_sweref(X, Y, E, N)
                out = alive & ~(geo.in_box(Ep, Np) & (X >= 0) & (X <= E.shape[1] - 1) & (Y >= 0) & (Y <= E.shape[0] - 1))
                texit[out] = (s + 1) / 24.0
                alive &= ~out
                if not alive.any():
                    break
            frac_left = np.isnan(texit).mean()
            res.append(dict(year=year, depth_m=dep, fields="daily_mean" if daily else "6_hourly",
                            release_day=d0 + 1, n=X.size,
                            median_exit_days=np.nanmedian(np.where(np.isnan(texit), np.inf, texit)),
                            mean_exit_days_lower_bound=np.nanmean(np.where(np.isnan(texit), nsteps / 24.0, texit)),
                            frac_remaining_at_end=frac_left, tracked_days=nsteps / 24.0,
                            frac_out_5d=np.mean(texit <= 5), frac_out_10d=np.mean(texit <= 10),
                            frac_out_20d=np.mean(texit <= 20), frac_out_30d=np.mean(texit <= 30)))
    write_csv(os.path.join(outdir, "lagrangian_exit_%s%s.csv" % (year, "_dailymean" if daily else "")), res)
    return res


def interp(uX, uY, tt, X, Y):
    nt = uX.shape[0]
    i0 = int(np.floor(tt))
    i0 = min(max(i0, 0), nt - 2)
    w = tt - i0
    ny, nx = uX.shape[1:]
    Xc = np.clip(X, 0, nx - 1.000001)
    Yc = np.clip(Y, 0, ny - 1.000001)
    x0 = np.floor(Xc).astype(int)
    y0 = np.floor(Yc).astype(int)
    fx, fy = Xc - x0, Yc - y0

    def bil(F):
        return ((1 - fx) * (1 - fy) * F[y0, x0] + fx * (1 - fy) * F[y0, x0 + 1]
                + (1 - fx) * fy * F[y0 + 1, x0] + fx * fy * F[y0 + 1, x0 + 1])
    ux = (1 - w) * bil(uX[i0]) + w * bil(uX[i0 + 1])
    uy = (1 - w) * bil(uY[i0]) + w * bil(uY[i0 + 1])
    return ux, uy


def index_to_sweref(X, Y, E, N):
    ny, nx = E.shape
    Xc = np.clip(X, 0, nx - 1.000001)
    Yc = np.clip(Y, 0, ny - 1.000001)
    x0 = np.floor(Xc).astype(int)
    y0 = np.floor(Yc).astype(int)
    fx, fy = Xc - x0, Yc - y0

    def bil(F):
        return ((1 - fx) * (1 - fy) * F[y0, x0] + fx * (1 - fy) * F[y0, x0 + 1]
                + (1 - fx) * fy * F[y0 + 1, x0] + fx * fy * F[y0 + 1, x0 + 1])
    return bil(E), bil(N)


def sweref_to_index(ee, nn, E, N):
    """Fractional field indices of SWEREF points (local affine fit, Newton steps)."""
    Y = np.zeros(ee.size)
    X = np.zeros(ee.size)
    # nearest start
    for i in range(ee.size):
        d = (E - ee[i]) ** 2 + (N - nn[i]) ** 2
        Y[i], X[i] = np.unravel_index(d.argmin(), d.shape)
    for _ in range(5):
        Ep, Np = index_to_sweref(X, Y, E, N)
        y0 = np.clip(np.round(Y).astype(int), 0, E.shape[0] - 2)
        x0 = np.clip(np.round(X).astype(int), 0, E.shape[1] - 2)
        dEdX = E[y0, x0 + 1] - E[y0, x0]
        dNdX = N[y0, x0 + 1] - N[y0, x0]
        dEdY = E[y0 + 1, x0] - E[y0, x0]
        dNdY = N[y0 + 1, x0] - N[y0, x0]
        det = dEdX * dNdY - dEdY * dNdX
        rE, rN = ee - Ep, nn - Np
        X += (rE * dNdY - rN * dEdY) / det
        Y += (-rE * dNdX + rN * dEdX) / det
    return Y, X


def wind_tables(root, days, pts, outdir, year):
    p = os.path.join(root, "data", "wind_%s.npz" % year)
    if not os.path.exists(p):
        return []
    w = np.load(p)
    t = w["time"]
    wu, wv = w["u10"], w["v10"]
    wday = np.array([np.datetime64(int(x), "s").astype("datetime64[D]").astype(str).replace("-", "") for x in t])
    rows = []
    # daily mean wind and wind stress-like vector |W| W
    Wd = {}
    for d in days:
        m = wday == d
        if m.sum() >= 12:
            Wd[d] = (wu[m].mean(), wv[m].mean(), np.mean(np.hypot(wu[m], wv[m]) * wu[m]),
                     np.mean(np.hypot(wu[m], wv[m]) * wv[m]))
    keep = np.array([d in Wd for d in days])
    WU = np.array([Wd[d][0] for d in np.array(days)[keep]])
    WV = np.array([Wd[d][1] for d in np.array(days)[keep]])
    TU = np.array([Wd[d][2] for d in np.array(days)[keep]])
    TV = np.array([Wd[d][3] for d in np.array(days)[keep]])
    rows.append(dict(year=year, point="C", depth_m="wind10m", lag_days=0, quantity="mean_wind_u_v_m_s",
                     r_u=WU.mean(), r_v=WV.mean(), gain=np.mean(np.hypot(WU, WV)), angle_deg=np.nan))
    for p_ in POINTS:
        for k, dep in enumerate(DEPTHS):
            ud = np.nanmean(pts[(p_, "u_eastward")][:, :, k], 1)[keep]
            vd = np.nanmean(pts[(p_, "v_northward")][:, :, k], 1)[keep]
            for lag in (0, 1, 2):
                a = slice(lag, None)
                b = slice(0, len(ud) - lag)
                # complex regression current = g * exp(i theta) * stress (vector correlation)
                c = ud[a] + 1j * vd[a]
                s = TU[b] + 1j * TV[b]
                c = c - c.mean()
                s = s - s.mean()
                g = np.sum(np.conj(s) * c) / np.sum(np.abs(s) ** 2)
                rho = np.abs(np.sum(np.conj(s) * c)) / np.sqrt(np.sum(np.abs(s) ** 2) * np.sum(np.abs(c) ** 2))
                rows.append(dict(year=year, point=p_, depth_m=dep, lag_days=lag,
                                 quantity="complex_corr_current_vs_wind_stress",
                                 r_u=np.corrcoef(ud[a], TU[b])[0, 1], r_v=np.corrcoef(vd[a], TV[b])[0, 1],
                                 gain=rho, angle_deg=np.degrees(np.angle(g))))
            if True:  # lag 0 regression on the wind itself (wind factor)
                c = (ud + 1j * vd) - np.mean(ud + 1j * vd)
                Wc = (WU + 1j * WV) - np.mean(WU + 1j * WV)
                g = np.sum(np.conj(Wc) * c) / np.sum(np.abs(Wc) ** 2)
                rho = np.abs(np.sum(np.conj(Wc) * c)) / np.sqrt(np.sum(np.abs(Wc) ** 2) * np.sum(np.abs(c) ** 2))
                rows.append(dict(year=year, point=p_, depth_m=dep, lag_days=0,
                                 quantity="complex_regression_current_on_wind (gain=|g| in percent of wind)",
                                 r_u=rho, r_v=np.nan, gain=100 * np.abs(g), angle_deg=np.degrees(np.angle(g))))
    write_csv(os.path.join(outdir, "wind_response_%s.csv" % year), rows)
    return rows


def main():
    root, year = sys.argv[1], sys.argv[2]
    outdir = os.path.join(root, "tables")
    os.makedirs(outdir, exist_ok=True)
    lat = parse_grid(open(os.path.join(root, "data", "lat_sub.txt")).read(), "lat")
    lon = parse_grid(open(os.path.join(root, "data", "lon_sub.txt")).read(), "lon")
    days, pts, fu, fv, ft = load(root, year)
    print("days", len(days))
    point_tables(days, pts, outdir, year)
    spatial_correlation(days, pts, outdir, year, lat, lon)
    fl, fo, E, N, gam, eX, eY = field_geometry(lat, lon)
    edge_flux(fu, fv, ft, days, E, N, gam, outdir, year)
    mean_field_map(fu, fv, E, N, gam, outdir, year)
    wind_tables(root, days, pts, outdir, year)
    if "--no-track" not in sys.argv:
        track(fu, fv, E, N, gam, eX, eY, outdir, year)
        track(fu, fv, E, N, gam, eX, eY, outdir, year, daily=True)
    print("done")


if __name__ == "__main__":
    main()
