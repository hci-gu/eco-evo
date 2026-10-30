#!/usr/bin/env python3
"""Fit stock-recruitment models to the lagged SAG pairs and derive the
compensation at today's SSB.

Usage:
    python3 -I scripts/sr_fit.py PAIRS_CSV SERIES_CSV REFPTS_DIR OUT_DIR [N_BOOT]
    e.g. python3 -I research_notes/Beståndsreglering/scripts/sr_fit.py \
             research_notes/Beståndsreglering/tables/sr_pairs.csv \
             research_notes/Beståndsreglering/tables/sr_ssb_series.csv \
             research_notes/Beståndsreglering/data/sag \
             research_notes/Beståndsreglering/tables 1000

Models (y = ln R, lognormal errors, least squares in log space = Gaussian ML):
  prop    ln R = ln a + ln S                    R/S constant: NO compensation (elast 0)
  const   ln R = ln R0                           R independent of S: FULL compensation (elast -1)
  bh      ln R = ln a + ln S - ln(1 + S/b)       Beverton-Holt; b = SSB at half Rmax
  ricker  ln R = ln a + ln S - S/b               Ricker; b = SSB at Rmax (1/b may go < 0)
  hs      ln R = ln a + ln min(S, Sbp)           segmented regression (ICES segreg),
                                                 Sbp searched within [min S, max S]
  *_shift the same with a separate ln a before / after the shift year (common b):
          prop_shift, const_shift, bh_shift, ricker_shift - a productivity change
          in place of (or on top of) compensation (full period only).
  *_trend the same with ln a linear in year: prop_trend, const_trend, bh_trend
          (both periods) - a gradual productivity trend.
  "elast_avg" averages the five stationary models by AICc weight;
  "elast_avg_all" averages all models fitted to that period (stationary, trend,
  and shift in the full period).
OUT_DIR/sr_boot_draws.csv keeps the bootstrap draws of elast_avg and
elast_avg_all per stock and period (for the group aggregation).
Parameters k count sigma. AICc = n ln(RSS/n) + 2k + 2k(k+1)/(n-k-1).
Compensation at today's SSB S_t (mean SSB of the last 3 years <= terminal year):
  ratio_origin = (R/S)(S_t) / (R/S)(S -> 0)
  ratio_minobs = (R/S)(S_t) / (R/S)(S_min observed in the fitted period)
  elast        = d ln(R/S) / d ln S at S_t
Descriptive (not in the AICc set): the power-law (Cushing) slope beta of
ln R = c + beta ln S, whose elasticity of R/S is beta - 1 at any S; its SE is
inflated for AR(1) residuals (n_eff = n (1 - rho)/(1 + rho)).
Uncertainty: moving-block bootstrap of the (S, R) pairs (block 5 years), refit.
Outputs: OUT_DIR/sr_fits.csv (every model x stock x period), OUT_DIR/sr_compensation.csv
(per stock x period: AICc winner, model-averaged elasticity and ratio with bootstrap
90 % intervals, per-model elasticities).
"""
import csv
import json
import math
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize_scalar

# shift year (first year class of the "new" regime) per stock; default 2000.
SHIFT = {"her.27.3a47d": 2002}
DEFAULT_SHIFT = 2000
POST_FROM = 2000  # "post2000" period: year classes >= max(2000, shift year)
MIN_N = 12
BLOCK = 5
SEED = 20261029

STOCKS = [
    "her.27.3a47d", "her.27.20-24", "spr.27.3a4", "nop.27.3a4", "whb.27.1-91214",
    "had.27.46a20", "cod.27.47d20", "cod.27.46a7d20_sum", "cod.27.46a7d20_viking",
    "cod.27.46a7d20_southern", "cod.27.46a7d20_northwestern", "cod.27.21",
    "ple.27.420", "ple.27.21-32", "ple.27.21-23", "wit.27.3a47d", "pok.27.3a46",
    "mac.27.nea", "hom.27.4bc7d", "hom.27.2a3a4a5b6a7a-ce-k8", "hke.27.3a46-8abd",
    "dgs.27.nea", "whg.27.47d", "anf.27.3a46",
]
REFKEY = {"cod.27.46a7d20_sum": None}


# ---------------------------------------------------------------- fitting

def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return beta, r


def fit(model, S, y, post, tt):
    """Return dict(rss, k, params..., resid). post: bool array (after shift)."""
    n = len(y)
    lS = np.log(S)
    one = np.ones(n)
    shift = model.endswith("_shift")
    trend = model.endswith("_trend")
    base = model.replace("_shift", "").replace("_trend", "")
    if shift:
        X0 = np.column_stack([one * (~post), one * post])
    elif trend:
        X0 = np.column_stack([one, tt - tt.mean()])
    else:
        X0 = one[:, None]
    if shift and (post.sum() < 3 or (~post).sum() < 3):
        return None
    out = {"model": model}
    if base == "prop":
        beta, r = _ols(X0, y - lS)
        out.update(la=beta, b=math.inf)
    elif base == "const":
        beta, r = _ols(X0, y)
        out.update(la=beta, b=0.0)
    elif base == "ricker":
        X = np.column_stack([X0, -S])
        beta, r = _ols(X, y - lS)
        inv_b = beta[-1]
        out.update(la=beta[:-1], b=(1.0 / inv_b) if inv_b != 0 else math.inf, inv_b=inv_b)
    elif base == "bh":
        lo, hi = math.log(S.min() * 1e-3), math.log(S.max() * 1e3)

        def rss(lb):
            _, r = _ols(X0, y - lS + np.log1p(S / math.exp(lb)))
            return float(r @ r)
        grid = np.linspace(lo, hi, 121)
        vals = [rss(g) for g in grid]
        i = int(np.argmin(vals))
        a_, b_ = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        res = minimize_scalar(rss, bounds=(a_, b_), method="bounded",
                              options={"xatol": 1e-6})
        lb = res.x if res.fun <= vals[i] else grid[i]
        beta, r = _ols(X0, y - lS + np.log1p(S / math.exp(lb)))
        out.update(la=beta, b=math.exp(lb), b_at_bound=bool(lb <= lo + 0.05 or lb >= hi - 0.05))
    elif base == "hs":
        cand = np.unique(S)
        best = None
        # profile Sbp on observed SSB values and midpoints (piecewise-smooth RSS)
        pts = np.unique(np.concatenate([cand, (cand[1:] + cand[:-1]) / 2]))
        for sb in pts:
            _, r = _ols(X0, y - np.log(np.minimum(S, sb)))
            v = float(r @ r)
            if best is None or v < best[0]:
                best = (v, sb)
        sb = best[1]
        beta, r = _ols(X0, y - np.log(np.minimum(S, sb)))
        out.update(la=beta, b=sb, sbp_at_bound=("min" if sb <= S.min() * 1.0001 else
                                                 "max" if sb >= S.max() * 0.9999 else ""))
    else:
        raise ValueError(model)
    k = X0.shape[1] + {"prop": 0, "const": 0, "ricker": 1, "bh": 1, "hs": 1}[base] + 1
    rss_ = float(r @ r)
    out.update(k=k, n=n, rss=rss_, resid=r, sigma=math.sqrt(rss_ / n))
    out["aicc"] = n * math.log(rss_ / n) + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else math.inf)
    return out


def rps_shape(base, b, S):
    """(R/S)(S) / a, i.e. the density-dependent factor."""
    S = np.asarray(S, float)
    if base == "prop":
        return np.ones_like(S)
    if base == "const":
        return 1.0 / S
    if base == "bh":
        return 1.0 / (1.0 + S / b)
    if base == "ricker":
        return np.exp(-S / b) if math.isfinite(b) else np.ones_like(S)
    if base == "hs":
        return np.minimum(S, b) / S
    raise ValueError(base)


def elasticity(base, b, S):
    if base == "prop":
        return 0.0
    if base == "const":
        return -1.0
    if base == "bh":
        x = S / b
        return -x / (1 + x)
    if base == "ricker":
        return -S / b if math.isfinite(b) else 0.0
    if base == "hs":
        return -1.0 if S > b else 0.0
    raise ValueError(base)


def derived(f, S_t, S_min):
    base = f["model"].replace("_shift", "").replace("_trend", "")
    b = f["b"]
    ro = math.nan if base == "const" else float(rps_shape(base, b, [S_t])[0])
    rm = float(rps_shape(base, b, [S_t])[0] / rps_shape(base, b, [S_min])[0])
    return dict(elast=elasticity(base, b, S_t), ratio_origin=ro, ratio_minobs=rm)


def ar1(r):
    if len(r) < 4:
        return math.nan
    a, b = r[:-1] - r.mean(), r[1:] - r.mean()
    return float((a @ b) / math.sqrt((a @ a) * (b @ b)))


NOSHIFT = ["prop", "const", "bh", "ricker", "hs"]
SHIFTED = ["prop_shift", "const_shift", "bh_shift", "ricker_shift"]
TREND = ["prop_trend", "const_trend", "bh_trend"]


def is_alt(m):
    return m.endswith("_shift") or m.endswith("_trend")


def weights(fits):
    aic = np.array([f["aicc"] for f in fits])
    d = aic - aic.min()
    w = np.exp(-d / 2)
    return d, w / w.sum()


def fit_all(S, y, post, tt, models):
    fs = [fit(m, S, y, post, tt) for m in models]
    return [f for f in fs if f is not None]


def avg_elast(fits, S_t, S_min):
    _, w = weights(fits)
    e = np.array([derived(f, S_t, S_min)["elast"] for f in fits])
    rm = np.array([derived(f, S_t, S_min)["ratio_minobs"] for f in fits])
    return float(w @ e), float(w @ rm)


def block_boot(n, rng):
    idx = []
    while len(idx) < n:
        s = rng.integers(0, max(n - BLOCK + 1, 1))
        idx.extend(range(s, min(s + BLOCK, n)))
    return np.array(idx[:n])


def q(v, p):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return float(np.quantile(v, p)) if len(v) else math.nan


# ---------------------------------------------------------------- main

def main():
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    pairs = list(csv.DictReader(open(sys.argv[1])))
    series = list(csv.DictReader(open(sys.argv[2])))
    refdir = pathlib.Path(sys.argv[3])
    out = pathlib.Path(sys.argv[4])
    nboot = int(sys.argv[5]) if len(sys.argv) == 6 else 1000
    keys = {lab: k for lab, k, _ in json.loads((refdir / "keys.json").read_text())}
    rng = np.random.default_rng(SEED)
    fit_rows, comp_rows, draw_rows = [], [], []
    for st in STOCKS:
        P = sorted([p for p in pairs if p["stock"] == st and p["use"] == "1"],
                   key=lambda p: int(p["year_class"]))
        if not P:
            continue
        ser = [s for s in series if s["stock"] == st and s["ssb"]]
        term = int(ser[0]["terminal_year"])
        last = [float(s["ssb"]) for s in ser if term - 3 < int(s["year"]) <= term]
        S_t = float(np.mean(last))
        rk = keys.get(st if st != "cod.27.46a7d20_sum" else "", None)
        blim = bpa = ""
        if rk is not None:
            rp = json.loads((refdir / f"refpts_{rk}.json").read_text())
            if rp:
                blim, bpa = rp[0].get("Blim") or "", rp[0].get("Bpa") or ""
        if st == "cod.27.46a7d20_sum":
            blim = 14027 + 14433 + 23422  # sum of substock Blims (not an ICES value)
            bpa = 19491 + 21549 + 32547
        shift_year = SHIFT.get(st, DEFAULT_SHIFT)
        post_from = max(POST_FROM, shift_year)  # NSAS herring: from 2002, its documented shift
        periods = [("full", P), ("post2000", [p for p in P if int(p["year_class"]) >= post_from])]
        for per, PP in periods:
            if len(PP) < MIN_N:
                continue
            yrs = np.array([int(p["year_class"]) for p in PP])
            S = np.array([float(p["ssb_t"]) for p in PP])
            R = np.array([float(p["rec_n"]) for p in PP])
            y = np.log(R)
            post = yrs >= shift_year
            S_min, S_max = float(S.min()), float(S.max())
            models = NOSHIFT + TREND + (SHIFTED if per == "full" else [])
            fits = fit_all(S, y, post, yrs.astype(float), models)
            d_all, w_all = weights(fits)
            ns = [f for f in fits if not is_alt(f["model"])]
            d_ns, w_ns = weights(ns)
            # bootstrap
            boot = {m: [] for m in ["bh", "ricker", "hs", "avg_ns", "avg_all",
                                    "bh_ratio", "ricker_ratio", "hs_ratio", "avg_ns_ratio",
                                    "bh_shift", "avg_all_ratio"]}
            win_count = {}
            for _ in range(nboot):
                ii = block_boot(len(y), rng)
                Sb, yb, pb = S[ii], y[ii], post[ii]
                Smb = float(Sb.min())
                fb = fit_all(Sb, yb, pb, yrs[ii].astype(float), models)
                nsb = [f for f in fb if not is_alt(f["model"])]
                for f in fb:
                    if f["model"] in ("bh", "ricker", "hs"):
                        dd = derived(f, S_t, Smb)
                        boot[f["model"]].append(dd["elast"])
                        boot[f["model"] + "_ratio"].append(dd["ratio_minobs"])
                    if f["model"] == "bh_shift":
                        boot["bh_shift"].append(derived(f, S_t, Smb)["elast"])
                e_ns, r_ns = avg_elast(nsb, S_t, Smb)
                e_all, r_all = avg_elast(fb, S_t, Smb)
                boot["avg_ns"].append(e_ns)
                boot["avg_ns_ratio"].append(r_ns)
                boot["avg_all"].append(e_all)
                boot["avg_all_ratio"].append(r_all)
                wb = min(fb, key=lambda f: f["aicc"])["model"]
                win_count[wb] = win_count.get(wb, 0) + 1
            for i_b, (e1, e2) in enumerate(zip(boot["avg_ns"], boot["avg_all"])):
                draw_rows.append(dict(stock=st, period=per, draw=i_b,
                                      elast_avg=f"{e1:.4f}", elast_avg_all=f"{e2:.4f}"))
            for f, d, w in zip(fits, d_all, w_all):
                dd = derived(f, S_t, S_min)
                la = np.atleast_1d(f["la"])
                fit_rows.append(dict(
                    stock=st, period=per, model=f["model"], n=f["n"], k=f["k"],
                    yc_first=int(yrs.min()), yc_last=int(yrs.max()),
                    shift_year=shift_year if f["model"].endswith("_shift") else "",
                    a=f"{math.exp(la[0]):.6g}",
                    a_post=f"{math.exp(la[1]):.6g}" if f["model"].endswith("_shift") else "",
                    trend_factor_per_decade=(f"{math.exp(10 * la[1]):.4g}"
                                             if f["model"].endswith("_trend") else ""),
                    b=f"{f['b']:.6g}" if math.isfinite(f["b"]) else "inf",
                    b_over_S_today=f"{f['b'] / S_t:.4g}" if math.isfinite(f["b"]) else "inf",
                    flag=("Ricker 1/b<0 (no compensation)" if f.get("inv_b", 1) < 0 else
                          "BH b at search bound" if f.get("b_at_bound") else
                          f"HS breakpoint at {f['sbp_at_bound']} SSB" if f.get("sbp_at_bound") else ""),
                    sigma=f"{f['sigma']:.4f}", ar1=f"{ar1(f['resid']):.3f}",
                    aicc=f"{f['aicc']:.3f}", d_aicc=f"{d:.3f}", w_aicc=f"{w:.4f}",
                    S_today=f"{S_t:.6g}", S_min=f"{S_min:.6g}", S_max=f"{S_max:.6g}",
                    elast_today=f"{dd['elast']:.4f}",
                    ratio_origin=f"{dd['ratio_origin']:.4f}",
                    ratio_minobs=f"{dd['ratio_minobs']:.4f}"))
            fd = {f["model"]: f for f in fits}
            best = min(ns, key=lambda f: f["aicc"])
            best_all = min(fits, key=lambda f: f["aicc"])
            e_ns, r_ns = avg_elast(ns, S_t, S_min)
            e_all, r_all = avg_elast(fits, S_t, S_min)
            X = np.column_stack([np.ones(len(y)), np.log(S)])
            cb, cr = _ols(X, y)
            rho = ar1(cr)
            neff = len(y) * (1 - rho) / (1 + rho) if rho > 0 else len(y)
            s2 = float(cr @ cr) / max(neff - 2, 1)
            se_b = math.sqrt(s2 / float(((np.log(S) - np.log(S).mean()) ** 2).sum()))
            rps_pre = np.median(np.exp(y[~post] - np.log(S[~post]))) if (~post).sum() else math.nan
            rps_post = np.median(np.exp(y[post] - np.log(S[post]))) if post.sum() else math.nan
            comp_rows.append(dict(
                stock=st, period=per, n=len(y), yc_first=int(yrs.min()), yc_last=int(yrs.max()),
                rec_age=PP[0]["rec_age"], S_today=f"{S_t:.6g}", S_min=f"{S_min:.6g}",
                S_max=f"{S_max:.6g}", Blim=blim, Bpa=bpa,
                S_today_over_Smax=f"{S_t / S_max:.3g}", sd_lnS=f"{np.std(np.log(S)):.3f}",
                cushing_elast=f"{cb[1] - 1:.3f}", cushing_elast_se_ar1=f"{se_b:.3f}",
                S_today_over_Blim=f"{S_t / float(blim):.3g}" if blim else "",
                best_noshift=best["model"],
                best_all=best_all["model"],
                d_aicc_prop_vs_best=f"{fd['prop']['aicc'] - best['aicc']:.2f}",
                d_aicc_const_vs_best=f"{fd['const']['aicc'] - best['aicc']:.2f}",
                d_aicc_bestnoshift_vs_bestshift=(f"{best['aicc'] - min(fd[m]['aicc'] for m in SHIFTED if m in fd):.2f}"
                                                 if per == "full" and any(m in fd for m in SHIFTED) else ""),
                d_aicc_bestnoshift_vs_besttrend=f"{best['aicc'] - min(fd[m]['aicc'] for m in TREND):.2f}",
                elast_bh_trend=f"{derived(fd['bh_trend'], S_t, S_min)['elast']:.3f}",
                trend_prop_per_decade=f"{math.exp(10 * fd['prop_trend']['la'][1]):.3f}",
                w_prop=f"{w_ns[[f['model'] for f in ns].index('prop')]:.3f}",
                w_const=f"{w_ns[[f['model'] for f in ns].index('const')]:.3f}",
                sigma_bh=f"{fd['bh']['sigma']:.3f}", ar1_bh=f"{ar1(fd['bh']['resid']):.3f}",
                bh_a=f"{math.exp(np.atleast_1d(fd['bh']['la'])[0]):.5g}",
                bh_b=f"{fd['bh']['b']:.5g}", bh_b_over_S_today=f"{fd['bh']['b'] / S_t:.3g}",
                ricker_b=f"{fd['ricker']['b']:.5g}" if math.isfinite(fd['ricker']['b']) else "inf",
                hs_sbp=f"{fd['hs']['b']:.5g}", hs_flag=fd['hs'].get('sbp_at_bound', ''),
                elast_bh=f"{derived(fd['bh'], S_t, S_min)['elast']:.3f}",
                elast_bh_lo=f"{q(boot['bh'], .05):.3f}", elast_bh_hi=f"{q(boot['bh'], .95):.3f}",
                elast_ricker=f"{derived(fd['ricker'], S_t, S_min)['elast']:.3f}",
                elast_ricker_lo=f"{q(boot['ricker'], .05):.3f}", elast_ricker_hi=f"{q(boot['ricker'], .95):.3f}",
                elast_hs=f"{derived(fd['hs'], S_t, S_min)['elast']:.3f}",
                frac_boot_hs_above_bp=f"{np.mean(np.array(boot['hs']) < -0.5):.2f}",
                elast_avg=f"{e_ns:.3f}", elast_avg_lo=f"{q(boot['avg_ns'], .05):.3f}",
                elast_avg_hi=f"{q(boot['avg_ns'], .95):.3f}",
                ratio_origin_bh=f"{derived(fd['bh'], S_t, S_min)['ratio_origin']:.3f}",
                ratio_minobs_bh=f"{derived(fd['bh'], S_t, S_min)['ratio_minobs']:.3f}",
                ratio_minobs_bh_lo=f"{q(boot['bh_ratio'], .05):.3f}",
                ratio_minobs_bh_hi=f"{q(boot['bh_ratio'], .95):.3f}",
                ratio_minobs_avg=f"{r_ns:.3f}", ratio_minobs_avg_lo=f"{q(boot['avg_ns_ratio'], .05):.3f}",
                ratio_minobs_avg_hi=f"{q(boot['avg_ns_ratio'], .95):.3f}",
                elast_bh_shift=(f"{derived(fd['bh_shift'], S_t, S_min)['elast']:.3f}" if 'bh_shift' in fd else ""),
                elast_bh_shift_lo=f"{q(boot['bh_shift'], .05):.3f}" if boot['bh_shift'] else "",
                elast_bh_shift_hi=f"{q(boot['bh_shift'], .95):.3f}" if boot['bh_shift'] else "",
                a_post_over_pre_bh_shift=(f"{math.exp(fd['bh_shift']['la'][1] - fd['bh_shift']['la'][0]):.3f}"
                                          if 'bh_shift' in fd else ""),
                elast_avg_all=f"{e_all:.3f}",
                elast_avg_all_lo=f"{q(boot['avg_all'], .05):.3f}",
                elast_avg_all_hi=f"{q(boot['avg_all'], .95):.3f}",
                ratio_minobs_avg_all=f"{r_all:.3f}",
                ratio_minobs_avg_all_lo=f"{q(boot['avg_all_ratio'], .05):.3f}",
                ratio_minobs_avg_all_hi=f"{q(boot['avg_all_ratio'], .95):.3f}",
                boot_winner_share=";".join(f"{k}:{v / nboot:.2f}" for k, v in
                                           sorted(win_count.items(), key=lambda kv: -kv[1])),
                rps_median_pre_shift=f"{rps_pre:.4g}", rps_median_post_shift=f"{rps_post:.4g}",
                shift_year=shift_year))
            print(st, per, len(y), "best", best["model"], "best_all", best_all["model"],
                  "e_avg %.2f" % e_ns, "e_bh %.2f" % derived(fd['bh'], S_t, S_min)['elast'])
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sr_fits.csv", fit_rows), ("sr_compensation.csv", comp_rows),
                       ("sr_boot_draws.csv", draw_rows)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", out / name, len(rows))


if __name__ == "__main__":
    main()
