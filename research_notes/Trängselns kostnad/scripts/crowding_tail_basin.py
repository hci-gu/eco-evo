"""Crowding, upper tail and crowding-vs-abundance (basin test) from tables/haul_density.csv.

Per survey (year x quarter, Q1 and Q3 only; Q2/Q4 1991-96 reported apart): Lloyd's mean crowding
over the mean (sum d^2 / sum d / mean d), area-share statistics, and the upper tail of haul
densities. Basin test: slope of ln(Lloyd/mean) on ln(survey mean density) across surveys, with a
resampling null: for each survey, haul densities are redrawn from the pooled (all-survey) relative
distribution d/mean at the survey's haul count, so the null has no abundance-dependent shape and
only the sampling noise of finite haul counts (incl. zero inflation at low catch numbers).
Lloyd's discreteness correction (Poisson sampling of individuals, sq_ columns of the haul table)
is applied where available ("lloyd_c"); the basin slope is also given on it, with a bootstrap over
surveys. Surveys with < 10 hauls are dropped. A second statistic, the elasticity of the survey's p90 haul density on the survey mean, is given
with the same null. Usage: python3 -I crowding_tail_basin.py <tables_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

tdir = sys.argv[1]
X = pd.read_csv(os.path.join(tdir, "haul_density.csv"))
TAXA = ["group", "haddock", "cod", "flatfish", "long_rough_dab", "dab", "plaice", "witch",
        "haddock_small", "cod_small"]
X["haddock_big"] = X.haddock - X.haddock_small
X["cod_big"] = X.cod - X.cod_small
X["group_big"] = X.group - X.haddock_small - X.cod_small
TAXA += ["haddock_big", "cod_big", "group_big"]
rng = np.random.default_rng(1)


def lloyd(x):
    x = np.asarray(x, float)
    s = x.sum()
    return np.nan if s <= 0 else (x ** 2).sum() / s / x.mean()


lines = []
pr = lambda s="": (print(s), lines.append(s))
Y = X[X.Quarter.isin([1, 3])].copy()
Y = Y[Y.groupby(["Year", "Quarter"]).Year.transform("size") >= 10]


def lloyd_c(x, sq):
    x = np.asarray(x, float)
    s = x.sum()
    return np.nan if s <= 0 else ((x ** 2).sum() - np.sum(sq)) / s / x.mean()
# ---- anchor: 2021-25
pr("Anchor 2021-25 (Q1+Q3; earlier note: haddock median 4.36 (2.64-9.53), cod 2.47, guild B 3.06)")
for t in ["group", "haddock", "cod", "plaice", "dab", "long_rough_dab"]:
    v = [lloyd(g[t]) for _, g in Y[Y.Year >= 2021].groupby(["Year", "Quarter"])]
    vc = [lloyd_c(g[t], g["sq_" + t]) for _, g in Y[Y.Year >= 2021].groupby(["Year", "Quarter"])]
    pr(f"  {t:16s} median {np.nanmedian(v):.2f} range {np.nanmin(v):.2f}-{np.nanmax(v):.2f}; "
       f"discreteness-corrected median {np.nanmedian(vc):.2f}")
# ---- per-survey table
rows = []
for (y, q), g in Y.groupby(["Year", "Quarter"]):
    for t in TAXA:
        x = g[t].values
        mu = x.mean()
        if mu <= 0:
            continue
        xs = np.sort(x)[::-1]
        cum = np.cumsum(xs) / xs.sum()
        lc = lloyd_c(x, g["sq_" + t].values) if "sq_" + t in g else np.nan
        rows.append(dict(taxon=t, year=y, quarter=q, n=len(x), mean=mu, lloyd=lloyd(x), lloyd_c=lc,
                         p90=np.quantile(x, 0.9), p99=np.quantile(x, 0.99), max=x.max(),
                         occupied=(x > 0).mean(),
                         hauls_for_half=np.searchsorted(cum, 0.5) + 1))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(tdir, "survey_crowding.csv"), index=False)
pr()
pr("Crowding by period (median over surveys of Lloyd/mean), Q1 / Q3")
for t in TAXA:
    s = []
    for q in (1, 3):
        for a, b in ((1991, 2000), (2001, 2010), (2011, 2020), (2021, 2025)):
            r = R[(R.taxon == t) & (R.quarter == q) & R.year.between(a, b)]
            s.append(f"{r.lloyd.median():5.2f}")
    pr(f"  {t:16s} Q1 " + " ".join(s[:4]) + " | Q3 " + " ".join(s[4:]))
pr("  (columns: 1991-2000, 2001-10, 2011-20, 2021-25)")
pr("Discreteness-corrected crowding, same layout")
for t in ["group", "haddock", "cod", "flatfish", "plaice", "dab", "long_rough_dab", "witch"]:
    s_ = []
    for q in (1, 3):
        for a, b in ((1991, 2000), (2001, 2010), (2011, 2020), (2021, 2025)):
            r = R[(R.taxon == t) & (R.quarter == q) & R.year.between(a, b)]
            s_.append(f"{r.lloyd_c.median():5.2f}")
    pr(f"  {t:16s} Q1 " + " ".join(s_[:4]) + " | Q3 " + " ".join(s_[4:]))
# ---- upper tail
pr()
pr("Upper tail of haul densities (t/km2, q = 1, nominal swept area), Q1 and Q3 1991-2025")
pr(f"  {'taxon':16s} {'Q':>2s} {'hauls':>5s} {'mean':>6s} {'p99':>6s} {'max':>6s} "
   f"{'>20':>5s} {'>50':>5s} {'>100':>5s} {'bio>20':>7s} {'bio>50':>7s} {'max/svmean':>10s}")
tail = []
for t in ["group", "haddock", "haddock_small", "haddock_big", "cod", "cod_big", "flatfish", "plaice", "dab"]:
    for q in (1, 3):
        g = Y[Y.Quarter == q]
        x = g[t].values
        rel = (g[t] / g.groupby(["Year"])[t].transform("mean")).values
        d = dict(taxon=t, quarter=q, hauls=len(x), mean=x.mean(), p99=np.quantile(x, 0.99),
                 max=x.max(), n_gt20=int((x > 20).sum()), n_gt50=int((x > 50).sum()),
                 n_gt100=int((x > 100).sum()), bio_gt20=x[x > 20].sum() / x.sum(),
                 bio_gt50=x[x > 50].sum() / x.sum(), max_rel=np.nanmax(rel))
        tail.append(d)
        pr(f"  {t:16s} {q:2d} {d['hauls']:5d} {d['mean']:6.2f} {d['p99']:6.1f} {d['max']:6.1f} "
           f"{d['n_gt20']:5d} {d['n_gt50']:5d} {d['n_gt100']:5d} {d['bio_gt20']:7.3f} {d['bio_gt50']:7.3f} {d['max_rel']:10.1f}")
pd.DataFrame(tail).to_csv(os.path.join(tdir, "haul_tail.csv"), index=False)
top = Y.sort_values("group", ascending=False).head(12)
pr()
pr("Twelve densest hauls (model group), t/km2")
pr(top[["Year", "Quarter", "Month", "StatRec", "Depth", "group", "haddock", "haddock_small", "cod",
         "flatfish"]].round(1).to_string(index=False))
# ---- basin test
pr()
pr("Basin test: slope of ln(Lloyd/mean) and ln(p90) on ln(survey mean) across surveys,")
pr("observed vs a resampling null (pooled relative distribution, survey haul counts; 1000 draws)")
basin = []
for t in ["group", "group_big", "haddock", "haddock_big", "cod", "cod_big", "flatfish", "plaice",
          "dab", "long_rough_dab"]:
    for q in (1, 3):
        g = Y[Y.Quarter == q]
        surv = [(k, s[t].values) for k, s in g.groupby("Year") if s[t].sum() > 0]
        mu = np.array([x.mean() for _, x in surv])
        ll = np.array([lloyd(x) for _, x in surv])
        p90 = np.array([np.quantile(x, 0.9) for _, x in surv])
        lm = np.log(mu)
        ok = p90 > 0
        b_obs = np.polyfit(lm, np.log(ll), 1)[0]
        e90 = np.polyfit(lm[ok], np.log(p90[ok]), 1)[0] if ok.sum() > 5 else np.nan
        pool = np.concatenate([x / x.mean() for _, x in surv])
        bn, en = [], []
        for _ in range(1000):
            sims = [rng.choice(pool, size=len(x), replace=True) * m for (_, x), m in zip(surv, mu)]
            mm = np.array([s.mean() for s in sims])
            okm = mm > 0
            l2 = np.array([lloyd(s) for s in sims])
            bn.append(np.polyfit(np.log(mm[okm]), np.log(l2[okm]), 1)[0])
            q90 = np.array([np.quantile(s, 0.9) for s in sims])
            o2 = okm & (q90 > 0)
            en.append(np.polyfit(np.log(mm[o2]), np.log(q90[o2]), 1)[0] if o2.sum() > 5 else np.nan)
        bn, en = np.array(bn), np.array(en)
        bc, bc_lo, bc_hi = np.nan, np.nan, np.nan
        if "sq_" + t in g:
            lc = np.array([lloyd_c(s[t].values, s["sq_" + t].values) for k, s in g.groupby("Year")
                           if s[t].sum() > 0])
            okc = lc > 0
            bc = np.polyfit(lm[okc], np.log(lc[okc]), 1)[0]
            bs = []
            idx = np.where(okc)[0]
            for _ in range(1000):
                j = rng.choice(idx, size=len(idx), replace=True)
                bs.append(np.polyfit(lm[j], np.log(lc[j]), 1)[0])
            bc_lo, bc_hi = np.quantile(bs, [0.025, 0.975])
        d = dict(taxon=t, quarter=q, surveys=len(surv), mean_range=mu.max() / mu.min(),
                 slope_lloyd=b_obs, null_lloyd_p2_5=np.nanquantile(bn, 0.025),
                 null_lloyd_med=np.nanmedian(bn), null_lloyd_p97_5=np.nanquantile(bn, 0.975),
                 p_lloyd=(bn <= b_obs).mean(), elast_p90=e90, null_p90_med=np.nanmedian(en),
                 null_p90_p2_5=np.nanquantile(en, 0.025), null_p90_p97_5=np.nanquantile(en, 0.975),
                 slope_lloyd_corrected=bc, boot_lo=bc_lo, boot_hi=bc_hi)
        basin.append(d)
        pr(f"  {t:15s} Q{q} n={d['surveys']:2d} range x{d['mean_range']:5.1f}  "
           f"Lloyd slope {b_obs:+.2f} (null {d['null_lloyd_med']:+.2f}, 95% {d['null_lloyd_p2_5']:+.2f}..{d['null_lloyd_p97_5']:+.2f}, "
           f"P(null<=obs) {d['p_lloyd']:.2f}) | p90 elast {e90:+.2f} (null {d['null_p90_med']:+.2f}, "
           f"{d['null_p90_p2_5']:+.2f}..{d['null_p90_p97_5']:+.2f}) | corrected Lloyd slope {bc:+.2f} "
           f"(boot {bc_lo:+.2f}..{bc_hi:+.2f})")
pd.DataFrame(basin).to_csv(os.path.join(tdir, "basin_test.csv"), index=False)
open(os.path.join(tdir, "crowding_tail_basin_stdout.txt"), "w").write("\n".join(lines) + "\n")
