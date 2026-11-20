"""Onsrud et al. 2004 (ICES JMS 61:1278, doi:10.1016/j.icesjms.2004.09.005) Table 3, transcribed by hand
from the rendered page (p. 1287). Derived statistics on krill counts per stomach and digestion state.

Run: python3 -I inl_onsrud_counts.py > ../tables/inl_onsrud_counts_stdout.txt

What it does (all [D]):
1. Candidate sample sizes n per cell: the smallest n (5..120) for which BOTH the % with krill and the % empty
   are k/n to two decimals. Table 3 gives no n, so this is a guess, not a datum.
2. Feasibility test: could "number of krill stomach^-1" be a mean over ALL stomachs (zeros included)?
   With a share p0 = 1 - FO of zeros, any distribution has SD >= m * sqrt(p0 / (1 - p0)).
   If the reported SD is below that bound, the statistic must be conditional on krill being present.
3. Conditional variance/mean, and the unconditional mean and variance rebuilt from FO:
   E[X] = FO m_c, Var[X] = FO (s_c^2 + m_c^2) - (FO m_c)^2.
4. A zero-truncated negative binomial (or zero-truncated Poisson when the positives are not
   overdispersed) fitted to (m_c, s_c); its implied P(0) is compared with the observed 1 - FO.
   P(0) observed > implied = more krill-free stomachs than the positives' spread predicts.
   When s_c exceeds the log-series SD (the k -> 0 limit of the zero-truncated NB), no NB fits: the
   positives have a heavier tail than any NB (a mixture of many 1-2 krill stomachs and a few large ones).
   FO here includes empty stomachs and stomachs with other prey only among the krill-free (an assumption:
   Table 3 does not define the denominator).
"""
import math

# species, month, period: fullness m (sd), krill FO %, krill per stomach m (sd), digestion m (sd), empty %
T3 = [
    ("whiting", "Nov2000", "day",   2.67, 1.37, 42.31, 1.82, 0.98, 2.89, 0.91, 25.00),
    ("whiting", "Nov2000", "night", 3.00, 1.20, 35.19, 3.47, 3.47, 1.88, 0.91, 14.81),
    ("whiting", "Mar2001", "day",   3.28, 1.03, 75.51, 3.95, 4.28, 3.03, 1.17, 8.16),
    ("whiting", "Mar2001", "night", 3.25, 0.97, 79.69, 5.92, 11.39, 2.77, 1.09, 6.25),
    ("whiting", "Jun2001", "day",   3.08, 1.21, 42.86, 2.17, 2.33, 3.32, 1.25, 7.14),
    ("whiting", "Jun2001", "night", 2.23, 1.03, 34.62, 1.44, 1.01, 2.92, 1.75, 26.92),
    ("pout",    "Nov2000", "day",   3.27, 1.42, 59.09, 1.54, 0.78, 3.45, 1.00, 22.73),
    ("pout",    "Nov2000", "night", 2.93, 1.08, 44.07, 1.54, 0.71, 3.72, 0.84, 15.25),
    ("pout",    "Mar2001", "day",   4.00, 1.31, 75.00, 2.17, 1.33, 3.69, 0.85, 0.00),
    ("pout",    "Mar2001", "night", 3.62, 0.86, 47.62, 2.10, 1.20, 3.41, 1.26, 0.00),
    ("herring", "Nov2000", "day",   2.91, 0.84, 31.03, 1.90, 1.10, 4.00, 0.69, 0.00),
    ("herring", "Nov2000", "night", 2.29, 0.46, 17.24, 2.20, 0.83, 4.08, 0.79, 0.00),
    ("herring", "Mar2001", "day",   3.43, 0.89, 50.00, 5.13, 4.64, 5.00, 0.00, 6.25),
    ("herring", "Mar2001", "night", 2.70, 0.47, 20.00, 1.50, 0.70, 2.79, 1.40, 0.00),
]


def candidate_n(fo_pct, empty_pct, nmax=120):
    out = []
    for n in range(5, nmax + 1):
        ok = True
        for pct in (fo_pct, empty_pct):
            k = round(pct / 100 * n)
            if abs(100 * k / n - pct) > 0.006:
                ok = False
        if ok:
            out.append(n)
    return out


def zt_moments(mu, k):
    """Zero-truncated NB (k=None: Poisson): P0, conditional mean, conditional variance."""
    if k is None:
        p0 = math.exp(-mu)
        ex2 = mu + mu * mu
    else:
        p0 = (k / (k + mu)) ** k
        ex2 = mu + mu * mu * (1 + 1 / k)
    mc = mu / (1 - p0)
    vc = ex2 / (1 - p0) - mc * mc
    return p0, mc, vc


def solve_mu(mc_target, k):
    lo, hi = 1e-9, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if zt_moments(mid, k)[1] < mc_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def logseries_sd(mc):
    """SD of the log-series distribution with mean mc: the k -> 0 limit, i.e. the largest conditional SD
    any zero-truncated NB can have at that conditional mean."""
    lo, hi = 1e-12, 1 - 1e-12
    for _ in range(200):
        th = 0.5 * (lo + hi)
        m = -th / ((1 - th) * math.log(1 - th))
        if m < mc:
            lo = th
        else:
            hi = th
    th = 0.5 * (lo + hi)
    m = -th / ((1 - th) * math.log(1 - th))
    return math.sqrt(m * (1 / (1 - th) - m))


def fit_zt(mc, sc):
    """Return (model, mu, k, P0). NB if the positives are more dispersed than a zero-truncated Poisson;
    no fit if they are more dispersed than any zero-truncated NB (log-series limit)."""
    lsd = logseries_sd(mc)
    if sc > lsd:
        return "beyond ZTNB family (log-series limit SD=%.2f)" % lsd, float("nan"), None, float("nan")
    mu_p = solve_mu(mc, None)
    p0_p, _, vc_p = zt_moments(mu_p, None)
    if sc * sc <= vc_p:
        return "ZTPoisson (positives not overdispersed; SD_ZTP=%.2f)" % math.sqrt(vc_p), mu_p, None, p0_p
    # conditional variance rises as k falls; bisect on log k
    lo, hi = math.log(1e-4), math.log(1e4)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        k = math.exp(mid)
        mu = solve_mu(mc, k)
        vc = zt_moments(mu, k)[2]
        if vc > sc * sc:
            lo = mid
        else:
            hi = mid
    k = math.exp(0.5 * (lo + hi))
    mu = solve_mu(mc, k)
    return "ZTNB", mu, k, zt_moments(mu, k)[0]


print("Onsrud et al. 2004 Table 3: krill per stomach, derived statistics [D]")
print("FO = share of stomachs with krill structures; m_c, s_c = reported mean and SD of krill per stomach")
print()
hdr = ("species month period | n candidates | FO | m_c s_c | SDmin if all-stomach -> verdict | "
       "var/mean cond | E[X] all | var/mean all | fit, mu, k | P0 implied vs observed (1-FO)")
print(hdr)
for sp, mo, per, fm, fs, fo, mc, sc, dm, ds, em in T3:
    p_fo = fo / 100
    p0 = 1 - p_fo
    sdmin = mc * math.sqrt(p0 / p_fo)
    verdict = "conditional" if sc < sdmin else "either"
    vm_c = sc * sc / mc
    ex = p_fo * mc
    vx = p_fo * (sc * sc + mc * mc) - ex * ex
    model, mu, k, p0_imp = fit_zt(mc, sc)
    ns = candidate_n(fo, em)
    kstr = "-" if k is None else "%.3f" % k
    print("%-7s %s %-5s | %-14s | %.2f | %.2f %.2f | %.2f -> %-11s | %5.2f | %.2f | %5.2f | %s mu=%.3f k=%s | %.3f vs %.3f"
          % (sp, mo, per, ",".join(str(x) for x in ns[:3]) or "none<=120", p_fo, mc, sc, sdmin, verdict,
             vm_c, ex, vx / ex, model, mu, kstr, p0_imp, p0))

print()
print("Digestion state of krill (1 undigested ... 5 fully digested), mean (SD); fullness 1 = empty ... 5 = full")
for sp, mo, per, fm, fs, fo, mc, sc, dm, ds, em in T3:
    print("%-7s %s %-5s digestion %.2f (%.2f)  fullness %.2f (%.2f)  empty %.1f %%" % (sp, mo, per, dm, ds, fm, fs, em))

print()
print("Night minus day digestion (negative = fresher krill at night):")
for sp in ("whiting", "pout", "herring"):
    for mo in ("Nov2000", "Mar2001", "Jun2001"):
        d = {r[2]: r for r in T3 if r[0] == sp and r[1] == mo}
        if "day" in d and "night" in d:
            print("  %-7s %s  %+.2f" % (sp, mo, d["night"][8] - d["day"][8]))
