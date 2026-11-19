"""Weight-at-length (condition) of individual fish against the local (haul) density of their own
species and of the whole benthivore group, Skagerrak + Kattegat NS-IBTS 1991-2025, Q1 and Q3.

Inputs: <lek_data_dir>/{datras_1991_99,datras,datras_2001q1_ca}/ca.csv (individual records,
IndWgt in g) and <tables>/haul_density_all.csv (from build_haul_density.py ... all).
Stage 1, per species x quarter: ln W ~ cubic in ln L + survey (year) effect + sex; residual r =
the fish's condition relative to fish of its length in the same survey.
Stage 2, per species x quarter, hauls with >= 3 weighed fish: haul mean r ~ beta ln(d_own + delta)
+ survey (year) effect + rectangle effect (additive, not survey x rectangle) + quadratic in ln
depth, weighted by the number of fish
(delta = 0.05 x the species' overall mean density, so empty-of-own-species hauls do not occur
here anyway: a weighed fish implies a catch). beta = elasticity of weight-at-length on local own
density, from the contrast between hauls after removing year and rectangle means.
Variant (e) "own_site": a SITE effect instead of the rectangle effect. NS-IBTS repeats tows at
near-fixed positions; a site is a cluster of shoot positions within 2 km (single-linkage on a
1-km grid). beta then comes from the same site in different years at different own density,
which removes station-scale habitat quality (dense hauls recur within ~0.6 km, dense_haul_sites).
Variants: (a) own species density, (b) whole-group density, (c) no rectangle effect (pooled
spatial contrast), (d) adults only (haddock >= 30 cm, cod >= 35 cm, plaice >= 27 cm, witch >= 28 cm).
Weight: total body weight (gonads included), so Q1 (pre-spawning) mixes gonad growth; Q3 is the
feeding season and is the primary test.
Usage: python3 -I condition_vs_local_density.py <lek_data_dir> <tables_dir>
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
lek, tdir = sys.argv[1:3]
SP = {126437: "haddock", 126436: "cod", 127143: "plaice", 127136: "witch"}
ADULT = {"haddock": 30, "cod": 35, "plaice": 27, "witch": 28}
KEY = ["Year", "Quarter", "Country", "Ship", "HaulNo", "StNo"]
ca = pd.concat([pd.read_csv(os.path.join(lek, d, "ca.csv"), low_memory=False,
                            dtype={"StNo": str, "AreaCode": str, "LngtCode": str, "Sex": str})
                for d in ("datras_1991_99", "datras", "datras_2001q1_ca")], ignore_index=True)
for c in ("Ship", "StNo", "Country"):
    ca[c] = ca[c].astype(str).str.strip()
ca = ca[ca.Valid_Aphia.isin(SP) & (ca.IndWgt > 0) & ca.Quarter.isin([1, 3])].copy()
code = ca.LngtCode.astype(str).str.strip()
ca["L"] = np.where(code == "1", ca.LngtClass + 0.5,
                   np.where(code == "0", ca.LngtClass / 10 + 0.25,
                            np.where(code == ".", ca.LngtClass / 10 + 0.05, np.nan)))
ca = ca[np.isfinite(ca.L) & (ca.L > 5)]
ca["sp"] = ca.Valid_Aphia.map(SP)
ca["lnW"] = np.log(ca.IndWgt)
ca["lnL"] = np.log(ca.L)
ca["sex"] = ca.Sex.where(ca.Sex.isin(["M", "F"]), "U")
ca = ca.drop_duplicates()
H = pd.read_csv(os.path.join(tdir, "haul_density_all.csv"), dtype={"StNo": str})
for c in ("Ship", "StNo", "Country"):
    H[c] = H[c].astype(str).str.strip()
# sites: cluster shoot positions within 2 km (union-find on pairwise distances)
pos = H[["ShootLat", "ShootLong"]].to_numpy()
yk = pos[:, 0] * 111.2
xk = pos[:, 1] * 111.2 * np.cos(np.radians(57.8))
par = list(range(len(H)))


def find(i):
    while par[i] != i:
        par[i] = par[par[i]]
        i = par[i]
    return i


order = np.argsort(yk)
for ii, i in enumerate(order):
    for j in order[ii + 1:]:
        if yk[j] - yk[i] > 2.0:
            break
        if np.hypot(xk[j] - xk[i], yk[j] - yk[i]) <= 2.0:
            par[find(i)] = find(j)
H["site"] = [find(i) for i in range(len(H))]
ca = ca.merge(H[KEY + ["StatRec", "Depth", "group", "haddock", "cod", "plaice", "witch", "site"]],
              on=KEY, how="inner")
lines = []
pr = lambda s="": (print(s), lines.append(s))
pr(f"weighed fish matched to valid hauls: {len(ca)}")
rows = []
for (sp, q), g in ca.groupby(["sp", "Quarter"]):
    g = g.copy()
    # outlier screen: |resid| > 0.5 in ln W from a plain length fit
    f0 = smf.ols("lnW ~ lnL + I(lnL**2)", g).fit()
    g = g[np.abs(f0.resid) < 0.5]
    f1 = smf.ols("lnW ~ lnL + I(lnL**2) + I(lnL**3) + C(Year) + C(sex)", g).fit()
    g["r"] = f1.resid
    for variant in ("own", "group", "own_norect", "own_adult", "own_site"):
        gg = g if variant != "own_adult" else g[g.L >= ADULT[sp]]
        hv = gg.groupby(KEY).agg(r=("r", "mean"), n=("r", "size"), StatRec=("StatRec", "first"),
                                 Depth=("Depth", "first"), own=(sp, "first"),
                                 grp=("group", "first"), site=("site", "first")).reset_index()
        hv = hv[(hv.n >= 3) & (hv.Depth > 0)]
        dcol = "grp" if variant == "group" else "own"
        hv = hv[hv[dcol] > 0]
        delta = 0.05 * H[sp if dcol == "own" else "group"].mean()
        hv["x"] = np.log(hv[dcol] + delta)
        hv["ld"] = np.log(hv.Depth)
        if variant == "own_site":
            hv = hv[hv.groupby("site").site.transform("size") >= 3]  # sites fished in >= 3 years
            form = "r ~ x + C(Year) + C(site)"
        else:
            form = "r ~ x + C(Year) + ld + I(ld**2)" + ("" if variant == "own_norect" else " + C(StatRec)")
        if len(hv) < 30:
            continue
        m = smf.wls(form, hv, weights=hv.n).fit(cov_type="HC1")
        sdx = np.sqrt(np.cov(hv.x, aweights=hv.n))
        # within-survey-and-rectangle spread of x (what identifies beta)
        wx = hv.x - hv.groupby(["site" if variant == "own_site" else "StatRec"]).x.transform("mean")
        rows.append(dict(species=sp, quarter=q, variant=variant, hauls=len(hv), fish=int(hv.n.sum()),
                         beta=m.params["x"], se=m.bse["x"], lo=m.params["x"] - 1.96 * m.bse["x"],
                         hi=m.params["x"] + 1.96 * m.bse["x"], sd_lnd=sdx,
                         sd_lnd_within=float(np.sqrt(np.average(wx ** 2, weights=hv.n))),
                         per_doubling_pct=100 * (2 ** m.params["x"] - 1),
                         sites=hv.site.nunique(),
                         resid_sd_fish=float(np.std(g.r))))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(tdir, "condition_vs_density.csv"), index=False)
pr("beta = d ln(weight-at-length) / d ln(local density); per doubling in %; 95% CI (HC1)")
pr(R[["species", "quarter", "variant", "hauls", "sites", "fish", "beta", "lo", "hi", "per_doubling_pct",
      "sd_lnd", "sd_lnd_within"]].round(4).to_string(index=False))
# group-weighted summary over the Q3 own-species within-rectangle estimates
share = {"haddock": 0.62, "cod": 0.15, "plaice": 0.06, "witch": 0.04}
open(os.path.join(tdir, "condition_vs_density_stdout.txt"), "w").write("\n".join(lines) + "\n")
