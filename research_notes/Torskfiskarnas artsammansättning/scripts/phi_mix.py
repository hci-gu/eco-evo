"""Box gadoid biomass shares and benthivore share phi from the haul table of gadoid_mix.py.

Usage: python3 -I phi_mix.py <tables_dir> <scores.csv> <catchability.csv>

scores.csv:       species,s_min,s_c,s_max          (benthivore score, morphology based)
catchability.csv: species,q_low,q_c,q_high         (GOV catchability relative to haddock = 1;
                                                   corrected biomass = CPUE / q; q_low -> more biomass)
In this study catchability.csv is all ones (so the corr=raw and corr=q rows are identical); the q
corrections are applied per length class by catchability.py, which writes tables/q_<variant>/, and
this script is then run on those directories.
Ling and tusk add-on (outside the GOV): extra share of the corrected group biomass, row
"ling_tusk_addon" in catchability.csv with q_low/q_c/q_high read as share high/central/low.

Estimators, each per quarter, pooled = mean of the Q1 and Q3 estimates:
- rect:  arithmetic haul mean over the four box rectangles 44G0-45G1 (120 hauls; shallow-biased);
- strict: hauls inside the grid's lat/lon bounds (41 hauls);
- depth: depth-stratified, bin means over region hauls (Skagerrak + box rectangles) weighted by the
  grid's area per depth bin (maps/depth.npz); the empty >= 300 m bin takes the 200-300 m means;
- depth_box: same with box-rectangle hauls only.
"""
import os
import sys

import numpy as np
import pandas as pd

tab, scores_csv, q_csv = sys.argv[1:4]
H = pd.read_csv(os.path.join(tab, "haul_species_kgh.csv"))
area = pd.read_csv(os.path.join(tab, "box_area_by_depth.csv"), index_col=0).iloc[:, 0]
S = pd.read_csv(scores_csv, index_col=0)
Q = pd.read_csv(q_csv, index_col=0)
addon = Q.loc["ling_tusk_addon"] if "ling_tusk_addon" in Q.index else None
Q = Q.drop(index="ling_tusk_addon", errors="ignore")

ZOOPL = ["Norway pout", "blue whiting"]
ALL = list(S.index)
CORE = [s for s in ALL if s not in ZOOPL]
BINS = ["<100", "100-200", "200-300", ">=300"]


def est(d, how):
    if how in ("rect", "strict"):
        return d[ALL].mean()
    g = d.groupby("dbin")[ALL].mean().reindex(BINS)
    g.loc[">=300"] = g.loc["200-300"]
    g = g.fillna(0.0)
    return g.mul(area.reindex(BINS), axis=0).sum()


SEL = {"rect": H.box, "strict": H.box_strict, "depth": H.region, "depth_box": H.box}


def estimate(how, quarter, frame=H):
    sel = SEL[how] if frame is H else frame["_sel_" + how]
    if quarter == "pooled":
        return 0.5 * (estimate(how, 1, frame) + estimate(how, 3, frame))
    return est(frame[sel & (frame.Quarter == quarter)], how)


def shares(mu, species, qcol=None):
    m = mu[species].copy()
    if qcol is not None:
        m = m / Q.loc[species, qcol]
    return m / m.sum()


def phi(w, scol, add=0.0):
    """phi over shares w; add = extra ling+tusk share (scored with their own scores)."""
    p = (w * S.loc[w.index, scol]).sum()
    if add > 0:
        lt = 0.5 * (S.loc["ling", scol] + S.loc["tusk", scol])
        p = (1 - add) * p + add * lt
    return p


rows, shrows = [], []
for how in SEL:
    for qu in [1, 3, "pooled"]:
        mu = estimate(how, qu)
        n = int((SEL[how] & ((H.Quarter == qu) if qu != "pooled" else True)).sum())
        for grp, sp in [("core", CORE), ("+NoP", CORE + ["Norway pout"]), ("+NoP+BW", ALL)]:
            for corr, qcol in [("raw", None), ("q", "q_c")]:
                w = shares(mu, sp, qcol)
                add = 0.0 if corr == "raw" or addon is None else addon["q_c"]
                rec = dict(estimator=how, quarter=qu, n=n, group=grp, corr=corr,
                           phi_min=phi(w, "s_min", add), phi_c=phi(w, "s_c", add), phi_max=phi(w, "s_max", add))
                if corr == "q" and addon is not None:
                    # widest: low q (more cod/saithe/hake) + high add-on, and the reverse
                    wl, wh = shares(mu, sp, "q_low"), shares(mu, sp, "q_high")
                    rec["phi_q_lowq"] = phi(wl, "s_c", addon["q_low"])
                    rec["phi_q_highq"] = phi(wh, "s_c", addon["q_high"])
                rows.append(rec)
                if grp in ("core", "+NoP+BW"):
                    shrows.append(dict(estimator=how, quarter=qu, n=n, group=grp, corr=corr,
                                       **(100 * w).round(2).to_dict()))
P = pd.DataFrame(rows)
SH = pd.DataFrame(shrows)
P.to_csv(os.path.join(tab, "phi_table.csv"), index=False)
SH.to_csv(os.path.join(tab, "share_table.csv"), index=False)

# bootstrap over hauls (stratified by quarter) for the depth and strict estimators, central scores
rng = np.random.default_rng(7)
boot = []
for how in ["depth", "strict", "rect"]:
    base = H[SEL[how]].copy()
    vals = {g: [] for g in ["core", "+NoP"]}
    for _ in range(1000):
        d = pd.concat([g.sample(len(g), replace=True, random_state=rng.integers(1 << 31))
                       for _, g in base.groupby(["Quarter", "dbin"])])
        mu = 0.5 * (est(d[d.Quarter == 1], how) + est(d[d.Quarter == 3], how))
        vals["core"].append(phi(shares(mu, CORE), "s_c"))
        vals["+NoP"].append(phi(shares(mu, CORE + ["Norway pout"]), "s_c"))
    for g, v in vals.items():
        boot.append(dict(estimator=how, group=g, p05=np.quantile(v, 0.05), p50=np.median(v),
                         p95=np.quantile(v, 0.95)))
B = pd.DataFrame(boot)
B.to_csv(os.path.join(tab, "phi_bootstrap.csv"), index=False)

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
main = ["haddock", "whiting", "saithe", "cod", "hake", "ling", "pollack", "poor cod", "fourbeard rockling",
        "greater forkbeard", "silvery pout", "Norway pout", "blue whiting"]
print(SH[[c for c in ["estimator", "quarter", "n", "group", "corr"] + main if c in SH]].to_string())
print(P.round(3).to_string())
print(B.round(3).to_string())
