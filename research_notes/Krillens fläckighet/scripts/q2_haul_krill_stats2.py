"""Revised haul statistics (pooled ICES records 1980-91, area != 'other'), after checks:
 - concentration computed on krill per stomach per haul, hauls with uncounted krill dropped;
 - within-haul association between length-class records judged against the CRUISE marginal
   (ship x year x month), and only in cruises with >= 1 krill record, to remove region/season;
 - ICC of record presence on residuals from cruise means (hauls nested in cruises);
 - the same for target sizes only (whiting >= 20 cm, saithe 35-70 cm, pout all, cod >= 20 cm);
 - pairs of hauls in the same ICES rectangle (positions are mostly rectangle centres) within 3 days.
Usage: python3 -I q2_haul_krill_stats2.py <records_linked.csv.gz>
"""
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
df = df[df["area"] != "other"].copy()
known = df["anyprey"].astype(str).eq("True") | (df["StomachEmpty"] == df["Number"])
df = df[known & (df["Number"] > 1) & (df["year"] <= 1995)].copy()
df["kr"] = (df["occ_krill"] > 0).astype(int)
# a krill-positive record without a krill count (only these make a haul uncountable)
df["unc_pos"] = ((df["kr"] == 1) & df["krill_count"].isna()).astype(int)
df["stom_pos"] = df["Number"] * df["kr"]
df["cruise"] = df["ship"].astype(str) + "_" + df["year"].astype(str) + "_" + df["month"].astype(str)
size_rule = {"whiting": (20, 999), "saithe": (35, 70), "Norway pout": (0, 999), "cod": (20, 999)}


def icc_resid(d):
    """One-way ANOVA ICC over hauls on presence minus its cruise mean."""
    d = d.copy()
    d["res"] = d["kr"] - d.groupby("cruise")["kr"].transform("mean")
    gs = [g["res"].to_numpy() for _, g in d.groupby("tblHaulID") if len(g) >= 2]
    if len(gs) < 5:
        return float("nan"), 0, 0
    n = np.array([len(g) for g in gs], float); N = n.sum(); k = len(gs)
    m = np.array([g.mean() for g in gs]); gm = np.concatenate(gs).mean()
    msb = (n * (m - gm) ** 2).sum() / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in gs) / (N - k)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw), k, n0


def assoc(d):
    """P(another class in the same haul has krill | one has) vs the expected value under the cruise marginal."""
    num = den = exp = 0.0
    pc = d.groupby("cruise")["kr"].mean()
    for hid, g in d.groupby("tblHaulID"):
        nr = len(g); nk = g["kr"].sum()
        if nr < 2:
            continue
        num += nk * (nk - 1); den += nk * (nr - 1)
        exp += nk * (nr - 1) * pc[g["cruise"].iloc[0]]
    return (num / den, exp / den, den) if den else (float("nan"), float("nan"), 0)


for sp in ["whiting", "saithe", "Norway pout", "cod"]:
    d0 = df[df["sp"] == sp]
    pos_cr = d0.groupby("cruise")["kr"].max()
    d0 = d0[d0["cruise"].isin(pos_cr[pos_cr == 1].index)]
    for label, d in (("all sizes", d0), ("target sizes", d0[d0["L_cm"].between(*size_rule[sp])])):
        h = d.groupby("tblHaulID").agg(kr=("kr", "max"), stom=("Number", "sum"), kc=("krill_count", "sum"),
                                       unc=("unc_pos", "max"), nposrec=("kr", "sum"), stom_pos=("stom_pos", "sum"),
                                       cruise=("cruise", "first"))
        icc, k, n0 = icc_resid(d)
        a, e, npair = assoc(d)
        print(f"== {sp}, {label}: cruises {d['cruise'].nunique()}, hauls {len(h)}, records {len(d)}, stomachs {int(d['Number'].sum())}")
        print(f"   share of hauls with krill (cruises with any krill): {h['kr'].mean():.3f}; record share {d['kr'].mean():.3f}")
        print(f"   P(krill in another length class | one) = {a:.3f} vs cruise-marginal expectation {e:.3f} (pair weight {int(npair)});"
              f" ICC on cruise-centred presence = {icc:.2f} (hauls {k}, n0 {n0:.1f})")
        hp = h[h["kr"] == 1]
        print(f"   per-fish FO bounds inside krill-positive hauls (n {len(hp)}, {int(hp['stom'].sum())} stomachs):"
              f" lower {hp['nposrec'].sum()/hp['stom'].sum():.3f} (positive records / stomachs),"
              f" upper {hp['stom_pos'].sum()/hp['stom'].sum():.3f} (stomachs in positive records / stomachs)")
        hc = h[h["unc"] == 0]
        ps = (hc["kc"].fillna(0) / hc["stom"]).to_numpy()
        if ps.sum() > 0 and len(hc) >= 10:
            srt = np.sort(ps)[::-1]; top = max(1, int(round(0.1 * len(ps))))
            pp = ps[ps > 0]
            print(f"   krill per stomach (hauls with counts, n {len(hc)}): mean {ps.mean():.2f}, CV {ps.std()/ps.mean():.2f};"
                  f" top 10% of hauls ({top}) hold {srt[:top].sum()/ps.sum():.2f} of the summed per-stomach means;"
                  f" positive hauls (n {len(pp)}) median {np.median(pp):.2f}, p90 {np.percentile(pp, 90):.1f}, max {pp.max():.1f}")
        # same-rectangle pairs within 3 days
        hh = d.groupby("tblHaulID").agg(kr=("kr", "max"), rect=("rect", "first"), day=("day", "first"),
                                        cruise=("cruise", "first"))
        num = den = 0
        for (c, r), g in hh.groupby(["cruise", "rect"]):
            if len(g) < 2:
                continue
            kk = g["kr"].to_numpy(); dd = g["day"].to_numpy()
            for i in range(len(g)):
                for j in range(len(g)):
                    if i != j and kk[i] == 1 and abs(dd[i] - dd[j]) <= 3:
                        den += 1; num += kk[j]
        if den:
            print(f"   same rectangle, <= 3 days: P(j krill | i krill) = {num/den:.2f} over {den} ordered pairs; haul marginal {hh['kr'].mean():.2f}")
