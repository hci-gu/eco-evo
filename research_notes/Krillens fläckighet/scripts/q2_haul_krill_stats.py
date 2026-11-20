"""Haul-level krill statistics from the ICES stomach records (North Sea, Skagerrak, Kattegat;
area != 'other'), to separate patch occupancy (which hauls have krill) from diet choice
(how many fish in a krill haul ate krill).

(a) Pooled records (Number > 1; 1980-91): occ_krill is binary per record (length class x haul).
    Per haul: any krill in any length class. Share of hauls with krill; among hauls with >= 2
    length-class records, the probability that a second class has krill given one has (vs the
    marginal), i.e. a within-haul association; concentration of krill counts across hauls.
(b) Single-stomach records (Number == 1): per haul FO; share of hauls with any krill; FO inside
    krill-positive hauls; ANOVA intraclass correlation of krill presence (hauls with >= 3 fed fish).

Only records with known content are used (anyprey True or StomachEmpty == Number).
Usage: python3 -I q2_haul_krill_stats.py <records_linked.csv.gz>
"""
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
df = df[df["area"] != "other"].copy()
known = df["anyprey"].astype(str).eq("True") | (df["StomachEmpty"] == df["Number"])
df = df[known]
df["kr"] = (df["occ_krill"] > 0).astype(int)


def icc_anova(groups):
    """One-way ANOVA ICC for unbalanced groups (binary or continuous)."""
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    N = n.sum()
    means = np.array([np.mean(g) for g in groups])
    grand = np.concatenate(groups).mean()
    ssb = (n * (means - grand) ** 2).sum()
    ssw = sum(((np.asarray(g) - m) ** 2).sum() for g, m in zip(groups, means))
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw), n0


print("=== (a) pooled records, 1980-91")
p = df[(df["Number"] > 1) & (df["year"] <= 1995)]
for sp in ["whiting", "saithe", "Norway pout", "cod"]:
    d = p[p["sp"] == sp]
    if d.empty:
        continue
    h = d.groupby("tblHaulID").agg(n_rec=("kr", "size"), any_kr=("kr", "max"), n_kr=("kr", "sum"),
                                   stom=("Number", "sum"), kcount=("krill_count", "sum"), lat=("lat", "mean"))
    print(f"-- {sp}: records {len(d)}, hauls {len(h)}, stomachs {int(d['Number'].sum())}")
    print(f"   share of hauls with krill in any length class: {h['any_kr'].mean():.3f}")
    print(f"   share of records (length class x haul) with krill: {d['kr'].mean():.3f}")
    m = h[h["n_rec"] >= 2]
    if len(m) >= 5:
        # pairwise: P(other class has krill | this class has krill) within haul
        num = 0.0
        den = 0.0
        for _, r in m.iterrows():
            nk, nr = r["n_kr"], r["n_rec"]
            num += nk * (nk - 1)
            den += nk * (nr - 1)
        cond = num / den if den else float("nan")
        groups = [g["kr"].to_numpy() for _, g in d[d["tblHaulID"].isin(m.index)].groupby("tblHaulID")]
        icc, n0 = icc_anova(groups)
        print(f"   hauls with >= 2 classes: {len(m)}; P(krill in another class | krill in one) = {cond:.3f}"
              f" vs marginal {d[d['tblHaulID'].isin(m.index)]['kr'].mean():.3f}; ICC(record presence) = {icc:.2f} (n0 {n0:.1f})")
    kc = h["kcount"].fillna(0).to_numpy()
    per_st = kc / h["stom"].to_numpy()
    if kc.sum() > 0:
        srt = np.sort(kc)[::-1]
        top = max(1, int(round(0.1 * len(kc))))
        print(f"   krill counted: {int(kc.sum())}; share in top 10% of hauls ({top}): {srt[:top].sum()/kc.sum():.2f};"
              f" krill per stomach over hauls: mean {per_st.mean():.2f}, CV {per_st.std()/per_st.mean():.2f};"
              f" among krill hauls median {np.median(per_st[per_st>0]):.2f}, p90 {np.percentile(per_st[per_st>0],90):.1f}")

print("=== (b) single-stomach records")
s = df[(df["Number"] == 1)]
for sp in ["saithe", "whiting", "cod", "hake", "horse mackerel", "mackerel", "haddock"]:
    d = s[(s["sp"] == sp) & (s["fed"] > 0)]
    if d.empty:
        continue
    h = d.groupby("tblHaulID")["kr"].agg(["size", "mean", "max"])
    hk = h[h["size"] >= 3]
    print(f"-- {sp}: fed stomachs {len(d)}, hauls {len(h)}, FO overall {d['kr'].mean():.3f},"
          f" years {sorted(d['year'].unique())}")
    print(f"   share of hauls with any krill: {h['max'].mean():.3f} (hauls with >= 3 fed: {hk['max'].mean() if len(hk) else float('nan'):.3f}, n {len(hk)})")
    pos = hk[hk["max"] == 1]
    if len(pos):
        print(f"   inside krill-positive hauls (>= 3 fed): mean FO {pos['mean'].mean():.3f}, median {pos['mean'].median():.3f},"
              f" range {pos['mean'].min():.2f}-{pos['mean'].max():.2f}, n hauls {len(pos)}, fish {int(pos['size'].sum())}")
    if len(hk) >= 5 and d["kr"].sum() > 0:
        groups = [g["kr"].to_numpy() for _, g in d[d["tblHaulID"].isin(hk.index)].groupby("tblHaulID")]
        icc, n0 = icc_anova(groups)
        print(f"   ICC(krill presence, fed fish, hauls >= 3 fed) = {icc:.2f} (n0 {n0:.1f})")
