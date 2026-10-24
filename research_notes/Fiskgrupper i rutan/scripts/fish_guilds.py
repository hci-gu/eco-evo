"""Fish biomass in the Mareld box by feeding group (planktivores, benthivores, predators).

Usage: python3 -I fish_guilds.py <extract_dir> <hauls_region_csv> <guilds_csv> <depth_npz> <out_dir>

Inputs (untrusted data, only parsed): the DATRAS NS-IBTS extracts (CPUEL.csv.gz, CA.csv.gz) and
the region haul list (hauls_region.csv) written by the species-mix scripts in
research_notes/"Torskfiskarnas artsammansattning" (Swedish letters); the guild table; maps/depth.npz.

Method (the gadoid species-mix convention, extended to every fish taxon):
- valid GOV hauls Q1 and Q3 2021-2025 in the Skagerrak + box rectangles; a taxon absent from a
  haul is a zero; invertebrates are dropped;
- CPUE per length class -> weight at the class midpoint (+0.25 cm for taxa measured to 0.5 cm,
  else +0.5 cm); W = a L^b fitted to NS-IBTS CA individual weights 2021-25 (n >= 50, 3-SD trim),
  else a body-form default (flagged; its biomass share is reported);
- depth-stratified box estimate: region hauls in depth bins, weighted by the grid's area per bin;
  the unsampled >= 300 m bin takes the 200-300 m composition; Q1 and Q3 separately, pooled = mean;
- swept-area density at q = 1, wing spread (a floor for pelagic and mesopelagic fish).
"""
import os
import sys

import numpy as np
import pandas as pd

ext, hauls_csv, guilds_csv, depth_npz, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)

G = pd.read_csv(guilds_csv).set_index("aphia")
FORM_LW = {  # body-form defaults (g, cm), rough FishBase-like values
    "fusiform": (0.0089, 3.05), "flat": (0.0105, 3.08), "eel": (0.0016, 3.0),
    "ray": (0.0045, 3.15), "shark": (0.0035, 3.1), "round": (0.02, 3.0),
}
HALF_CM = {126417, 126425, 126426, 126421, 125516, 126751, 126756, 126754, 125909}

# ---------------------------------------------------------------- length-weight
ca = pd.read_csv(os.path.join(ext, "CA.csv.gz"), low_memory=False)
ca["LngtCode"] = ca["LngtCode"].astype(str).str.strip()
ca = ca[(ca.Year >= 2021) & (ca.Year <= 2025) & ca.Quarter.isin([1, 3]) & (ca.IndWgt > 0)]
scale = {"1": 1.0, ".": 0.1, "0": 0.1, "5": 1.0}
ca["L"] = ca.LngtClass * ca.LngtCode.map(scale) + ca.LngtCode.map({"1": 0.5, ".": 0.05, "0": 0.25, "5": 2.5})
lw = {}
for aph, row in G.iterrows():
    k = ca[(ca.Valid_Aphia == aph) & (ca.L > 0)]
    if len(k) >= 50:
        x, y = np.log(k.L.values), np.log(k.IndWgt.values)
        for _ in range(2):
            b, la = np.polyfit(x, y, 1)
            r = y - (la + b * x)
            keep = np.abs(r) < 3 * r.std()
            x, y = x[keep], y[keep]
        b, la = np.polyfit(x, y, 1)
        lw[aph] = (np.exp(la), b, len(x), "CA fit")
    else:
        a, b = FORM_LW[row.form]
        lw[aph] = (a, b, 0, "form default")
LW = pd.DataFrame(lw, index=["a", "b", "n", "source"]).T
LW.insert(0, "species", G.species)
LW.to_csv(os.path.join(out, "length_weight_all.csv"))

# ---------------------------------------------------------------- hauls and CPUE
H = pd.read_csv(hauls_csv)
H["Ship"] = H.Ship.astype(str).str.strip()
H["key"] = list(zip(H.Year, H.Quarter, H.Ship, H.HaulNo))
cp = pd.read_csv(os.path.join(ext, "CPUEL.csv.gz"))
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp["key"] = list(zip(cp.Year, cp.Quarter, cp.Ship, cp.HaulNo))
cp = cp[cp.key.isin(set(H.key))]
unknown = cp[~cp.AphiaID.isin(G.index)].groupby("Species").CPUE_number_per_hour.sum()
cp = cp[cp.AphiaID.isin(G.index)].copy()
cp["L"] = cp.LngtClas / 10.0 + np.where(cp.AphiaID.isin(HALF_CM), 0.25, 0.5)
cp["kg_h"] = cp.CPUE_number_per_hour * cp.AphiaID.map(LW.a).astype(float) * \
    cp.L ** cp.AphiaID.map(LW.b).astype(float) / 1000
m = cp.groupby(["key", "AphiaID"]).kg_h.sum().unstack(fill_value=0.0)
X = H.set_index("key")[["Year", "Quarter", "dbin", "km2_per_h", "box_strict"]].join(m, how="left").fillna(0.0)
taxa = [c for c in X.columns if isinstance(c, (int, np.integer))]
for c in G.index:
    if c not in X:
        X[c] = 0.0
        taxa.append(c)

z = np.load(depth_npz)["depth"].astype(float)
z = np.abs(z[np.isfinite(z)])
BINS = ["<100", "100-200", "200-300", ">=300"]
area = pd.cut(pd.Series(z), [0, 100, 200, 300, 600], labels=BINS, right=False).value_counts(
    normalize=True).reindex(BINS)


def strat(df, cols):
    """Depth-stratified box mean of cols; >= 300 m takes the 200-300 m bin."""
    mu = df.groupby("dbin")[cols].mean().reindex(BINS)
    mu.loc[">=300"] = mu.loc["200-300"]
    return mu.mul(area, axis=0).sum()


def by_quarter(df, cols):
    q1 = strat(df[df.Quarter == 1], cols)
    q3 = strat(df[df.Quarter == 3], cols)
    return q1, q3, (q1 + q3) / 2


dens = X[taxa].div(X.km2_per_h, axis=0) / 1000  # t/km2, wing spread, q = 1
D = X[["Year", "Quarter", "dbin"]].join(dens)
k1, k3, kp = by_quarter(X, taxa)
d1, d3, dp = by_quarter(D, taxa)

names = {"P": "planktivorous_fish", "B": "benthivorous_fish", "R": "predatory_fish", "X": "other"}
sp = pd.DataFrame(dict(species=G.species, common=G.common, guild=G.guild.map(names),
                       lw=LW.source, kgh_Q1=k1, kgh_Q3=k3, kgh_pooled=kp,
                       t_km2_Q1=d1, t_km2_Q3=d3, t_km2_pooled=dp))
sp["share_all_fish_pct"] = 100 * sp.kgh_pooled / sp.kgh_pooled.sum()
sp = sp.sort_values("kgh_pooled", ascending=False)
sp.to_csv(os.path.join(out, "species_box_depth_strat.csv"))

g = sp.groupby("guild")[["kgh_Q1", "kgh_Q3", "kgh_pooled", "t_km2_Q1", "t_km2_Q3", "t_km2_pooled"]].sum()
for c in ["kgh_Q1", "kgh_Q3", "kgh_pooled"]:
    g[c.replace("kgh", "pct")] = 100 * g[c] / g[c].sum()
g.to_csv(os.path.join(out, "guild_box_depth_strat.csv"))

# guild share without the badly sampled pelagic/mesopelagic taxa' leverage: strict box raw mean
S = X[X.box_strict]
sb = S[taxa].mean()
gs = sb.groupby(G.guild.map(names)).sum()

# bootstrap over hauls within quarter x depth bin
rng = np.random.default_rng(1)
gg = G.guild.map(names).reindex(taxa)
Xg = X[["Year", "Quarter", "dbin"]].join(X[taxa].T.groupby(gg).sum().T)
gcols = [c for c in names.values() if c in Xg]
boot = []
for _ in range(1000):
    d = pd.concat([grp.sample(len(grp), replace=True, random_state=rng.integers(1 << 31))
                   for _, grp in Xg.groupby(["Quarter", "dbin"], observed=True)])
    _, _, p = by_quarter(d, gcols)
    boot.append(100 * p / p.sum())
boot = pd.DataFrame(boot)
ci = pd.DataFrame(dict(p05=boot.quantile(0.05), p50=boot.quantile(0.5), p95=boot.quantile(0.95)))
ci.to_csv(os.path.join(out, "guild_share_bootstrap.csv"))

# per year (pooled Q1/Q3, depth-stratified)
yr = {}
for y, d in Xg.groupby("Year"):
    _, _, p = by_quarter(d, gcols)
    yr[y] = 100 * p / p.sum()
yr = pd.DataFrame(yr).T
yr.to_csv(os.path.join(out, "guild_share_by_year.csv"))

# guild shares without herring/sprat/mackerel (the GOV-poor pelagic schoolers)
pel = [126417, 126425, 127023]
rest = sp.drop(index=[a for a in pel if a in sp.index]).groupby("guild").kgh_pooled.sum()

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
print("box area by depth bin\n", area.round(3))
print("\nhauls", len(X), "strict box", int(X.box_strict.sum()))
print("\ninvertebrates / unassigned taxa dropped (numbers/h summed):\n", unknown.sort_values(ascending=False).round(0).to_string())
print("\nlength-weight source counts\n", LW.source.value_counts())
fd = sp[sp.lw == "form default"]
print("biomass share on form-default LW: %.2f %%" % fd.share_all_fish_pct.sum())
print(fd[fd.share_all_fish_pct > 0.05][["common", "share_all_fish_pct"]].round(2).to_string())
print("\nguilds, depth-stratified box\n", g.round(2).to_string())
print("\nguild shares, bootstrap 5-50-95 %\n", ci.round(1).to_string())
print("\nguild shares by year\n", yr.round(1).to_string())
print("\nstrict box raw mean kg/h by guild\n", gs.round(1), "\n", (100 * gs / gs.sum()).round(1))
print("\nwithout herring, sprat, mackerel (pooled kg/h, %)\n", rest.round(1), "\n", (100 * rest / rest.sum()).round(1))
print("\nspecies (pooled share >= 0.1 % of all fish)\n",
      sp[sp.share_all_fish_pct >= 0.1][["common", "guild", "lw", "kgh_Q1", "kgh_Q3", "kgh_pooled",
                                         "t_km2_pooled", "share_all_fish_pct"]].round(3).to_string())
