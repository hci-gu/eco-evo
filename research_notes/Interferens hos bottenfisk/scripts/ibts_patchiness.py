"""Haul-scale crowding of benthivorous fish in the Skagerrak region, NS-IBTS 2021-25.

Field analogue of the model's "biomass-weighted own density / grid mean".
Reuses the earlier extracts and conventions (no new download):
  hauls:  research_notes/Torskfiskarnas artsammansattning/tables/hauls_region.csv
          (valid GOV hauls Q1, Q3 2021-25, Skagerrak + Mareld-box rectangles,
           km2_per_h = wing-spread swept area per hour, q = 1)
  CPUE:   .../Torskfiskarnas artsammansattning/data/extract/CPUEL.csv.gz
  L-W:    research_notes/Fiskgrupper i rutan/tables/length_weight_all.csv
  guilds: research_notes/Fiskgrupper i rutan/inputs/guilds.csv (guild B = benthivores)
For each survey (year x quarter): Lloyd's mean crowding / mean
  = sum(d^2)/sum(d) / mean(d)   (density experienced by the average tonne),
and quantiles of d/mean, for the benthivore guild and its main species.
Usage: python3 -I ibts_patchiness.py <torsk_dir> <guild_dir> <out_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

torsk, guild, out = sys.argv[1:4]
H = pd.read_csv(os.path.join(torsk, "tables", "hauls_region.csv"))
H["Ship"] = H.Ship.astype(str).str.strip()
H["key"] = list(zip(H.Year, H.Quarter, H.Ship, H.HaulNo))
G = pd.read_csv(os.path.join(guild, "inputs", "guilds.csv")).set_index("aphia")
LW = pd.read_csv(os.path.join(guild, "tables", "length_weight_all.csv"), index_col=0)
cp = pd.read_csv(os.path.join(torsk, "data", "extract", "CPUEL.csv.gz"))
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp["key"] = list(zip(cp.Year, cp.Quarter, cp.Ship, cp.HaulNo))
cp = cp[cp.key.isin(set(H.key)) & cp.AphiaID.isin(G.index)].copy()
HALF_CM = {126417, 126425, 126426, 126421, 125516, 126751, 126756, 126754, 125909}
cp["L"] = cp.LngtClas / 10.0 + np.where(cp.AphiaID.isin(HALF_CM), 0.25, 0.5)
cp["kg_h"] = (cp.CPUE_number_per_hour * cp.AphiaID.map(LW.a).astype(float)
              * cp.L ** cp.AphiaID.map(LW.b).astype(float) / 1000)
m = cp.groupby(["key", "AphiaID"]).kg_h.sum().unstack(fill_value=0.0)
X = H.set_index("key")[["Year", "Quarter", "km2_per_h", "Depth"]].join(m, how="left").fillna(0.0)
ben = [a for a in G.index[G.guild == "B"] if a in X.columns]
dens = X[ben].div(X.km2_per_h, axis=0) / 1000.0  # t/km2, q = 1
named = {}
for a in ben:
    named[G.loc[a, "common"]] = dens[a]
D = pd.DataFrame(named)
D["benthivore_guild"] = dens.sum(axis=1)
D["Year"], D["Quarter"] = X.Year, X.Quarter
cols = ["benthivore_guild"] + list(D[ben and [G.loc[a, "common"] for a in ben]]
                                   .mean().sort_values(ascending=False).index[:6])
rows = []
for c in cols:
    for (y, q), g in D.groupby(["Year", "Quarter"]):
        x = g[c].values
        if x.sum() <= 0:
            continue
        mu = x.mean()
        rows.append(dict(taxon=c, year=y, quarter=q, n_hauls=len(x), mean_t_km2=mu,
                         lloyd_over_mean=(x ** 2).sum() / x.sum() / mu,
                         p50_over_mean=np.median(x) / mu,
                         p90_over_mean=np.quantile(x, 0.9) / mu,
                         max_over_mean=x.max() / mu, max_t_km2=x.max(),
                         share_biomass_in_hauls_ge5x=x[x >= 5 * mu].sum() / x.sum()))
R = pd.DataFrame(rows)
S = R.groupby("taxon").agg(surveys=("year", "size"), hauls_per_survey=("n_hauls", "median"),
                           mean_t_km2=("mean_t_km2", "median"),
                           lloyd_median=("lloyd_over_mean", "median"),
                           lloyd_min=("lloyd_over_mean", "min"),
                           lloyd_max=("lloyd_over_mean", "max"),
                           p90_over_mean=("p90_over_mean", "median"),
                           max_over_mean=("max_over_mean", "median"),
                           max_t_km2=("max_t_km2", "max"),
                           share_ge5x=("share_biomass_in_hauls_ge5x", "median"))
S = S.loc[cols]
# pooled over all surveys, each survey normalised by its own mean
D2 = D.copy()
for c in cols:
    D2[c] = D[c] / D.groupby(["Year", "Quarter"])[c].transform("mean")
pooled = {c: (D2[c] ** 2).sum() / D2[c].sum() for c in cols}
S["lloyd_pooled_rel"] = pd.Series(pooled)
os.makedirs(out, exist_ok=True)
R.to_csv(os.path.join(out, "ibts_patchiness_by_survey.csv"), index=False)
S.to_csv(os.path.join(out, "ibts_patchiness_summary.csv"))
pd.set_option("display.width", 220)
print("hauls", len(X))
print(S.round(2).to_string())
