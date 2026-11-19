"""Field crowding (Lloyd mean crowding / mean) per model group, NS-IBTS GOV Q1+Q3 2021-25,
Skagerrak + box rectangles. Copy of research_notes/"Interferens hos bottenfisk"/scripts/ibts_patchiness.py
conventions (same haul table, CPUE extract, L-W, guilds); groups by species name."""
import os, sys
import numpy as np, pandas as pd
torsk, guild = sys.argv[1:3]
H = pd.read_csv(os.path.join(torsk, "tables", "hauls_region.csv"))
H["Ship"] = H.Ship.astype(str).str.strip(); H["key"] = list(zip(H.Year, H.Quarter, H.Ship, H.HaulNo))
G = pd.read_csv(os.path.join(guild, "inputs", "guilds.csv")).set_index("aphia")
LW = pd.read_csv(os.path.join(guild, "tables", "length_weight_all.csv"), index_col=0)
cp = pd.read_csv(os.path.join(torsk, "data", "extract", "CPUEL.csv.gz"))
cp["Ship"] = cp.Ship.astype(str).str.strip(); cp["key"] = list(zip(cp.Year, cp.Quarter, cp.Ship, cp.HaulNo))
cp = cp[cp.key.isin(set(H.key)) & cp.AphiaID.isin(G.index)].copy()
HALF_CM = {126417, 126425, 126426, 126421, 125516, 126751, 126756, 126754, 125909}
cp["L"] = cp.LngtClas / 10.0 + np.where(cp.AphiaID.isin(HALF_CM), 0.25, 0.5)
cp["kg_h"] = cp.CPUE_number_per_hour * cp.AphiaID.map(LW.a).astype(float) * cp.L ** cp.AphiaID.map(LW.b).astype(float) / 1000
m = cp.groupby(["key", "AphiaID"]).kg_h.sum().unstack(fill_value=0.0)
X = H.set_index("key")[["Year", "Quarter", "km2_per_h"]].join(m, how="left").fillna(0.0)
dens = X[[c for c in X.columns if c in G.index]].div(X.km2_per_h, axis=0) / 1000.0
name = G.common.str.lower()
def cols(names): return [a for a in dens.columns if name.get(a, '') in names]
groups = {
    'benthivorous_fish (guild B)': [a for a in dens.columns if G.loc[a, 'guild'] == 'B'],
    'whiting': cols({'whiting'}),
    'predatory_fish (saithe, spurdog, hake, anglerfish, ling, velvet belly, pollack)':
        cols({'saithe', 'spurdog', 'hake', 'anglerfish', 'ling', 'velvet belly', 'pollack', 'velvet belly lanternshark'}),
    'saithe only': cols({'saithe'}),
    'planktivorous_fish (guild P; GOV q low for pelagics)': [a for a in dens.columns if G.loc[a, 'guild'] == 'P'],
    'mackerel (GOV q ~0.1)': cols({'mackerel', 'horse mackerel'}),
}
for g, c in groups.items():
    d = dens[c].sum(axis=1); out = []
    for (y, q), x in d.groupby([X.Year, X.Quarter]):
        x = x.values
        if x.sum() > 0: out.append((x ** 2).sum() / x.sum() / x.mean())
    out = np.array(out)
    print(f"{g}: members {len(c)}; Lloyd/mean median {np.median(out):.2f} (range {out.min():.2f}-{out.max():.2f}, {len(out)} surveys)")
