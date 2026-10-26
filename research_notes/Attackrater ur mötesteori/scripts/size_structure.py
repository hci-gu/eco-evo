"""Size structure of the planktivore members in the Mareld box -> individuals per tonne and
length moments for an encounter model.

Usage:
  python3 -I size_structure.py <cpuel_csv_gz> <hauls_region_csv> <length_weight_csv> <depth_npz>

Inputs (untrusted data, only parsed):
- NS-IBTS CPUE per length class, GOV, 2021-25 (extract written by
  research_notes/"Torskfiskarnas artsammansattning"/scripts/extract_datras.py; Swedish letters);
- the region haul list of the same folder (Skagerrak + box rectangles, valid hauls, depth bin,
  swept area);
- the length-weight fits of research_notes/"Fiskgrupper i rutan"/tables/length_weight_all.csv
  (W = a L^b, g and cm, NS-IBTS CA 2021-25);
- maps/depth.npz (area share per depth bin of the grid).

Method: the convention of "Fiskgrupper i rutan"/scripts/fish_guilds.py. Length class midpoint
(+0.25 cm for 0.5-cm taxa herring and sprat, +0.5 cm otherwise); numbers per hour per class;
depth-stratified box mean per quarter (>= 300 m takes the 200-300 m bin); Q1 and Q3 pooled as
the mean. Output per species: individuals per tonne N/t = sum n / sum n W, the biomass-weighted
mean length, and the length moments M_k = sum n L^k / sum n W(L) [cm^k per tonne] for k = 0..4,
so that any encounter volume V(L) = c L^k gives V per tonne = c M_k without redoing the size
structure. Also an adults-only (L >= 20 cm for herring) bound, because the GOV under-samples
large pelagic herring, and an L < 29 cm subset (blue whiting: the small fish that rise to the surface at night, Degnbol & Munch-Petersen 1985).
"""
import sys

import numpy as np
import pandas as pd

cpuel, hauls_csv, lw_csv, depth_npz = sys.argv[1:5]
SPECIES = {126417: "herring", 126439: "blue whiting", 126444: "Norway pout", 126425: "sprat"}
HALF_CM = {126417, 126425}
BINS = ["<100", "100-200", "200-300", ">=300"]

H = pd.read_csv(hauls_csv)
H["Ship"] = H.Ship.astype(str).str.strip()
H["key"] = list(zip(H.Year, H.Quarter, H.Ship, H.HaulNo))
cp = pd.read_csv(cpuel)
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp["key"] = list(zip(cp.Year, cp.Quarter, cp.Ship, cp.HaulNo))
cp = cp[cp.key.isin(set(H.key)) & cp.AphiaID.isin(SPECIES)].copy()
cp["dbin"] = cp.key.map(H.set_index("key").dbin)
cp["L"] = cp.LngtClas / 10.0 + np.where(cp.AphiaID.isin(HALF_CM), 0.25, 0.5)
LW = pd.read_csv(lw_csv, index_col=0)

z = np.load(depth_npz)["depth"].astype(float)
z = np.abs(z[np.isfinite(z)])
area = pd.cut(pd.Series(z), [0, 100, 200, 300, 600], labels=BINS, right=False).value_counts(
    normalize=True).reindex(BINS)
hinfo = H.set_index("key")[["Quarter", "dbin"]]
nh = H.groupby(["Quarter", "dbin"]).size()


def strat_lf(df):
    """Depth-stratified box mean numbers/h per length class, Q1 and Q3 pooled."""
    out = []
    for q in (1, 3):
        d = df[df.Quarter == q]
        s = d.groupby(["dbin", "L"]).CPUE_number_per_hour.sum()  # summed over hauls
        lf = 0.0
        for b in BINS:
            bb = "200-300" if b == ">=300" else b
            if bb not in s.index.get_level_values(0):
                continue
            mean_b = s.loc[bb] / nh.loc[(q, bb)]  # mean over all hauls of the bin (zeros incl.)
            lf = mean_b * area[b] if isinstance(lf, float) else lf.add(mean_b * area[b], fill_value=0)
        out.append(lf if not isinstance(lf, float) else pd.Series(dtype=float))
    return out


rows = []
pd.set_option("display.width", 200)
print("grid area by depth bin:", area.round(3).to_dict())
print("hauls per quarter x bin:", nh.to_dict())
for aph, name in SPECIES.items():
    d = cp[cp.AphiaID == aph]
    a, b = float(LW.loc[aph, "a"]), float(LW.loc[aph, "b"])
    q1, q3 = strat_lf(d)
    for label, lf in (("Q1", q1), ("Q3", q3), ("pooled", None)):
        if lf is None:
            lf = (q1 * 0.5).add(q3 * 0.5, fill_value=0)
        if len(lf) == 0 or lf.sum() == 0:
            rows.append(dict(species=name, set=label))
            continue
        for sub, sel in (("all", lf.index > 0), ("L>=20cm", lf.index >= 20.0), ("L<29cm", lf.index < 29.0)):
            n = lf[sel]
            if n.sum() == 0:
                continue
            L = n.index.values
            W = a * L ** b  # g
            mass_t = (n * W).sum() / 1e6  # t per (hour of trawl) - units cancel
            mass_all = (lf * a * lf.index.values ** b).sum() / 1e6
            r = dict(species=name, set=label, sizes=sub, biomass_share=mass_t / mass_all,
                     N_per_t=n.sum() / mass_t,
                     Lbar_biomass=(n * W * L).sum() / (n * W).sum(),
                     Lbar_numbers=(n * L).sum() / n.sum(),
                     Wbar_g=(n * W).sum() / n.sum())
            for k in range(5):
                r["M%d" % k] = (n * L ** k).sum() / mass_t
            # L-quantiles by biomass
            cw = np.cumsum((n * W).values) / (n * W).sum()
            r["L_b10"], r["L_b50"], r["L_b90"] = [float(L[np.searchsorted(cw, p)]) for p in (0.1, 0.5, 0.9)]
            rows.append(r)

T = pd.DataFrame(rows)
print(T.to_string(float_format=lambda x: "%.4g" % x))
