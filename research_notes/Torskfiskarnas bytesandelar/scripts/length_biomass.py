"""Depth-stratified gadoid biomass per 1-cm length class in the Mareld box, per quarter.

Usage: python3 -I length_biomass.py <species_mix_dir> <out_dir>

<species_mix_dir> = research_notes/Torskfiskarnas artsammansättning (read only). Uses its
data/extract/CPUEL.csv.gz (untrusted DATRAS extract, parsed only), tables/hauls_region.csv,
tables/length_weight.csv and tables/box_area_by_depth.csv.

Same haul set and estimator as that note's phi_mix.py "depth" estimator: region hauls (Skagerrak +
box rectangles), NS-IBTS GOV Q1/Q3 2021-25, haul-mean kg/h per depth bin, weighted by the grid's
area per bin; the >= 300 m bin takes the 200-300 m means. Pooled = mean of Q1 and Q3 kg/h.

Variants of the length-dependent catchability (Fraser, Greenstreet & Piet 2007, functions as
transcribed in that note's scripts/catchability.py):
  raw        q = 1;
  A_floor    Fraser q; haddock/whiting converted to the wing basis with the haul's door/wing ratio;
             saithe = mean of the printed cod, haddock and whiting q (Fraser's rule);
  A_sw_floor as A_floor, saithe from the wing-converted haddock and whiting q.
(The floor only concerns poor cod and rocklings, which are not used here.)

Output: length_biomass.csv (variant, quarter, species, L_cm, kg_h), length_class_pct.csv,
haul_length_kgh.csv.gz (per haul x species x 1-cm class: raw kg/h and Fraser-corrected kg/h),
hauls_used.csv, and a printed summary.
"""
import os
import sys

import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

SP = {126437: "haddock", 126438: "whiting", 126441: "saithe", 126436: "cod", 126484: "hake"}
BINS = ["<100", "100-200", "200-300", ">=300"]


def q_cod(l):
    return np.where(l <= 25, 0.15963, np.where(l >= 97, 0.66822, 0.20935 - 0.00434 * l + 0.00009 * l ** 2))


def q_had(l):
    return np.where(l <= 11, 0.0116121, np.where(l >= 46, 0.123268, -0.370453 + 0.04227 * l - 0.000685 * l ** 2))


def q_whg(l):
    return np.where(l <= 21, 0.616878, np.where(l >= 38, 0.138165, -0.028159 * l + 1.208229))


hh = pd.read_csv(os.path.join(src, "tables", "hauls_region.csv"))
hh["Ship"] = hh.Ship.astype(str).str.strip()
hh = hh[hh.region]
ratio = hh.DoorSpread.where(hh.DoorSpread > 0) / hh.WingSpread.where(hh.WingSpread > 0)
hh["dw"] = ratio.fillna(ratio.median())
area = pd.read_csv(os.path.join(src, "tables", "box_area_by_depth.csv"), index_col=0).iloc[:, 0]
lw = pd.read_csv(os.path.join(src, "tables", "length_weight.csv"), index_col=0)

cp = pd.read_csv(os.path.join(src, "data", "extract", "CPUEL.csv.gz"))
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp = cp[cp.AphiaID.isin(SP)].copy()
cp["species"] = cp.AphiaID.map(SP)
cp["L"] = cp.LngtClas / 10.0 + 0.5
cp = cp.merge(hh[["Year", "Quarter", "Ship", "HaulNo", "dw", "dbin"]], on=["Year", "Quarter", "Ship", "HaulNo"],
              how="inner")
cp["kg_h"] = cp.CPUE_number_per_hour * lw.loc[cp.species, "a"].values * cp.L ** lw.loc[cp.species, "b"].values / 1000
cp["Lc"] = np.floor(cp.L).astype(int)
L, dw, s = cp.L.values, cp.dw.values, cp.species.values

# haul-level long table for the bootstrap in prey_shares.py
qv = {}
for var in ["A_floor", "A_sw_floor"]:
    qh, qw = q_had(L) * dw, q_whg(L) * dw
    sai = (q_cod(L) + q_had(L) + q_whg(L)) / 3 if var == "A_floor" else (q_cod(L) + qh + qw) / 3
    qv[var] = np.select([s == "cod", s == "haddock", s == "whiting", s == "saithe", s == "hake"],
                        [q_cod(L), qh, qw, sai, np.full(len(L), 0.0909)])
HL = cp[["Year", "Quarter", "Ship", "HaulNo", "dbin", "species", "Lc", "kg_h"]].copy()
for var, q in qv.items():
    HL["kg_" + var] = cp.kg_h.values / q
HL = HL.groupby(["Year", "Quarter", "Ship", "HaulNo", "dbin", "species", "Lc"], as_index=False).sum()
HL.to_csv(os.path.join(out, "haul_length_kgh.csv.gz"), index=False)
hh[["Year", "Quarter", "Ship", "HaulNo", "dbin", "Depth"]].to_csv(os.path.join(out, "hauls_used.csv"), index=False)

nhaul = hh.groupby(["Quarter", "dbin"]).size()
rows = []
for var in ["raw", "A_floor", "A_sw_floor"]:
    if var == "raw":
        q = np.ones(len(cp))
    else:
        qh, qw = q_had(L) * dw, q_whg(L) * dw
        sai = (q_cod(L) + q_had(L) + q_whg(L)) / 3 if var == "A_floor" else (q_cod(L) + qh + qw) / 3
        q = np.select([s == "cod", s == "haddock", s == "whiting", s == "saithe", s == "hake"],
                      [q_cod(L), qh, qw, sai, np.full(len(L), 0.0909)])
    cp["w"] = cp.kg_h / q
    # haul-mean kg/h per (quarter, depth bin, species, L): sum over hauls / number of hauls in the bin
    g = cp.groupby(["Quarter", "dbin", "species", "Lc"]).w.sum().reset_index()
    g["n"] = [nhaul.get((qq, d), np.nan) for qq, d in zip(g.Quarter, g.dbin)]
    g["mean"] = g.w / g.n
    for qu in [1, 3]:
        d = g[g.Quarter == qu].pivot_table(index=["species", "Lc"], columns="dbin", values="mean", aggfunc="sum",
                                           fill_value=0.0).reindex(columns=BINS, fill_value=0.0)
        d[">=300"] = d["200-300"]
        strat = (d[BINS] * area.reindex(BINS).values).sum(axis=1)
        for (sp, lc), v in strat.items():
            rows.append(dict(variant=var, quarter=qu, species=sp, L_cm=lc, kg_h=v))
R = pd.DataFrame(rows)
P = R.groupby(["variant", "species", "L_cm"]).kg_h.sum().reset_index()  # Q1+Q3 sum = 2 x pooled mean
P["kg_h"] /= 2.0
P["quarter"] = "pooled"
R = pd.concat([R.astype({"quarter": str}), P[R.columns]])
R.to_csv(os.path.join(out, "length_biomass.csv"), index=False)

# summary: biomass share per 10-cm class
R["cls"] = pd.cut(R.L_cm, [0, 10, 15, 20, 25, 30, 40, 50, 60, 70, 200], right=False)
T = R.groupby(["variant", "quarter", "species", "cls"], observed=False).kg_h.sum()
T = (100 * T / T.groupby(["variant", "quarter", "species"]).transform("sum")).unstack("cls").round(1)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
print(T.to_string())
T.to_csv(os.path.join(out, "length_class_pct.csv"))
tot = R.groupby(["variant", "quarter", "species"]).kg_h.sum().unstack("species")
print((100 * tot.div(tot.sum(axis=1), axis=0)).round(1))
