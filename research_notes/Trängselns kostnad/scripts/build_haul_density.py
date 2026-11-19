"""Haul-level biomass density (t/km2, q = 1) of the benthivore species, Skagerrak, NS-IBTS 1991-2025.

Inputs (DATRAS extracts already in the repo, untrusted data, parsed only):
  <lek>/data/datras/{hh,hl}.csv          2000-2025 Q1/Q3 (research_notes/"Lek i rutan")
  <lek>/data/datras_1991_99/{hh,hl}.csv  1991-1999 Q1-Q4
  <lw>  length_weight_all.csv            (research_notes/"Fiskgrupper i rutan"/tables), W = a L^b, g, cm
Region: the 10 ICES rectangles of the earlier crowding analysis (43F8, 43F9, 44F8, 44F9, 44G0,
44G1, 45F9, 45G0, 45G1, 46G0; Skagerrak + Mareld box; Kattegat 43G0-43G2 excluded).
Numbers per hour: DataType C -> HLNoAtLngt * SubFactor; R -> HLNoAtLngt * SubFactor * 60 / HaulDur.
Length: LngtCode '.' (1 mm) and '0' (5 mm) in mm, '1' in cm; mid-class added.
Swept area: these HH files carry no wing spread / distance, so a nominal 0.1416 km2/h (the mean
wing-spread swept area per hour of the 265 hauls 2021-25 in tables/hauls_region.csv) is used.
Crowding statistics are scale-free; only absolute t/km2 depend on it.
Output: <out>/haul_density.csv (one row per valid GOV haul; columns per species, the model group,
guild B species not in the model group are not included), and the 0-group split for haddock
(< 20 cm) and cod (< 25 cm) as separate columns.
Discreteness: columns sq_<taxon> = sum over caught individuals of (w / A)^2 (t^2/km^4), w the
individual weight and A the haul's swept area, for Lloyd's (1967) correction of mean crowding for
Poisson sampling of individuals: corrected = (sum d^2 - sum sq) / sum d / mean d.
Optional 4th argument "all": every rectangle in the Lek i rutan extract (adds Kattegat 43G0-43G2);
output then haul_density_all.csv.
Usage: python3 -I build_haul_density.py <lek_data_dir> <lw_csv> <out_dir> [all]
"""
import os
import sys

import numpy as np
import pandas as pd

lek, lwf, out = sys.argv[1:4]
RECTS = {"43F8", "43F9", "44F8", "44F9", "44G0", "44G1", "45F9", "45G0", "45G1", "46G0"}
SP = {126437: "haddock", 126436: "cod", 127137: "long_rough_dab", 127139: "dab",
      127143: "plaice", 127136: "witch"}
SMALL = {126437: 20.0, 126436: 25.0}  # small-fish split (cm): mostly 0-group in Q3, age 1 in Q1
SWEPT = 0.1416  # km2/h
KEY = ["Year", "Quarter", "Country", "Ship", "HaulNo", "StNo"]

hh = pd.concat([pd.read_csv(os.path.join(lek, d, "hh.csv"), dtype={"StNo": str, "StatRec": str})
                for d in ("datras_1991_99", "datras")], ignore_index=True)
hl = pd.concat([pd.read_csv(os.path.join(lek, d, "hl.csv"), dtype={"StNo": str, "LngtCode": str})
                for d in ("datras_1991_99", "datras")], ignore_index=True)
for df in (hh, hl):
    df["Ship"] = df.Ship.astype(str).str.strip()
    df["StNo"] = df.StNo.astype(str).str.strip()
    df["Country"] = df.Country.astype(str).str.strip()
ALL = len(sys.argv) > 4 and sys.argv[4] == "all"
hh = hh[(hh.HaulVal == "V") & (hh.Gear == "GOV") & (ALL | hh.StatRec.isin(RECTS)) & (hh.HaulDur > 0)].copy()
hh = hh.drop_duplicates(KEY)
lw = pd.read_csv(lwf, index_col=0)

hl = hl[hl.Valid_Aphia.isin(SP)].copy()
hl = hl.merge(hh[KEY + ["HaulDur", "DataType"]], on=KEY, how="inner")
hl = hl[(hl.HLNoAtLngt > 0) & (hl.LngtClass > 0)]
sub = hl.SubFactor.where(hl.SubFactor > 0, 1.0)
n_h = hl.HLNoAtLngt * sub
n_h = np.where(hl.DataType == "C", n_h, n_h * 60.0 / hl.HaulDur)
code = hl.LngtCode.astype(str).str.strip()
L = np.where(code == "1", hl.LngtClass + 0.5,
             np.where(code == "0", hl.LngtClass / 10.0 + 0.25,
                      np.where(code == ".", hl.LngtClass / 10.0 + 0.05, np.nan)))
a = hl.Valid_Aphia.map(lw.a).astype(float)
b = hl.Valid_Aphia.map(lw.b).astype(float)
hl["L"] = L
hl["kg_h"] = n_h * a * L ** b / 1000.0
w_t = a * L ** b / 1e6                      # t per individual
n_raw = n_h * hl.HaulDur / 60.0             # individuals caught
area = SWEPT * hl.HaulDur / 60.0            # km2 swept
hl["sq"] = n_raw * (w_t / area) ** 2        # t^2/km^4
hl = hl[np.isfinite(hl.kg_h)]
hl["sp"] = hl.Valid_Aphia.map(SP)
hl["small"] = [l < SMALL.get(s, -1) for s, l in zip(hl.Valid_Aphia, hl.L)]
tab = hl.groupby(KEY + ["sp"]).kg_h.sum().unstack(fill_value=0.0)
tab_q = hl.groupby(KEY + ["sp"]).sq.sum().unstack(fill_value=0.0).add_prefix("sq_")
tab_s = hl[hl.small].groupby(KEY + ["sp"]).kg_h.sum().unstack(fill_value=0.0).add_suffix("_small")
X = hh.set_index(KEY)[["Month", "Day", "ShootLat", "ShootLong", "StatRec", "Depth", "DayNight",
                       "HaulDur", "DataType"]]
X = X.join(tab, how="left").join(tab_s, how="left").join(tab_q, how="left").fillna(0.0)
cols = list(SP.values()) + [c + "_small" for c in ("haddock", "cod")]
for c in cols:
    if c not in X:
        X[c] = 0.0
    X[c] = X[c] / SWEPT / 1000.0  # t/km2
X["group"] = X[list(SP.values())].sum(axis=1)
X["flatfish"] = X[["long_rough_dab", "dab", "plaice", "witch"]].sum(axis=1)
for c in SP.values():
    if "sq_" + c not in X:
        X["sq_" + c] = 0.0
X["sq_group"] = X[["sq_" + c for c in SP.values()]].sum(axis=1)
X["sq_flatfish"] = X[["sq_" + c for c in ("long_rough_dab", "dab", "plaice", "witch")]].sum(axis=1)
os.makedirs(out, exist_ok=True)
X.reset_index().to_csv(os.path.join(out, "haul_density_all.csv" if ALL else "haul_density.csv"), index=False)
print("hauls", len(X), "by quarter", X.reset_index().groupby("Quarter").size().to_dict())
print(X.reset_index().groupby(["Year"]).size().to_string())
