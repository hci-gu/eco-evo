"""Length-based GOV catchability correction (Fraser, Greenstreet & Piet 2007, Appendix; transcribed
in evidence_B_catchability.md A1) applied to the haul x species table of gadoid_mix.py.

Usage: python3 -I catchability.py <extract_dir> <tables_dir> <variant> <out_dir>

Corrected kg/h_i = sum_L CPUE(L) W(L) / q_wing(L). Only ratios between species are used for shares;
the absolute value is "kg per hour of fishing, as if q = 1 on the wing-swept area".

q_wing: Fraser's q is per wing (net) swept area except haddock and whiting, which are per door-spread
swept area (herding). Converted with the haul's door/wing spread ratio (region median 3.75 where
missing).

Variants:
  A   Fraser as printed, haddock/whiting converted to wing basis; saithe = mean of the cod, haddock
      and whiting q-at-length as printed (Fraser's rule, mixed basis); tusk, pollack, forkbeard,
      blue ling ("roundfish average", value not found) take the ling q 0.105; bib = mean of Norway
      pout and poor cod; silvery pout and blue whiting (pelagic, excluded by Fraser) take the
      Norway pout q.
  A_sw  as A, saithe from the wing-converted haddock and whiting q (more saithe-efficient gear, less
      saithe after correction).
  A_floor  as A, minimum q 0.10 for the small species (poor cod, rocklings, bib), whose q of 0.01-0.11
      come from indirect methods.
  A_sw_floor  both changes.
  B   all q as printed on one basis (no door/wing conversion): a sensitivity, not a reading of Fraser.
"""
import os
import sys

import numpy as np
import pandas as pd

ext, tab, variant, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)


def q_cod(l):
    return np.where(l <= 25, 0.15963, np.where(l >= 97, 0.66822, 0.20935 - 0.00434 * l + 0.00009 * l ** 2))


def q_had(l):
    return np.where(l <= 11, 0.0116121, np.where(l >= 46, 0.123268, -0.370453 + 0.04227 * l - 0.000685 * l ** 2))


def q_whg(l):
    return np.where(l <= 21, 0.616878, np.where(l >= 38, 0.138165, -0.028159 * l + 1.208229))


def q_nop(l):
    return np.where(l <= 11, 0.066204, 0.191855)


def q_poc(l):
    return np.where(l <= 5, 0.038387, np.where(l >= 15, 0.01004, -0.002834 * l + 0.05256))


def q_rock(l):
    return np.where(l <= 10, 0.03298, np.where(l >= 25, 0.03647, -0.34995 + 0.05352 * l - 0.00152 * l ** 2))


H = pd.read_csv(os.path.join(tab, "haul_species_kgh.csv"))
H["Ship"] = H.Ship.astype(str).str.strip()
ratio = (H.DoorSpread.where(H.DoorSpread > 0) / H.WingSpread.where(H.WingSpread > 0))
H["dw"] = ratio.fillna(ratio[H.region].median())
lw = pd.read_csv(os.path.join(tab, "length_weight.csv"), index_col=0)

SP = {126437: "haddock", 126438: "whiting", 126441: "saithe", 126436: "cod", 126484: "hake",
      126461: "ling", 126459: "blue ling", 126447: "tusk", 126440: "pollack", 126501: "greater forkbeard",
      126446: "poor cod", 126445: "bib", 126435: "silvery pout", 126450: "fourbeard rockling",
      126458: "threebeard rockling", 126444: "Norway pout", 126439: "blue whiting"}
cp = pd.read_csv(os.path.join(ext, "CPUEL.csv.gz"))
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp = cp[cp.AphiaID.isin(SP)].copy()
cp["species"] = cp.AphiaID.map(SP)
cp["L"] = cp.LngtClas / 10.0 + 0.5
cp = cp.merge(H[["Year", "Quarter", "Ship", "HaulNo", "dw"]], on=["Year", "Quarter", "Ship", "HaulNo"], how="inner")
cp["kg_h"] = cp.CPUE_number_per_hour * lw.loc[cp.species, "a"].values * cp.L ** lw.loc[cp.species, "b"].values / 1000
L, dw = cp.L.values, cp.dw.values
conv = dw if variant != "B" else 1.0
qh, qw = q_had(L) * conv, q_whg(L) * conv
q = np.full(len(cp), np.nan)
s = cp.species.values
q[s == "cod"] = q_cod(L)[s == "cod"]
q[s == "haddock"] = qh[s == "haddock"]
q[s == "whiting"] = qw[s == "whiting"]
sai = (q_cod(L) + q_had(L) + q_whg(L)) / 3 if "sw" not in variant else (q_cod(L) + qh + qw) / 3
q[s == "saithe"] = sai[s == "saithe"]
q[s == "hake"] = 0.0909
for sp in ["ling", "tusk", "pollack", "greater forkbeard", "blue ling"]:
    q[s == sp] = 0.104976
q[s == "Norway pout"] = q_nop(L)[s == "Norway pout"]
for sp in ["silvery pout", "blue whiting"]:
    q[s == sp] = q_nop(L)[s == sp]
q[s == "poor cod"] = q_poc(L)[s == "poor cod"]
q[s == "bib"] = ((q_nop(L) + q_poc(L)) / 2)[s == "bib"]
for sp in ["fourbeard rockling", "threebeard rockling"]:
    q[s == sp] = q_rock(L)[s == sp]
if "floor" in variant:
    small = np.isin(s, ["poor cod", "bib", "fourbeard rockling", "threebeard rockling"])
    q[small] = np.maximum(q[small], 0.10)
assert not np.isnan(q).any()
cp["q"] = q
cp["kg_h_corr"] = cp.kg_h / cp.q

m = cp.groupby(["Year", "Quarter", "Ship", "HaulNo", "species"]).kg_h_corr.sum().unstack(fill_value=0.0)
names = list(SP.values())
C = H.drop(columns=[c for c in names if c in H]).merge(m.reset_index(), on=["Year", "Quarter", "Ship", "HaulNo"],
                                                       how="left")
for n in names:
    if n not in C:
        C[n] = 0.0
C[names] = C[names].fillna(0.0)
C.to_csv(os.path.join(out, "haul_species_kgh.csv"), index=False)
# carry the area table so phi_mix.py can run on this directory
pd.read_csv(os.path.join(tab, "box_area_by_depth.csv"), index_col=0).to_csv(os.path.join(out, "box_area_by_depth.csv"))

# biomass-weighted mean q per species (region hauls), for the record
wq = cp.groupby("species").apply(lambda d: d.kg_h.sum() / d.kg_h_corr.sum(), include_groups=False)
wq.rename("effective_q_wing").to_csv(os.path.join(out, "effective_q.csv"))
print(variant, "effective q (raw kg / corrected kg):")
print(wq.round(3).to_string())
