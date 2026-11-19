"""Q1 ogive from BOX fish only vs the region, and the box Q1 mature biomass share with each.

Usage: python3 -I ogive_box_vs_region.py DATA_DIR [DATA_DIR ...]
Same conventions as datras_maturity.py (2-cm classes, unweighted by haul depth).
"""
import sys

import numpy as np
import pandas as pd

BOX = {"44G0", "44G1", "45G0", "45G1"}
KEYS = ["Year", "Quarter", "Country", "Ship", "Gear", "HaulNo", "StNo"]
SP = {126436: "cod", 126437: "haddock", 126438: "whiting", 126441: "saithe", 126444: "Norway pout",
      126417: "herring", 126425: "sprat", 127143: "plaice", 127136: "witch"}
STAGE = {"1": 0, "61": 0, "A": 0, "2": 1, "62": 1, "B": 1, "3": 1, "63": 1, "C": 1, "4": 1, "64": 1,
         "D": 1, "65": 1, "E": 1, "5": 1, "6": 1}
rd = lambda f: pd.concat([pd.read_csv(f"{d}/{f}", dtype=str) for d in sys.argv[1:]]).apply(lambda c: c.str.strip())
hh, hl, ca = rd("hh.csv").drop_duplicates(KEYS), rd("hl.csv"), rd("ca.csv")
hh = hh[(hh.HaulVal == "V") & (hh.Gear == "GOV") & (hh.Quarter == "1")]
hh["box"] = hh.StatRec.isin(BOX)
L = lambda d: np.where(d.LngtCode.isin([".", "0"]), pd.to_numeric(d.LngtClass, errors="coerce") / 10,
                       pd.to_numeric(d.LngtClass, errors="coerce"))
ca = ca[(ca.Quarter == "1") & (ca.Gear == "GOV")].merge(hh[KEYS + ["StatRec"]], on=KEYS, how="left")
ca["box"] = ca.StatRec.isin(BOX) | ca.AreaCode.isin(BOX)
ca["mat"] = ca.Maturity.map(STAGE)
ca["n"] = pd.to_numeric(ca.CANoAtLngt, errors="coerce")
ca["L2"] = (L(ca) // 2) * 2
ca["sp"] = pd.to_numeric(ca.Valid_Aphia, errors="coerce").map(SP)
ca["w"] = pd.to_numeric(ca.IndWgt, errors="coerce")
hl = hl.merge(hh[KEYS + ["box", "HaulDur", "DataType"]], on=KEYS)
hl = hl[hl.box]
hl["N"] = pd.to_numeric(hl.HLNoAtLngt, errors="coerce") * pd.to_numeric(hl.SubFactor, errors="coerce")
hl.loc[hl.DataType == "R", "N"] *= 60 / pd.to_numeric(hl.loc[hl.DataType == "R", "HaulDur"])
hl["L2"] = (L(hl) // 2) * 2
hl["sp"] = pd.to_numeric(hl.Valid_Aphia, errors="coerce").map(SP)
for sp in SP.values():
    c = ca[(ca.sp == sp) & ca.mat.notna()]
    wl = c[(c.w > 0)]
    b, lna = np.polyfit(np.log(wl.L2 + 1), np.log(wl.w), 1)
    h = hl[hl.sp == sp].copy()
    h["wt"] = h.N * np.exp(lna) * (h.L2 + 1) ** b
    res = []
    for lab, cc in (("region", c), ("box", c[c.box])):
        og = cc.groupby("L2").apply(lambda d: (d.n * d.mat).sum() / d.n.sum(), include_groups=False)
        og = og.reindex(sorted(set(og.index) | set(h.L2))).interpolate(limit_direction="both")
        res.append(f"{lab} ogive (n {int(cc.n.sum())}): {(h.wt * h.L2.map(og)).sum() / h.wt.sum():.2f}")
    print(f"{sp:12s} Q1 box mature biomass share: " + "; ".join(res))
