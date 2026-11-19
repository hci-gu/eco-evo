"""Detail of spawning-stage fish: herring stage 'spawning' in the box by year, scale, length;
and the developing/spawning/spent mix of MATURE fish by month (region), per species.

Usage: python3 -I datras_stage_detail.py DATA_DIR [DATA_DIR ...]
"""
import sys

import numpy as np
import pandas as pd

sys.argv = sys.argv
BOX = {"44G0", "44G1", "45G0", "45G1"}
KEYS = ["Year", "Quarter", "Country", "Ship", "Gear", "HaulNo", "StNo"]
SP = {126436: "cod", 126437: "haddock", 126438: "whiting", 126441: "saithe", 126444: "Norway pout",
      126417: "herring", 126425: "sprat", 127143: "plaice", 127136: "witch", 126484: "hake"}
STAGE = {"1": "imm", "61": "imm", "A": "imm", "2": "dev", "62": "dev", "B": "dev", "3": "spawn",
         "63": "spawn", "C": "spawn", "4": "spent", "64": "spent", "D": "spent", "65": "rest", "E": "rest"}
hh = pd.concat([pd.read_csv(f"{d}/hh.csv", dtype=str) for d in sys.argv[1:]]).apply(lambda c: c.str.strip())
ca = pd.concat([pd.read_csv(f"{d}/ca.csv", dtype=str) for d in sys.argv[1:]]).apply(lambda c: c.str.strip())
hh = hh.drop_duplicates(KEYS)
ca = ca.merge(hh[KEYS + ["Month", "StatRec", "Day"]], on=KEYS, how="left")
ca["sp"] = pd.to_numeric(ca.Valid_Aphia, errors="coerce").map(SP)
ca["stage"] = ca.Maturity.map(STAGE)
ca["n"] = pd.to_numeric(ca.CANoAtLngt, errors="coerce")
ca["L"] = np.where(ca.LngtCode.isin([".", "0"]), pd.to_numeric(ca.LngtClass, errors="coerce") / 10,
                   pd.to_numeric(ca.LngtClass, errors="coerce"))
ca["box"] = ca.StatRec.isin(BOX) | ca.AreaCode.isin(BOX)
h = ca[(ca.sp == "herring") & ca.box & (ca.stage == "spawn")]
print("herring 'spawning' stage in the box: by year, quarter, month, scale, code; median length")
print(h.groupby(["Year", "Quarter", "Month", "MaturityScale", "Maturity"]).agg(n=("n", "sum"), L=("L", "median")).to_string())
m = ca[ca.stage.isin(["dev", "spawn", "spent", "rest"]) & ca.Month.notna()]
print("\nMATURE fish by month (region incl. box): share in each stage (rows sum 1), n")
for sp in ["cod", "haddock", "whiting", "saithe", "plaice", "witch", "Norway pout", "herring", "sprat", "hake"]:
    d = m[m.sp == sp]
    t = d.groupby(["Month", "stage"]).n.sum().unstack(fill_value=0)
    t = t.reindex(sorted(t.index, key=lambda x: int(float(x))))
    tot = t.sum(axis=1)
    t = t.div(tot, axis=0).round(2)
    t["n"] = tot
    print(f"--- {sp}")
    print(t.to_string())
