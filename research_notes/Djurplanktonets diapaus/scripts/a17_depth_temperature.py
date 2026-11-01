"""Monthly temperature and oxygen by depth at A17 (and A16) - diapause habitat check.

Usage: python3 -I a17_depth_temperature.py <shark_tsv>

<shark_tsv>: the SHARK physical/chemical download for A16/A17 1991-2024 already
in the repo (research_notes/"Exponering av bottenbyten"/data/E_shark/
shark_A16_A17_temperature_1991_2024.tsv; its request is documented in that
note's scripts/bottom_temperature.py). Reused read-only.

Method: CTD temperature (bottle fallback), CTD oxygen (bottle fallback),
depth windows 0-10, 20-30, 45-55, 95-105, 140-160, 190-210, 240-260,
290-310 m; per visit mean -> year-month mean -> mean over 1991-2020 years.
Station A17 only (A16 is ~ 200 m and is printed separately for 0-200 m).
"""
import sys

import numpy as np
import pandas as pd

SH = pd.read_csv(sys.argv[1], sep="\t", low_memory=False)
t = pd.to_numeric(SH["Temperature CTD (C)"], errors="coerce")
tb = pd.to_numeric(SH["Temperature bottle (C)"], errors="coerce")
SH["T"] = t.where(t.notna(), tb)
o = pd.to_numeric(SH["Dissolved oxygen O2 CTD (ml/l)"], errors="coerce")
ob = pd.to_numeric(SH["Dissolved oxygen O2 bottle (ml/l)"], errors="coerce")
SH["O2"] = o.where(o.notna(), ob)
SH["z"] = pd.to_numeric(SH["sample_depth_m"], errors="coerce")
SH["year"] = pd.to_numeric(SH["visit_year"], errors="coerce")
SH["month"] = pd.to_numeric(SH["visit_month"], errors="coerce")
SH = SH[(SH.year >= 1991) & (SH.year <= 2020)]
WIN = [("0-10", 0, 10), ("25", 20, 30), ("50", 45, 55), ("100", 95, 105),
       ("150", 140, 160), ("200", 190, 210), ("250", 240, 260),
       ("300", 290, 310)]
months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()

for station in sorted(SH.station_name.unique()):
    S = SH[SH.station_name == station]
    print(f"Station {station}: water depth median "
          f"{pd.to_numeric(S.water_depth_m, errors='coerce').median():.0f} m,"
          f" visits {S.visit_date.nunique()}")
    for var, fmt in (("T", "{:6.1f}"), ("O2", "{:6.2f}")):
        label = "temperature C" if var == "T" else "oxygen ml/l"
        print(f"  {label} (mean of year-month means, 1991-2020)")
        print("  depth   " + "".join(f"{m:>6s}" for m in months) + "   ann")
        for name, lo, hi in WIN:
            W = S[(S.z >= lo) & (S.z <= hi) & S[var].notna()]
            if W.empty:
                continue
            v = W.groupby(["visit_date", "year", "month"])[var].mean()
            ym = v.groupby(level=["year", "month"]).mean()
            mon = ym.groupby(level="month").mean()
            row = [mon.get(m, np.nan) for m in range(1, 13)]
            ann = np.nanmean(row)
            print(f"  {name:>6s}  " + "".join(fmt.format(x) for x in row)
                  + "  " + fmt.format(ann))
    print()
