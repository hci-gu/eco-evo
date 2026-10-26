"""Bottom-water temperature for the benthivore grid (6.5-8 C claim, resume 223.14 / task step 4).

Usage: python3 -I bottom_temperature.py <shark_tsv> <depth_npz>

  <shark_tsv>: SHARK physical/chemical download for stations A16/A17 (Swedish ring-A letters in the
               station names), 1991-2024, Temperature CTD and bottle. Request (public, no key):
               POST https://shark.smhi.se/api/sample/download with
               {"params":{"tableView":"sharkdata_physicalchemical_columns","delimiters":"point-tab",
                "lineEnd":"unix","encoding":"utf_8","headerLang":"internal_key"},
                "query":{"fromYear":1991,"toYear":2024,"dataTypes":["Physical and Chemical"],
                         "parameters":["Temperature CTD","Temperature bottle"],
                         "stationName":["<A16>","<A17>"]}}
               (same method as research_notes/"Vintermetabolism sill och tumlare"/skagerrak_temperature.md).
  <depth_npz>: maps/depth.npz (60x60 mean cell depth, m).

Method: CTD temperature, bottle as fallback, -2..30 C. Per visit, mean inside depth windows centred on
50, 75, 100, 125, 150, 200, 250, 300 m (+/-5 m below 100 m, +/-10 m at 150-300 m since casts are coarse);
visits -> year-month means -> mean over years (1991-2020 reference, and 2011-2024). Then each grid cell
gets the monthly temperature at its own depth (linear interpolation between levels; cells shallower than
50 m get the 50 m value, deeper than 300 m the 300 m value), and the grid mean / percentiles are printed.
Also: the biomass-weighted grid mean using today's benthivore start map is not available here (it is the
equilibrium map of the engine); the plain cell mean and the depth histogram are given instead.
"""
import sys

import numpy as np
import pandas as pd

SH = pd.read_csv(sys.argv[1], sep="\t", low_memory=False)
DEPTH = np.load(sys.argv[2])["depth"].astype(float).ravel()

t = pd.to_numeric(SH["Temperature CTD (C)"], errors="coerce")
tb = pd.to_numeric(SH["Temperature bottle (C)"], errors="coerce")
SH["T"] = t.where(t.notna(), tb)
SH = SH[(SH["T"] > -2) & (SH["T"] < 30)].copy()
SH["z"] = pd.to_numeric(SH["sample_depth_m"], errors="coerce")
SH["year"] = pd.to_numeric(SH["visit_year"], errors="coerce").astype(int)
SH["month"] = pd.to_numeric(SH["visit_month"], errors="coerce").astype(int)
SH["wd"] = pd.to_numeric(SH["water_depth_m"], errors="coerce")
print("rows", len(SH), "stations", SH.station_name.unique().tolist())
print("max sample depth by station:", SH.groupby("station_name").z.max().to_dict())

LEVELS = {50: 5, 75: 5, 100: 5, 125: 5, 150: 10, 200: 10, 250: 10, 300: 10}


def table(y0, y1):
    d = SH[(SH.year >= y0) & (SH.year <= y1)]
    out = {}
    for lev, half in LEVELS.items():
        w = d[(d.z >= lev - half) & (d.z <= lev + half)]
        v = w.groupby(["station_name", "visit_date"]).agg(T=("T", "mean"), year=("year", "first"),
                                                         month=("month", "first"))
        ym = v.groupby(["year", "month"]).T.mean().reset_index()
        m = ym.groupby("month").T.mean()
        n = ym.groupby("month").T.size()
        m[n.reindex(m.index).fillna(0) < 5] = np.nan   # drop cells with < 5 year-months (250 m is rarely sampled)
        out[lev] = (m.reindex(range(1, 13)), n.reindex(range(1, 13)))
    return out


for (y0, y1) in ((1991, 2020), (2011, 2024)):
    tab = table(y0, y1)
    print(f"\nMonthly mean temperature (C), A16+A17, {y0}-{y1} (n year-months per cell in brackets)")
    print("month " + " ".join(f"{lev:>10d}m" for lev in LEVELS))
    for mo in range(1, 13):
        print(f"{mo:5d} " + " ".join(
            f"{tab[l][0][mo]:6.2f}({int(tab[l][1][mo]) if not np.isnan(tab[l][1][mo]) else 0:2d})" for l in LEVELS))
    ann = {l: float(np.nanmean(tab[l][0].values)) for l in LEVELS}
    rng = {l: (float(np.nanmin(tab[l][0].values)), float(np.nanmax(tab[l][0].values))) for l in LEVELS}
    print("annual " + " ".join(f"{ann[l]:10.2f} " for l in LEVELS))
    print("min-max " + " ".join(f"{rng[l][0]:4.1f}-{rng[l][1]:4.1f}  " for l in LEVELS))

    # Grid cells at their own depth
    levs = np.array(list(LEVELS))
    monthly = np.array([[tab[l][0][mo] for l in LEVELS] for mo in range(1, 13)])  # 12 x L
    # fill any NaN level by the neighbour mean (deep levels are sparse in some months)
    for i in range(12):
        row = monthly[i]
        ok = ~np.isnan(row)
        monthly[i] = np.interp(levs, levs[ok], row[ok])
    zc = np.clip(DEPTH, levs[0], levs[-1])
    cellT = np.array([np.interp(zc, levs, monthly[i]) for i in range(12)])  # 12 x cells
    gm = cellT.mean(axis=1)
    print(f"grid cells at own depth ({y0}-{y1}): monthly grid mean " + " ".join(f"{x:.2f}" for x in gm))
    ca = cellT.mean(axis=0)
    print(f"  annual grid mean {ca.mean():.2f} C; cell annual means p10/p50/p90 "
          f"{np.percentile(ca, 10):.2f}/{np.percentile(ca, 50):.2f}/{np.percentile(ca, 90):.2f}; "
          f"monthly grid-mean range {gm.min():.2f}-{gm.max():.2f}")
    deep = DEPTH >= 100
    print(f"  cells >= 100 m ({deep.mean():.0%} of cells): annual mean {ca[deep].mean():.2f} C; "
          f"cells < 100 m: {ca[~deep].mean():.2f} C")

print("\nDepth histogram of grid cells (m): " + ", ".join(
    f"{a}-{b}: {np.mean((DEPTH >= a) & (DEPTH < b)):.2f}" for a, b in
    ((0, 75), (75, 100), (100, 150), (150, 200), (200, 300), (300, 800))))
