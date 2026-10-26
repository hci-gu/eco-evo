"""Share of 0-100 m copepod biomass in 0-25 m, Skagerrak, from Tonnesson et al. 2006
(MEPS 314:213, Table 2: pump samples, 45 um, March 1999 and Aug-Sep 2000, stations K, H,
T1-T4 at 200-600 m water depth; transcribed to tonnesson2006_table2.csv).

Usage: python3 -I skagerrak_copepod_strata.py <tonnesson2006_table2.csv>

Biomass per individual = L^b (cube-law, constant cancels in shares), L = the mean size per
taxon and season given in the table header (um). b = 3.0 central, 2.7 / 3.3 sensitivity.
Prey sets: 'all' copepods; 'herring' = without Microsetella (harpacticoid on aggregates) and
Oithona (small ambush feeder below most herring size selectivity) - judgement.
"""
import sys
import pandas as pd

d = pd.read_csv(sys.argv[1])
size = {"spring": dict(Cal=1810, Par=510, Pseu=560, Mic=440, Tem=890, Ac=700, Oith=400, Cent=690, Micr=460),
        "autumn": dict(Cal=1080, Par=760, Pseu=780, Mic=400, Tem=900, Ac=720, Oith=370, Cent=840, Micr=420)}
taxa = list(size["spring"])
d[["z0", "z1"]] = d.stratum.str.split("-", expand=True).astype(float)
d["dz"] = d.z1 - d.z0
sets = {"all": taxa, "herring": [t for t in taxa if t not in ("Micr", "Oith")], "Calanus": ["Cal"]}
rows = []
for b in (2.7, 3.0, 3.3):
    for name, ts in sets.items():
        for (season, st), g in d.groupby(["season", "station"]):
            w = sum(g[t] * (size[season][t] / 1000.0) ** b for t in ts)  # rel. biomass m-3
            col = (w * g.dz)
            upper = col[g.z1 <= 25].sum()
            lower = col[g.z0 >= 25].sum()
            tot = upper + lower
            if tot <= 0:
                continue
            c_up = upper / 25.0
            c_lo = lower / 75.0
            zc = ((g.z0 + g.z1) / 2 * col).sum() / tot
            rows.append(dict(b=b, set=name, season=season, station=st, share_0_25=upper / tot,
                             conc_ratio_25_100_vs_0_25=c_lo / c_up if c_up > 0 else float("nan"),
                             zc_m=zc, H_eff_m=tot / c_up if c_up > 0 else float("nan")))
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("Per station (b = 3.0):")
print(r[r.b == 3.0].round(2).to_string(index=False))
print("\nMeans over stations by season and prey set (share of 0-100 m in 0-25 m; concentration in 25-100 m relative to 0-25 m;")
print("centre of mass; H_eff = 0-100 m column / mean 0-25 m concentration):")
s = r.groupby(["b", "set", "season"])[["share_0_25", "conc_ratio_25_100_vs_0_25", "zc_m", "H_eff_m"]].agg(["mean", "min", "max"])
print(s.round(2).to_string())

# Concentration in 25-60 m and 60-100 m relative to the 0-25 m mean (5-strata stations only)
print("\nFive-strata stations: concentration relative to the 0-25 m mean, b = 3.0")
out = []
for name, ts in sets.items():
    for (season, st), g in d.groupby(["season", "station"]):
        if len(g) < 5:
            continue
        w = sum(g[t] * (size[season][t] / 1000.0) ** 3.0 for t in ts)
        c = dict(zip(g.stratum, w))
        up = (c["0-10"] * 10 + c["10-25"] * 15) / 25
        if up <= 0:
            continue
        mid = (c["25-40"] * 15 + c["40-60"] * 20) / 35
        out.append(dict(set=name, season=season, station=st, r_25_60=mid / up, r_60_100=c["60-100"] / up,
                        r_0_10=c["0-10"] / up, r_10_25=c["10-25"] / up))
o = pd.DataFrame(out)
print(o.round(3).to_string(index=False))
print(o.groupby(["set", "season"])[["r_0_10", "r_10_25", "r_25_60", "r_60_100"]].agg(["mean", "min", "max"]).round(3).to_string())
