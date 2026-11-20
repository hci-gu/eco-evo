"""Spatial structure of krill-in-stomach presence between hauls of the same cruise
(ship x year x month), ICES stomach records, North Sea/Skagerrak/Kattegat (area != 'other').
A haul is krill-positive if any record (any length class, pooled or single) has krill.
Reports, per species: share of hauls with krill within cruises that had >= 1 krill haul, and
P(haul j positive | haul i positive) by distance bin for pairs within 3 days, against the
marginal share in the same cruises.
Usage: python3 -I q2_haul_pairs.py <records_linked.csv.gz>
"""
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
df = df[df["area"] != "other"].copy()
known = df["anyprey"].astype(str).eq("True") | (df["StomachEmpty"] == df["Number"])
df = df[known & (df["fed"] > 0)]
df["kr"] = (df["occ_krill"] > 0).astype(int)
bins = [0, 10, 25, 50, 100, 200, 1e9]
labels = ["0-10", "10-25", "25-50", "50-100", "100-200", ">200"]
for sp in ["whiting", "saithe", "Norway pout", "cod"]:
    d = df[df["sp"] == sp]
    h = d.groupby("tblHaulID").agg(kr=("kr", "max"), lat=("lat", "mean"), lon=("lon", "mean"),
                                   ship=("ship", "first"), year=("year", "first"), month=("month", "first"),
                                   day=("day", "first")).dropna(subset=["lat", "lon"])
    h["cruise"] = h["ship"].astype(str) + "_" + h["year"].astype(str) + "_" + h["month"].astype(str)
    cr = h.groupby("cruise")["kr"].agg(["size", "sum"])
    good = cr[(cr["size"] >= 3) & (cr["sum"] >= 1)].index
    hh = h[h["cruise"].isin(good)]
    print(f"== {sp}: hauls {len(h)}, cruises with >= 3 hauls and >= 1 krill haul: {len(good)}, their hauls {len(hh)},"
          f" share positive in them {hh['kr'].mean():.3f} (all hauls {h['kr'].mean():.3f})")
    rows = []
    for c, g in hh.groupby("cruise"):
        la = np.radians(g["lat"].to_numpy()); lo = np.radians(g["lon"].to_numpy())
        k = g["kr"].to_numpy(); dd = g["day"].to_numpy()
        for i in range(len(g)):
            for j in range(len(g)):
                if i == j:
                    continue
                if not np.isnan(dd[i]) and not np.isnan(dd[j]) and abs(dd[i] - dd[j]) > 3:
                    continue
                a = np.sin((la[j]-la[i])/2)**2 + np.cos(la[i])*np.cos(la[j])*np.sin((lo[j]-lo[i])/2)**2
                rows.append((6371*2*np.arcsin(np.sqrt(a)), k[i], k[j]))
    if not rows:
        continue
    r = pd.DataFrame(rows, columns=["km", "ki", "kj"])
    r["bin"] = pd.cut(r["km"], bins, labels=labels, right=False)
    pos = r[r["ki"] == 1]
    tab = pos.groupby("bin", observed=False)["kj"].agg(["size", "mean"])
    tab_all = r.groupby("bin", observed=False)["kj"].mean()
    tab["marginal_j"] = tab_all
    print("   P(j positive | i positive) by distance (ordered pairs within 3 days):")
    print(tab.round(3).to_string())
