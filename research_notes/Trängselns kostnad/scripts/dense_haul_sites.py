"""Where do the densest benthivore hauls occur, and do they recur at the same site?
For each Q3 survey, the haul with the highest model-group density; distance (km) to the nearest
top-decile haul of every OTHER Q3 survey, against the same statistic for a random haul.
Usage: python3 -I dense_haul_sites.py <tables_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

t = sys.argv[1]
X = pd.read_csv(os.path.join(t, "haul_density.csv"))
X = X[X.Quarter == 3]
X = X[X.groupby("Year").Year.transform("size") >= 10]
rng = np.random.default_rng(2)


def km(a, b, c, d):
    return np.hypot((a - c) * 111.2, (b - d) * 111.2 * np.cos(np.radians(57.8)))


out = []
for col in ("group", "haddock", "cod", "flatfish"):
    top = X.loc[X.groupby("Year")[col].idxmax()]
    dec = X[X[col] >= X.groupby("Year")[col].transform(lambda s: s.quantile(0.9))]
    dist_top, dist_rand = [], []
    for _, r in top.iterrows():
        o = dec[dec.Year != r.Year]
        dist_top.append(km(r.ShootLat, r.ShootLong, o.ShootLat, o.ShootLong).min())
        rr = X[X.Year == r.Year].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        dist_rand.append(km(rr.ShootLat, rr.ShootLong, o.ShootLat, o.ShootLong).min())
    out.append(dict(taxon=col, surveys=len(top), median_km_top=np.median(dist_top),
                    median_km_random=np.median(dist_rand),
                    top_rects=top.StatRec.value_counts().head(4).to_dict(),
                    top_depth_median=top.Depth.median()))
R = pd.DataFrame(out)
print(R.to_string(index=False))
R.to_csv(os.path.join(t, "dense_haul_sites.csv"), index=False)
