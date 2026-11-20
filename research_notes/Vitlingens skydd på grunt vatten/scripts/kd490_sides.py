"""Surface Kd490 (MODIS Aqua, monthly 4 km, R2022 SQ; NOAA CoastWatch ERDDAP erdMH1kd490mday_R2022SQ, 2015-2022)
over the box by bed-depth class and by coast side: is the Danish shallow side as turbid as the Swedish one the gliders
measured?

Usage: python3 -I kd490_sides.py <etopo1 csvp.gz> <kd490 csv> > ../tables/kd490_sides_stdout.txt
Bed depth per pixel: ETOPO1 (1 arc-minute) nearest point, from research_notes/"Krill över grunda bottnar"/stomachs/
data/etopo1_northsea/etopo1_54_64N_m6_15E.csvp.gz (NOAA, public domain; read in place). Box 57.7-58.4 N,
9.9-11.3 E. Sides: Danish (lat < 57.95, lon < 10.9) and Swedish (lon >= 10.9) for
< 100 m beds, as in the GOV split. Statistic: per pixel and calendar month the median over years; then the median
over pixels of a group; annual = mean over months with >= 3 valid pixels in every group. Satellite K is the first
optical depth only (~ top 5-15 m), so it shows the water mass, not the 20-90 m column.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

import gzip

etopo, f = sys.argv[1], sys.argv[2]
ET = {}
with gzip.open(etopo, "rt") as fh:
    rr = csv.reader(fh)
    next(rr)
    for la, lo, z in rr:
        la, lo = float(la), float(lo)
        if 57.6 <= la <= 58.5 and 9.8 <= lo <= 11.4:
            ET[(round(la * 60), round(lo * 60))] = -float(z)


def bed_at(lat, lon):
    if not (57.7 <= lat < 58.4 and 9.9 <= lon < 11.3):
        return np.nan
    v = ET.get((round(lat * 60), round(lon * 60)))
    return v if v is not None and v > 0 else np.nan


vals = defaultdict(lambda: defaultdict(list))  # (lat, lon) -> month -> values
with open(f) as fh:
    r = csv.reader(fh)
    next(r)
    next(r)
    for t, la, lo, k in r:
        if k == "NaN":
            continue
        vals[(float(la), float(lo))][int(t[5:7])].append(float(k))


def group(lat, lon, bed):
    if not np.isfinite(bed):
        return None
    if bed < 100:
        if lon >= 10.9:
            return "< 100 m, Swedish side"
        if lat < 57.95:
            return "< 100 m, Danish side"
        return "< 100 m, other"
    if bed < 150:
        return "100-150 m"
    if bed < 250:
        return "150-250 m"
    return ">= 250 m"


G = defaultdict(lambda: defaultdict(list))
for (la, lo), mm in vals.items():
    g = group(la, lo, bed_at(la, lo))
    if g is None:
        continue
    for mo, v in mm.items():
        G[g][mo].append(float(np.median(v)))
names = ["< 100 m, Danish side", "< 100 m, Swedish side", "< 100 m, other", "100-150 m", "150-250 m", ">= 250 m"]
print("median over pixels of the per-pixel monthly median Kd490 (m-1), n pixels:")
print("  group                    " + " ".join(f"{m:>6d}" for m in range(1, 13)))
for g in names:
    print(f"  {g:24s} " + " ".join(f"{np.median(G[g][m]):6.3f}" if len(G[g][m]) >= 3 else "     -" for m in range(1, 13))
          + f"   (n pix {max((len(G[g][m]) for m in range(1, 13)), default=0)})")
ok = [m for m in range(1, 13) if all(len(G[g][m]) >= 3 for g in names if g != "< 100 m, other")]
print(f"\nmonths with every group >= 3 pixels: {ok}")
for g in names:
    if g == "< 100 m, other":
        continue
    print(f"  {g:24s} mean over those months {np.mean([np.median(G[g][m]) for m in ok]):.3f}")
