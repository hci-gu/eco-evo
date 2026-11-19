"""Night readings of the MPE-PAR gliders at depth: the dark floor and any excess (bioluminescence upper bound).

Usage: python3 -I night_floor.py > ../tables/night_floor_stdout.txt
Input: ../data/voto/<dataset>_bin5.csv (5 m bin means, so single flashes are averaged over ~25-50 s of flight).
Night = sun < -12 deg. Per mission and depth band: percentiles of the night bin means, in umol m-2 s-1
(instrument unit uE cm-2 s-1 x 1e4), and the excess of the p90 / p99 over the band's p10 (taken as dark).
The glider itself stimulates luminescence (hull, wings), so any excess is an upper bound on ambient
bioluminescence, and part of it may be dark drift with temperature.
"""
import json
import os

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this scripts/ folder only
import voto_k as vk  # noqa: E402

BANDS = [(50, 100), (100, 150), (150, 200), (200, 260)]


def main():
    meta = json.load(open(os.path.join(vk.DATA, "meta.json")))
    print("MPE-PAR night casts (sun < -12 deg), 5 m bin means, umol m-2 s-1:")
    print("  mission              band     n   p10       p50       p90       p99      (p90-p10)  (p99-p10)  frac>2xp10")
    allex = {b: [] for b in BANDS}
    for ds, m in meta.items():
        if "MPE" not in m["attrs"].get("irradiance", ""):
            continue
        casts, _ = vk.load(ds)
        night = [c for c in casts if c["sun"] < -12]
        if len(night) < 20:
            continue
        for lo, hi in BANDS:
            v = np.array([row["downwelling_PAR"] for c in night for k, row in c["bins"].items()
                          if lo <= 5 * k + 2.5 < hi and np.isfinite(row["downwelling_PAR"])]) * 1e4
            if len(v) < 50:
                continue
            p10, p50, p90, p99 = np.percentile(v, [10, 50, 90, 99])
            allex[(lo, hi)].append((p90 - p10, p99 - p10))
            print(f"  {ds:20s} {lo:3d}-{hi:3d} {len(v):5d} {p10:9.2e} {p50:9.2e} {p90:9.2e} {p99:9.2e}  {p90 - p10:9.2e}  {p99 - p10:9.2e}"
                  f"  {np.mean(v > 2 * p10):.3f}")
    print("\nMedian over missions of the excess (umol m-2 s-1; x 54 for a nominal lx):")
    for b, ex in allex.items():
        if ex:
            a = np.array(ex)
            print(f"  {b[0]}-{b[1]} m: p90-p10 {np.median(a[:, 0]):.2e}, p99-p10 {np.median(a[:, 1]):.2e}"
                  f" (max over missions {a[:, 1].max():.2e})")


if __name__ == "__main__":
    main()
