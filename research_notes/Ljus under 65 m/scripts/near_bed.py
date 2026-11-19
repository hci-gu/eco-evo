"""K_PAR near the bed: slab K by height above the EMODnet bed, for casts over 150-250 m beds only, and a
bed-referenced light estimate at the bed.

Usage: python3 -I near_bed.py > ../tables/near_bed_stdout.txt   (needs voto_k.py, the data and emodnet_bed.json)
Same dark floor, usability rule and slabs as voto_k.py (MPE-PAR casts, sun > 5 deg). Height above bed = bed depth
under the cast position minus the slab mid-depth. The glider turns 2-12 m above the bed (altimeter setting),
so the lowest metres are not sampled.
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this scripts/ folder only
import voto_k as vk  # noqa: E402

HC = ((0, 25), (25, 50), (50, 75), (75, 100), (100, 1e4))


def main():
    meta = json.load(open(os.path.join(vk.DATA, "meta.json")))
    pool = defaultdict(list)
    for ds, m in meta.items():
        if "MPE" not in m["attrs"].get("irradiance", ""):
            continue
        casts, _ = vk.load(ds)
        dark, noise, _ = vk.darkfloor(casts, vk.CH_PAR)
        for c in casts:
            if c["sun"] <= 5:
                continue
            bed = vk.bed_of(c)
            if not np.isfinite(bed):
                continue
            E = vk.corrected(c, vk.CH_PAR, dark, noise, True)
            for lo, hi in vk.SLABS[2:]:
                k = vk.slab_k(E, lo, hi)
                if k is None:
                    continue
                hab = bed - (lo + hi) / 2
                if hab < 0:  # slab below the rounded-position bed (steep slope): skip
                    continue
                h = next(f"{a}-{b:g}" for a, b in HC if a <= hab < b)
                cls = "150-250" if 150 <= bed < 250 else ">=250" if bed >= 250 else "<150"
                pool[(cls, h)].append(k)
                pool[(cls, h, lo)].append(k)
                if hab >= 0 and c["kmax"] * 5 + 5 >= bed - 15:
                    pool[(cls, "reached bed", h)].append(k)
    print("Slab K_PAR (median [p25-p75] n) by height of the slab mid above the bed (m), MPE-PAR casts, sun > 5 deg:")
    res = {}
    for cls in ("<150", "150-250", ">=250"):
        for a, b in HC:
            h = f"{a}-{b:g}"
            n, p25, p50, p75 = vk.q(pool[(cls, h)])
            if n < 10:
                continue
            by = "; ".join(f"{lo}-{lo + 15} {np.median(pool[(cls, h, lo)]):.3f} n{len(pool[(cls, h, lo)])}"
                           for lo, _ in vk.SLABS[2:] if len(pool[(cls, h, lo)]) >= 10)
            res[f"{cls} {h}"] = (n, p25, p50, p75)
            print(f"  bed {cls:>7s}, {h:>8s} m above bed: {p50:.3f} [{p25:.3f}-{p75:.3f}] n{n} | {by}")
    print("\nBed-referenced profile for a bed at depth D (150-250 m class) [D/J]:")
    print("  K above the bed by the class medians: >=100 m above: K_far; 50-100 m: K_50; 25-50: K_25; 0-25: K_0.")
    get = lambda h: res.get(f"150-250 {h}", (0, np.nan, np.nan, np.nan))  # noqa: E731
    k_far = get("100-10000")[2]
    k_75 = get("75-100")[2]
    k_50 = get("50-75")[2]
    k_25 = get("25-50")[2]
    k_0 = get("0-25")[2]
    print(f"  class medians: >=100 {k_far:.3f}; 75-100 {k_75:.3f}; 50-75 {k_50:.3f}; 25-50 {k_25:.3f}; 0-25 {k_0:.3f}")
    print("  Extra optical depth of the lowest 100 m against a uniform K_far: "
          f"{25 * (k_75 - k_far) + 25 * (k_50 - k_far) + 25 * (k_25 - k_far) + 25 * (k_0 - k_far):.2f} "
          f"(x{math.exp(-(25 * (k_75 - k_far) + 25 * (k_50 - k_far) + 25 * (k_25 - k_far) + 25 * (k_0 - k_far))):.2g} light at the bed)")
    json.dump(res, open(os.path.join(vk.TAB, "near_bed.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
