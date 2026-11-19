"""Daytime light at a 150, 200 and 250 m bed: mid-water K plus the measured near-bed layer.

Usage: python3 -I bed_light.py > ../tables/bed_light_stdout.txt   (needs ../tables/voto_k.json, near_bed.json)

Profile above 65 m and surface: the prior note's convention (deep_light.py, imported).
Below 65 m, for a bed at depth D (150-250 m class statistics from near_bed.py, MPE-PAR 2024-26):
  slab more than 100 m above the bed: K_far (casts >= 100 m above a 150-250 m bed)
  75-100, 50-75, 25-50, 0-25 m above the bed: the class medians of those bands.
The near-bed bands are measured only where the light was above the floor (slab mid <= 170 m, i.e. beds of
150-195 m); for deeper beds the same bed-referenced layer is assumed [J].
Variants: median / p25 (bright) / p75 (dark) of each band; 'uniform' = no near-bed layer (K_far to the bed);
'glider anchor' = start from the glider-measured daytime median at 120 m over 150-250 m beds instead of the
convention (the measured absolute level is ~0.3-0.4 of the convention's there); old K_deep 0.06 / 0.05 / 0.09 for
comparison. lx at 54 per umol (headline, prior convention) and at the spectral estimate (spectral.py: 45 at
150 m, 42 at 200 m, 40 at 250 m; flat-quanta case).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this scripts/ folder only
import deep_light as dl  # noqa: E402

TAB = os.path.join(dl.HERE, "..", "tables")
LXQ = {150: 45.3, 200: 41.9, 250: 39.7}


def layers(D, k_far, k75, k50, k25, k0):
    """K below 65 m for a bed at D; bands referenced to the bed, k_far above D - 100."""
    lay = []
    edges = [(65, D - 100, k_far), (D - 100, D - 75, k75), (D - 75, D - 50, k50), (D - 50, D - 25, k25), (D - 25, D, k0)]
    for top, bot, k in edges:
        top = max(top, 65)
        if bot > top:
            lay.append((top, bot, k))
    return lay


def main():
    nb = json.load(open(os.path.join(TAB, "near_bed.json")))
    D_ = json.load(open(os.path.join(TAB, "voto_k.json")))["daymed"]
    bands = ["100-10000", "75-100", "50-75", "25-50", "0-25"]
    stats = {lab: [nb[f"150-250 {b}"][i] for b in bands] for lab, i in (("p25 (bright)", 1), ("median", 2), ("p75 (dark)", 3))}
    print("K bands (>=100, 75-100, 50-75, 25-50, 0-25 m above the bed), 150-250 m beds, MPE-PAR:")
    for lab, ks in stats.items():
        print(f"  {lab:12s} " + " ".join(f"{k:.3f}" for k in ks))
    out = {}
    print("\nDaytime median (sun > 0, all months) at the bed, lx at 54 per umol [umol m-2 s-1] {lx at the spectral lx/umol}:")
    rows = []
    for lab, ks in stats.items():
        rows.append((f"near-bed layer, {lab}", lambda D, ks=ks: layers(D, *ks)))
    rows.append(("uniform K_far median, no near-bed layer", lambda D: [(65, D, stats["median"][0])]))
    for kd in (0.05, 0.06, 0.09):
        rows.append((f"old judged K_deep {kd}", lambda D, kd=kd: [(65, D, kd)]))
    for name, f in rows:
        cells = []
        for D in (150, 200, 250):
            lay = f(D) + [(D, 1e4, 0.1)]
            lx = dl.day_median(D, lay)
            out[f"{name} {D}"] = lx
            cells.append(f"{D} m {lx:.2g} lx [{lx / 54:.2g}] {{{lx / 54 * LXQ[D]:.2g}}}")
        print(f"  {name:42s} " + "; ".join(cells))
    # glider-anchored: measured daytime median PAR at 120-125 m over 150-250 m beds (5 m bin 120-125)
    m120 = D_.get("MPE-PAR bed 150-250 120")
    if m120:
        print(f"\nGlider anchor: measured daytime median PAR at 120-125 m over 150-250 m beds {m120:.2g} umol "
              f"({m120 * 54:.2g} lx at 54)")
        for lab, ks in stats.items():
            cells = []
            for D in (150, 200, 250):
                lay = layers(D, *ks)
                tau = sum(k * (min(bot, D) - max(top, 122.5)) for top, bot, k in lay if bot > 122.5)
                v = m120 * math.exp(-tau)
                out[f"anchor {lab} {D}"] = v * 54
                cells.append(f"{D} m {v * 54:.2g} lx [{v:.2g}]")
            print(f"  {lab:12s} " + "; ".join(cells))
    json.dump(out, open(os.path.join(TAB, "bed_light.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
