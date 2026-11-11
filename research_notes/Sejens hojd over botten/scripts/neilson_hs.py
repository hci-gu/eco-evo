"""Scale height Hs of an exponential profile exp(-h/Hs) implied by Neilson et al. 2003 (Scotian Shelf pollock =
saithe, pre-spawning aggregations, Sep 30 - Oct 3 1999, 38 kHz, Sa integrated in 1-5, 6-10, ..., 26-30 m off
bottom; the first metre is the dead zone): the 1-5 m share of the 1-30 m Sa read from their Fig. 7 as given in
research_notes/"Vattenpelarens bottenfiskbyte"/evidence_literature.md A1 (Banana ~27 % day / ~37 % night; Shelf
Edge ~68 % day / ~80 % night) [S, read from a figure]. Also DFO CSAS 2003/075: 75 % of the biomass within 10 m
of the bed in Dec-Jan (echo logger; 3-13 m by bimonth for the 75 % height) [S].
Usage: python3 -I neilson_hs.py
"""
import math
from scipy.optimize import brentq


def share(Hs, a, b, lo, hi):
    f = lambda x: math.exp(-x / Hs)
    return (f(a) - f(b)) / (f(lo) - f(hi))


print("Neilson 2003: 1-5 m share of the 1-30 m layer -> Hs (exponential from the bed)")
for site, val in (("Banana day", 0.27), ("Banana night", 0.37), ("Shelf Edge day", 0.68), ("Shelf Edge night", 0.80)):
    # share falls from 1 (Hs -> 0) to 4/29 (uniform); solve
    if val <= 4 / 29:
        print(f"  {site:18s} {val:.2f}: below the uniform share 0.138 -> no decaying exponential")
        continue
    hs = brentq(lambda H: share(H, 1, 5, 1, 30) - val, 0.05, 1e4)
    print(f"  {site:18s} {val:.2f}: Hs {hs:6.2f} m; bottom-2-m share of the full profile {1 - math.exp(-2 / hs):.3f}")
    for v2 in (val - 0.05, val + 0.05):
        h2 = brentq(lambda H: share(H, 1, 5, 1, 30) - v2, 0.05, 1e4) if v2 > 4 / 29 else float('inf')
        print(f"      share {v2:.2f}: Hs {h2:.2f} m")
print("\nDFO CSAS 2003/075: height holding 75 % of the biomass -> Hs = h75 / ln 4")
for h75 in (3.0, 10.0, 13.0):
    hs = h75 / math.log(4)
    print(f"  h75 {h75:5.1f} m: Hs {hs:5.2f} m; bottom-2-m share {1 - math.exp(-2 / hs):.3f}")
print("\nNote: both are deliberately sampled near-bed aggregations (Neilson: only 0-30 m integrated, so any saithe"
      " above 30 m is outside the share); they bound Hs from below for a pre-spawning / winter aggregation.")
