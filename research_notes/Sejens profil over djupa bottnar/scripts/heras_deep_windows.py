"""Adult saithe in HERAS / IESSNS pelagic hauls over DEEP beds (>= 150 m), by the fished window's absolute depth
and height above the bed, day and night; with the prey caught in the same hauls (blue whiting, Norway pout,
herring, krill) so the windows that fished the prey layers can be seen.

Usage: python3 -I heras_deep_windows.py <hauls.tsv> <catch.tsv> <individuals.tsv> <haul_meta.tsv>
  (the inputs of research_notes/"Sejens hojd over botten"/scripts/heras_saithe_height.py, whose data-preparation
  head is exec'd unchanged: windows, sun elevation, adult share by individual weights, kg/h).
Region 54.6-59.1 N, 5-12.4 E, June-July. Hauls were aimed at acoustic marks (mostly herring): occupancy within
the hauls, not a survey design [J].
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "Sejens hojd over botten", "scripts", "heras_saithe_height.py")
src = open(SRC).read()
head = src[:src.index('print("=" * 110)')]
G = {"__name__": "heras_head", "__file__": SRC}
exec(compile(head, SRC, "exec"), G)
H, C = G["H"], G["C"]
PREY = {"126439": "blue whiting", "126444": "pout", "126417": "herring", "126425": "sprat"}
for aid, nm in PREY.items():
    w = C[C.species == aid].groupby(["cruise", "haul"]).weight_kg.sum().rename(nm)
    H = H.merge(w.reset_index(), on=["cruise", "haul"], how="left")
    H[nm] = H[nm].fillna(0.0) / H.hours
kr = C[C.species.isin([str(x) for x in (110671, 110690, 110672, 110676, 1128)])]
H = H.merge(kr.groupby(["cruise", "haul"]).weight_kg.sum().rename("euphausiids").reset_index(),
            on=["cruise", "haul"], how="left")
H["euphausiids"] = H.euphausiids.fillna(0.0) / H.hours
H["top"] = H.min_depth
H["foot"] = H.min_depth + H.open
H["mid"] = 0.5 * (H.top + H.foot)
D = H[(H.bottom >= 150)].copy()


def band(r):
    if r.hab_foot <= 3:
        return "a: footrope 0-3 m above bed"
    if r.hab_foot <= 30:
        return "b: footrope 3-30 m above bed"
    if r.hab_foot <= 60:
        return "c: footrope 30-60 m above bed"
    if r.foot >= 60 and r.top <= 160:
        return "d: >60 m up, window in 60-160 m depth"
    return "e: >60 m up, window above 60 m"


D["band"] = D.apply(band, axis=1)
D["yy"] = D.y.fillna(-1)
D["has"] = D.y > 0
cols = ["blue whiting", "pout", "herring", "sprat", "euphausiids"]
for per, dd in (("DAY (sun > 0 deg)", D[D.day]), ("NIGHT (sun <= 0 deg)", D[~D.day])):
    print("=" * 112)
    print(f"{per}, beds >= 150 m: hauls / with adult saithe / mean adult kg/h (unmeasured catches dropped) / bed median /"
          " window depth median; prey: share of hauls with the prey and mean kg/h")
    print("=" * 112)
    for b, g in dd.groupby("band"):
        ok = g[g.y.notna()]
        prey = "; ".join(f"{c} {float((g[c] > 0).mean()):.2f} ({g[c].mean():.0f})" for c in cols)
        print(f"  {b:42s} n {len(g):3d}  saithe {int(ok.has.sum()):2d}  kg/h {ok.y.mean() if len(ok) else float('nan'):7.1f}"
              f"  bed {g.bottom.median():5.0f}  window {g.top.median():4.0f}-{g.foot.median():4.0f} m |  {prey}")
    for lo, hi in ((150, 250), (250, 1000)):
        g = dd[(dd.bottom >= lo) & (dd.bottom < hi)]
        print(f"    bed {lo}-{hi} m: " + ", ".join(f"{b[:1]} {int(((g.band == b) & g.has).sum())}/{int((g.band == b).sum())}"
                                             for b in sorted(D.band.unique())))
print("\nDay hauls over beds >= 150 m whose window lay inside 60-160 m depth and > 60 m above the bed (the krill and")
print("blue-whiting day layers): every haul")
g = D[D.day & (D.band.str.startswith("d"))]
print(g[["cruise", "haul", "gear", "bottom", "top", "foot", "elev", "y"] + cols].round(1).to_string(index=False))
print("\nNight hauls over beds >= 150 m with adult saithe: every haul")
g = D[(~D.day) & D.has]
print(g[["cruise", "haul", "gear", "bottom", "top", "foot", "hab_foot", "elev", "y", "med_len"] + cols].round(1).to_string(index=False))
