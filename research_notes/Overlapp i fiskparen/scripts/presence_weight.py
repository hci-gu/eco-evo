"""Prey-presence weighting of the band a's: the engine multiplies a constant a by the prey biomass B(t), so the
constant that reproduces the season-resolved encounter sum_t a(t) B(t) is the B-weighted mean of a(t), not the
equal-time mean. Prey seasons = the library's outside_season (12 monthly factors) of the prey group (mackerel:
the migration; planktivores: the adult window; krill: the open pool). Resident prey (benthivores, whiting): 1.

Usage: python3 -I presence_weight.py <repo_root> <band_pairs.json>
"""
import json
import os
import sys

import numpy as np
import yaml

ROOT = sys.argv[1]
R = json.load(open(sys.argv[2]))
SP = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8-sig"))["species_definitions"]
print("pair: presence-weighted a / equal-time a (month_rel = monthly a / annual a), per variant")
for var in ("lib_band", "A_band", "A_band_Hs5", "A_band_Hs40"):
    for name, r in R["variants"][var]["fish"].items():
        if "month_rel" not in r:
            continue
        prey = name.split("->")[1]
        s = SP[prey].get("outside_season") or [1.0] * 12
        s = np.asarray(s, float)
        m = np.asarray(r["month_rel"])
        f = float((m * s).sum() / s.sum())
        print(f"  {var:11s} {name:38s} x{f:.3f}   months " + " ".join(f"{x:.2f}" for x in m))
