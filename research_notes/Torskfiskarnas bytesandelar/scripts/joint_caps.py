"""Joint caps: the share of gadoid biomass that can feed on F or Z (pelagic prey) at the full rate.

Usage: python3 -I joint_caps.py <note_dir>
From tables/phi_contributions.csv (central scores): per species x stage, score_j = contribution /
biomass share. A stage can time-share between pools but not feed on two at the full rate at once, so
the joint cap of a set of pools is sum_stage w x max_j score_j. Writes tables/joint_caps.csv.
"""
import os
import sys

import pandas as pd

note = sys.argv[1]
C = pd.read_csv(os.path.join(note, "tables", "phi_contributions.csv"))
rows = []
for (v, q), c in C.groupby(["variant", C.quarter.astype(str)]):
    c = c[c.biomass_share > 0].copy()
    c["score"] = c.contribution / c.biomass_share
    P = c.pivot_table(index=["species", "stage"], columns="pool", values="score")
    w = c.groupby(["species", "stage"]).biomass_share.first()
    r = dict(variant=v, quarter=q)
    for name, pools in [("F_or_Z", ["fish", "zoo"]), ("B_or_F_or_Z", ["benthos", "fish", "zoo"]),
                        ("B", ["benthos"]), ("F", ["fish"]), ("Z", ["zoo"])]:
        r[name] = float((w * P[pools].max(axis=1)).sum())
    r["forced_B_min_if_caps_F_Z"] = max(0.0, 1 - r["F"] - r["Z"])
    r["forced_B_min_if_joint_cap"] = max(0.0, 1 - r["F_or_Z"])
    rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(note, "tables", "joint_caps.csv"), index=False)
print(R.round(3).to_string(index=False))
