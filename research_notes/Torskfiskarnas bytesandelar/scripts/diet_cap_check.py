"""Cap test of phi against diet by mass (check only; diets never set a score).

Usage: python3 -I diet_cap_check.py <note_dir>
Reads inputs/diet_check.csv (species-level diet by mass mapped to model pools, low / c / high, from
evidence_C_diet_check.md) and tables/phi_contributions.csv (raw, central). Writes
tables/diet_cap_check.csv: per species the stage-weighted score (phi_i,j = contribution / species
biomass share) next to its diet share, and the group-level diet share weighted by the box mix (Q1, Q3,
pooled) next to phi_j. phi_j caps the group's eat allocation to pool j, so the test is diet <= phi.
Assumes equal ration per unit biomass across species (not measured).
"""
import os
import sys

import pandas as pd

note = sys.argv[1]
D = pd.read_csv(os.path.join(note, "inputs", "diet_check.csv"))
C = pd.read_csv(os.path.join(note, "tables", "phi_contributions.csv"))
C = C[C.variant == "raw"]
MAIN = ["haddock", "whiting", "saithe", "cod", "hake"]
POOLMAP = {"benthos": "B", "fish": "F", "zoo": "Z"}
rows = []
for q in ["1", "3", "pooled"]:
    c = C[C.quarter.astype(str) == q]
    w = c[c.pool == "benthos"].groupby("species").biomass_share.sum()
    for p, col in POOLMAP.items():
        sc = c[c.pool == p].groupby("species").contribution.sum()
        phi = sc.sum()
        for s in MAIN:
            r = dict(quarter=q, pool=p, species=s, biomass_share=w[s], score=sc[s] / w[s])
            for b in ["low", "c", "high"]:
                r["diet_" + b] = float(D[(D.species == s) & (D.bound == b)][col].iloc[0])
            rows.append(r)
        g = {b: sum(w[s] * float(D[(D.species == s) & (D.bound == b)][col].iloc[0]) for s in MAIN) for b in
             ["low", "c", "high"]}
        rows.append(dict(quarter=q, pool=p, species="GROUP", biomass_share=w[MAIN].sum(), score=phi,
                         diet_low=g["low"], diet_c=g["c"], diet_high=g["high"]))
    of = {b: sum(w[s] * float(D[(D.species == s) & (D.bound == b)]["Ofish"].iloc[0]) for s in MAIN) for b in
          ["low", "c", "high"]}
    rows.append(dict(quarter=q, pool="other_fish", species="GROUP", biomass_share=w[MAIN].sum(), score=float("nan"),
                     diet_low=of["low"], diet_c=of["c"], diet_high=of["high"]))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(note, "tables", "diet_cap_check.csv"), index=False)
pd.set_option("display.width", 200, "display.max_rows", 200)
print(R.round(3).to_string(index=False))
