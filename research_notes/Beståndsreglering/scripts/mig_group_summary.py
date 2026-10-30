"""Collapse the per-species migration rates onto the model groups (biomass-share weights).

Usage: python3 -I mig_group_summary.py <tables_dir>

Reads mig_q1_over_q3_biomass.csv (relative Q1/Q3 presence, ages 1+), mig_density_ratio_by_age.csv
(profile slope of the unit/stock density ratio over age) and mig_cohort_net_emigration.csv.
Members without NS-IBTS age data get the value given in MISSING (J; the literature sections of
migration.md justify them) and are flagged in the output.
"""
import os
import sys

import numpy as np
import pandas as pd

t = sys.argv[1]
GROUPS = {
    "planktivorous_fish": {"herring": 0.79, "blue_whiting": 0.105, "norway_pout": 0.075, "sprat": 0.03},
    "benthivorous_fish": {"haddock": 0.386, "cod": 0.314, "flatfish_rays_hagfish": 0.30},
    "whiting": {"whiting": 1.0},
    "predatory_fish": {"saithe": 0.651, "mackerel": 0.128, "spurdog_horse_mackerel_hake_other": 0.221},
}
# value used for members without age data: (Q1/Q3 relative presence, profile emigration /yr)
MISSING = {"blue_whiting": (1.0, 0.0), "flatfish_rays_hagfish": (1.0, 0.0),
           "spurdog_horse_mackerel_hake_other": (0.3, 0.0)}
Q = pd.read_csv(os.path.join(t, "mig_q1_over_q3_biomass.csv"))
R = pd.read_csv(os.path.join(t, "mig_density_ratio_by_age.csv"))


def profile(sp, u, q, a0, a1):
    x = R[(R.species == sp) & (R.unit == u) & (R.quarter == q) & R.age.between(a0, a1) & (R.ratio_pooled > 0)]
    return -np.polyfit(x.age, np.log(x.ratio_pooled), 1)[0] if len(x) >= 3 else np.nan


rows = []
for g, mem in GROUPS.items():
    for u in ("BOX", "SKA50"):
        rho = prof = 0.0
        flags = []
        for sp, wgt in mem.items():
            if sp in MISSING:
                r, p = MISSING[sp]
                flags.append(sp)
            elif sp == "mackerel":   # absent in Q1: no age profile; adults (2+) for the presence
                r = float(Q[(Q.species == sp) & (Q.unit == u) & (Q.ages == "2+")].rel_Q1_over_Q3.iloc[0])
                p = 0.0
            else:
                r = float(Q[(Q.species == sp) & (Q.unit == u) & (Q.ages == "1+")].rel_Q1_over_Q3.iloc[0])
                p = profile(sp, u, 1, 1, 4)
            rho += wgt * r
            prof += wgt * p
        rows.append(dict(group=g, unit=u, rel_Q1_over_Q3=rho, profile_emigration_Q1_age1to4=prof,
                         assumed_members=",".join(flags)))
G = pd.DataFrame(rows)
G.to_csv(os.path.join(t, "mig_group_summary.csv"), index=False, float_format="%.3g")
print(G.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
