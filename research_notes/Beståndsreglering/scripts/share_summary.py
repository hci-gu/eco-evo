"""Per-member box share of stock biomass, SSB and recruits; upper bound of self-recruitment
(WP3, box_share.md).

Usage: python3 -I share_summary.py <share_sag.csv> <share_sms_tsb.csv> <share_survey.csv> <out_csv>

- SAG share: box biomass (model density x 3600 km2 x member share) / SAG SSB, and / TSB where
  TSB = SAG SSB x the SMS 2023 TSB/SSB ratio (2001-22 mean) or SAG TBiomass (hake), else a [J] ratio.
- survey shares: the climatological range (R and S estimators) over the survey area that best
  matches the stock (column 'area').
- f_self_max = min(1, s_spawners / s_recruits): the largest possible share of the box's recruits
  that the box's own spawners could have produced, if every one of their offspring stayed in the
  box and survived as well as the immigrants (equal survival). [D under that assumption]
"""
import sys

import numpy as np
import pandas as pd

sag_csv, sms_csv, surv_csv, out = sys.argv[1:5]
S = pd.read_csv(sag_csv).set_index("member")
M = pd.read_csv(sms_csv).set_index("species")
V = pd.read_csv(surv_csv)
V = V[~V.stage.str.endswith("_kg")]

# member -> (SAG row, survey species, survey area, TSB/SSB ratio source, recruit stage)
MAP = [
    ("herring NSAS", "herring", "4+3a", ("sms", "herring"), "age1"),
    ("herring WBSS", "herring", "3a (20+21)", ("J", 1.5), None),
    ("blue whiting", "blue whiting", "4+3a", ("J", 1.3), None),
    ("Norway pout", "Norway pout", "4+3a", ("sms", "Norway pout"), "age1"),
    ("sprat", "sprat", "4+3a", ("sms", "sprat"), "age1"),
    ("haddock", "haddock", "4+3a20", ("sms", "haddock"), "age0"),
    ("cod (Viking substock)", "cod", "Viking>2E", ("sms", "cod"), "age1"),
    ("cod (whole NS stock)", "cod", "4+3a20", ("sms", "cod"), "age1"),
    ("long rough dab", "long rough dab", "4+3a", None, None),
    ("dab", "dab", "4+3a", None, None),
    ("plaice (ple.27.420)", "plaice", "4+3a20", ("sms", "plaice"), "age1"),
    ("witch", "witch", "4+3a", ("J", 1.3), None),
    ("saithe", "saithe", "4+3a", ("sms", "saithe"), None),
    ("mackerel", "mackerel", "4+3a", ("J", 1.2), None),
    ("spurdog", "spurdog", "4+3a", ("J", 1.0), None),
    ("horse mackerel (western)", "horse mackerel", "4+3a", ("J", 1.3), None),
    ("hake", "hake", "4+3a", ("tb", None), "age0"),
    ("whiting (whg.27.3a)", "whiting", "3a (20+21)", None, "age1"),
    ("whiting (whg.27.47d)", "whiting", "4+3a", ("sms", "whiting"), "age1"),
]


def rng(sp, stage, area, q=None):
    v = V[(V.species == sp) & (V.stage == stage) & (V.area == area)]
    if q is not None:
        v = v[v.quarter == q]
    if v.empty:
        return np.nan, np.nan
    x = v[["share_R", "share_S"]].values.ravel()
    return x.min(), x.max()


rows = []
for m, sp, area, rat, rstage in MAP:
    s = S.loc[m]
    r = dict(member=m, group=s.group, box_t=s.box_t, stock=s.get("stock"), survey_area=area)
    ssb = s.get("SSB_0125")
    if s.get("ssb_desc") == "SSB" and pd.notna(ssb):
        r["SSB_0125_t"] = ssb
        r["share_SSB"] = s.box_t / ssb
        r["share_SSB_2125"] = s.box_t / s.SSB_2125
        if rat is not None:
            if rat[0] == "sms":
                ratio = M.loc[rat[1], "ratio_mean"]
            elif rat[0] == "tb":
                ratio = s.TBiomass_0125 / ssb
            else:
                ratio = rat[1]
            r["tsb_ssb_ratio"] = ratio
            r["ratio_src"] = rat[0]
            r["share_TSB"] = s.box_t / (ssb * ratio)
    # survey: biomass (Q1, Q3), spawners (Q1; herring and blue whiting Q3), recruits
    for q in (1, 3):
        lo, hi = rng(sp, "kg", area, q)
        r[f"surv_kg_Q{q}_lo"], r[f"surv_kg_Q{q}_hi"] = lo, hi
    mq = 3 if sp in ("herring", "blue whiting") else 1
    r["surv_mature_lo"], r["surv_mature_hi"] = rng(sp, "mature", area, mq)
    for st, q in (("age0", 3), ("age1", 1)):
        r[f"surv_{st}_lo"], r[f"surv_{st}_hi"] = rng(sp, st, area, q)
    if rstage is not None and pd.notna(r["surv_mature_lo"]):
        rec_lo, rec_hi = r[f"surv_{rstage}_lo"], r[f"surv_{rstage}_hi"]
        r["recruit_stage"] = rstage
        r["f_self_max_lo"] = min(1.0, r["surv_mature_lo"] / rec_hi)
        r["f_self_max_hi"] = min(1.0, r["surv_mature_hi"] / rec_lo)
    rows.append(r)

T = pd.DataFrame(rows)
T.to_csv(out, index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
pct = [c for c in T.columns if c.startswith(("share_", "surv_"))]
P = T.copy()
P[pct] = (P[pct] * 100).round(3)
print("percent of the stock (survey columns: lo-hi of the R and S estimators)\n")
print(P[["member", "box_t", "SSB_0125_t", "share_SSB", "share_SSB_2125", "tsb_ssb_ratio", "share_TSB",
         "survey_area", "surv_kg_Q1_lo", "surv_kg_Q1_hi", "surv_kg_Q3_lo", "surv_kg_Q3_hi"]].to_string(index=False))
print()
print(P[["member", "survey_area", "surv_mature_lo", "surv_mature_hi", "surv_age0_lo", "surv_age0_hi",
         "surv_age1_lo", "surv_age1_hi", "recruit_stage", "f_self_max_lo", "f_self_max_hi"]].to_string(index=False))
