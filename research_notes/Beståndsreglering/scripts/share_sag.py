"""Box share of each member stock's biomass and SSB from ICES SAG (WP3, box_share.md).

Usage: python3 -I share_sag.py <sag_dir> <out_csv>
Box biomass = model box density (t/km2) x 3600 km2 x member share (group tables in the task brief).
Stock values: SAG SummaryTable means over 2001-2025 and 2021-2025 (terminal-year rows beyond the
last assessed year are skipped when empty).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sag, out_csv = sys.argv[1:3]
AREA = 3600.0
DENS = {"planktivorous_fish": 3.3, "benthivorous_fish": 1.53, "predatory_fish": 0.754, "whiting": 0.446}
# group, member, share in group, SAG key (None = not assessed), share of the member's box biomass in this stock
MEMBERS = [
    ("planktivorous_fish", "herring NSAS", 0.79, 22468, 0.68),
    ("planktivorous_fish", "herring WBSS", 0.79, 21294, 0.32),
    ("planktivorous_fish", "blue whiting", 0.105, 25908, 1.0),
    ("planktivorous_fish", "Norway pout", 0.075, 26011, 1.0),
    ("planktivorous_fish", "sprat", 0.03, 21311, 1.0),
    ("benthivorous_fish", "haddock", 0.386, 22417, 1.0),
    ("benthivorous_fish", "cod (whole NS stock)", 0.314, 26010, 1.0),
    ("benthivorous_fish", "cod (Viking substock)", 0.314, 26001, 1.0),
    ("benthivorous_fish", "long rough dab", 0.126, None, 1.0),
    ("benthivorous_fish", "dab", 0.045, 19480, 1.0),
    ("benthivorous_fish", "plaice (ple.27.420)", 0.038, 22489, 1.0),
    ("benthivorous_fish", "witch", 0.017, 22534, 1.0),
    ("predatory_fish", "saithe", 0.651, 22504, 1.0),
    ("predatory_fish", "mackerel", 0.128, 25875, 1.0),
    ("predatory_fish", "spurdog", 0.08, 25850, 1.0),
    ("predatory_fish", "horse mackerel (western)", 0.055, 25892, 1.0),
    ("predatory_fish", "hake", 0.04, 22627, 1.0),
    ("whiting", "whiting (whg.27.3a)", 1.0, 22563, 1.0),
    ("whiting", "whiting (whg.27.47d)", 1.0, 22488, 1.0),
]


def load(key):
    d = json.load(open(os.path.join(sag, f"summary_{key}.json")))
    L = pd.DataFrame(d["Lines"]).set_index("Year")
    return d, L


rows = []
for g, m, sh, key, stock_sh in MEMBERS:
    box_t = DENS[g] * AREA * sh * stock_sh
    r = dict(group=g, member=m, box_t=round(box_t), key=key)
    if key is not None:
        d, L = load(key)
        r.update(stock=d["FishStock"], ay=d["AssessmentYear"], ssb_desc=d["StockSizeDescription"],
                 ssb_units=d["StockSizeUnits"], rec_age=d["RecruitmentAge"], units=d["Units"])
        for col in ("SSB", "TBiomass", "Recruitment"):
            s = pd.to_numeric(L[col], errors="coerce").dropna()
            r[col + "_0125"] = s.loc[2001:2025].mean() if len(s.loc[2001:2025]) else np.nan
            r[col + "_2125"] = s.loc[2021:2025].mean() if len(s.loc[2021:2025]) else np.nan
            r[col + "_last"] = s.iloc[-1] if len(s) else np.nan
            r[col + "_lastyr"] = s.index[-1] if len(s) else np.nan
        tonnes = (str(r["ssb_units"]) in ("tonnes", "None", "") and r["ssb_desc"] in ("SSB", "Total biomass", "TSB"))
        for per in ("0125", "2125"):
            if tonnes:
                r["share_SSB_" + per] = box_t / r["SSB_" + per] if r["SSB_" + per] > 0 else np.nan
            r["share_TB_" + per] = box_t / r["TBiomass_" + per] if r["TBiomass_" + per] > 0 else np.nan
    rows.append(r)

# whole North Sea cod: the 2026 whole-stock entry (26010) has no SSB; sum the three substocks
sub = pd.DataFrame({k: pd.to_numeric(load(k)[1]["SSB"], errors="coerce") for k in (26001, 26002, 26003)}).dropna()
tot = sub.sum(axis=1)
for r in rows:
    if r["member"] == "cod (whole NS stock)":
        r.update(ssb_desc="SSB", ssb_units="tonnes", SSB_0125=tot.loc[2001:2025].mean(),
                 SSB_2125=tot.loc[2021:2025].mean(), SSB_last=tot.iloc[-1], SSB_lastyr=tot.index[-1])
        r["share_SSB_0125"] = r["box_t"] / r["SSB_0125"]
        r["share_SSB_2125"] = r["box_t"] / r["SSB_2125"]
        r["stock"] = "cod.27.46a7d20 (sum of substocks)"
print("Viking share of the summed cod SSB, 2001-25 mean: %.3f" % (sub[26001] / tot).loc[2001:2025].mean())

T = pd.DataFrame(rows)
T.to_csv(out_csv, index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
cols = ["member", "box_t", "stock", "ay", "ssb_desc", "ssb_units", "rec_age", "SSB_0125", "SSB_2125",
        "TBiomass_0125", "TBiomass_2125", "Recruitment_0125", "share_SSB_0125", "share_SSB_2125",
        "share_TB_0125", "share_TB_2125"]
print(T[cols].to_string(float_format=lambda x: "%.4g" % x))
