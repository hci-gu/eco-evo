"""Stock-level biomass shares by age and by length class, Q1 and Q3, from assessment N-at-age.

Usage: python3 -I stock_age_shares.py <tables_dir> <data_stock_dir> <species_mix_dir>

Reads <tables_dir>/stock_naa.csv (stock_parse.py), stock_alk_nla.csv and stock_length_at_age.csv
(stock_age_length.py), <data_stock_dir>/sag/SummaryTable_22627.json (hake, parsed only) and
<species_mix_dir>/tables/length_weight.csv.

Survival to survey time: N(a, t) = N(a, Jan 1) exp(-(F + M) t), t = 0.10 (Q1, Jan-Feb) and
t = 0.62 (Q3, mid Aug). Age 0 is absent in Q1. SAM age 0 at Jan 1 is notional (back-projected with
the full M0), so N0(Q3) = N0 exp(-Z0 0.62) is model-consistent. Cod starts at age 1: age-0 cod in Q3
is back-projected from N1 of the next year, N0(Q3, y) = N1(y + 1) exp(M0 0.38), M0 = 1.5 (1.0-2.5).
Weights: Q1 = stock weight-at-age (Jan 1). Q3 = (a) "survey": Skagerrak-region GOV mean weight-at-
age (stock_length_at_age.csv, raw variant; ages with < 20 aged fish fall back to b), or (b)
"sw_interp": stock weights interpolated 0.6 yr forward, sw(a)^0.4 sw(a+1)^0.6 (plus group: sw).
Age-0 weight in Q3 always from the survey with the Fraser-corrected length mix (small fish restored).
Length classes: B(a) spread over length with the region's biomass-weighted P(L | a) from the ALK.

Outputs (in <tables_dir>): stock_age_biomass_share.csv, stock_length_class_share.csv,
stock_pelagic_0group.csv, and a printed summary.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

tab, dstock, smix = sys.argv[1], sys.argv[2], sys.argv[3]
naa = pd.read_csv(os.path.join(tab, "stock_naa.csv"))
nla = pd.read_csv(os.path.join(tab, "stock_alk_nla.csv"))
laa = pd.read_csv(os.path.join(tab, "stock_length_at_age.csv"))
lw = pd.read_csv(os.path.join(smix, "tables", "length_weight.csv"), index_col=0)
T = {1: 0.10, 3: 0.62}
YEARS = {"2021-2025": range(2021, 2026), "2019-2024": range(2019, 2025)}
RUNS = [("haddock", "had.27.46a20", "NShaddock_WGNSSK2026_Run1", "primary"),
        ("haddock", "had.27.46a20", "NShaddock_WGNSSK2024_Run2", "sens"),
        ("whiting", "whg.27.47d", "NSwhiting_2026n", "primary"),
        ("whiting", "whg.27.47d", "NSwhiting_2025", "sens"),
        ("cod", "cod.27.46a7d20_Viking", "WGNSSK2026_multistockSAM", "primary"),
        ("cod", "cod.27.46a7d20_Northwestern", "WGNSSK2026_multistockSAM", "sens"),
        ("cod", "cod.27.46a7d20_Southern", "WGNSSK2026_multistockSAM", "sens"),
        ("saithe", "pok.27.3a46", "WGNSSK2026_SAM", "primary")]
M0_COD = {"central": 1.5, "low": 1.0, "high": 2.5}
CLS = [0, 10, 15, 20, 25, 30, 40, 50, 60, 70, 200]


def w_survey(sp, qu, a, var="raw"):
    r = laa[(laa.species == sp) & (laa.quarter == qu) & (laa.variant == var) & (laa.age == a)]
    if len(r) and r.n_aged_ca.iloc[0] >= 20:
        return r.mean_w_g.iloc[0] / 1000.0
    return np.nan


def biomass_by_age(sp, stock, run, years, wq3, m0="central"):
    d = naa[(naa.stock == stock) & (naa.run == run)].set_index(["year", "age"]).sort_index()
    ages = sorted(d.index.get_level_values("age").unique())
    amax = ages[-1]
    out = {1: {}, 3: {}}
    num = {1: {}, 3: {}}
    have = set(d.index.get_level_values("year"))
    for y in [y for y in years if y in have and (0 in ages or y + 1 in have)]:
        for qu in (1, 3):
            for a in ages:
                r = d.loc[(y, a)]
                Z = r.F + r.M
                if qu == 1:
                    if a == 0:
                        continue
                    out[1].setdefault(a, []).append(r.N_thousands * np.exp(-Z * T[1]) * r.sw_kg)
                    num[1].setdefault(a, []).append(r.N_thousands * np.exp(-Z * T[1]))
                    continue
                num[3].setdefault(a, []).append(r.N_thousands * np.exp(-Z * T[3]))
                if a == 0:
                    w = w_survey(sp, 3, 0, "fraser")
                    out[3].setdefault(0, []).append(r.N_thousands * np.exp(-Z * T[3]) * w)
                    continue
                if wq3 == "survey" and not np.isnan(w_survey(sp, 3, a)):
                    w = w_survey(sp, 3, a)
                else:
                    w1 = d.loc[(y, a + 1)].sw_kg if a < amax else r.sw_kg
                    w = r.sw_kg ** 0.4 * w1 ** 0.6
                out[3].setdefault(a, []).append(r.N_thousands * np.exp(-Z * T[3]) * w)
            if qu == 3 and sp == "cod":  # cod: back-project age 0 from N1 of next year
                n1 = d.loc[(y + 1, 1)].N_thousands
                out[3].setdefault(0, []).append(n1 * np.exp(M0_COD[m0] * (1 - T[3])) * w_survey(sp, 3, 0, "fraser"))
                num[3].setdefault(0, []).append(n1 * np.exp(M0_COD[m0] * (1 - T[3])))
    NUM.clear()
    NUM.update({qu: pd.Series({a: np.mean(v) for a, v in num[qu].items()}).sort_index() for qu in num})
    return {qu: pd.Series({a: np.mean(v) for a, v in out[qu].items()}).sort_index() for qu in out}


NUM = {}


def numbers_age0_q3(stock, run, years, m0="central"):
    d = naa[(naa.stock == stock) & (naa.run == run)].set_index(["year", "age"]).sort_index()
    have = set(d.index.get_level_values("year"))
    years = [y for y in years if y in have and y + 1 in have] if 0 not in d.index.get_level_values("age") else \
        [y for y in years if y in have]
    if 0 in d.index.get_level_values("age"):
        return np.mean([d.loc[(y, 0)].N_thousands * np.exp(-(d.loc[(y, 0)].F + d.loc[(y, 0)].M) * T[3]) for y in years]), \
            np.mean([d.loc[(y, 0)].M for y in years])
    return np.mean([d.loc[(y + 1, 1)].N_thousands * np.exp(M0_COD[m0] * (1 - T[3])) for y in years]), M0_COD[m0]


def pl_given_a(sp, qu, var):
    d = nla[(nla.species == sp) & (nla.quarter == qu) & (nla.variant == var)].copy()
    d["b"] = d.n_rel * lw.loc[sp, "a"] * (d.L_cm + 0.5) ** lw.loc[sp, "b"]
    return d.pivot_table(index="L_cm", columns="age", values="b", aggfunc="sum", fill_value=0.0)


rows_age, rows_len, rows_pel = [], [], []
for sp, stock, run, role in RUNS:
    for yl, years in YEARS.items():
        for wq3 in ("survey", "sw_interp"):
            B = biomass_by_age(sp, stock, run, years, wq3)
            for qu in (1, 3):
                b = B[qu]
                nn = NUM[qu]
                for a, v in b.items():
                    rows_age.append(dict(species=sp, stock=stock, run=run, role=role, years=yl, q3_weights=wq3,
                                         quarter=qu, age=a, biomass_t=round(v), share_pct=round(100 * v / b.sum(), 2),
                                         N_millions=round(nn[a] / 1000, 1),
                                         numbers_share_pct=round(100 * nn[a] / nn.sum(), 2)))
                for var in ("raw", "fraser"):
                    P = pl_given_a(sp, qu, var)
                    BL = pd.Series(0.0, index=P.index)
                    for a, v in b.items():
                        col = a if a in P.columns else min(P.columns, key=lambda c: abs(c - a))
                        BL += v * P[col] / P[col].sum()
                    c = pd.cut(BL.index + 0.5, CLS, right=False)
                    s = BL.groupby(c, observed=False).sum()
                    rows_len.append(dict(species=sp, stock=stock, run=run, role=role, years=yl, q3_weights=wq3,
                                         quarter=qu, alk_variant=var,
                                         **{str(k): round(100 * x / s.sum(), 1) for k, x in s.items()}))
        # pelagic 0-group (3-7 cm, mid June, t = 0.45) vs stock biomass in Q3
        for m0 in (["central", "low", "high"] if sp == "cod" else ["central"]):
            if sp == "saithe":
                continue
            n0_q3, M0 = numbers_age0_q3(stock, run, years, m0)
            B3 = biomass_by_age(sp, stock, run, years, "survey", m0)[3]
            w5 = lw.loc[sp, "a"] * 5.0 ** lw.loc[sp, "b"] / 1000.0
            for mult in (1, 3):
                n_jun = n0_q3 * np.exp(mult * M0 * (T[3] - 0.45))
                rows_pel.append(dict(species=sp, stock=stock, run=run, years=yl, m0_case=m0, M0=round(M0, 2),
                                     jun_aug_mortality_multiplier=mult, N0_jun_millions=round(n_jun / 1000),
                                     w_5cm_g=round(w5 * 1000, 2), pelagic_biomass_t=round(n_jun * w5),
                                     stock_biomass_q3_t=round(B3.sum()),
                                     pelagic_pct_of_q3_stock=round(100 * n_jun * w5 / B3.sum(), 3),
                                     age0_q3_pct=round(100 * B3[0] / B3.sum(), 2),
                                     age0_q3_numbers_pct=None))

# hake (SS3, no N-at-age here): R(age 0) and TSB from SAG, M 0.4, age-0 weight at 8-12 cm
j = json.load(open(os.path.join(dstock, "sag", "SummaryTable_22627.json")))
L = pd.DataFrame(j["Lines"]).set_index("Year")
for yl, years in YEARS.items():
    R = L.loc[list(years), "Recruitment"].mean()
    tsb = L.loc[list(years), "TBiomass"].mean()
    for lcm in (8, 10, 12):
        w = lw.loc["hake", "a"] * lcm ** lw.loc["hake", "b"] / 1000.0
        b0 = R * np.exp(-0.4 * T[3]) * w
        rows_age.append(dict(species="hake", stock="hke.27.3a46-8abd", run="SAG 22627 (SS3)", role="crude",
                             years=yl, q3_weights=f"age0 at {lcm} cm", quarter=3, age=0, biomass_t=round(b0),
                             share_pct=round(100 * b0 / (tsb + b0), 2)))

A = pd.DataFrame(rows_age)
A.to_csv(os.path.join(tab, "stock_age_biomass_share.csv"), index=False)
Lr = pd.DataFrame(rows_len)
Lr.to_csv(os.path.join(tab, "stock_length_class_share.csv"), index=False)
Pp = pd.DataFrame(rows_pel).drop(columns="age0_q3_numbers_pct")
Pp.to_csv(os.path.join(tab, "stock_pelagic_0group.csv"), index=False)

pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 30)
k = A[A.age <= 3].pivot_table(index=["species", "stock", "run", "years", "q3_weights", "quarter"], columns="age",
                              values="share_pct")
print(k.to_string())
k = A[(A.age <= 2) & (A.q3_weights == "survey")].pivot_table(index=["species", "stock", "run", "years", "quarter"],
                                                            columns="age", values="numbers_share_pct")
print("numbers share (%)\n", k.to_string())
print("total biomass (t)\n", A[A.q3_weights != "sw_interp"].groupby(["stock", "run", "years", "quarter"]).biomass_t.sum().to_string())
print(Lr[(Lr.alk_variant == "raw") & (Lr.q3_weights == "survey")].drop(columns=["q3_weights", "alk_variant"]).to_string(index=False))
print(Lr[(Lr.alk_variant == "fraser") & (Lr.q3_weights == "survey") & (Lr.role == "primary")].drop(columns=["q3_weights"]).to_string(index=False))
print(Pp.to_string(index=False))
