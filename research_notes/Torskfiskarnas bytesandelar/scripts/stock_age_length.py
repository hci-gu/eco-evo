"""Survey age-length keys for the Skagerrak region: length-at-age and weight-at-age per quarter.

Usage: python3 -I stock_age_length.py <species_mix_dir> <out_tables_dir>

<species_mix_dir> = research_notes/Torskfiskarnas artsammansättning (read only). Uses its untrusted
DATRAS extracts (parsed only): data/extract/CA.csv.gz (age-length records, rows with IndWgt > 0) and
data/extract/CPUEL.csv.gz (numbers/h per 1-cm class), plus tables/hauls_region.csv (region hauls:
Skagerrak + box rectangles, NS-IBTS GOV Q1/Q3 2021-25) and tables/length_weight.csv (W = a L^b, g, cm).

Method: ALK P(age | L, quarter) from CA records in the Skagerrak rectangles (counts CANoAtLngt; CA is
length-stratified, so it is used only conditionally on length). Missing lengths borrow the nearest
aged length within 3 cm; lengths below / above the aged range take the youngest / oldest age seen at
the edge. The ALK is applied to the region haul-mean CPUE-at-length, giving n(L, age). Two variants:
raw (q = 1) and fraser (CPUE divided by the Fraser et al. 2007 length-dependent q; only the shape of
q within a species matters here, so the door/wing basis does not). Hake has no ages in CA: skipped.

Outputs: stock_alk_nla.csv (species, quarter, variant, L_cm, age, n_rel), stock_length_at_age.csv
(median, p10, p90 length, survey mean weight at age), stock_ca_weight_at_age.csv (direct CA IndWgt
means, Skagerrak and NS-wide, for reference).
"""
import os
import sys

import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
SP = {126437: "haddock", 126438: "whiting", 126441: "saithe", 126436: "cod"}
REG = set("43F8 43F9 43G0 43G1 44F8 44F9 44G0 44G1 45F8 45F9 45G0 45G1 46F9 46G0 46G1".split())
PLUS = {"haddock": 8, "whiting": 8, "saithe": 10, "cod": 7}


def q_cod(l):
    return np.where(l <= 25, 0.15963, np.where(l >= 97, 0.66822, 0.20935 - 0.00434 * l + 0.00009 * l ** 2))


def q_had(l):
    return np.where(l <= 11, 0.0116121, np.where(l >= 46, 0.123268, -0.370453 + 0.04227 * l - 0.000685 * l ** 2))


def q_whg(l):
    return np.where(l <= 21, 0.616878, np.where(l >= 38, 0.138165, -0.028159 * l + 1.208229))


QF = {"cod": q_cod, "haddock": q_had, "whiting": q_whg,
      "saithe": lambda l: (q_cod(l) + q_had(l) + q_whg(l)) / 3}

lw = pd.read_csv(os.path.join(src, "tables", "length_weight.csv"), index_col=0)
ca = pd.read_csv(os.path.join(src, "data", "extract", "CA.csv.gz"), dtype={"AreaCode": str, "LngtCode": str})
ca = ca[ca.Valid_Aphia.isin(SP) & (ca.Age >= 0)].copy()
ca["sp"] = ca.Valid_Aphia.map(SP)
ca["L"] = np.where(ca.LngtCode.isin([".", "0"]), ca.LngtClass / 10.0, ca.LngtClass.astype(float))
ca["Lc"] = np.floor(ca.L).astype(int)
ca["n"] = ca.CANoAtLngt.clip(lower=1)
ca["agep"] = np.minimum(ca.Age, ca.sp.map(PLUS))
ca["reg"] = ca.AreaCode.isin(REG)

# direct CA mean weight at age (reference only; length-stratified sampling biases it)
w = []
for scope, d in [("skagerrak", ca[ca.reg]), ("ns_wide", ca)]:
    g = d.groupby(["sp", "Quarter", "agep"]).apply(
        lambda x: pd.Series({"n_fish": x.n.sum(), "mean_w_g": np.average(x.IndWgt, weights=x.n),
                             "mean_L_cm": np.average(x.L, weights=x.n)}), include_groups=False).reset_index()
    g["scope"] = scope
    w.append(g)
pd.concat(w).rename(columns={"sp": "species", "agep": "age"}).round(2).to_csv(
    os.path.join(out, "stock_ca_weight_at_age.csv"), index=False)

hh = pd.read_csv(os.path.join(src, "tables", "hauls_region.csv"))
hh["Ship"] = hh.Ship.astype(str).str.strip()
hh = hh[hh.region]
nh = hh.groupby("Quarter").size()
cp = pd.read_csv(os.path.join(src, "data", "extract", "CPUEL.csv.gz"))
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp = cp[cp.AphiaID.isin(SP)].merge(hh[["Year", "Quarter", "Ship", "HaulNo"]], on=["Year", "Quarter", "Ship", "HaulNo"])
cp["sp"] = cp.AphiaID.map(SP)
cp["Lc"] = (cp.LngtClas // 10).astype(int)

rows, summ = [], []
for sp in SP.values():
    a_, b_ = lw.loc[sp, "a"], lw.loc[sp, "b"]
    for qu in (1, 3):
        alk = ca[(ca.sp == sp) & ca.reg & (ca.Quarter == qu)].pivot_table(
            index="Lc", columns="agep", values="n", aggfunc="sum", fill_value=0)
        if alk.empty:
            continue
        alk = alk.div(alk.sum(axis=1), axis=0)
        f = cp[(cp.sp == sp) & (cp.Quarter == qu)].groupby("Lc").CPUE_number_per_hour.sum() / nh[qu]
        Ls = sorted(set(f.index))
        key = {}
        aged = np.array(alk.index)
        for L in Ls:
            if L in alk.index:
                key[L] = alk.loc[L].values
            elif L < aged.min():
                v = np.zeros(alk.shape[1]); v[0] = 1; key[L] = v
            elif L > aged.max():
                v = np.zeros(alk.shape[1]); v[-1] = 1; key[L] = v
            else:
                near = aged[np.argsort(np.abs(aged - L))[:2]]
                near = near[np.abs(near - L) <= 3]
                key[L] = alk.loc[near].mean().values if len(near) else alk.iloc[np.argmin(np.abs(aged - L))].values
        for var in ("raw", "fraser"):
            q = np.ones(len(Ls)) if var == "raw" else QF[sp](np.array(Ls) + 0.5)
            nla = pd.DataFrame({L: key[L] * f[L] / qq for L, qq in zip(Ls, q)}, index=alk.columns).T
            nla.index.name = "L_cm"
            nla = nla / nla.values.sum()
            for L in nla.index:
                for a in nla.columns:
                    if nla.loc[L, a] > 0:
                        rows.append([sp, qu, var, L, a, nla.loc[L, a]])
            for a in nla.columns:
                p = nla[a]
                if p.sum() <= 0:
                    continue
                c = (p / p.sum()).cumsum()
                Lm = np.array(p.index) + 0.5
                wL = a_ * Lm ** b_
                summ.append(dict(species=sp, quarter=qu, variant=var, age=a,
                                 n_aged_ca=int(ca[(ca.sp == sp) & ca.reg & (ca.Quarter == qu) & (ca.agep == a)].n.sum()),
                                 share_of_numbers=round(p.sum(), 4),
                                 L_p10=Lm[np.searchsorted(c.values, 0.1)], L_median=Lm[np.searchsorted(c.values, 0.5)],
                                 L_p90=Lm[np.searchsorted(c.values, 0.9)],
                                 mean_w_g=round(float(np.average(wL, weights=p)), 1)))

pd.DataFrame(rows, columns=["species", "quarter", "variant", "L_cm", "age", "n_rel"]).to_csv(
    os.path.join(out, "stock_alk_nla.csv"), index=False)
S = pd.DataFrame(summ)
S.to_csv(os.path.join(out, "stock_length_at_age.csv"), index=False)
pd.set_option("display.width", 200, "display.max_rows", 300)
print(S[S.age <= 4].to_string(index=False))
