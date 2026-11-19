"""Adult saithe (>= 35 cm) stomachs by bed class: fullness and prey composition by taxon (krill, Norway pout, blue
whiting, herring + sprat, juvenile gadoids, sandeel, other fish, other), by quarter and length class.

Usage: python3 -I saithe_stomach_bed.py <records_linked.csv.gz> <ices_stomach_all_dir> <out_csv>

  records_linked.csv.gz: research_notes/"Krill över grunda bottnar"/stomachs/tables (one row per predator record
    with bed depth, rectangle sea-area shares per bed class, area, quarter; built from the ICES stomach database,
    full download 2026-10-26, ICES data licence).
  ices_stomach_all_dir: research_notes/"Födonivå och mättnad i fält"/data/A_ices_stomach/all (PreyInformation.csv,
    PredatorInformation.csv) for prey identity and mass per predator record.

Area: N of 57 N (NE + NW North Sea, Skagerrak). Bed class: "true" = haul position or reported depth; "spread" =
rectangle-only records (1980s) split over classes by the rectangle's sea-area shares (ecological inference, as in
the sibling note). Wet weight only (dry-weight records dropped). A record = one stomach or a pooled sample of
`Number` stomachs; mass per stomach = record mass / Number (empties included). Predator weight 0.0086 L^3.06 g
(HERAS script's length-weight). CIs: station bootstrap (ship + date + station), 5-95 %, 200 draws.
Juvenile gadoids = haddock, cod, whiting identified (any size: prey lengths are rarely given); Gadidae / Gadiformes
unidentified kept apart ("Gadidae unid": may hold Norway pout or other gadoids). Trisopterus esmarkii is POUT.
Areas: "N57" = NE + NW North Sea + Skagerrak; "NE" = NE North Sea + Skagerrak (the box side; NW is 0.6 of N57).
"""
import os
import sys

import numpy as np
import pandas as pd

REC, ADIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
R = pd.read_csv(REC, low_memory=False)
R = R[(R.sp == "saithe") & (R.L_cm >= 35) & (R.lat >= 57.0) & R.area.isin(["NE North Sea", "NW North Sea", "Skagerrak"])].copy()
R["tblPredatorInformationID"] = R.tblPredatorInformationID.astype(str)
Y = pd.read_csv(os.path.join(ADIR, "PreyInformation.csv"), dtype=str, on_bad_lines="skip")
Y = Y[Y.tblPredatorInformationID.isin(R.tblPredatorInformationID)].copy()
Y["w"] = pd.to_numeric(Y.Weight, errors="coerce")
Y.loc[Y.UnitWgt == "mg", "w"] = Y.w / 1000.0
Y = Y[~Y.GravMethod.fillna("").str.upper().str.contains("DRY")]
TAX = {"126444": "pout", "126439": "blue whiting", "126417": "herring+sprat", "126425": "herring+sprat",
       "125464": "herring+sprat", "126437": "juv gadoids", "126436": "juv gadoids", "126438": "juv gadoids",
       "125469": "Gadidae unid", "10313": "Gadidae unid", "125467": "Gadidae unid",
       "126751": "sandeel", "125516": "sandeel", "126752": "sandeel", "126754": "sandeel", "125909": "sandeel",
       "126756": "sandeel"}
# krill / other classes from the record builder's lineage classes: re-derive per prey row via the record table's
# class masses is not possible per taxon, so classify here: euphausiid ids seen in the download
EUPH = {"110671", "110690", "110672", "110676", "110673", "110681", "110674", "1128", "110675", "110679"}
FISH_CLS = set()


def tax(aid):
    if aid in TAX:
        return TAX[aid]
    if aid in EUPH:
        return "krill"
    return None


Y["tax"] = Y.AphiaIDPrey.map(tax)
Y.loc[Y.AphiaIDPrey.isin(["107052", "106790"]), "tax"] = "pasiphaeid shrimp"
Y.loc[Y.AphiaIDPrey.isin(["127312"]), "tax"] = "pearlside"
occ = pd.crosstab(Y.tblPredatorInformationID, Y.tax.fillna("unnamed")).clip(upper=1).add_prefix("o_")
m = Y[Y.w > 0].pivot_table(index="tblPredatorInformationID", columns="tax", values="w", aggfunc="sum")
tot = Y[Y.w > 0].groupby("tblPredatorInformationID").w.sum().rename("g_total")
R = R.merge(occ, left_on="tblPredatorInformationID", right_index=True, how="left").merge(m, left_on="tblPredatorInformationID", right_index=True, how="left").merge(
    tot, left_on="tblPredatorInformationID", right_index=True, how="left")
NAMED = ["krill", "pout", "blue whiting", "herring+sprat", "juv gadoids", "Gadidae unid", "sandeel", "pasiphaeid shrimp", "pearlside"]
OCC = NAMED
for c in OCC:
    R["o_" + c] = R.get("o_" + c, pd.Series(0, index=R.index)).fillna(0)
R["o_krill"] = np.maximum(R.o_krill, R.occ_krill.fillna(0))
for c in NAMED:
    if c not in R:
        R[c] = 0.0
    R[c] = R[c].fillna(0.0)
# krill mass from the record builder where the per-id map missed (its lineage classifier is complete)
R["krill"] = np.maximum(R.krill, R.g_krill.fillna(0.0))
R["g_total"] = R.g_total.fillna(0.0)
R["other fish"] = (R.g_fish.fillna(0.0) - R[["pout", "blue whiting", "herring+sprat", "juv gadoids", "Gadidae unid", "sandeel", "pearlside"]].sum(1)).clip(lower=0)
R["other"] = (R.g_total - R[NAMED + ["other fish"]].sum(1)).clip(lower=0)
R["weighed"] = R.g_total > 0
R["Wpred"] = 0.0086 * R.L_cm ** 3.06
R["stn"] = R.ship.astype(str) + "_" + R.year.astype(str) + "_" + R.month.astype(str) + "_" + R.day.astype(str) + "_" + R.station.astype(str)
R["Lc"] = pd.cut(R.L_cm, [35, 50, 70, 200], right=False, labels=["35-50", "50-70", ">=70"])
CLS = ["<100", "100-150", "150-250", ">250"]
SH = {"<100": "sh100", "100-150": "sh150", "150-250": "sh250", ">250": "shdeep"}
# masses only on weighed stations; occurrence (FO) on every station
wst = R.groupby("stn").weighed.max()
R["wst"] = R.stn.map(wst)

rows = []
for _, r in R.iterrows():
    if r.pos == "haul":
        if r.bed in CLS:
            rows.append((r.name, "true", r.bed, 1.0))
    else:
        for c in CLS:
            s = r[SH[c]]
            if np.isfinite(s) and s > 0:
                rows.append((r.name, "spread", c, float(s)))
X = pd.DataFrame(rows, columns=["i", "kind", "cls", "wt"]).merge(R, left_on="i", right_index=True)
COMP = NAMED + ["other fish", "other"]


def stats(d):
    w = d.wt
    num = d.Number.fillna(1) * w
    dw = d[d.wst]
    mass = {c: float((dw[c] * dw.wt).sum()) for c in COMP}
    tm = sum(mass.values())
    full = float((dw.g_total * dw.wt).sum() / (dw.Number.fillna(1) * dw.Wpred * dw.wt).sum()) if len(dw) else np.nan
    fedrec = d[d.anyprey.astype(bool)]
    fo = {f"fo_{c}": float((fedrec["o_" + c] * fedrec.wt).sum() / fedrec.wt.sum()) if len(fedrec) else np.nan for c in OCC}
    return dict(stomachs=float(num.sum()), stations=d.stn.nunique(), weighed_stations=dw.stn.nunique(), fullness_permille=1000 * full, **fo,
                **{f"sh_{c}": (mass[c] / tm if tm > 0 else np.nan) for c in COMP})


rng = np.random.default_rng(20261118)


def boot(d, key, n=200):
    st = d.stn.unique()
    if len(st) < 3:
        return (np.nan, np.nan)
    idx = {s: g.index.values for s, g in d.groupby("stn")}
    vals = []
    for _ in range(n):
        pick = rng.choice(st, len(st))
        dd = d.loc[np.concatenate([idx[s] for s in pick])]
        vals.append(stats(dd)[key])
    v = np.array(vals, float)
    v = v[np.isfinite(v)]
    return (np.percentile(v, 5), np.percentile(v, 95)) if len(v) else (np.nan, np.nan)


X["areaset"] = np.where(X.area == "NW North Sea", "NW", "NE")
XALL = X
out = []
SETS = [("all quarters", lambda d: d), ("Q1", lambda d: d[d.quarter == 1]), ("Q3", lambda d: d[d.quarter == 3]),
        ("Q2+Q4", lambda d: d[d.quarter.isin([2, 4])])]
for aset in ("N57", "NE"):
  X = XALL if aset == "N57" else XALL[XALL.areaset == "NE"]
  for kind in ("spread", "true"):
    for sname, f in SETS:
        for lc in ["all", "35-50", "50-70", ">=70"]:
            for c in CLS:
                d = f(X[(X.kind == kind) & (X.cls == c)])
                if lc != "all":
                    d = d[d.Lc == lc]
                if len(d) == 0:
                    continue
                s = stats(d)
                s.update(kind=kind, season=sname, Lc=lc, cls=c, areaset=aset)
                if lc == "all" and s["stations"] >= 3:
                    keys = (("fullness_permille", "sh_krill", "sh_pout", "sh_blue whiting", "sh_juv gadoids") if kind == "true"
                            else ("fo_krill", "fo_pout", "fo_juv gadoids", "fo_Gadidae unid"))
                    for k in keys:
                        lo, hi = boot(d, k)
                        s[k + "_lo"], s[k + "_hi"] = lo, hi
                out.append(s)
O = pd.DataFrame(out)
O.to_csv(OUT, index=False, float_format="%.4g")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
for aset, kind in (("N57", "spread"), ("NE", "spread"), ("N57", "true")):
    print("=" * 120)
    print(f"{kind.upper()} positions; area {aset}; saithe >= 35 cm; masses on weighed stations only, FO on all")
    print("=" * 120)
    for sname, _ in SETS:
        d = O[(O.kind == kind) & (O.season == sname) & (O.Lc == "all") & (O.areaset == aset)]
        if len(d) == 0:
            continue
        print(f"\n  {sname}")
        for _, r in d.iterrows():
            ci = lambda k: (f" ({r[k + '_lo']:.2f}-{r[k + '_hi']:.2f})" if (k + "_lo") in r and np.isfinite(r.get(k + "_lo", np.nan)) else "")
            print(f"    {r.cls:8s} stomachs {r.stomachs:7.1f} stations {r.stations:3d} (weighed {r.weighed_stations:3d})  FO krill {r.fo_krill:.2f}{ci('fo_krill')} pout {r.fo_pout:.2f}{ci('fo_pout')} BW {r['fo_blue whiting']:.2f} herr {r['fo_herring+sprat']:.2f} juvgad {r['fo_juv gadoids']:.2f}{ci('fo_juv gadoids')} Gadidae {r['fo_Gadidae unid']:.2f}{ci('fo_Gadidae unid')} pasiph {r['fo_pasiphaeid shrimp']:.2f}\n             fullness {r.fullness_permille:5.1f} permille"
                  f"{ci('fullness_permille')}  | krill {r.sh_krill:.2f}{ci('sh_krill')} pout {r.sh_pout:.2f}{ci('sh_pout')}"
                  f" BW {r['sh_blue whiting']:.2f}{ci('sh_blue whiting')} herr+spr {r['sh_herring+sprat']:.2f}"
                  f" juvgad {r['sh_juv gadoids']:.3f}{ci('sh_juv gadoids')} sandeel {r.sh_sandeel:.2f} pasiph {r['sh_pasiphaeid shrimp']:.2f} pearlside {r.sh_pearlside:.2f}"
                  f" oth.fish {r['sh_other fish']:.2f} other {r.sh_other:.2f}")
    print(f"\n  by length class ({kind}, all quarters): fullness permille / mass krill / pout / BW / juv gadoids (stomachs); FO krill / pout / juv gadoids")
    d = O[(O.kind == kind) & (O.season == "all quarters") & (O.Lc != "all") & (O.areaset == aset)]
    for lc, g in d.groupby("Lc"):
        print(f"    {lc:6s} " + " | ".join(f"{r.cls} {r.fullness_permille:.1f} / {r.sh_krill:.2f} / {r.sh_pout:.2f} / "
                                        f"{r['sh_blue whiting']:.2f} / {r['sh_juv gadoids']:.3f} ({r.stomachs:.0f}); FO {r.fo_krill:.2f} / {r.fo_pout:.2f} / {r['fo_juv gadoids']:.2f}"
                                        for _, r in g.iterrows()))
print("\nUnclassified fish (other fish) and other prey are kept; shares are of total weighed prey mass.")
