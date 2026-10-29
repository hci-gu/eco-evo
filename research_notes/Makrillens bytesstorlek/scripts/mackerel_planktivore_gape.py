"""Share of the planktivore pool inside the mackerel and horse mackerel gape, Mareld box.

Usage: python3 -I mackerel_planktivore_gape.py <extract_old> <extract_new> <lw_csv> <depth_npz>
       <pred_length_biomass_csv> <sms_summary_stom> <heras_derived_dir> <out_dir>

1. Prey-size window: SMS 2023 key run stomachs (summary_stom.out; North Sea 1981 + 1991), fish
   prey lengths eaten by mackerel (predator 20) and horse mackerel (11, 12), weighted by
   stomach content share x hauls; the weighted 95th percentile of prey/predator length is the gape.
2. Predator sizes in the box: NS-IBTS GOV length biomass (pred_length_biomass.csv, 5-cm classes).
3. Pool length composition per species: NS-IBTS GOV in the box (same hauls, box and depth strata as
   ../../Sejens diet pa unga torskfiskar/scripts/box_gape_by_quarter.py), Q1 (Jan-Feb) and Q3
   (Aug); and HERAS (RV Dana, late Jun-Jul, pelagic trawls) individuals in the same box.
4. Species weights in the pool (params_planktivorous_fish.md section 0, judgement on the season
   split): annual herring 0.788, blue whiting 0.106, Norway pout 0.076, sprat 0.030; Q3 0.78 /
   0.15 / 0.04 / 0.03; Q1 0.88 / 0 / 0.11 / 0.01.
f = sum over predator length classes (biomass weights) of sum over species (pool weight x share of
the species' biomass below ratio x predator length).
"""
import os
import sys

import numpy as np
import pandas as pd

old, new, lw_csv, depth_npz, plb_csv, stom, heras, out = sys.argv[1:9]
os.makedirs(out, exist_ok=True)
LW = pd.read_csv(lw_csv, index_col=0)
PREY = {126417: "herring", 126425: "sprat", 126444: "Norway pout", 126439: "blue whiting"}
HALF_CM = {126417, 126425}
W_POOL = {"annual": {"herring": 0.788, "blue whiting": 0.106, "Norway pout": 0.076, "sprat": 0.030},
          3: {"herring": 0.78, "blue whiting": 0.15, "Norway pout": 0.04, "sprat": 0.03},
          1: {"herring": 0.88, "blue whiting": 0.0, "Norway pout": 0.11, "sprat": 0.01}}

# ---------------------------------------------------------------- 1. prey/predator length ratio
rows = [l.split() for l in open(stom)][1:]
RATIO = {}
for pname, preds in (("mackerel", {20}), ("horse mackerel", {11, 12})):
    r, w = [], []
    for x in rows:
        if int(x[3]) in preds and int(x[15]) > 0 and float(x[26]) > 0:
            r.append(float(x[17]) / float(x[7]))
            w.append(float(x[26]) * float(x[24]))
    r, w = np.array(r), np.array(w)
    o = np.argsort(r)
    cw = np.cumsum(w[o]) / w.sum()
    q = {p: float(r[o][np.searchsorted(cw, p)]) for p in (0.5, 0.9, 0.95, 0.99)}
    RATIO[pname] = q
    print(f"{pname}: SMS fish prey records {len(r)}; prey/predator length p50 {q[0.5]:.2f}, p90 {q[0.9]:.2f}, "
          f"p95 {q[0.95]:.2f}, p99 {q[0.99]:.2f}, max {r.max():.2f}")

# ---------------------------------------------------------------- 2. predator sizes in the box
PLB = pd.read_csv(plb_csv)
PRED_L = {}
for pname in ("mackerel", "horse mackerel"):
    t = PLB[PLB.common == pname].copy()
    t["Lmid"] = t.L5 + 2.5
    PRED_L[pname] = {k: (t.Lmid.values, (t[c] / t[c].sum()).values) for k, c in ((1, "q1"), (3, "q3"))}
    m3 = float((t.Lmid * t.q3).sum() / t.q3.sum())
    print(f"{pname}: box biomass-mean length Q3 {m3:.1f} cm")

# ---------------------------------------------------------------- 3a. GOV pool length composition
hh = pd.concat([pd.read_csv(os.path.join(d, "HH.csv.gz")) for d in (old, new)])
hh = hh[(hh.HaulVal.astype(str).str.strip() == "V") & hh.Quarter.isin([1, 3])
        & (hh.Gear.astype(str).str.strip() == "GOV") & (hh.BySpecRecCode == 1)]
for c in ["Ship", "StatRec"]:
    hh[c] = hh[c].astype(str).str.strip()
hh["la"], hh["lo"] = hh.ShootLat.round(3), hh.ShootLong.round(3)
hh = hh.drop_duplicates(["Year", "Quarter", "Ship", "HaulNo", "la", "lo"])


def in_box(lat, lon):
    x1, y1, x2, y2 = 8.60, 57.12, 7.05, 57.98
    east = ((x2 - x1) * (lat - y1) - (y2 - y1) * (lon - x1)) < 0
    return east & (lat < 59.2) & ~((lon > 10.6) & (lat < 57.75)) & (lon < 12.5) & (lat > 57.0)


hh = hh[in_box(hh.ShootLat, hh.ShootLong) | hh.StatRec.isin(["44G0", "44G1", "45G0", "45G1"])].copy()
hh = hh[(hh.Year >= 2001) & (hh.Year <= 2025)]
BINS = ["<100", "100-200", "200-300", ">=300"]
hh["dbin"] = pd.cut(hh.Depth, [0, 100, 200, 300, 600], labels=BINS, right=False)
hh["key"] = list(zip(hh.Year, hh.Quarter, hh.Ship, hh.HaulNo, hh.la, hh.lo))
cp = pd.concat([pd.read_csv(os.path.join(d, "CPUEL.csv.gz")) for d in (old, new)])
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp["key"] = list(zip(cp.Year, cp.Quarter, cp.Ship, cp.HaulNo, cp.ShootLat.round(3), cp.ShootLon.round(3)))
cp = cp[cp.key.isin(set(hh.key)) & cp.AphiaID.isin(PREY)].copy()
cp["L"] = cp.LngtClas / 10.0 + np.where(cp.AphiaID.isin(HALF_CM), 0.25, 0.5)
cp["Lcm"] = np.floor(cp.L).astype(int)
cp["w_g"] = cp.AphiaID.map(LW.a) * cp.L ** cp.AphiaID.map(LW.b)
cp["kg_h"] = cp.CPUE_number_per_hour * cp.w_g / 1000
cp["col"] = cp.AphiaID.astype(str) + "_" + cp.Lcm.astype(str)
z = np.load(depth_npz)["depth"].astype(float)
z = np.abs(z[np.isfinite(z)])
area = pd.cut(pd.Series(z), [0, 100, 200, 300, 600], labels=BINS, right=False).value_counts(
    normalize=True).reindex(BINS)
m = cp.groupby(["key", "col"]).kg_h.sum().unstack(fill_value=0.0)
cols = list(m.columns)
X = hh.set_index("key")[["Year", "Quarter", "dbin"]].join(m, how="left").fillna(0.0)


def strat(df):
    mu = df.groupby("dbin", observed=False)[cols].mean().reindex(BINS)
    if mu.loc["200-300"].isna().any():
        mu.loc["200-300"] = mu.loc["100-200"]
    mu.loc[">=300"] = mu.loc["200-300"]
    if mu.loc[BINS[:3]].isna().any().any():
        return None
    return mu.mul(area, axis=0).sum()


per_q = {1: [], 3: []}
for y, d in X.groupby("Year"):
    s = {k: strat(d[d.Quarter == k]) for k in (1, 3)}
    if s[1] is not None and s[3] is not None:
        for k in (1, 3):
            per_q[k].append(s[k])
print(f"GOV years with both quarters: {len(per_q[1])}")
CUM = {}  # (source, quarter, species) -> (L upper edges, cumulative biomass share)
for k in (1, 3):
    s = pd.DataFrame(per_q[k]).mean()
    for aph, sp in PREY.items():
        v = {int(c.split("_")[1]): s[c] for c in s.index if int(c.split("_")[0]) == aph}
        if not v or sum(v.values()) <= 0:
            continue
        L = np.arange(0, 61)
        b = np.array([v.get(int(l), 0.0) for l in L])
        CUM[("GOV", k, sp)] = (L + 1.0, np.cumsum(b) / b.sum())

# ---------------------------------------------------------------- 3b. HERAS pool length composition
hh2 = pd.read_csv(os.path.join(heras, "biotic_region_hauls.tsv"), sep="\t")
ca = pd.read_csv(os.path.join(heras, "biotic_region_catch.tsv"), sep="\t")
iv = pd.read_csv(os.path.join(heras, "biotic_region_individuals.tsv"), sep="\t")
hh2 = hh2[in_box(hh2.lat, hh2.lon) & hh2.gear.isin(["FOT", "PEL", "EXP"])]
keys = set(zip(hh2.cruise, hh2.haul))
ca = ca[np.array([k in keys for k in zip(ca.cruise, ca.haul)]) & ca.species.isin(PREY).values]
iv = iv[np.array([k in keys for k in zip(iv.cruise, iv.haul)]) & iv.species.isin(PREY).values].dropna(subset=["length_mm"])
print(f"HERAS hauls in the box (pelagic gears): {len(hh2)}")
for aph, sp in PREY.items():
    acc = np.zeros(61)
    nh = 0
    cw = ca[ca.species == aph].groupby(["cruise", "haul"]).weight_kg.sum()
    for (cr, ha), kg in cw.items():
        f = iv[(iv.cruise == cr) & (iv.haul == ha) & (iv.species == aph)]
        if len(f) == 0 or kg <= 0:
            continue
        wgt = f.weight_kg.where(f.weight_kg > 0, LW.loc[aph, "a"] * (f.length_mm / 10.0) ** LW.loc[aph, "b"] / 1000)
        Lc = np.clip(np.floor(f.length_mm / 10.0).astype(int), 0, 60)
        h = np.bincount(Lc, weights=wgt, minlength=61)
        acc += kg * h / h.sum()
        nh += 1
    if nh:
        CUM[("HERAS", 7, sp)] = (np.arange(0, 61) + 1.0, np.cumsum(acc) / acc.sum())
        print(f"  HERAS {sp}: {nh} hauls with lengths, catch-weighted")


def share_below(src, k, sp, L):
    edges, c = CUM[(src, k, sp)]
    return float(np.interp(L, np.concatenate([[0.0], edges]), np.concatenate([[0.0], c])))


print("\nshare of each species' biomass below L cm")
print("  L    " + "  ".join(f"{s[0]}Q{s[1]}:{s[2][:6]:>6}" for s in CUM))
for L in (6, 8, 9, 10, 11, 12, 14, 16, 20):
    print(f"  {L:3d}  " + "  ".join(f"{share_below(*s, L):>14.3f}" for s in CUM))

# ---------------------------------------------------------------- 4. f per predator and quarter
res = []


def f_member(pname, ratio, src, k, wq):
    Ls, ws = PRED_L[pname][3 if k == 7 else k]
    tot = 0.0
    for Lp, wp in zip(Ls, ws):
        g = ratio * Lp
        acc = sum(wq[sp] * share_below(src, k, sp, g) for sp in wq if (src, k, sp) in CUM)
        norm = sum(wq[sp] for sp in wq if (src, k, sp) in CUM)
        tot += wp * acc / norm
    return tot


print("\naccessible share f of the pool (predator-biomass weighted gape = ratio x length)")
for pname in ("mackerel", "horse mackerel"):
    for lab, ratio in (("p90", RATIO[pname][0.9]), ("p95", RATIO[pname][0.95]), ("p99", RATIO[pname][0.99])):
        f1 = f_member(pname, ratio, "GOV", 1, W_POOL[1])
        f3 = f_member(pname, ratio, "GOV", 3, W_POOL[3])
        fh = f_member(pname, ratio, "HERAS", 7, W_POOL[3]) if ("HERAS", 7, "herring") in CUM else float("nan")
        print(f"  {pname:15s} {lab} ratio {ratio:.2f}: GOV Q1 {f1:.3f}, GOV Q3 (Aug) {f3:.3f}, HERAS Jun-Jul {fh:.3f}")
        res.append(dict(predator=pname, gape=lab, ratio=ratio, gov_q1=f1, gov_q3=f3, heras_jul=fh))
R = pd.DataFrame(res)
R.to_csv(os.path.join(out, "mackerel_planktivore_f.csv"), index=False)

# group: shares of the group's planktivore a (Rovfiskens uppdelning, split_encounter_223: mackerel
# 0.49, horse mackerel 0.12 of the pooled wc a) -> 0.803 / 0.197
A = {"mackerel": 0.49 / 0.61, "horse mackerel": 0.12 / 0.61}
print("\ngroup f (a-weighted: mackerel 0.803, horse mackerel 0.197)")
for lab in ("p90", "p95", "p99"):
    g = {c: sum(A[p] * float(R[(R.predator == p) & (R.gape == lab)][c].iloc[0]) for p in A)
         for c in ("gov_q1", "gov_q3", "heras_jul")}
    summer = np.nanmean([g["gov_q3"], g["heras_jul"]])
    print(f"  {lab}: Q1 {g['gov_q1']:.3f}, Jun-Jul (HERAS) {g['heras_jul']:.3f}, Aug (GOV Q3) {g['gov_q3']:.3f}; "
          f"summer mean {summer:.3f}; presence-weighted (Q3 0.95 / Q1 0.05) {0.95 * summer + 0.05 * g['gov_q1']:.3f}")
