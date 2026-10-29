"""Age-0 and age-1 share of the planktivore, benthivore and whiting biomass per quarter, Mareld box.

Usage: python3 -I juvenile_share_by_quarter.py <extract_old> <extract_new> <lw_csv> <depth_npz>
       <q2q4_dir> <out_dir>

Q1 (Jan-Feb) and Q3 (Aug): NS-IBTS GOV 2001-25 in the box (hauls, box and depth strata as
../../Sejens diet pa unga torskfiskar/scripts/box_gape_by_quarter.py), ages from the 2021-25 CA
records in the Skagerrak rectangles (43-47 F7-G1) as an age-length key per species, quarter and
1-cm class. Q2 and Q4: NS-IBTS 1991-97 (the only years the quarterly survey ran), box hauls from
datras_q2q4_fetch.py, unstratified means (few hauls), their own CA key (same rectangles).
Catchability: Walker et al. 2017 q-at-length for cod, haddock, whiting and Norway pout
(../../Torskfiskarnas artsammansattning/evidence_B_catchability.md); herring, sprat and flatfish
raw (no published q; the GOV under-catches small pelagic fish).
Group weights: planktivores from the pool weights (params_planktivorous_fish.md); benthivores and
whiting by the q-corrected GOV biomass of their species.
"""
import os
import sys

import numpy as np
import pandas as pd

old, new, lw_csv, depth_npz, q24, out = sys.argv[1:7]
os.makedirs(out, exist_ok=True)
LW = pd.read_csv(lw_csv, index_col=0)
SP = {126417: "herring", 126425: "sprat", 126444: "Norway pout", 126439: "blue whiting",
      126437: "haddock", 126436: "cod", 127143: "plaice", 127139: "dab", 127137: "long rough dab",
      127140: "lemon sole", 127136: "witch", 150637: "grey gurnard", 105865: "starry ray",
      126438: "whiting"}
GROUP = {"herring": "plk", "sprat": "plk", "Norway pout": "plk", "blue whiting": "plk",
         "haddock": "ben", "cod": "ben", "plaice": "ben", "dab": "ben", "long rough dab": "ben",
         "lemon sole": "ben", "witch": "ben", "grey gurnard": "ben", "starry ray": "ben",
         "whiting": "wht"}
HALF_CM = {126417, 126425}
POOL_W = {1: {"herring": 0.88, "blue whiting": 0.0, "Norway pout": 0.11, "sprat": 0.01},
          2: {"herring": 0.788, "blue whiting": 0.106, "Norway pout": 0.076, "sprat": 0.030},
          3: {"herring": 0.78, "blue whiting": 0.15, "Norway pout": 0.04, "sprat": 0.03},
          4: {"herring": 0.788, "blue whiting": 0.106, "Norway pout": 0.076, "sprat": 0.030}}
# benthivore member weights: long-run 2001-25 GOV biomass shares (params_benthivorous_fish.md, the
# model group's own weighting); whiting is one species
MEMBER_W = {"haddock": 38.6, "cod": 31.4, "long rough dab": 12.6, "dab": 4.5, "plaice": 3.8, "witch": 1.7,
            "starry ray": 1.5, "grey gurnard": 0.9, "lemon sole": 0.9, "whiting": 1.0}
SKAG_RECT = {f"{r}{c}" for r in range(43, 48) for c in ("F7", "F8", "F9", "G0", "G1")}


def q_walker(sp, L):
    L = np.asarray(L, dtype=float)
    if sp == "haddock":
        return np.where(L <= 11, 0.0116121, np.where(L >= 46, 0.123268,
                                                     -0.370453 + 0.04227 * L - 0.000685 * L ** 2))
    if sp == "cod":
        return np.where(L <= 25, 0.15963, np.where(L >= 97, 0.66822, 0.20935 - 0.00434 * L + 0.00009 * L ** 2))
    if sp == "whiting":
        return np.where(L <= 21, 0.616878, np.where(L >= 38, 0.138165, -0.028159 * L + 1.208229))
    if sp == "Norway pout":
        return np.where(L <= 11, 0.066204, 0.191855)
    return np.ones_like(L)


def len_cm(code, cls):
    """DATRAS length class to cm: LngtCode '.' (mm) and '0' (half cm, in mm) / 10, '1' as is."""
    mm = pd.Series(code).astype(str).str.strip().isin([".", "0"]).values
    cls = pd.to_numeric(pd.Series(cls), errors="coerce").values
    return np.where(mm, cls / 10.0, cls)


def alk(ca):
    """{(species, quarter): DataFrame index L 0..80, columns ages 0, 1, 2, 3+} from CA rows."""
    ca = ca.copy()
    ca["L"] = np.floor(len_cm(ca.LngtCode.astype(str), ca.LngtClass)).astype(float)
    ca["Age"] = pd.to_numeric(ca.Age, errors="coerce")
    ca["n"] = pd.to_numeric(ca.CANoAtLngt, errors="coerce").fillna(1.0)
    ca = ca[(ca.Age >= 0) & ca.L.notna()]
    keys = {}
    for (aph, q), d in ca.groupby(["Valid_Aphia", "Quarter"]):
        aph = int(aph)
        if aph not in SP:
            continue
        t = d.groupby(["L", d.Age.clip(upper=3).astype(int)]).n.sum().unstack().fillna(0.0)
        t = t.reindex(columns=[0, 1, 2, 3], fill_value=0.0)
        Ls = np.arange(0, 81)
        full = t.reindex(Ls, fill_value=0.0)
        # smooth sparse classes: pool +-2 cm where a class has < 5 aged fish
        pooled = full.rolling(5, center=True, min_periods=1).sum()
        use = np.where(full.sum(axis=1).values[:, None] >= 5, full.values, pooled.values)
        tot = use.sum(axis=1)
        p = np.divide(use, tot[:, None], out=np.full_like(use, np.nan), where=tot[:, None] > 0)
        p = pd.DataFrame(p, index=Ls, columns=[0, 1, 2, 3])
        lo, hi = t.index.min(), t.index.max()
        p.loc[p.index < lo] = [1.0, 0.0, 0.0, 0.0] if p.loc[lo, 0] > 0.5 else list(p.loc[lo].values)
        p.loc[p.index > hi] = [0.0, 0.0, 0.0, 1.0]
        p = p.ffill().bfill()
        keys[(SP[aph], int(q))] = p
    return keys


def by_age(sp_bio, key, sp, q, correct):
    """sp_bio: Series kg/h by 1-cm class -> (age0, age1, total) biomass, q-corrected if asked."""
    L = sp_bio.index.values.astype(float)
    v = sp_bio.values / (q_walker(sp, L + 0.5) if correct else 1.0)
    if key is None:
        return np.nan, np.nan, float(v.sum())
    p = key.reindex(np.clip(L, 0, 80).astype(int)).values
    return float((v * p[:, 0]).sum()), float((v * p[:, 1]).sum()), float(v.sum())


# ------------------------------------------------------------------ Q1 / Q3, 2001-25
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
cp = cp[cp.key.isin(set(hh.key)) & cp.AphiaID.isin(SP)].copy()
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
BIO = {}  # (quarter, species) -> Series kg/h by 1-cm class (long-run mean)
for k in (1, 3):
    s = pd.DataFrame(per_q[k]).mean()
    for aph, sp in SP.items():
        v = {int(c.split("_")[1]): s[c] for c in s.index if int(c.split("_")[0]) == aph}
        if v:
            BIO[(k, sp)] = pd.Series(v).sort_index()
ca13 = pd.read_csv(os.path.join(new, "CA.csv.gz"), low_memory=False)
ca13 = ca13[ca13.AreaCode.astype(str).str.strip().isin(SKAG_RECT)]
KEY = alk(ca13)
print(f"Q1/Q3: GOV years with both quarters {len(per_q[1])}; CA 2021-25 Skagerrak rectangles: "
      f"{len(ca13)} rows; keys {sorted(set(k[0] for k in KEY))}")

# ------------------------------------------------------------------ DATRAS downloads, 1991-96
def datras_bio(q_dir, quarters, label):
    """Box biomass by species and 1-cm class (unstratified: haul mean per year, then year mean) and the
    age-length keys from the Skagerrak CA rows, from datras_q2q4_fetch.py output."""
    h = pd.read_csv(os.path.join(q_dir, "hh.csv"))
    h = h[(h.HaulVal.astype(str).str.strip() == "V") & (h.Gear.astype(str).str.strip() == "GOV")]
    for c in ("Country", "Ship", "StNo"):
        h[c] = h[c].astype(str).str.strip()
    h["k"] = list(zip(h.Country, h.Ship, h.StNo, h.HaulNo.astype(str), h.Year, h.Quarter))
    hl = pd.read_csv(os.path.join(q_dir, "hl.csv"))
    for c in ("Country", "Ship", "StNo"):
        hl[c] = hl[c].astype(str).str.strip()
    hl["k"] = list(zip(hl.Country, hl.Ship, hl.StNo, hl.HaulNo.astype(str), hl.Year, hl.Quarter))
    meta = h.drop_duplicates("k").set_index("k")[["DataType", "HaulDur"]]
    hl = hl[hl.k.isin(meta.index)].join(meta, on="k")
    hl["L"] = len_cm(hl.LngtCode.astype(str), hl.LngtClass) + np.where(
        hl.LngtCode.astype(str).str.strip() == "1", 0.5, 0.25)
    sub = pd.to_numeric(hl.SubFactor, errors="coerce")
    hl["n_h"] = pd.to_numeric(hl.HLNoAtLngt, errors="coerce") * sub.where(sub > 0, 1.0)
    raw_rows = hl.DataType.astype(str).str.strip() == "R"
    hl.loc[raw_rows, "n_h"] *= 60.0 / pd.to_numeric(hl.loc[raw_rows, "HaulDur"], errors="coerce")
    hl["aph"] = hl.Valid_Aphia.astype(int)
    hl = hl[hl.aph.isin(SP)]
    hl["kg_h"] = hl.n_h * hl.aph.map(LW.a) * hl.L ** hl.aph.map(LW.b) / 1000
    hl["Lcm"] = np.floor(hl.L).astype(int)
    ca = pd.read_csv(os.path.join(q_dir, "ca.csv"))
    ca = ca[ca.AreaCode.astype(str).str.strip().isin(SKAG_RECT)]
    keys = alk(ca)
    bio = {}
    for q in quarters:
        hq, lq = h[h.Quarter == q], hl[hl.Quarter == q]
        yrs = sorted(hq.Year.unique())
        print(f"{label} Q{q}: box GOV hauls {len(hq)} in years {yrs}; CA Skagerrak rows {len(ca[ca.Quarter == q])}")
        if len(hq) == 0:
            continue
        per_y = [lq[lq.Year == y].groupby(["aph", "Lcm"]).kg_h.sum() / len(hq[hq.Year == y]) for y in yrs]
        s_ = pd.concat(per_y, axis=1).fillna(0.0).mean(axis=1)
        for aph, sp in SP.items():
            if aph in s_.index.get_level_values(0):
                bio[(q, sp)] = s_.loc[aph].sort_index()
    return bio, keys, ca


def shift_q1_key(keys, q1_keys, ca, label):
    """Q4 has few aged fish; below 50 aged fish a species' Q4 key is the Q1 key shifted one year (Q1 age 1
    = the Q4 age-0 cohort ~2-3 months later; winter growth is small)."""
    n = ca.assign(n=pd.to_numeric(ca.CANoAtLngt, errors="coerce").fillna(1.0),
                  a=pd.to_numeric(ca.Age, errors="coerce")).query("a >= 0").groupby(["Valid_Aphia", "Quarter"]).n.sum()
    used = []
    for aph, sp in SP.items():
        if n.get((aph, 4), 0.0) < 50 and (sp, 1) in q1_keys:
            k1 = q1_keys[(sp, 1)]
            keys[(sp, 4)] = pd.DataFrame({0: k1[1], 1: k1[2], 2: k1[3], 3: 0.0}, index=k1.index)
            used.append(sp)
    print(f"{label}: Q4 keys from the shifted Q1 key: {used}")


B24, K24, C24 = datras_bio(q24, (2, 4), "1991-96")
q13 = os.path.join(os.path.dirname(q24.rstrip("/")), "datras_q1q3_1991_96")
B13, K13, C13 = datras_bio(q13, (1, 3), "1991-96")
K90 = dict(K13)
K90.update({k: v for k, v in K24.items()})
shift_q1_key(K90, K13, C24, "1991-96")
B90 = dict(B13)
B90.update(B24)

# ------------------------------------------------------------------ shares
def species_rows(label, bio, keys, quarters):
    rows = []
    for q in quarters:
        for sp in SP.values():
            if (q, sp) not in bio:
                continue
            b = bio[(q, sp)]
            key = keys.get((sp, q))
            a0, a1, tot = by_age(b, key, sp, q, True)
            r0, r1, rtot = by_age(b, key, sp, q, False)
            L0 = np.nan
            if key is not None:
                p0 = key.reindex(np.clip(b.index.values, 0, 80)).values[:, 0]
                wt = b.values / q_walker(sp, b.index.values + 0.5) * p0
                L0 = float((wt * (b.index.values + 0.5)).sum() / wt.sum()) if wt.sum() > 0 else np.nan
            rows.append(dict(period=label, quarter=q, species=sp, group=GROUP[sp], kg_h_q=tot, kg_h_raw=rtot,
                             age0=a0 / tot if tot > 0 else np.nan, age1=a1 / tot if tot > 0 else np.nan,
                             age0_raw=r0 / rtot if rtot > 0 else np.nan,
                             age1_raw=r1 / rtot if rtot > 0 else np.nan, age0_mean_L=L0, key=key is not None))
    return rows


R = pd.DataFrame(species_rows("2001-25", BIO, KEY, (1, 3)) + species_rows("1991-96", B90, K90, (1, 2, 3, 4)))
R.to_csv(os.path.join(out, "juvenile_share_by_species.csv"), index=False)
print("\nper species: age-0 and age-1 share of the species' biomass (q-corrected where Walker gives q; raw "
      "alongside); age0_mean_L = biomass-mean length of the age-0 fish (cm)")
with pd.option_context("display.width", 220, "display.max_rows", 300):
    print(R[R.key].drop(columns="key").round(3).to_string(index=False))

print("\ngroup shares of biomass, age 0 | age 1 (within-species q-corrected; raw in brackets). Planktivores by "
      "pool weight, benthivores by member weight; species without an age key count as 0 (lower bound)")
G = []
for (label, q), d in R.groupby(["period", "quarter"]):
    for g in ("plk", "ben", "wht"):
        e = d[d.group == g]
        if e.empty:
            continue
        W = POOL_W[q] if g == "plk" else MEMBER_W
        w = e.species.map(W).fillna(0.0)
        tot = sum(W.values()) if g == "plk" else sum(MEMBER_W[x] for x in e.species if x in MEMBER_W)
        w = w / tot
        k = e.key.values
        a0, a1 = float((w * e.age0.fillna(0) * k).sum()), float((w * e.age1.fillna(0) * k).sum())
        r0, r1 = float((w * e.age0_raw.fillna(0) * k).sum()), float((w * e.age1_raw.fillna(0) * k).sum())
        cover = float((w * k).sum())
        G.append(dict(period=label, quarter=q, group=g, age0=a0, age1=a1, age0_raw=r0, age1_raw=r1, keyed=cover))
        print(f"  {label} Q{q} {g}: age 0 {a0:.3f} ({r0:.3f}), age 1 {a1:.3f} ({r1:.3f}); keyed weight {cover:.2f}")
pd.DataFrame(G).to_csv(os.path.join(out, "juvenile_share_by_group.csv"), index=False)


# ------------------------------------------------------------------ size-defined stage (for stage_access)
def graduation_length(ca):
    """Mean length (cm) at age 1 in Q3 per species: the size at the second summer, where the stage ends."""
    c = ca.copy()
    c["L"] = len_cm(c.LngtCode.astype(str), c.LngtClass)
    c["Age"] = pd.to_numeric(c.Age, errors="coerce")
    c["n"] = pd.to_numeric(c.CANoAtLngt, errors="coerce").fillna(1.0)
    c = c[(c.Age == 1) & (c.Quarter == 3) & c.L.notna()]
    return {SP[int(a)]: float((d.L * d.n).sum() / d.n.sum()) for a, d in c.groupby("Valid_Aphia")
            if int(a) in SP and d.n.sum() >= 20}


LG = graduation_length(ca13)
for sp, v in graduation_length(C13).items():
    LG.setdefault(sp, v)
print("\ngraduation length L_g (mean length at age 1 in Q3, cm; 2021-25 CA, else 1991-96): "
      + ", ".join(f"{k} {v:.1f}" for k, v in sorted(LG.items())))


def stage_lengths(label, bio, quarters):
    """Biomass (q-corrected kg/h) by 1-cm class and stage: J = below the species' L_g, A = the rest."""
    rows = []
    for q in quarters:
        for sp in SP.values():
            if (q, sp) not in bio or sp not in LG:
                continue
            b = bio[(q, sp)]
            L = b.index.values.astype(int)
            v = b.values / q_walker(sp, L + 0.5)
            jj = (L + 0.5 < LG[sp]).astype(float)
            for Lc, vv, j in zip(L, v, jj):
                rows.append(dict(period=label, quarter=q, species=sp, L_cm=int(Lc), J=vv * j, A=vv * (1 - j)))
    return rows


SL = pd.DataFrame(stage_lengths("2001-25", BIO, (1, 3)) + stage_lengths("1991-96", B90, (1, 2, 3, 4)))
SL.to_csv(os.path.join(out, "stage_length_biomass.csv"), index=False)
print("size-defined stage: J/B by group (planktivores pool-weighted, benthivores member-weighted; species without "
      "L_g count as adult)")
GS = []
for (label, q), d in SL.groupby(["period", "quarter"]):
    for g in ("plk", "ben", "wht"):
        W = POOL_W[q] if g == "plk" else MEMBER_W
        num = den = 0.0
        for sp, w in W.items():
            if GROUP.get(sp) != g:
                continue
            e = d[d.species == sp]
            den += w
            if not e.empty and (e.J + e.A).sum() > 0:
                num += w * e.J.sum() / (e.J + e.A).sum()
        if den > 0:
            GS.append(dict(period=label, quarter=q, group=g, J_share=num / den))
            print(f"  {label} Q{q} {g}: J/B {num / den:.3f}")
pd.DataFrame(GS).to_csv(os.path.join(out, "size_stage_share_by_group.csv"), index=False)
