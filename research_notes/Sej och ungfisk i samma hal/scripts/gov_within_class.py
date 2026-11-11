"""Within-class (haul-level) co-occurrence of adult saithe and the benthivores' juveniles in NS-IBTS GOV hauls,
box and a wider Skagerrak / Norwegian Trench window, Q1 and Q3 2000-25: the factor by which the juvenile density
met by the mean saithe tonne INSIDE a bed-depth class differs from that class's mean juvenile density.

Usage: python3 -I gov_within_class.py <extract_2000_20> <extract_2021_25> <lw_csv> <depth_npz> <fg_library.yaml> <out_dir>
  <extract_*> = research_notes/"Fiskgrupper i rutan"/data/extract and research_notes/"Torskfiskarnas
                artsammansattning" (a-umlaut)/data/extract (HH.csv.gz, CPUEL.csv.gz; ICES DATRAS, CC BY 4.0). Parsed only.
  <lw_csv>    = research_notes/"Fiskgrupper i rutan"/tables/length_weight_all.csv
  <depth_npz> = maps/depth.npz (cell shares per bed class, as gov_juv_bed.py)
  <fg_library.yaml> = the live library, READ ONLY (the pair's depth_access_table)

Haul construction: copied from research_notes/"Rekryternas bottendjup"/scripts/gov_juv_bed.py (HaulVal V, GOV,
Q1/Q3, BySpecRecCode 1 so zeros are real, the 263.7 key, bed classes < 100 / 100-150 / 150-250 / >= 250 m, Walker
q-at-length for haddock and cod, J cuts haddock 27.0, cod 27.1, plaice 16.0, witch 14.8 cm, member weights of the
library), with saithe (126441) added: S = saithe >= 35 cm, kg/h (raw: a length-independent q cancels in every
ratio below). Group stage density per haul G_X,h = sum_s m_s X_s,h / D^B_s (D^B_s the species' cell-weighted
B in that region and quarter), X = J, Js (J below 19 cm, the in-gape part), A, B.

Factors, per class k (ratio of sums, zeros kept; w_h = year multiplicity in a bootstrap draw):
  W_k      = sum_h S_h Y_h / sum_h S_h Ybar_k             haul level / class level (all years pooled): the
             within-class co-occurrence factor asked for (Ybar_k the class mean prey; LOO check: Ybar_k,-h).
  E_k(g)   = sum_h S_h Ybar_g(h) / sum_h S_h Ybar_k       the W that co-occurrence ONLY at the scale of the strata g
             would give (= the mean of W_k under permutation of Y inside g, exactly); W_k / E_k(g) = what remains
             inside g. Strata (nested in the class):
             'band'  25 m bed-depth band (years pooled): the depth gradient inside the class;
             'rect'  ICES statistical rectangle (years pooled): position at ~30 x 55 km;
             'year'  survey year: interannual covariation (not spatial). Small strata (2-10 hauls per year and
                     class) put each haul's own S*Y into Ybar_g with weight 1/n_g, so E(year) is biased towards W;
                     'yearLOO' (Ybar_g,-h, strata of one haul dropped) brackets it from the other side.
  aggregate = sum_k e_k X_k / sum_k e_k, e_k = c_k m_k Sbar_k Ybar_k (the class-level encounter of the pair: cell
             share x the pair's depth_access_table (mean 1) x saithe x prey); 'nom' = without m_k.
  Identity: F_haul (cell-weighted) = F_class x W (aggregate without m).
  Decomposition W(J) = W(B) x R: R = (J/B met by saithe, S*B-weighted) / (class J/B) - the B part (saithe at the
  benthivores' hauls; the engine's policies can make it) and the J/B part (the juvenile share where saithe stand;
  the engine carries J inside B).
Nulls (1000 permutations, S fixed): Y shuffled among hauls inside each class (independence at the observed
presence and catch distributions: mean 1, the noise spread), and inside each (band, class) (residual beyond the
depth gradient). Year-block bootstrap: 1000 resamples of whole years, the same draw for box and window.
Robustness: saithe as presence/absence; the largest saithe haul per class dropped; per swept km2; raw (no Walker
q); saithe >= 40 cm.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

E1, E2, LWF, DEP, LIBF, OUT = sys.argv[1:7]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261111)
NBOOT = int(os.environ.get("NBOOT", "1000"))
NPERM = int(os.environ.get("NPERM", "1000"))

SP = {126437: "haddock", 126436: "cod", 127137: "long rough dab", 127139: "dab", 127143: "plaice",
      127136: "witch", 105865: "starry ray", 101170: "hagfish", 127140: "lemon sole", 150637: "grey gurnard",
      126441: "saithe"}
MEMBER = {"haddock": 0.386, "cod": 0.314, "long rough dab": 0.126, "dab": 0.045, "plaice": 0.038, "witch": 0.017,
          "starry ray": 0.015, "hagfish": 0.013, "lemon sole": 0.009, "grey gurnard": 0.009}
CUT = {"haddock": 27.0, "cod": 27.1, "plaice": 16.0, "witch": 14.8}
EDGES = [100.0, 150.0, 250.0]
CLS = ["<100", "100-150", "150-250", ">=250"]
REG = {"box": (57.7, 58.4, 9.9, 11.3), "wide": (57.3, 59.2, 8.0, 12.0)}
PRESENCE = {1: 0.1, 3: 0.9}   # saithe box presence by quarter (research_notes/"Sejens hojd over botten")

z = np.abs(np.load(DEP)["depth"].astype(float))
z = z[np.isfinite(z)]
CELL = np.array([(np.digitize(z, EDGES) == k).mean() for k in range(4)])
LIB = yaml.safe_load(open(LIBF, encoding="utf-8-sig"))
rows = LIB["interaction_definitions"]["predatory_fish_preys_on_benthivorous_fish"]["depth_access_table"]
M = np.array([rows[0]["factor"], rows[1]["factor"], rows[3]["factor"], rows[5]["factor"]], dtype=float)
M = M / (CELL * M).sum()


def q_walker(sp, L):
    L = np.asarray(L, dtype=float)
    if sp == "haddock":
        return np.where(L <= 11, 0.0116121, np.where(L >= 46, 0.123268, -0.370453 + 0.04227 * L - 0.000685 * L ** 2))
    if sp == "cod":
        return np.where(L <= 25, 0.15963, np.where(L >= 97, 0.66822, 0.20935 - 0.00434 * L + 0.00009 * L ** 2))
    return np.ones_like(L)


# ------------------------------------------------------------------ hauls (as gov_juv_bed.py)
hh = pd.concat([pd.read_csv(os.path.join(d, "HH.csv.gz")) for d in (E1, E2)])
hh = hh[(hh.HaulVal.astype(str).str.strip() == "V") & (hh.Gear.astype(str).str.strip() == "GOV")
        & hh.Quarter.isin([1, 3]) & (hh.BySpecRecCode == 1) & (hh.Depth > 0)].copy()
hh["Ship"] = hh.Ship.astype(str).str.strip()
hh["la"], hh["lo"] = hh.ShootLat.round(3), hh.ShootLong.round(3)
a0, a1, o0, o1 = REG["wide"]
hh = hh[(hh.ShootLat >= a0) & (hh.ShootLat <= a1) & (hh.ShootLong >= o0) & (hh.ShootLong <= o1)]
hh = hh.drop_duplicates(["Year", "Quarter", "Ship", "HaulNo", "la", "lo"]).copy()
hh["key"] = list(zip(hh.Year, hh.Quarter, hh.Ship, hh.HaulNo, hh.la, hh.lo))
hh["cl"] = np.digitize(hh.Depth, EDGES)
a0, a1, o0, o1 = REG["box"]
hh["box"] = (hh.ShootLat >= a0) & (hh.ShootLat <= a1) & (hh.ShootLong >= o0) & (hh.ShootLong <= o1)
ws = hh.WingSpread.where(hh.WingSpread > 0)
med = ws.groupby(hh.cl).median()
hh["ws"] = ws.fillna(hh.cl.map(med))
dist = hh.Distance.where(hh.Distance > 0)
hh["dist"] = dist.fillna(hh.GroundSpeed.where(hh.GroundSpeed > 0, 3.8) * 1852.0 * hh.HaulDur / 60.0)
hh["km2_h"] = hh.ws * hh.dist / (hh.HaulDur / 60.0) / 1e6
hh = hh.set_index("key")
print(f"hauls in window {len(hh)} (box {int(hh.box.sum())}); years {hh.Year.min()}-{hh.Year.max()}")
print(f"model map cell shares per bed class {np.round(CELL, 4).tolist()}; pair depth table (mean 1) "
      f"{np.round(M, 3).tolist()}")
print(f"median tow length {hh.dist.median() / 1000:.2f} km, wing spread {hh.ws.median():.1f} m (model cell 1 km)")

lw = pd.read_csv(LWF, index_col=0)
cp = pd.concat([pd.read_csv(os.path.join(d, "CPUEL.csv.gz")) for d in (E1, E2)])
cp = cp[cp.AphiaID.isin(SP)].copy()
cp["Ship"] = cp.Ship.astype(str).str.strip()
cp["key"] = list(zip(cp.Year, cp.Quarter, cp.Ship, cp.HaulNo, cp.ShootLat.round(3), cp.ShootLon.round(3)))
cp = cp[cp.key.isin(set(hh.index))].copy()
cp["sp"] = cp.AphiaID.map(SP)
cp["L"] = cp.LngtClas / 10.0 + 0.5
cp["kg"] = cp.CPUE_number_per_hour * cp.AphiaID.map(lw.a) * cp.L ** cp.AphiaID.map(lw.b) / 1000.0
cp["q"] = 1.0
for s in ("haddock", "cod"):
    m = cp.sp == s
    cp.loc[m, "q"] = q_walker(s, cp.loc[m, "L"].values)
cp["kgq"] = cp.kg / cp.q
print(f"catch rows matched {len(cp)}")

cols = {}


def add(name, mask, val):
    cols[name] = cp[mask].groupby("key")[val].sum()


for s in MEMBER:
    m = cp.sp == s
    for v, tag in (("kg", "raw"), ("kgq", "q")):
        add(f"{s}|B|{tag}", m, v)
        cut = CUT.get(s)
        add(f"{s}|J|{tag}", m & (cp.L < cut) if cut else m & False, v)
        add(f"{s}|Js|{tag}", m & (cp.L < min(cut, 19.0)) if cut else m & False, v)
ms = cp.sp == "saithe"
add("S35", ms & (cp.L >= 35), "kg")
add("S40", ms & (cp.L >= 40), "kg")
X = pd.DataFrame(cols).reindex(hh.index).fillna(0.0)
for s in MEMBER:
    for tag in ("raw", "q"):
        X[f"{s}|A|{tag}"] = X[f"{s}|B|{tag}"] - X[f"{s}|J|{tag}"]
XS = X.div(hh.km2_h, axis=0)
META = hh[["Year", "Quarter", "cl", "box", "Depth", "StatRec"]].copy()
META["band"] = (META.Depth // 25).astype(int)
YEARS = sorted(META.Year.unique())
YIDX = {y: i for i, y in enumerate(YEARS)}


def group_stage(Xm, sel, stage, tag, wy):
    """Member-weighted stage density per haul (relative units), species scaled by their cell-weighted B in the
    selection (region x quarter), with the year weights of the draw."""
    meta = META[sel]
    w = wy[meta.Year.map(YIDX).values]
    g = np.zeros(int(sel.sum()))
    for s in MEMBER:
        b = Xm.loc[sel, f"{s}|B|{tag}"].values
        dB = 0.0
        for k in range(4):
            mk = meta.cl.values == k
            if w[mk].sum() > 0:
                dB += CELL[k] * (w[mk] * b[mk]).sum() / w[mk].sum()
        if dB <= 0:
            continue
        g += MEMBER[s] * Xm.loc[sel, f"{s}|{stage}|{tag}"].values / dB
    return g


def strata_ids(meta, kind):
    cl = meta.cl.values.astype(np.int64)
    if kind == "band":
        return meta.band.values.astype(np.int64) * 10 + cl
    if kind == "rect":
        return pd.factorize(meta.StatRec.astype(str))[0].astype(np.int64) * 10 + cl
    if kind in ("year", "yearLOO"):
        return meta.Year.values.astype(np.int64) * 10 + cl
    return cl


SKINDS = ("band", "rect", "year", "yearLOO")


def class_terms(S, Y, w, meta):
    """Per class: Sbar, Ybar, W (plain), W_LOO, E(g) for every stratification."""
    out = {"Sb": np.full(4, np.nan), "Yb": np.full(4, np.nan), "W": np.full(4, np.nan),
           "Wloo": np.full(4, np.nan), "n": np.zeros(4)}
    for kd in SKINDS:
        out[kd] = np.full(4, np.nan)
    cl = meta.cl.values
    for k in range(4):
        mk = (cl == k) & (w > 0)
        if mk.sum() < 2:
            continue
        s, y, ww = S[mk], Y[mk], w[mk]
        n = ww.sum()
        out["n"][k] = n
        Sb, Yb = (ww * s).sum() / n, (ww * y).sum() / n
        out["Sb"][k], out["Yb"][k] = Sb, Yb
        num = (ww * s * y).sum()
        base = (ww * s).sum() * Yb
        if base <= 0:
            continue
        out["W"][k] = num / base
        yloo = ((ww * y).sum() - ww * y) / np.maximum(n - ww, 1e-12)
        out["Wloo"][k] = num / (ww * s * yloo).sum()
        sub = meta[mk]
        for kd in SKINDS:
            g = strata_ids(sub, kd)
            u, inv = np.unique(g, return_inverse=True)
            n_g = np.bincount(inv, ww)
            y_g = np.bincount(inv, ww * y)
            if kd == "yearLOO":
                rest = n_g[inv] - ww
                keep = rest > 0
                yg = np.where(keep, (y_g[inv] - ww * y) / np.where(keep, rest, 1.0), 0.0)
                b2 = (ww * s * yloo)[keep].sum()
                out[kd][k] = (ww * s * yg)[keep].sum() / b2 if b2 > 0 else np.nan
            else:
                out[kd][k] = (ww * s * (y_g / n_g)[inv]).sum() / base
    return out


def agg(t, key, use_m=True):
    e = CELL * (M if use_m else 1.0) * t["Sb"] * t["Yb"]
    v = t[key]
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
    return float((e[ok] * v[ok]).sum() / e[ok].sum()) if ok.any() else np.nan


def one(reg, q, wy, stage="J", tag="q", unit="h", svar="S35", spa=False, drop_top=False, prey_override=None,
        prey_from=None):
    sel = (META.Quarter.values == q) & (META.box.values if reg == "box" else np.ones(len(META), bool))
    sel = pd.Series(sel, index=META.index)
    Xm = X if unit == "h" else XS
    meta = META[sel]
    w = wy[meta.Year.map(YIDX).values].astype(float)
    S = Xm.loc[sel, svar].values.astype(float)
    if prey_from is not None:            # a benthivore stage in the predator's place (J vs A check)
        S = group_stage(Xm, sel, prey_from, tag, wy)
    if spa:
        S = (S > 0).astype(float)
    if drop_top:
        w = w.copy()
        for k in range(4):
            mk = np.where((meta.cl.values == k) & (w > 0))[0]
            if len(mk):
                w[mk[np.argmax(S[mk])]] = 0.0
    Y = prey_override if prey_override is not None else group_stage(Xm, sel, stage, tag, wy)
    t = class_terms(S, Y, w, meta)
    r = {"t": t}
    for key in ("W", "Wloo") + SKINDS:
        r[key] = agg(t, key)
        r[key + "_nom"] = agg(t, key, use_m=False)
    ok = np.isfinite(t["Sb"]) & np.isfinite(t["Yb"])
    c, mm = CELL[ok], M[ok]
    Ybox = (c * t["Yb"][ok]).sum() / c.sum()
    r["F_class"] = float((c * t["Sb"][ok] * t["Yb"][ok]).sum() / ((c * t["Sb"][ok]).sum() * Ybox))
    r["F_class_m"] = float((c * mm * t["Sb"][ok] * t["Yb"][ok]).sum() / ((c * mm * t["Sb"][ok]).sum() * Ybox))
    num = 0.0
    for k in np.where(ok)[0]:
        mk = (meta.cl.values == k) & (w > 0)
        num += CELL[k] * (w[mk] * S[mk] * Y[mk]).sum() / w[mk].sum()
    r["F_haul"] = float(num / ((c * t["Sb"][ok]).sum() * Ybox))
    return r, S, Y, w, meta


W1 = np.ones(len(YEARS))
VARIANTS = {
    "J": dict(stage="J"),
    "Js": dict(stage="Js"),
    "A": dict(stage="A"),
    "B": dict(stage="B"),
    "J_raw": dict(stage="J", tag="raw"),
    "J_sa": dict(stage="J", unit="sa"),
    "J_S40": dict(stage="J", svar="S40"),
    "J_Spa": dict(stage="J", spa=True),
    "J_droptop": dict(stage="J", drop_top=True),
    "JvsA": dict(stage="J", prey_from="A"),
}
QS = (1, 3)
central = {(reg, q, vn): one(reg, q, W1, **kw)[0] for reg in REG for q in QS for vn, kw in VARIANTS.items()}


def fmt4(v, d=2):
    return "/".join("-" if not np.isfinite(x) else f"{x:.{d}f}" for x in v)


# ---------------------------------------------------------------- descriptive
print("\nHauls per class (< 100 / 100-150 / 150-250 / >= 250 m), share with saithe >= 35 cm, share with J; hauls per"
      " (year, class) stratum (median):")
for reg in REG:
    for q in QS:
        r, S, Y, w, meta = one(reg, q, W1, stage="J")
        n = [int((meta.cl.values == k).sum()) for k in range(4)]
        ps = [((S > 0) & (meta.cl.values == k)).sum() / max(n[k], 1) for k in range(4)]
        pj = [((Y > 0) & (meta.cl.values == k)).sum() / max(n[k], 1) for k in range(4)]
        per = [np.median(np.bincount(pd.factorize(meta.Year.values[meta.cl.values == k])[0])) if n[k] else 0
               for k in range(4)]
        print(f"  {reg:4s} Q{q}: n {n}; S>0 {fmt4(ps)}; J>0 {fmt4(pj)}; hauls per year-class {per}")

print("\nClass means and the class-level factor (saithe kg/h; J relative units):")
for reg in REG:
    for q in QS:
        r = central[(reg, q, "J")]
        t = r["t"]
        print(f"  {reg:4s} Q{q}: saithe {fmt4(t['Sb'])}  J {fmt4(t['Yb'], 3)}  F_class {r['F_class']:.3f} (with the"
              f" depth table {r['F_class_m']:.3f}); F_haul {r['F_haul']:.3f} = F_class x W(nom) "
              f"{r['F_class'] * r['W_nom']:.3f}")

# ---------------------------------------------------------------- bootstrap and nulls
print(f"\nBootstrap: {NBOOT} year resamples; permutation nulls: {NPERM} ...", flush=True)
boot = []
for b in range(NBOOT):
    draw = rng.choice(len(YEARS), size=len(YEARS), replace=True)
    wy = np.bincount(draw, minlength=len(YEARS)).astype(float)
    boot.append({(reg, q, vn): one(reg, q, wy, **VARIANTS[vn])[0]
                 for reg in REG for q in QS for vn in ("J", "Js", "A", "B")})

NULL = {}
for reg in REG:
    for q in QS:
        for vn in ("J", "Js"):
            r, S, Y, w, meta = one(reg, q, W1, **VARIANTS[vn])
            for nk, g in (("class", meta.cl.values.astype(np.int64)), ("band", strata_ids(meta, "band"))):
                idx = [np.where(g == u)[0] for u in np.unique(g)]
                vals = []
                for p in range(NPERM):
                    Yp = Y.copy()
                    for ii in idx:
                        Yp[ii] = Y[rng.permutation(ii)]
                    vals.append(one(reg, q, W1, prey_override=Yp, **VARIANTS[vn])[0]["W"])
                NULL[(reg, q, vn, nk)] = np.array(vals)


def qcomb(d, reg, vn, fn, mode="pres"):
    pw = PRESENCE if mode == "pres" else {1: 0.5, 3: 0.5}
    return sum(pw[q] * fn(d[(reg, q, vn)]) for q in QS)


def ci(fn):
    arr = np.array([fn(bb) for bb in boot])
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


QUANT = {
    "W": lambda r: r["W"],
    "W_nom": lambda r: r["W_nom"],
    "Wloo": lambda r: r["Wloo"],
    "E_band": lambda r: r["band"],
    "W/E_band": lambda r: r["W"] / r["band"],
    "E_rect": lambda r: r["rect"],
    "W/E_rect": lambda r: r["W"] / r["rect"],
    "E_year": lambda r: r["year"],
    "W/E_year": lambda r: r["W"] / r["year"],
    "E_yearLOO": lambda r: r["yearLOO"],
    "W/E_yearLOO": lambda r: r["Wloo"] / r["yearLOO"],
}
OUTJ = {"cell_shares": CELL.tolist(), "depth_table_mean1": M.tolist(), "rows": {}}
print("\n=== Within-class factor W = haul level / class level, aggregated over the classes with the pair's encounter"
      " weights (cell share x depth table x saithe x prey). 95 % year-bootstrap in brackets. E(g) = the W that"
      " co-occurrence only at the scale of g gives; W / E(g) = what remains inside g. ===")
for reg in REG:
    print(f"\n--- {reg} ---")
    for vn in ("J", "Js", "A", "B"):
        for q in list(QS) + ["pres", "eq"]:
            line = f"  {vn:2s} {('Q' + str(q)) if q in QS else ('Q3 0.9/Q1 0.1' if q == 'pres' else 'equal Q'):13s}"
            for qn, fn in QUANT.items():
                if q in QS:
                    v = fn(central[(reg, q, vn)])
                    lo, hi = ci(lambda bb, fn=fn: fn(bb[(reg, q, vn)]))
                else:
                    v = qcomb(central, reg, vn, fn, q)
                    lo, hi = ci(lambda bb, fn=fn: qcomb(bb, reg, vn, fn, q))
                OUTJ["rows"][f"{reg}|{q}|{vn}|{qn}"] = dict(central=v, lo=lo, hi=hi)
                if qn in ("W", "E_band", "W/E_band", "E_rect", "W/E_rect", "W/E_year", "W/E_yearLOO"):
                    line += f" {qn} {v:.2f} [{lo:.2f}-{hi:.2f}]"
            print(line)
            if q in QS:
                t = central[(reg, q, vn)]["t"]
                print(f"{'':17s}per class: W {fmt4(t['W'])}; E_band {fmt4(t['band'])}; W LOO {fmt4(t['Wloo'])};"
                      f" W(nom) {central[(reg, q, vn)]['W_nom']:.3f}")
                for nk in ("class", "band"):
                    if (reg, q, vn, nk) in NULL:
                        nv = NULL[(reg, q, vn, nk)]
                        obs = central[(reg, q, vn)]["W"]
                        OUTJ["rows"][f"{reg}|{q}|{vn}|null_{nk}"] = [float(nv.mean()), float(np.percentile(nv, 2.5)),
                                                                     float(np.percentile(nv, 97.5)),
                                                                     float((nv <= obs).mean())]
                        print(f"{'':17s}null (shuffle inside {nk}): mean {nv.mean():.3f}, 95 % "
                              f"{np.percentile(nv, 2.5):.2f}-{np.percentile(nv, 97.5):.2f}; p(null <= obs) "
                              f"{(nv <= obs).mean():.3f}")

print("\n=== Decomposition W(J) = W(B) x R; R = J/B met by saithe relative to the class J/B ===")
for reg in REG:
    for q in list(QS) + ["pres"]:
        if q in QS:
            wj, wb = central[(reg, q, "J")]["W"], central[(reg, q, "B")]["W"]
            lo, hi = ci(lambda bb: bb[(reg, q, "J")]["W"] / bb[(reg, q, "B")]["W"])
            lo2, hi2 = ci(lambda bb: bb[(reg, q, "B")]["W"])
        else:
            wj = qcomb(central, reg, "J", QUANT["W"])
            wb = qcomb(central, reg, "B", QUANT["W"])
            lo, hi = ci(lambda bb: qcomb(bb, reg, "J", QUANT["W"]) / qcomb(bb, reg, "B", QUANT["W"]))
            lo2, hi2 = ci(lambda bb: qcomb(bb, reg, "B", QUANT["W"]))
        OUTJ["rows"][f"{reg}|{q}|R"] = dict(central=wj / wb, lo=lo, hi=hi, WB=wb, WB_lo=lo2, WB_hi=hi2)
        print(f"  {reg:4s} {str(q):4s}: W(J) {wj:.3f} = W(B) {wb:.3f} [{lo2:.2f}-{hi2:.2f}] x R {wj / wb:.3f} "
              f"[{lo:.2f}-{hi:.2f}]")

print("\n=== J against the benthivore adults inside the class (the engine carries J inside B: W(J vs A) = 1 is"
      " 'J spread like A') ===")
for reg in REG:
    for q in QS:
        r = central[(reg, q, "JvsA")]
        print(f"  {reg:4s} Q{q}: per class {fmt4(r['t']['W'])}; E_band {fmt4(r['t']['band'])}")

print("\n=== Robustness of W(J) (aggregate, with the depth table) and W / E_band ===")
for reg in REG:
    for q in list(QS) + ["pres"]:
        line = f"  {reg:4s} {str(q):4s}:"
        for vn in ("J", "J_raw", "J_sa", "J_S40", "J_Spa", "J_droptop"):
            if q in QS:
                v, v2 = central[(reg, q, vn)]["W"], central[(reg, q, vn)]["W"] / central[(reg, q, vn)]["band"]
            else:
                v = qcomb(central, reg, vn, QUANT["W"])
                v2 = qcomb(central, reg, vn, QUANT["W/E_band"])
            OUTJ["rows"][f"{reg}|{q}|{vn}|robust"] = dict(W=v, W_over_Eband=v2)
            line += f"  {vn} {v:.2f}/{v2:.2f}"
        print(line)

json.dump(OUTJ, open(os.path.join(OUT, "within_class.json"), "w"), indent=1, default=float)
print(f"\nwrote {OUT}/within_class.json")
