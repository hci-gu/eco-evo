"""Bed-depth shape of the benthivores' juveniles (J), adults (A) and recruits in NS-IBTS GOV, box and a wider
Skagerrak / Norwegian Trench window, Q1 and Q3 2000-25, with year-block bootstrap CIs.

Usage: python3 -I gov_juv_bed.py <extract_2000_20> <extract_2021_25> <lw_csv> <depth_npz> <out_dir>
  <extract_*>  = research_notes/"Fiskgrupper i rutan"/data/extract and research_notes/"Torskfiskarnas
                 artsammansattning" (a-umlaut)/data/extract (HH.csv.gz, CPUEL.csv.gz: every species; ICES DATRAS,
                 CC BY 4.0). Parsed only.
  <lw_csv>     = research_notes/"Fiskgrupper i rutan"/tables/length_weight_all.csv
  <depth_npz>  = maps/depth.npz (the model's 60 x 60 bed depth; the equal-cell normalisation)

Definitions
  Hauls: HaulVal V, GOV, Q1 / Q3, BySpecRecCode 1 (all species recorded: zeros are real), keyed on year, quarter,
    ship, haul number, shoot lat / lon (the 263.7 key). Bed class from the haul's Depth: < 100 / 100-150 /
    150-250 / >= 250 m (the depth_access_table rows).
  Regions: box 57.7-58.4 N, 9.9-11.3 E (field_diet.FIELD_BED_SHARES, gov_juv_overlap.py); window 57.3-59.2 N, 8-12 E.
  Stage (Js = the J below 19 cm, the in-gape part, a gate diagnostic): J = below the species' length at age 1 in Q3 (library juvenile_stage comment; Arsungarnas andel section
    0): haddock 27.0, cod 27.1, plaice 16.0, witch 14.8 cm; dab, long rough dab, lemon sole, rays, gurnard,
    hagfish have no cut and count as adult (the J/B 0.15 convention). Variant 'flatJ': dab, long rough dab, lemon
    sole J below 12 cm [J proxy: no age readings in CA 2021-25].
  Recruits (box_share.md conventions, the recruit_import source): haddock age 0 = Q3 < 20 cm; cod age 1 = Q1
    10-27 cm; cod age 0 = Q3 < 20 cm.
  Biomass: n/h x W(L) (CA fits), L = class + 0.5 cm; Walker et al. 2017 q-at-length for haddock and cod ('q')
    or raw. Per hour, or per swept km2 ('sa': wing spread x towed distance; missing wing spread imputed by the
    window's median per bed class).
  Density per class: pooled haul mean. The >= 250 m class in the box (0 hauls Q1, 3 Q3): box 150-250 value x the
    window's (>= 250 / 150-250) ratio of the same quantity, same resample [J]; variant 'nb' = neighbour copy.
  Shape: f_k = d_k / sum_j c_j d_j, c = the model map's cell shares per class (equal-cell mean 1); class share
    s_k = c_k f_k.
  Group (member-weighted): G_X,k = sum_s m_s d^X_s,k / D^B_s, m_s the library member shares, D^B_s the species'
    cell-weighted total density in that quarter (each species scaled to its member weight, its J/B from data).
Bootstrap: 1000 resamples of whole years (2000-25), the same years for box and window.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

E1, E2, LWF, DEP, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261111)
NBOOT = int(os.environ.get("NBOOT", "1000"))

SP = {126437: "haddock", 126436: "cod", 127137: "long rough dab", 127139: "dab", 127143: "plaice",
      127136: "witch", 105865: "starry ray", 101170: "hagfish", 127140: "lemon sole", 150637: "grey gurnard"}
# library member_species shares (fg_library.yaml benthivorous_fish); lemon sole and grey gurnard from
# params_benthivorous_fish (0.9 % each, inside the library's 'others')
MEMBER = {"haddock": 0.386, "cod": 0.314, "long rough dab": 0.126, "dab": 0.045, "plaice": 0.038, "witch": 0.017,
          "starry ray": 0.015, "hagfish": 0.013, "lemon sole": 0.009, "grey gurnard": 0.009}
CUT = {"haddock": 27.0, "cod": 27.1, "plaice": 16.0, "witch": 14.8}
CUT_FLAT = dict(CUT, **{"dab": 12.0, "long rough dab": 12.0, "lemon sole": 12.0})
EDGES = [100.0, 150.0, 250.0]
CLS = ["<100", "100-150", "150-250", ">=250"]
REG = {"box": (57.7, 58.4, 9.9, 11.3), "wide": (57.3, 59.2, 8.0, 12.0)}

z = np.abs(np.load(DEP)["depth"].astype(float))
z = z[np.isfinite(z)]
CELL = np.array([(np.digitize(z, EDGES) == k).mean() for k in range(4)])


def q_walker(sp, L):
    L = np.asarray(L, dtype=float)
    if sp == "haddock":
        return np.where(L <= 11, 0.0116121, np.where(L >= 46, 0.123268, -0.370453 + 0.04227 * L - 0.000685 * L ** 2))
    if sp == "cod":
        return np.where(L <= 25, 0.15963, np.where(L >= 97, 0.66822, 0.20935 - 0.00434 * L + 0.00009 * L ** 2))
    return np.ones_like(L)


# ------------------------------------------------------------------ hauls
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
print(f"model map cell shares per bed class {np.round(CELL, 4).tolist()}  ({CLS})")
print(f"wing spread median by class (m) {np.round(med.values, 1).tolist()}; share of hauls with a recorded "
      f"wing spread {float((hh.WingSpread > 0).mean()):.2f}")
for reg in ("box", "wide"):
    h = hh[hh.box] if reg == "box" else hh
    print(f"  {reg}: hauls by quarter x class " + "; ".join(
        f"Q{q} {[int(((h.Quarter == q) & (h.cl == k)).sum()) for k in range(4)]}" for q in (1, 3))
          + f"; day share Q3 {float((h[h.Quarter == 3].DayNight == 'D').mean()):.2f}")

# ------------------------------------------------------------------ catches
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

# per-haul quantities: columns '<sp>|<stage>|<raw|q>'
cols = {}


def add(name, mask, val):
    cols[name] = cp[mask].groupby("key")[val].sum()


for s in SP.values():
    m = cp.sp == s
    for v, tag in (("kg", "raw"), ("kgq", "q")):
        add(f"{s}|B|{tag}", m, v)
        cut = CUT.get(s)
        add(f"{s}|J|{tag}", m & (cp.L < cut) if cut else m & False, v)
        cutf = CUT_FLAT.get(s)
        add(f"{s}|Jf|{tag}", m & (cp.L < cutf) if cutf else m & False, v)
        # Js: the small juveniles inside a 19 cm gape (saithe p95; Sejens hojd over botten finding 3)
        add(f"{s}|Js|{tag}", m & (cp.L < min(cut, 19.0)) if cut else m & False, v)
for v, tag in (("kg", "raw"), ("kgq", "q")):
    add(f"had0|R|{tag}", (cp.sp == "haddock") & (cp.L < 20) & (cp.Quarter == 3), v)
    add(f"cod1|R|{tag}", (cp.sp == "cod") & (cp.L >= 10) & (cp.L < 27) & (cp.Quarter == 1), v)
    add(f"cod0|R|{tag}", (cp.sp == "cod") & (cp.L < 20) & (cp.Quarter == 3), v)
X = pd.DataFrame(cols).reindex(hh.index).fillna(0.0)
for s in SP.values():  # A = B - J
    for tag in ("raw", "q"):
        X[f"{s}|A|{tag}"] = X[f"{s}|B|{tag}"] - X[f"{s}|J|{tag}"]
        X[f"{s}|Af|{tag}"] = X[f"{s}|B|{tag}"] - X[f"{s}|Jf|{tag}"]
XS = X.div(hh.km2_h, axis=0)  # per swept km2
META = hh[["Year", "Quarter", "cl", "box", "DayNight"]]


def class_means(Xm, meta, years_w):
    """Weighted class means per region and quarter; years_w: year -> resample count."""
    w = meta.Year.map(years_w).fillna(0.0).values
    out = {}
    for reg in ("box", "wide"):
        rm = meta.box.values if reg == "box" else np.ones(len(meta), bool)
        for q in (1, 3):
            out[(reg, q)] = np.full((4, Xm.shape[1]), np.nan)
            for k in range(4):
                m = rm & (meta.Quarter.values == q) & (meta.cl.values == k)
                ww = w[m]
                if ww.sum() > 0:
                    out[(reg, q)][k] = (Xm.values[m] * ww[:, None]).sum(0) / ww.sum()
    # fill empty classes: >= 250 in the box by the window ratio; any other gap by the neighbour
    for q in (1, 3):
        b, wd = out[("box", q)], out[("wide", q)]
        ratio = np.divide(wd[3], wd[2], out=np.ones_like(wd[3]), where=wd[2] > 0)
        b_nb = b.copy()
        b[3] = b[2] * ratio
        b_nb[3] = b_nb[2] if np.isnan(b_nb[3]).any() else b_nb[3]
        out[("box_nb", q)] = b_nb
        out[("box_3h", q)] = np.where(np.isnan(out[("box", q)]), b_nb, out[("box", q)])
        for key in (("box", q), ("wide", q), ("box_nb", q)):
            arr = out[key]
            for k in range(4):
                if np.isnan(arr[k]).any():
                    arr[k] = arr[k - 1] if k > 0 else arr[k + 1]
    return out


def box_raw3(Xm, meta, years_w, q):
    """The box's own >= 250 m class (3 Q3 hauls) where it exists."""
    w = meta.Year.map(years_w).fillna(0.0).values
    m = meta.box.values & (meta.Quarter.values == q) & (meta.cl.values == 3)
    if w[m].sum() <= 0:
        return None
    return (Xm.values[m] * w[m][:, None]).sum(0) / w[m].sum()


IDX = {c: i for i, c in enumerate(X.columns)}


def shape(d):
    tot = (CELL * d).sum()
    return d / tot if tot > 0 else np.full(4, np.nan)


def group(cm, stage, tag, members, jkey="J"):
    """Member-weighted class densities of a stage (relative units)."""
    g = np.zeros(4)
    for s in members:
        dB = cm[:, IDX[f"{s}|B|{tag}"]]
        DB = (CELL * dB).sum()
        if DB <= 0:
            continue
        st = {"J": jkey, "A": "A" if jkey == "J" else "Af", "B": "B", "Js": "Js"}[stage]
        g += MEMBER[s] * cm[:, IDX[f"{s}|{st}|{tag}"]] / DB
    return g


ALL = list(MEMBER)
GAD = ["haddock", "cod"]
FLAT = [s for s in ALL if s not in GAD]
SAITHE = np.array([0.043, 0.127, 0.782, 0.048])   # field_diet.FIELD_BED_SHARES predatory_fish (class shares)
WHITING = np.array([0.769, 0.204, 0.027, 0.001])  # field_diet.FIELD_BED_SHARES whiting
IMPORT = {"had0": 256.5, "cod1": 402.5}  # t/yr midpoints, library recruit_import comment (190-323, 336-469)


def quantities(cm_all):
    """Every reported number from one set of class means."""
    r = {}
    for unit, cmset in cm_all.items():
        for reg in ("box", "wide", "box_nb"):
            for tag in ("q", "raw"):
                p = f"{unit}|{reg}|{tag}"
                for q in (1, 3):
                    cm = cmset[(reg, q)]
                    for nm, mem, jk in (("all", ALL, "J"), ("gad", GAD, "J"), ("flatJ", ALL, "Jf"),
                                        ("had", ["haddock"], "J"), ("cod", ["cod"], "J"), ("flat", FLAT, "Jf")):
                        for st in ("J", "A", "B", "Js"):
                            g = group(cm, st, tag, mem, jk)
                            r[f"{p}|Q{q}|{nm}|{st}"] = shape(g)
                        gJ, gB = group(cm, "J", tag, mem, jk), group(cm, "B", tag, mem, jk)
                        r[f"{p}|Q{q}|{nm}|JB"] = np.array([(CELL * gJ).sum() / (CELL * gB).sum()])
                        r[f"{p}|Q{q}|{nm}|JBk"] = gJ / np.where(gB > 0, gB, np.nan)
                    for rc in ("had0", "cod1", "cod0"):
                        if (rc == "cod1") == (q == 1):
                            r[f"{p}|Q{q}|rec_{rc}"] = shape(cm[:, IDX[f"{rc}|R|{tag}"]])
                for nm in ("all", "gad", "flatJ", "had", "cod", "flat"):
                    for st in ("J", "A", "B", "Js"):
                        r[f"{p}|ann|{nm}|{st}"] = 0.5 * (r[f"{p}|Q1|{nm}|{st}"] + r[f"{p}|Q3|{nm}|{st}"])
                # import-tonne weighted arrival shape: haddock age 0 (Q3) + cod age 1 (Q1)
                w = IMPORT["had0"] / sum(IMPORT.values())
                r[f"{p}|rec_import"] = w * r[f"{p}|Q3|rec_had0"] + (1 - w) * r[f"{p}|Q1|rec_cod1"]
    return r


def run(years_w):
    cm = {"h": class_means(X, META, years_w), "sa": class_means(XS, META, years_w)}
    return quantities(cm), cm


years = sorted(hh.Year.unique())
central, CM = run({y: 1.0 for y in years})
boot = []
for b in range(NBOOT):
    draw = rng.choice(years, size=len(years), replace=True)
    yw = pd.Series(draw).value_counts().to_dict()
    boot.append(run(yw)[0])


def ci(key):
    arr = np.array([bb[key] for bb in boot])
    return np.nanpercentile(arr, 2.5, axis=0), np.nanpercentile(arr, 97.5, axis=0)


def fmt(v, d=3):
    return "/".join(f"{x:.{d}f}" for x in np.atleast_1d(v))


def show(key, label):
    v = central[key]
    lo, hi = ci(key)
    sh = CELL * v
    print(f"  {label:44s} f {fmt(v, 2):>23s}  [{fmt(lo, 2)} - {fmt(hi, 2)}]   shares {fmt(sh)}")


print("\nShape f per bed class (< 100 / 100-150 / 150-250 / >= 250 m), equal-cell mean 1 on the model map,"
      " 95 % year-bootstrap range; shares = f x cell share. Member-weighted, Walker q, per hour. Js = J below 19 cm.")
for reg in ("box", "wide"):
    print(f"\n== {reg} ==")
    for q in ("Q1", "Q3", "ann"):
        for nm, lab in (("all", "all members"), ("gad", "haddock + cod"), ("had", "haddock"), ("cod", "cod"),
                        ("flat", "flatfish etc. (J cut 12 cm proxy)")):
            for st in ("J", "Js", "A", "B"):
                if st == "Js" and nm not in ("all", "had"):
                    continue
                show(f"h|{reg}|q|{q}|{nm}|{st}", f"{q} {lab} {st}")
    for key, lab in (("Q3|rec_had0", "recruits: haddock age 0 Q3 (< 20 cm)"),
                     ("Q3|rec_cod0", "recruits: cod age 0 Q3 (< 20 cm)"),
                     ("Q1|rec_cod1", "recruits: cod age 1 Q1 (10-27 cm)")):
        show(f"h|{reg}|q|{key}", lab)
    show(f"h|{reg}|q|rec_import", "import-tonne weighted (had0 0.39 + cod1 0.61)")

print("\nJ/B (box mean, member-weighted) and J/B per class; library box J/B 0.15 (Q1) / 0.16 (Q3)")
for reg in ("box", "wide"):
    for q in ("Q1", "Q3"):
        for tag in ("q", "raw"):
            k = f"h|{reg}|{tag}|{q}|all"
            lo, hi = ci(k + "|JB")
            print(f"  {reg:4s} {q} {tag:3s}: J/B {central[k + '|JB'][0]:.3f} [{lo[0]:.3f}-{hi[0]:.3f}]; by class "
                  f"{fmt(central[k + '|JBk'])}")

print("\nSensitivities of the J shape (all members), f per class:")
for q in ("Q1", "Q3", "ann"):
    print(f" {q}")
    for key, lab in ((f"h|box|q|{q}|all|J", "central (box, q, per hour, window ratio >= 250)"),
                     (f"h|box|raw|{q}|all|J", "raw (no Walker q)"),
                     (f"sa|box|q|{q}|all|J", "per swept km2"),
                     (f"h|box_nb|q|{q}|all|J", ">= 250 m: 150-250 copy (Q1), own 3 hauls (Q3)"),
                     (f"h|box|q|{q}|flatJ|J", "flatfish J cut 12 cm (dab, LRD, lemon sole)"),
                     (f"h|box|q|{q}|gad|J", "haddock + cod only"),
                     (f"h|wide|q|{q}|all|J", "window 57.3-59.2 N, 8-12 E")):
        show(key, lab)

print("\nDay-only check (Q3 box, J all members, per hour, q): ")
D = META.DayNight.values == "D"
cmD = class_means(X[D], META[D], {y: 1.0 for y in years})
gJ = group(cmD[("box", 3)], "J", "q", ALL)
print(f"  day hauls only f {fmt(shape(gJ), 2)} (central {fmt(central['h|box|q|Q3|all|J'], 2)})")

print("\nThe box's own >= 250 m hauls (Q3, 3 hauls) against the imputed class (J and A, all members):")
yw = {y: 1.0 for y in years}
r3 = box_raw3(X, META, yw, 3)
cmq = CM["h"][("box", 3)]
for st in ("J", "A"):
    g = group(cmq, st, "q", ALL)
    alt = cmq.copy()
    alt[3] = r3
    g2 = group(alt, st, "q", ALL)
    print(f"  {st}: imputed {fmt(shape(g), 2)}; with the 3 hauls {fmt(shape(g2), 2)}")

print("\nCo-occurrence with the predators' field spread (F = juvenile density met by the mean predator tonne /"
      " box mean J density):")
for q in ("Q1", "Q3", "ann"):
    for nm, stg in (("all", "J"), ("had", "J"), ("all", "Js")):
        k = f"h|box|q|{q}|{nm}|{stg}"
        fs = [float((SAITHE * bb[k]).sum()) for bb in boot]
        fw = [float((WHITING * bb[k]).sum()) for bb in boot]
        print(f"  {q} {nm:3s} {stg:2s}: F saithe-WC {float((SAITHE * central[k]).sum()):.3f} "
              f"[{np.percentile(fs, 2.5):.3f}-{np.percentile(fs, 97.5):.3f}]  F whiting "
              f"{float((WHITING * central[k]).sum()):.3f} [{np.percentile(fw, 2.5):.3f}-{np.percentile(fw, 97.5):.3f}]")
for nm in ("A", "B"):
    k = f"h|box|q|ann|all|{nm}"
    print(f"  ann {nm} (all members): F saithe-WC {float((SAITHE * central[k]).sum()):.3f}, whiting "
          f"{float((WHITING * central[k]).sum()):.3f}")


def member_j_shares(cmset, reg, tag, jkey):
    """Each member's share of the group's J biomass: m_s (J/B)_s / sum (cell-weighted, quarter)."""
    out = {}
    for q in (1, 3):
        cm = cmset[(reg, q)]
        v = {}
        for s in ALL:
            dB = (CELL * cm[:, IDX[f"{s}|B|{tag}"]]).sum()
            dJ = (CELL * cm[:, IDX[f"{s}|{jkey}|{tag}"]]).sum()
            v[s] = MEMBER[s] * dJ / dB if dB > 0 else 0.0
        tot = sum(v.values())
        out[q] = {s: x / tot for s, x in v.items() if x > 0}
    return out


print("\nMember shares of J biomass (member weight x the species' own J/B, box, Walker q, per hour); central and "
      "95 % year-bootstrap range:")
for jkey, lab in (("J", "library J definition (cuts: haddock, cod, plaice, witch)"),
                  ("Jf", "plus dab, long rough dab, lemon sole J < 12 cm [J proxy]")):
    cen = member_j_shares(CM["h"], "box", "q", jkey)
    bs = []
    for b in range(200):
        draw = rng.choice(years, size=len(years), replace=True)
        bs.append(member_j_shares({"x": None, **class_means(X, META, pd.Series(draw).value_counts().to_dict())},
                                  "box", "q", jkey))
    print(f"  {lab}")
    for q in (1, 3):
        parts = []
        for s, v in sorted(cen[q].items(), key=lambda kv: -kv[1]):
            arr = [bb[q].get(s, 0.0) for bb in bs]
            parts.append(f"{s} {v:.3f} [{np.percentile(arr, 2.5):.3f}-{np.percentile(arr, 97.5):.3f}]")
        print(f"    Q{q}: " + ", ".join(parts))
    ann = {s: 0.5 * (cen[1].get(s, 0.0) + cen[3].get(s, 0.0)) for s in set(cen[1]) | set(cen[3])}
    print("    annual (Q1/Q3 mean): " + ", ".join(f"{s} {v:.3f}" for s, v in sorted(ann.items(), key=lambda kv: -kv[1])))

# save central + CI for the gate and the proposal
outj = {k: {"central": central[k].tolist(), "lo": ci(k)[0].tolist(), "hi": ci(k)[1].tolist()}
        for k in central if k.startswith(("h|", "sa|"))}
outj["_cell_shares"] = CELL.tolist()
json.dump(outj, open(os.path.join(OUT, "gov_juv_bed.json"), "w"), indent=0)
rows = []
for k, v in outj.items():
    if k.startswith("_") or len(v["central"]) != 4:
        continue
    for c in range(4):
        rows.append(dict(key=k, bed_class=CLS[c], f=v["central"][c], lo=v["lo"][c], hi=v["hi"][c],
                         share=v["central"][c] * CELL[c]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "gov_juv_bed_shapes.csv"), index=False)
print(f"\nwrote {OUT}/gov_juv_bed.json and gov_juv_bed_shapes.csv")
