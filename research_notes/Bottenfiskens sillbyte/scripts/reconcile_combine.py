"""Reconcile herring's daytime near-bed (0-10 m) share: HERAS profile vs GOV/acoustic ratio on one footing.

Usage: python3 -I reconcile_combine.py <tables_dir> <depth_npz>
  reads reconcile_heras_by_class.csv, reconcile_gov_hauls.csv, reconcile_gov_by_class.csv (written by
  reconcile_heras_profile.py and reconcile_gov_nearbed.py) and maps/depth.npz.

Footing [D]: the model grid's cells, bed-depth classes <80 / 80-110 / 110-130 / 130-150 / 150-200 /
200-300 / >=300 m weighted by their cell shares in maps/depth.npz; summer (HERAS late June - mid July;
GOV Q3 = Aug-Sep, the nearest quarter); day (sun > 6 deg; GOV day hauls).

Column herring per class (July, t/km2) = fish NASC per grid log in the class (2010-26) x C, with C set
so that the log-mean equals the grid July herring 4.3 t/km2 (../Ekolodsdata planktivorer, grid
correction) [F/D].

Two readings of the same data (they differ only in where the GOV's surplus over the acoustic
near-bed layer sits):
  H2 "acoustic profile is real": band share = HERAS 0-10 m share (+ the blind 0-2 m filled at the
     density of the 2-4 m bins); >=150 m beds from GOV Q3 / column divided by q_eff, the GOV-to-band
     ratio calibrated on the four classes where both see the bed. GOV's surplus over the acoustic
     0-4 m layer is then fish brought into the net from above (vessel/trawl diving, sweep herding).
  H1 "dead-zone layer": GOV catches the undisturbed 0-4 m layer with in-net efficiency q_in (1;
     0.7 as a variant) and its surplus over the acoustic 0-4 m layer lies in the 0-2 m dead zone,
     invisible to HERAS: band = A10 + D, column = A + D, D = max(0, G/q_in - A4).
"""
import csv
import math
import os
import sys

import numpy as np

tdir, depth_npz = sys.argv[1:3]
LAB = ["<80", "80-110", "110-130", "130-150", "150-200", "200-300", ">=300"]
EDGES = [0, 80, 110, 130, 150, 200, 300, 1e4]
SEEN = LAB[:4]
JULY_GRID_HERRING = 4.3      # t/km2, grid July 2010-26 [F/D]
ANNUAL_HERRING = 2.60        # t/km2 (July x 0.6) [F/D]
NMI2_KM2 = 1.852 ** 2

z = np.load(depth_npz)["depth"].astype(float).ravel()
area = {LAB[i]: float(np.mean((z >= EDGES[i]) & (z < EDGES[i + 1]))) for i in range(len(LAB))}

H = list(csv.DictReader(open(os.path.join(tdir, "reconcile_heras_by_class.csv"))))
GH = list(csv.DictReader(open(os.path.join(tdir, "reconcile_gov_hauls.csv"))))
GC = list(csv.DictReader(open(os.path.join(tdir, "reconcile_gov_by_class.csv"))))
f = lambda x: float(x) if x not in ("", None) else float("nan")

allc = {r["cls"]: r for r in H if r["region"] == "grid_allcruises"}
logs = {k: f(allc[k]["logs"]) for k in LAB}
nasc = {k: f(allc[k]["nasc_per_log"]) for k in LAB}
C = JULY_GRID_HERRING * sum(logs.values()) / sum(logs[k] * nasc[k] for k in LAB)
col = {k: nasc[k] * C for k in LAB}
strat_col = sum(area[k] * col[k] for k in LAB)
deep = {r["cls"]: r for r in H if r["region"] == "grid" and r["sun"] == "day>6"}
cru = [r for r in H if r["region"] == "grid_cruise"]
cruises = sorted(set(r["cruise"] for r in cru))
t_per_nasc = f(next(r for r in H if r["region"] == "conversion")["t_per_nasc"])
wsig_dana = t_per_nasc * 4 * math.pi * NMI2_KM2 * 1000      # kg per m2 sigma_bs, Dana herring lengths

print("=== Footing ===")
print("class    cell share | column herring July t/km2 (NASC/log) | HERAS 1-m logs seen | GOV Q3 grid day hauls")
gq3 = {k: [r for r in GH if r["grid"] == "True" and r["Quarter"] == "3" and r["k"] == k] for k in LAB}
gq1 = {k: [r for r in GH if r["grid"] == "True" and r["Quarter"] == "1" and r["k"] == k] for k in LAB}
for k in LAB:
    nseen = int(f(deep[k]["seen"])) if k in deep else 0
    print(f"{k:8s} {area[k]:6.3f} | {col[k]:5.2f} ({nasc[k]:4.0f}) | {nseen:4d} | {len(gq3[k]):3d}")
print(f"C = {C:.4f} t/km2 per NASC unit (fish NASC -> herring, calibrated to {JULY_GRID_HERRING}); "
      f"depth-stratified grid column {strat_col:.2f} t/km2 (log mean {JULY_GRID_HERRING})")

# ---------------------------------------------------------------- class estimates (point values)
lqf = {r["cls"]: f(r["her_lq_factor"]) for r in GC if r["footing"] == "grid" and r["quarter"] == "3"}
ben = {}
for k in LAB:
    b3 = [f(r["ben"]) for r in gq3[k]]
    b1 = [f(r["ben"]) for r in gq1[k]]
    vals = [np.mean(v) for v in (b3, b1) if len(v) >= 3]
    ben[k] = float(np.mean(vals)) if vals else float("nan")
ben[">=300"] = ben["200-300"]          # no hauls >= 300 m: borrow [J]
wht = {}
for k in LAB:
    vals = [np.mean([f(r["wht"]) for r in g[k]]) for g in (gq3, gq1) if len(g[k]) >= 3]
    wht[k] = float(np.mean(vals)) if vals else float("nan")
wht[">=300"] = wht["200-300"]


def heras_class(k, sel):
    rr = [r for r in cru if r["cls"] == k and r["cruise"] in sel]
    s = lambda key: sum(f(r[key]) * sel.count(r["cruise"]) for r in rr)
    tot, b10, b4 = s("sum_tot"), s("sum_last10"), s("sum_last4")
    b0, b1, b2, b3 = s("sum_bin0"), s("sum_bin1"), s("sum_bin2"), s("sum_bin3")
    dz = max(0.0, (b2 + b3) - (b0 + b1))     # blind 0-2 m filled at the 2-4 m density
    return tot, b10, b4, dz


def estimate(sel_cruises, gov_q3, q_in=1.0, lam=None, verbose=False):
    """Return per-class F for H2 and H1, and q_eff."""
    s10, s10dz, s4, R = {}, {}, {}, {}
    for k in SEEN:
        tot, b10, b4, dz = heras_class(k, sel_cruises)
        s10[k] = b10 / tot if tot > 0 else 0.0
        s10dz[k] = (b10 + dz) / (tot + dz) if tot > 0 else 0.0
        s4[k] = b4 / tot if tot > 0 else 0.0
        if lam:
            l = lam[k]
            s10dz[k] = s10dz[k] * l / (1 + s10dz[k] * (l - 1))
    for k in LAB:
        g = gov_q3.get(k)
        R[k] = (np.mean(g) * lqf.get(k, 1.0) / col[k]) if g is not None and len(g) else float("nan")
    R[">=300"] = R["200-300"]                       # no Q3 hauls >= 300 m in the grid [J]
    w = {k: area[k] * col[k] for k in SEEN}
    q_eff = sum(w[k] * R[k] for k in SEEN) / sum(w[k] * s10dz[k] for k in SEEN)
    F2 = {k: s10dz[k] for k in SEEN}
    for k in LAB[4:]:
        F2[k] = R[k] / q_eff
    F1 = {}
    for k in LAB:
        G = R[k] * col[k] / q_in
        if k in SEEN:
            A10, A4 = s10[k] * col[k], s4[k] * col[k]
            D = max(0.0, G - A4)
            F1[k] = (A10 + D) / (col[k] + D)
        else:
            F1[k] = G / (col[k] + G)              # band = the GOV 0-4 m layer only (floor) [J]
    return F2, F1, q_eff, s10, s10dz, s4, R


def weighted(F, scheme):
    if scheme == "equal-cell":
        w = {k: area[k] for k in LAB}
    elif scheme == "herring":
        w = {k: area[k] * col[k] for k in LAB}
    elif scheme == "co":
        w = {k: area[k] * col[k] * ben[k] for k in LAB}
    else:
        w = {k: area[k] * col[k] * wht[k] for k in LAB}
    return sum(w[k] * F[k] for k in LAB) / sum(w.values())


gov_pt = {k: [f(r["her_raw"]) for r in gq3[k]] for k in LAB}
F2, F1, q_eff, s10, s10dz, s4, R = estimate(cruises, gov_pt, verbose=True)
F1b = estimate(cruises, gov_pt, q_in=0.7)[1]
# length (TS per kg) correction: band lengths = GOV Q3 class herring, column = Dana trawl herring
lam = {}
for k in SEEN:
    kg = sum(f(r["her_kg"]) for r in gq3[k]); sg = sum(f(r["her_sig"]) for r in gq3[k])
    lam[k] = (kg / sg) / wsig_dana if sg > 0 else 1.0
F2L = estimate(cruises, gov_pt, lam=lam)[0]

print("\n=== Per class (grid, day, summer) ===")
print("class    HERAS s10  +blind 0-2 m  s(0-4 m) | GOV Q3 t/km2 (Lq) | GOV/column R | R/s10dz | "
      "F(H2) F(H1, q_in 1) F(H1, 0.7) | benthivores / whiting t/km2 (GOV Q1+Q3) | lambda (TS/kg)")
for k in LAB:
    g = np.mean(gov_pt[k]) * lqf.get(k, 1.0) if gov_pt[k] else float("nan")
    a = f"{s10[k]:.3f}      {s10dz[k]:.3f}        {s4[k]:.3f}" if k in SEEN else "  -          -             -  "
    rr = f"{R[k] / s10dz[k]:.2f}" if k in SEEN else "  - "
    print(f"{k:8s} {a}   | {g:6.3f} | {R[k]:.3f} | {rr} | {F2[k]:.3f} {F1[k]:.3f} {F1b[k]:.3f} | {ben[k]:.2f} / {wht[k]:.3f} | "
          f"{lam.get(k, float('nan')):.2f}")
print(f"q_eff (GOV Q3 catch / HERAS 0-10 m band biomass, herring-weighted over the 4 seen classes) = {q_eff:.2f}")

print("\n=== Box means of the daytime near-bed share ===")
for lab, F in (("H2: HERAS band, deep beds from GOV/q_eff", F2), ("H2 + TS-per-kg length correction", F2L),
               ("H1: dead-zone layer, q_in 1", F1), ("H1: dead-zone layer, q_in 0.7", F1b),
               ("HERAS as measured (no blind fill), deep from GOV/q_eff", dict(F2, **{k: s10[k] for k in SEEN}))):
    print(f"{lab:56s} equal-cell {weighted(F, 'equal-cell'):.3f}  herring-weighted {weighted(F, 'herring'):.3f}  "
          f"herring x benthivore {weighted(F, 'co'):.3f}  herring x whiting {weighted(F, 'wht'):.3f}")

# sensitivity: class weights from the 2015-20 column only (the years of the vertical shares)
c1520 = {r["cls"]: f(r["nasc_per_log"]) for r in H if r["region"] == "grid_1520"}
if len(c1520) == len(LAB):
    col_save = dict(col)
    C2 = JULY_GRID_HERRING * sum(logs.values()) / sum(logs[k] * c1520[k] for k in LAB)
    col.update({k: c1520[k] * C2 for k in LAB})
    a2s = estimate(cruises, gov_pt)
    print(f"sensitivity, column weights 2015-20 only: H2 herring-weighted {weighted(a2s[0], 'herring'):.3f}, "
          f"herring x benthivore {weighted(a2s[0], 'co'):.3f}; H1 herring-weighted {weighted(a2s[1], 'herring'):.3f}; "
          f"q_eff {a2s[2]:.2f}")
    col.clear(); col.update(col_save)
# sensitivity: GOV hauls from the whole bbox instead of the grid (same classes)
gbox = {k: [f(r["her_raw"]) for r in GH if r["bbox"] == "True" and r["Quarter"] == "3" and r["k"] == k] for k in LAB}
a2b = estimate(cruises, gbox)
print(f"sensitivity, GOV Q3 hauls from the bbox: q_eff {a2b[2]:.2f}; H2 herring-weighted {weighted(a2b[0], 'herring'):.3f}; "
      f"H1 herring-weighted {weighted(a2b[1], 'herring'):.3f}")
# floor from the GOV herring catch if the trawl gathers the whole 0-20 m layer (q_eff 2-3 vs the 0-10 m band)
fl = {k: R[k] / qq for qq in (2.0, 3.0) for k in LAB}
for qq in (2.0, 3.0):
    Fq = {k: R[k] / qq for k in LAB}
    print(f"floor if GOV catch = {qq:.0f} x the 0-10 m band (gathered from ~0-20 m; MIX near-bed gadoid-heavy): "
          f"herring-weighted {weighted(Fq, 'herring'):.3f}, per class " + " ".join(f"{Fq[k]:.3f}" for k in LAB))

# ---------------------------------------------------------------- bootstrap (cruises, GOV hauls)
rng = np.random.default_rng(7)
out = {key: [] for key in ("H2_eq", "H2_her", "H2_co", "H2_wht", "H1_her", "H1_co", "H1_wht", "q_eff", "m1_q3")}
for b in range(4000):
    sel = list(rng.choice(cruises, len(cruises), replace=True))
    gb = {k: (list(np.array(v)[rng.integers(0, len(v), len(v))]) if v else v) for k, v in gov_pt.items()}
    try:
        a2, a1, qe = estimate(sel, gb)[:3]
    except ZeroDivisionError:
        continue
    if not np.isfinite(qe) or qe <= 0:
        continue
    out["H2_eq"].append(weighted(a2, "equal-cell")); out["H2_her"].append(weighted(a2, "herring"))
    out["H2_co"].append(weighted(a2, "co")); out["H1_her"].append(weighted(a1, "herring"))
    out["H1_co"].append(weighted(a1, "co")); out["q_eff"].append(qe)
    out["H2_wht"].append(weighted(a2, "wht")); out["H1_wht"].append(weighted(a1, "wht"))
    out["m1_q3"].append((sum(area[k] * np.mean(gb[k]) for k in LAB[:6]) + area[">=300"] * np.mean(gb["200-300"]))
                        / JULY_GRID_HERRING)
print("\nbootstrap (6 HERAS cruises x GOV Q3 hauls per class, 4000 draws): median (10-90 %)")
for key, v in out.items():
    v = np.array(v)
    print(f"  {key:7s} {np.median(v):.3f} ({np.percentile(v, 10):.3f}-{np.percentile(v, 90):.3f})")

# ---------------------------------------------------------------- method 1 restated on the grid footing
g3 = sum(area[k] * np.mean(gov_pt[k]) for k in LAB[:6]) + area[">=300"] * np.mean(gov_pt["200-300"])
g1v = {k: [f(r["her_raw"]) for r in gq1[k]] for k in LAB}
g1 = sum(area[k] * np.mean(g1v[k]) for k in LAB if len(g1v[k]) >= 3)
g1 += area["150-200"] * 0.5 * (np.mean(g1v["130-150"]) + np.mean(g1v["200-300"])) + area[">=300"] * np.mean(g1v["200-300"])
print("\n=== Method 1 (GOV / acoustic column) restated on the grid footing ===")
print(f"GOV raw, grid hauls, depth-stratified with the grid's cell shares: Q3 {g3:.2f}, Q1 {g1:.2f}, mean {(g1 + g3) / 2:.2f} t/km2 "
      f"(method 1 used Skagerrak-wide hauls: 0.70 annual, Q3 0.87, Q1 0.53)")
print(f"Q3 / July column {g3 / JULY_GRID_HERRING:.3f}; annual / annual column {(g1 + g3) / 2 / ANNUAL_HERRING:.3f}; "
      f"Q1 / winter column (0.3-0.6 x July, J) {g1 / (0.6 * JULY_GRID_HERRING):.2f}-{g1 / (0.3 * JULY_GRID_HERRING):.2f}")

# ---------------------------------------------------------------- consequences for the a values
W_DAY, W_NIGHT = 0.5 * 0.36, 0.5 * 0.09          # window_by_species.py
S = {"herring": 0.788, "Norway pout": 0.076, "blue whiting": 0.106, "sprat": 0.030}
F_CUR = {"herring": (0.15, 0.02), "Norway pout": (0.7, 0.4), "blue whiting": (0.3, 0.15), "sprat": (0.1, 0.01)}
F_PROP = dict(F_CUR, **{"blue whiting": (0.6, 0.15), "sprat": (0.5, 0.01)})


def pool(Fd_her, others):
    F = dict(others, herring=(Fd_her, 0.02))
    return sum(S[s] * (W_DAY * d + W_NIGHT * n) for s, (d, n) in F.items())


base = pool(0.15, F_CUR)
A_ENC, M_C = 0.019, 0.377          # per capable tonne (section 220); membership share (Bottenfiskens sillbyte 3)
print("\n=== benthivores -> planktivores a (= 0.019 x overlap factor x membership 0.377) ===")
print(f"pool sum at herring 0.15, current others = {base:.4f}")
for Fh in (0.10, 0.12, 0.15, 0.17, 0.20, 0.25, 0.30, 0.34):
    k_cur = pool(Fh, F_CUR) / base; k_prop = pool(Fh, F_PROP) / base
    print(f"herring F_day {Fh:.2f}: overlap x{k_cur:.2f} -> a {A_ENC * M_C * k_cur:.4f} (others current) | "
          f"x{k_prop:.2f} -> a {A_ENC * M_C * k_prop:.4f} (sprat 0.5, blue whiting 0.6 as proposed)")
print("\n=== whiting -> planktivores (step 2 overlap a 0.0116 at herring 0.15; herring 0.59 of the a) ===")
for Fh in (0.10, 0.12, 0.15, 0.17, 0.20, 0.25, 0.34):
    print(f"herring F_day {Fh:.2f}: a ~ {0.0116 * (0.41 + 0.59 * Fh / 0.15):.4f}  (day c per m {Fh / 10:.3f})")
