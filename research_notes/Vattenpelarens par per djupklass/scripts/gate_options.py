"""Water-column diet gate per bed class, the factor decomposition of the krill : fish ratio, and the options.

Usage: python3 -I gate_options.py <repo_root> <wc_terms.json> <out_json>

Nothing tuned, no model output read. Gate = encounter shares a_j(c) B_vis,j(c) (concurrent search, fixed
attack), as tools/probes/budget_gate.py, evaluated on the 60 x 60 depth map (maps/depth.npz):
  equal cells       sum_c w_c a_j(c) B_j(c), w_c the cell share of each 5-m bed bin;
  saithe spread     the cells weighted by the saithe GOV class shares 0.043 / 0.127 / 0.782 / 0.048
                    (lib/diagnostics/field_diet.py), whiting as prey at its GOV shape, krill at its depth shape;
  per class         one bed class alone (cells of that class, predator uniform inside it).
Booked shares (budget_gate, today's library): krill 0.361, planktivores 0.498, benthivores 0.077, whiting 0.042,
mackerel 0.011, zooplankton 0.010.

Variants of the a's (every one from an existing derivation or a cited field value; none chosen to hit a share):
  V_lib    library: krill class table (262), fish a's constant over cells (in-layer, 220/243/243.3).
  V_p      field capture p (Juanes et al. 2002 review: piscivore attacks ~0.5 successful; fish 0.45 (0.25-0.6)
           [F review]; krill 0.6 (0.4-0.8) [J], as 252.6 A2): krill a x 1.2, every fish a x 2.25.
  V_one    one convention (band overlap, one vision function: R^2 = R_nv^2 (1-g) + R_sat^2 g, R_sat =
           min(R0(L_prey), 2 L_pred)) under the measured light with the 262 krill layer; research_notes/
           "Vattenpelarens par i uppmatt ljus" (a-umlaut) meas_pairs.py, exec'd, layer patched. Library p.
  V_one_p  V_one with the field p.
  V_fv_lo/hi the fish pairs' vision convention on krill (krill R 0.1 / 0.17 m), fish as the library.
  V_dark   V_one with the fish prey's dark floor R_nv kept for fish at every depth (no light gain on fish:
           the 220 convention for fish, the band O kept) - isolates the overlap part of V_one.
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
TERMS = json.load(open(sys.argv[2]))
OUTJ = sys.argv[3]
RN = os.path.join(ROOT, "research_notes")
ARGV0 = list(sys.argv)

# ------------------------------------------------------------------ meas_pairs head (unchanged), 262 layer
MP = os.path.join(RN, "Vattenpelarens par i uppmätt ljus", "scripts", "meas_pairs.py")
src = open(MP).read()
head = src[:src.index("\n# ================================================================ central\nqc = level(1)")]
sys.argv = [MP, ROOT, os.devnull, "10"]
M = {"__name__": "meas_pairs_head", "__file__": MP}
exec(compile(head, MP, "exec"), M)
sys.argv = ARGV0
G = M["G"]
ANCHOR_B = [150.0, 160.0, 170.0, 250.0, 350.0]
ANCHOR_Z = [95.0, 140.0, 140.0, 120.0, 145.0]


def new_layer(bed, q):
    old_zc = q["zc"] + q["zc_slope"] * np.clip(bed - 150.0, 0, 200) / 100.0
    shift = float(np.interp(bed, ANCHOR_B, ANCHOR_Z)) - (95.0 + 25.0 * np.clip(bed - 150.0, 0, 200) / 100.0)
    zc = old_zc + shift
    H = q["Hk"] * (1.0 + (2.0 / 3.0) * np.clip((bed - 150.0) / 100.0, 0, 1))
    bot = min(zc + H / 2, bed - 1.0)
    top = max(bot - H, q["ztop_min"])
    if bot - top < 5.0:
        top = bot - 5.0
    return max(top, 0.0), bot


G["krill_day_layer"] = new_layer
BEDS, WCELL, CLS_IDX, ELEV, MONTH = M["BEDS"], M["WCELL"], M["CLS_IDX"], M["ELEV"], M["MONTH"]
WPRES = np.full(12, 0.43)
WPRES[[4, 5, 6, 7]] = 0.05
WT = WPRES[MONTH]


def cls_a(a_t_b, krill):
    out = []
    for k, idx in enumerate(CLS_IDX):
        s = a_t_b[:, idx] @ WCELL[idx] / WCELL[idx].sum()
        out.append(float((s * WT).sum() / WT.sum()) if (k == 0 and krill) else float(s.mean()))
    return np.array(out)


qc = M["level"](1)
fa, _ = M["prey_a"](qc, "meas")
pa = M["pairs_from"](fa)
ONE = {pair: cls_a(a, pair == "krill") for pair, a in pa.items()}
# V_dark: the same band O, but R = R_nv (dark floor) on the fish prey at every depth (light gain only on krill)
qd = dict(qc)
fa_d, _ = M["prey_a"](dict(qc, Ke_gad=1e12), "meas")      # g -> 0 for every prey: R^2 = R_nv^2
pa_d = M["pairs_from"](fa_d)
DARK = {pair: cls_a(a, pair == "krill") for pair, a in pa_d.items()}
KRILL_DARK = DARK["krill"].copy()          # krill at its non-visual floor R_nv 0.1 m everywhere
DARK["krill"] = ONE["krill"]

# ------------------------------------------------------------------ cells, shapes, densities
dep = np.load(os.path.join(ROOT, "maps", "depth.npz"))["depth"].ravel()
cls = np.digitize(dep, [100.0, 150.0, 250.0])
ncell = np.array([(cls == k).mean() for k in range(4)])
tab_d = [49.9, 50.0, 99.9, 100.0, 150.0, 250.0, 530.0]
tab_w = [0.025, 0.38, 0.38, 0.8, 1.0, 1.2, 1.4]
shape = np.interp(dep, tab_d, tab_w)
shape = shape / shape.mean()
B_KRILL = np.array([5.0 * shape[cls == k].mean() for k in range(4)])
SAI = np.array([0.043, 0.127, 0.782, 0.048])
WHI = np.array([0.769, 0.204, 0.027, 0.001])
BVIS = {"planktivores": 1.776, "benthivores": 0.254, "whiting": 0.060, "mackerel": 0.119, "zooplankton": 9.722}
BOOK = {"krill": 0.361, "planktivores": 0.498, "benthivores": 0.077, "whiting": 0.042, "mackerel": 0.011,
        "zooplankton": 0.010}
PREYS = list(BOOK)
LIB = {"krill": np.array([0.068, 0.031, 0.0058, 0.00014]), "planktivores": 0.0060, "benthivores": 0.0055,
       "whiting": 0.013, "mackerel": 0.00088, "zooplankton": 3.13e-5}


def as_cls(x):
    return np.asarray(x, float) * np.ones(4) if np.ndim(x) == 0 else np.asarray(x, float)


def B_cls(prey, spread):
    if prey == "krill":
        return B_KRILL
    b = BVIS[prey] * np.ones(4)
    if prey == "whiting" and spread:
        b = BVIS[prey] * WHI / ncell
    return b


def gate(A):
    """encounter shares at equal cells, the saithe spread, and per class."""
    out = {}
    for name, w, spread in (("equal", ncell, False), ("saithe", SAI, True)):
        e = {p: float((w * as_cls(A[p]) * B_cls(p, spread)).sum()) for p in PREYS}
        t = sum(e.values())
        out[name] = {p: e[p] / t for p in PREYS}
    for k in range(4):
        e = {p: float(as_cls(A[p])[k] * B_cls(p, False)[k]) for p in PREYS}
        t = sum(e.values())
        out[f"class{k}"] = {p: e[p] / t for p in PREYS}
    return out


def odds_factor(sh):
    """the factor on the krill a (equivalently 1/x on every fish a) that brings krill to its booked share."""
    k = sh["krill"]
    return (k / (1 - k)) / (BOOK["krill"] / (1 - BOOK["krill"]))


PMAP = {"krill": "krill", "planktivores": "planktivorous_fish", "benthivores": "benthivorous_fish",
        "whiting": "whiting", "mackerel": "mackerel", "zooplankton": "zooplankton"}
VAR = {"V_lib": LIB}
VAR["V_p"] = {p: as_cls(v) * (1.2 if p == "krill" else 1.0 if p == "zooplankton" else 2.25) for p, v in LIB.items()}
VAR["V_one"] = {p: ONE[PMAP[p]] for p in PREYS}
VAR["V_one_p"] = {p: ONE[PMAP[p]] * (1.2 if p == "krill" else 1.0 if p == "zooplankton" else 2.25) for p in PREYS}
VAR["V_dark"] = {p: DARK[PMAP[p]] for p in PREYS}
# V_fv: the fish pairs' vision convention applied to krill (the reverse of V_one). Section 220 fish R: lake-trout
# RD(I) = 0.2549 I^0.4747 (cap 1.012 m) on fish prey (pred_encounter_a.py), set to 0.2 m (dark) / 0.5 m (lit herring).
# Scaled to a 25 mm krill by prey length (R ~ sqrt(prey area), the krill pair's own k_v sqrt(A) rule) from a 15 cm
# fish prey [J]: R <= 1.012 x 25/150 = 0.17 m, i.e. at or near the krill floor R_nv 0.1 m. Bounds: krill R 0.1 m
# everywhere (lo) and 0.17 m everywhere (hi = lo x (0.17/0.1)^2); fish pairs as in the library.
VAR["V_fv_lo"] = dict(LIB, krill=KRILL_DARK)
VAR["V_fv_hi"] = dict(LIB, krill=KRILL_DARK * (1.012 * 25.0 / 150.0 / 0.1) ** 2)
# V_one keeps the library krill table exactly (the generic evaluator's krill = the 262 table) - check:
print("cells per class (60 x 60):", np.round(ncell, 3), " krill B by class:", np.round(B_KRILL, 2))
print("generic evaluator krill classes (V_one):", np.round(ONE["krill"], 5), " library", LIB["krill"])
RES = {}
print("\nGate: krill encounter share (needed factor on the krill a, or 1/x on all fish a's) | planktivores | "
      "benthivores | whiting | mackerel")
for v, A in VAR.items():
    g = gate(A)
    RES[v] = dict(a={p: as_cls(A[p]).tolist() for p in PREYS}, gate=g)
    print(f"  {v}")
    for sp in ("equal", "saithe", "class0", "class1", "class2", "class3"):
        s = g[sp]
        print(f"    {sp:7s} krill {s['krill']:.3f} (x{1 / odds_factor(s):.3f}) | plk {s['planktivores']:.3f} "
              f"({s['planktivores'] / BOOK['planktivores']:.2f} of booked) | ben {s['benthivores']:.3f} "
              f"({s['benthivores'] / BOOK['benthivores']:.2f}) | whi {s['whiting']:.3f} ({s['whiting'] / BOOK['whiting']:.2f}) | "
              f"mac {s['mackerel']:.4f} ({s['mackerel'] / BOOK['mackerel']:.2f})")
    print("    a by class: " + "; ".join(f"{p} " + "/".join(f"{x:.4g}" for x in as_cls(A[p])) for p in PREYS[:4]))

# ------------------------------------------------------------------ factor decomposition, library (saithe < 60 cm)
km = TERMS["krill_members"]["saithe < 60 cm"]
ft = TERMS["fish_terms_saithe"]
qf = TERMS["qf"]
S = {"herring": 0.554 + 0.021, "pout": 0.350, "bw": 0.075}
vk = km["S1d"] / TERMS["saithe_adult"]["Nt"]                 # krill v_rel by day (m/s)
hd, hn = ft["herring+sprat day"][0], ft["herring+sprat night"][0]
po_d, po_n = ft["pout (day and night)"]
bw_d, bw_n = ft["blue whiting (day and night)"]
terms = [(S["herring"], hd, 0.5), (S["herring"], hn, 0.5), (S["pout"], po_d, 0.5), (S["pout"], po_n, 0.5),
         (S["bw"], bw_d, 0.5), (S["bw"], bw_n, 0.5)]
O_p = sum(s * f * t["O"] * t["vrel"] / vk for s, t, f in terms)
OR_p = sum(s * f * t["O"] * t["vrel"] / vk * t["Reff2"] for s, t, f in terms)
R2_p = OR_p / O_p
p_k, p_f = TERMS["qk"]["p_cap"], qf["p_fish"]
print("\nFactor decomposition, saithe < 60 cm, library derivations: krill a / planktivore a per t/km2")
print(f"  planktivore pool (in-layer): O_eff {1000 * O_p:.2f} e-3/m (speed-weighted), R_eff {math.sqrt(R2_p):.3f} m, p {p_f}")
DEC = []
grp_ratio = np.array(LIB["krill"]) / LIB["planktivores"]
for k, c in enumerate(km["classes"]):
    fp = p_k / p_f
    fo = c["O"] / O_p
    fr = c["R2"] / R2_p
    sai = fp * fo * fr
    mix = grp_ratio[k] / sai
    DEC.append(dict(cls=k, p=fp, O=fo, R2=fr, saithe=sai, mix=mix, group=grp_ratio[k], O_k=c["O"], R_k=math.sqrt(c["R2"])))
    print(f"  class {k}: p x{fp:.2f}  O x{fo:.3f} (krill {1000 * c['O']:.2f} e-3/m)  R^2 x{fr:.2f} (krill R_eff "
          f"{math.sqrt(c['R2']):.2f} m)  = saithe x{sai:.3f}; members x{mix:.2f}; group a ratio x{grp_ratio[k]:.3f}")

# required a ratio and the excess, per spread
print("\nRequired krill/planktivore a ratio for the booked diet, and the excess (= needed factor's krill:plk part)")
EXC = {}
lib_g = RES["V_lib"]["gate"]
for sp, w, spread in (("equal", ncell, False), ("saithe", SAI, True)):
    ak_eff = float((w * LIB["krill"] * B_KRILL).sum()) / float((w * LIB["krill"]).sum())   # a-weighted krill B
    a_ratio = float((w * LIB["krill"]).sum()) / LIB["planktivores"]
    b_ratio = ak_eff / BVIS["planktivores"]
    e = lib_g[sp]
    fish_e = 1 - e["krill"]
    plk_of_fish_enc = e["planktivores"] / fish_e
    plk_of_fish_book = BOOK["planktivores"] / (1 - BOOK["krill"])
    need = 1 / odds_factor(e)
    req = (BOOK["krill"] / BOOK["planktivores"]) / b_ratio
    EXC[sp] = dict(a_ratio=a_ratio, B_ratio=b_ratio, krill_B_eff=ak_eff, required=req, excess=a_ratio / req,
                   plk_share_fish_enc=plk_of_fish_enc, plk_share_fish_book=plk_of_fish_book, needed=need)
    print(f"  {sp:7s}: a ratio {a_ratio:.3f} x B ratio {b_ratio:.2f} (krill B seen {ak_eff:.2f} t/km2 vs plk 1.776) -> "
          f"required a ratio {req:.3f}; excess x{a_ratio / req:.2f}; planktivores' share of fish encounter {plk_of_fish_enc:.2f} "
          f"vs booked {plk_of_fish_book:.2f}; needed: krill a x{need:.3f} or every fish a x{1 / need:.2f}")
for k in range(4):
    a_ratio = LIB["krill"][k] / LIB["planktivores"]
    b_ratio = B_KRILL[k] / BVIS["planktivores"]
    req = (BOOK["krill"] / BOOK["planktivores"]) / b_ratio
    e = lib_g[f"class{k}"]
    EXC[f"class{k}"] = dict(a_ratio=a_ratio, B_ratio=b_ratio, required=req, excess=a_ratio / req,
                            needed=1 / odds_factor(e))
    print(f"  class {k}: a ratio {a_ratio:.3f} x B ratio {b_ratio:.2f} -> required {req:.3f}; excess x{a_ratio / req:.2f}; "
          f"needed: krill a x{1 / odds_factor(e):.3f} or every fish a x{odds_factor(e):.2f}")

# share of the krill encounter from each class
print("\nWhere the krill encounter comes from (library): class share of sum_c w_c a_c B_c")
WH = {}
for sp, w in (("equal", ncell), ("saithe", SAI)):
    x = w * LIB["krill"] * B_KRILL
    WH[sp] = (x / x.sum()).tolist()
    print(f"  {sp:7s}: " + " / ".join(f"{v:.3f}" for v in x / x.sum()) + f"  (predator weights {np.round(w, 3)})")
json.dump(dict(variants=RES, decomposition=DEC, excess=EXC, krill_from_class=WH, ncell=ncell.tolist(),
               B_krill=B_KRILL.tolist(), O_pool=O_p, R2_pool=R2_p), open(OUTJ, "w"), indent=1)
