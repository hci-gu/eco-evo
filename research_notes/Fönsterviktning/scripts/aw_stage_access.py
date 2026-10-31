"""Library-wide pass on the stage_access windows: predator-length weighting by the a ("aw") and the
normalisation over the whole prey group, for every stage_access pair in fgconfig/fg_library.yaml.

Usage: python3 -I aw_stage_access.py <research_notes_dir> <scratch_dir>

Nothing is written outside stdout (<scratch_dir> only receives the empty temporary directory the
encounter script makes and removes). No model output is read.

Imported unchanged (as modules / exec'd heads):
  * research_notes/"Arsungarnas andel" (A-ring) scripts/stage_access.py: prey stage length tables
    (stage_length_biomass.csv), windows (p95 prey/predator length ratios, mammal ranges), predator box
    length mixes (pred_length_biomass.csv), pool and member weights, the 4-quarter mean.
  * research_notes/"Rovfiskens uppdelning" scripts/split_encounter_223.py (which exec's
    research_notes/"Attackrater ur motesteori" scripts/pred_encounter_a.py): the section 220 encounter
    (layer convention, central factors) that produced the a of every water-column, mackerel and
    whiting pair. Used here only for the SHAPE a(L) per tonne of one predator length class.
  * The benthivore fish-pair encounter (research_notes/"Attackrater ur motesteori"
    scripts/benth_encounter.py section 3) is re-stated in two lines (its per-fish volume is
    pi (R_sat L)^2 v_rel(0.4 L, 0.10) p; light and prey-overlap factors do not depend on L).

Steps, applied one at a time so each is attributable:
  lib   the library method (reproduces the current values; check).
  thr   predator lengths the pair's a does not count are dropped from the window average (layer
        convention: a stage with capability score < 0.1 is not in the a). This is the section 232
        whiting fix; it also bites whiting -> benthivores (juvenile whiting benth_fish score 0.05).
  norm  bw+all: species weighted by their STAGE biomass (w x stage share), normalised over ALL the
        prey group's tonnes the engine applies f to: the unkeyed benthivores (long rough dab, dab, lemon
        sole, grey gurnard, starry ray; all adult in the box J/B) and blue whiting (all adult in the
        box J/B) get their own box length distributions instead of being skipped; a species outside a
        member's diet identity counts 0 inside.
  aw    each member's window averaged over its box length mix with weights biomass(L) x m(L) x a(L),
        a(L) = the encounter's a per tonne of predators of length L (central factors). Central.
  range aw with a(L) x L^+0.5 and x L^-0.5 (the exponent uncertainty of a(L); 5-cm classes).
Benthivores -> planktivores additionally: encounter-weighted prey species with the CURRENT overlap
table (research_notes/"Bottenfiskens sillbyte" window_by_species.py ENC; not the revised herring
overlap ENC'), and membership without gape (cod >= 20 cm at 1; haddock >= 20 cm as a member at a
non-gape capability 0.5 (0.3-1) as a sensitivity).
"""
import importlib.util
import math
import os
import sys
from collections import namedtuple

import numpy as np
import pandas as pd

RN = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
AD = os.path.join(RN, "Årsungarnas andel")
FG = os.path.join(RN, "Fiskgrupper i rutan")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------ import stage_access.py (unchanged)
_mk = os.makedirs
os.makedirs = lambda *a, **k: None
argv = sys.argv
p = os.path.join(AD, "scripts", "stage_access.py")
sys.argv = [p, os.path.join(AD, "tables", "stage_length_biomass.csv"), "/nonexistent_not_written"]
spec = importlib.util.spec_from_file_location("stage_access_mod", p)
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)
# ------------------------------------------------------------------ the section 220 encounter (heads only)
p2 = os.path.join(RN, "Rovfiskens uppdelning", "scripts", "split_encounter_223.py")
src = open(p2).read()
src = src[:src.index("ALL = [")]
sys.argv = [p2, os.path.join(RN, "Attackrater ur mötesteori"), os.path.join(FG, "predators"), SCR]
GE = {"__name__": "split_head", "__file__": p2}
exec(compile(src, p2, "exec"), GE)
os.makedirs = _mk
sys.argv = argv
GP = GE["G"]  # pred_encounter_a namespace
GP["SCHOOLS"] = False  # library convention: dispersed herring (220.3 decision 2)
QC = GE["level"](1)
stage_pair, score_pe = GE["stage_pair"], GE["sc"]

SL, PERIOD, POOL_W, BEN_W = sa.SL, sa.PERIOD, sa.POOL_W, sa.BEN_W
LW = pd.read_csv(os.path.join(FG, "tables", "length_weight_all.csv"), index_col=0)
PLB = pd.read_csv(os.path.join(FG, "predators", "tables", "pred_length_biomass.csv"))
APH = {"cod": 126436, "haddock": 126437, "whiting": 126438}
for c in PLB.common.unique():
    APH[c] = int(PLB[PLB.common == c].aphia.iloc[0])

# ------------------------------------------------------------------ prey length distributions
LB = pd.read_csv(os.path.join(FG, "benthivores", "tables", "length_biomass_longrun.csv"))
UNKEYED = {sp: (g.L_cm.values + 0.5, (g.share_2001_25 / g.share_2001_25.sum()).values)
           for sp, g in LB.groupby("species") if sp not in set(SL.species)}
BW = pd.read_csv(os.path.join(HERE, "tables", "blue_whiting_lengths.csv"))
BLUE = {1: (BW.L_cm.values + 0.5, BW.share_q1.values)}
for q in (2, 3, 4):  # Q2/Q4 have no box survey after 1997; blue whiting is a Q3 visitor (pool weight 0 in Q1)
    BLUE[q] = (BW.L_cm.values + 0.5, BW.share_q3.values)


def prey_dist(q, sp, stage):
    """(lengths, weights) of species sp's stage biomass in quarter q, and the stage share of sp's biomass."""
    if sp in UNKEYED or sp == "blue whiting":
        if stage == "J":
            return None, 0.0
        L, w = BLUE[q] if sp == "blue whiting" else UNKEYED[sp]
        return (L, w), 1.0
    d = SL[(SL.period == PERIOD[q]) & (SL.quarter == q) & (SL.species == sp)]
    J, A = d.J.sum(), d.A.sum()
    if J + A <= 0:
        return None, 0.0
    v = d[stage].values
    share = (J if stage == "J" else A) / (J + A)
    if v.sum() <= 0:
        return None, share
    return (d.L_cm.values + 0.5, v / v.sum()), share


GROUP_W = {"plk": lambda q: POOL_W[q], "ben": lambda q: BEN_W, "wht": lambda q: {"whiting": 1.0}}

# ------------------------------------------------------------------ predator a(L) per tonne (shape)
Row = namedtuple("Row", "sp st W Nt L")
CUT = {"whiting": 20, "saithe": 60, "hake": 20}
ST_NAMES = {"whiting": ("juvenile", "adult"), "saithe": ("adult", "large"), "hake": ("juvenile", "adult")}
PT = {"plk": "plank_fish", "ben": "benth_fish", "wht": "whiting_prey"}


def w_g(sp, L):
    a, b = LW.loc[APH[sp], "a"], LW.loc[APH[sp], "b"]
    return a * L ** b


def stage_of(sp, L5):
    if sp in CUT:
        return ST_NAMES[sp][0] if L5 < CUT[sp] else ST_NAMES[sp][1]
    return "all"


def a_pe(sp, L5, Lmid, group, prey_only=None):
    """encounter a (km2/t/tick, annual mean of day and night) per tonne of length class Lmid (cm) of a
    pred_encounter_a member on the pair, and the layer membership m (score >= 0.1)."""
    st = stage_of(sp, L5)
    r = Row(sp, st, 1.0, 1e6 / w_g(sp, Lmid), Lmid / 100.0)
    m = 1.0 if score_pe(r, PT[group]) >= 0.1 else 0.0
    saved = dict(GP["PLANK_SHARE"])
    if prey_only is not None:
        for k in GP["PLANK_SHARE"]:
            GP["PLANK_SHARE"][k] = 1.0 if k == prey_only else 0.0
    a = 0.5 * (stage_pair(r, PT[group], QC, False) + stage_pair(r, PT[group], QC, True))
    GP["PLANK_SHARE"].clear()
    GP["PLANK_SHARE"].update(saved)
    return a, m


def vrel(v, u):
    return (u * u + 3 * v * v) / (3 * v) if v >= u else (v * v + 3 * u * u) / (3 * u)


def a_benth(sp, Lmid):
    """benthivore fish-pair encounter per tonne of length class Lmid (cm): N/t x pi (2 L)^2 v_rel(0.4 L, 0.1)
    (benth_encounter.py section 3 central; light, capture, band and prey-overlap factors are L-free)."""
    L = Lmid / 100.0
    return (1e6 / w_g(sp, Lmid)) * math.pi * (2.0 * L) ** 2 * vrel(0.4 * L, 0.10)


def plen_pred(sp, col="pooled"):
    t = PLB[PLB.common == sp]
    return t.L5.values, t.L5.values + 2.5, (t[col] / t[col].sum()).values


def plen_benth(sp, Lmin=20):
    """cod / haddock box length mix (stage table 2001-25, all quarters, >= Lmin cm), as stage_access.py."""
    t = SL[(SL.species == sp) & (SL.period == "2001-25")].groupby("L_cm")[["J", "A"]].sum().sum(axis=1)
    t = t[t.index >= Lmin]
    return t.index.values, t.index.values + 0.5, (t / t.sum()).values


# ------------------------------------------------------------------ member predator-length weights
def lweights(member, group, mode, kexp=0.0, prey_only=None, score_tab=None):
    """(lengths cm, weights) over the member's box length mix. mode: 'bio' (library), 'thr' (biomass x m),
    'aw' (biomass x m x a(L)); kexp multiplies a(L) by L^kexp (range)."""
    if member in ("cod", "haddock"):
        L5, Lm, b = plen_benth(member)
        a = np.array([a_benth(member, x) for x in Lm])
        m = np.ones_like(b)
        if score_tab is not None:  # the library a's phi stage scores (sensitivity)
            m = np.array([score_tab(member, x) for x in Lm])
    else:
        col = "q3" if member in ("mackerel", "horse mackerel") else "pooled"
        L5, Lm, b = plen_pred(member, col)
        am = [a_pe(member, l5, lm, group, prey_only) for l5, lm in zip(L5, Lm)]
        a = np.array([x[0] for x in am])
        m = np.array([x[1] for x in am])
    if mode == "bio":
        w = b.copy()
    elif mode == "thr":
        w = b * m
    elif mode == "pow":  # power-law a(L) ~ L^kexp (the audit's approximation), with the layer membership
        w = b * m * Lm ** kexp
    else:
        w = b * m * a * Lm ** kexp
    return Lm, w / w.sum()


# ------------------------------------------------------------------ windows
def inside_share(dist, window, lw):
    L, v = dist
    if window[0] == "ratio":
        Ls, ws = lw
        return float(sum(wp * v[L <= window[1] * Lp].sum() for Lp, wp in zip(Ls, ws)))
    return float(v[(L >= window[1]) & (L <= window[2])].sum())


ZERO_UNKEYED = False  # sensitivity: unkeyed benthivores in the denominator only (Vattenpelarens audit's bw+all)


def member_f(q, group, window, ident, stage, lw, variant, enc=None):
    """variant 'lib': stage_access.py (group weights, keyed species only, identity-normalised);
    'norm': stage-biomass weights over ALL species of the group (identity outside -> 0 inside)."""
    num = den = 0.0
    for sp, w in GROUP_W[group](q).items():
        if w <= 0:
            continue
        dist, share = prey_dist(q, sp, stage)
        if variant == "lib":
            if (ident is not None and sp not in ident) or sp in UNKEYED or sp == "blue whiting" or dist is None:
                continue
            k = w * (enc.get(sp, 1.0) if enc else 1.0)
            num += k * inside_share(dist, window, lw)
            den += k
            continue
        k = w * share * (enc.get(sp, 1.0) if enc else 1.0)
        if k <= 0:
            continue
        den += k
        if ZERO_UNKEYED and sp in UNKEYED:
            continue
        if dist is not None and (ident is None or sp in ident):
            num += k * inside_share(dist, window, lw)
    return num / den if den > 0 else np.nan


POW_K = {"anglerfish": -3.0}


def pair_access(group, members, variant, mode, kexp=0.0, enc=None, enc_lw=False, score_tab=None):
    """members: [(member, share of the pair's a, identity set or None)]; returns (f_J, f_A, by quarter)."""
    out = {}
    for stage in ("J", "A"):
        fq = []
        for q in (1, 2, 3, 4):
            f = 0.0
            for m, s, ident in members:
                window = sa.WINDOWS[m]
                if window[0] == "ratio":
                    k = POW_K.get(m, -2.0) if mode == "pow" else kexp
                    lw = lweights(m, group, mode, k, score_tab=score_tab)
                else:
                    lw = None
                f += s * member_f(q, group, window, ident, stage, lw, variant, enc)
            fq.append(f)
        out[stage] = (float(np.mean(fq)), fq)
    return out["J"][0], out["A"][0], out


# ------------------------------------------------------------------ pairs
GAD, FLATH = sa.GADOIDS, sa.FLAT | {"haddock"}
WC_BEN = [("saithe", 0.77, GAD), ("spurdog", 0.10, FLATH), ("hake", 0.07, None), ("velvet belly", 0.04, None),
          ("anglerfish", 0.01, None)]
WC_WHT = [("saithe", 0.77, None), ("spurdog", 0.10, None), ("hake", 0.07, None), ("velvet belly", 0.04, None),
          ("anglerfish", 0.01, None)]
_rp_old = [("saithe", 0.33), ("hake", 0.03), ("spurdog", 0.02), ("velvet belly", 0.01)]
_t = sum(w for _, w in _rp_old)
WC_PLK_OLD = [(m, w / _t, None) for m, w in _rp_old]
# member contributions to the CURRENT rest -> planktivores a 0.00559 (split_encounter_mack_field_stdout.txt)
_rp = [("saithe", 0.84), ("hake", 0.06), ("spurdog", 0.06), ("velvet belly", 0.03)]
_t = sum(w for _, w in _rp)
WC_PLK = [(m, w / _t, None) for m, w in _rp]
MACK = [("mackerel", 0.9642, None), ("horse mackerel", 0.0358, None)]
JB = {"plk": 0.38, "ben": 0.15, "wht": 0.165}
LIB = {"water column -> benthivores": [0.43, 0.027], "water column -> planktivores": [0.69, 0.27],
       "water column -> whiting": [0.57, 0.15], "mackerel -> planktivores": [0.078, 0.0037],
       "whiting -> planktivores": [0.16, 0.010], "whiting -> benthivores": [0.049, 0.0],
       "benthivores -> planktivores": [0.49, 0.14], "benthivores -> whiting": [0.38, 0.042],
       "seals -> benthivores": [1.0, 0.34], "seals -> whiting": [1.0, 0.84],
       "porpoises -> benthivores": [1.0, 0.55], "porpoises -> whiting": [1.0, 0.95]}

# benthivores -> planktivores: encounter species weights with the CURRENT overlap table (Bottenfiskens
# sillbyte window_by_species.py F_CUR, w_day 0.5 x 0.36, w_night 0.5 x 0.09)
F_CUR = {"herring": (0.15, 0.02), "blue whiting": (0.3, 0.15), "Norway pout": (0.7, 0.4), "sprat": (0.1, 0.01)}
ENC_BEN = {s: 0.5 * 0.36 * d + 0.5 * 0.09 * n for s, (d, n) in F_CUR.items()}


# the library a's phi stage scores (group_params.py SC, central): planktivores cod 0.3/0.7/0.9 (<30, 30-50,
# >=50 cm), haddock 0.1/0.25 (<40, >=40); whiting cod 0/0.15/0.4, haddock 0
def phi_plk(sp, L):
    if sp == "cod":
        return 0.3 if L < 30 else (0.7 if L < 50 else 0.9)
    return 0.1 if L < 40 else 0.25


def phi_wht(sp, L):
    if sp == "cod":
        return 0.0 if L < 30 else (0.15 if L < 50 else 0.4)
    return 0.0


def eff(fj, fa, g):
    return JB[g] * fj + (1 - JB[g]) * fa


rows = []


def report(name, group, steps, central, rng, note=""):
    lib = LIB[name]
    print(f"\n== {name}  (box J/B {JB[group]})   library {lib}, f_eff {eff(*lib, group):.4f}")
    for lab, (fj, fa, det) in steps:
        qj = " ".join(f"{x:.3f}" for x in det["J"][1])
        qa = " ".join(f"{x:.4f}" for x in det["A"][1])
        print(f"  {lab:46s} [{fj:.3f}, {fa:.4f}]  f_eff {eff(fj, fa, group):.4f}   Q f_J {qj} | f_A {qa}")
    fj, fa = central[0], central[1]
    lo, hi = rng
    e0, e1 = eff(*lib, group), eff(fj, fa, group)
    print(f"  CENTRAL [{fj:.3f}, {fa:.4f}]  range f_J {min(lo[0], hi[0]):.3f}-{max(lo[0], hi[0]):.3f}, "
          f"f_A {min(lo[1], hi[1]):.4f}-{max(lo[1], hi[1]):.4f}; f_eff {e0:.4f} -> {e1:.4f} "
          f"(x{e1 / e0:.2f}; range {min(eff(*lo, group), eff(*hi, group)):.4f}-{max(eff(*lo, group), eff(*hi, group)):.4f}) {note}")
    rows.append(dict(pair=name, lib_fJ=lib[0], lib_fA=lib[1], fJ=fj, fA=fa, fJ_lo=min(lo[0], hi[0]),
                     fJ_hi=max(lo[0], hi[0]), fA_lo=min(lo[1], hi[1]), fA_hi=max(lo[1], hi[1]),
                     JB=JB[group], eff_lib=e0, eff_new=e1, ratio=e1 / e0,
                     juv_take_lib=JB[group] * lib[0] / e0, juv_take_new=JB[group] * fj / e1))


def fish_pair(name, group, members, lib_members=None, thr_matters=False):
    lm = lib_members or members
    s_lib = pair_access(group, lm, "lib", "bio")
    steps = [("lib (reproduction)", s_lib)]
    if lib_members is not None:
        steps.append(("lib, current-a member shares", pair_access(group, members, "lib", "bio")))
    s_thr = pair_access(group, members, "lib", "thr")
    if thr_matters:
        steps.append(("+ thr (window over the a's lengths only)", s_thr))
    s_norm = pair_access(group, members, "norm", "thr")
    steps.append(("+ norm (stage biomass, whole group)", s_norm))
    s_aw = pair_access(group, members, "norm", "aw")
    steps.append(("+ aw (biomass x m x a(L))  = central", s_aw))
    lo = pair_access(group, members, "norm", "aw", kexp=-0.5)
    hi = pair_access(group, members, "norm", "aw", kexp=+0.5)
    steps.append(("  range: a(L) x L^-0.5", lo))
    steps.append(("  range: a(L) x L^+0.5", hi))
    # audit approximation a ~ L^-2 (L^-3 anglerfish), for comparison with Vattenpelarens bottenfiskbyte
    steps.append(("  sens.: a(L) ~ L^-2, anglerfish L^-3 (audit)", pair_access(group, members, "norm", "pow")))
    report(name, group, steps, s_aw[:2], (lo[:2], hi[:2]))
    per_member(group, members)
    return s_aw


def per_member(group, members, enc=None):
    print("  per member (share of a): lib [f_J, f_A] | norm, biomass windows | central aw")
    for m, s, ident in members:
        one = [(m, 1.0, ident)]
        a = pair_access(group, one, "lib", "bio", enc=enc)
        b = pair_access(group, one, "norm", "thr", enc=enc)
        c = pair_access(group, one, "norm", "aw", enc=enc)
        print(f"    {m:14s} {s:.3f}: [{a[0]:.3f}, {a[1]:.4f}] | [{b[0]:.3f}, {b[1]:.4f}] | [{c[0]:.3f}, {c[1]:.4f}]")


print("a(L) shape check (encounter, per tonne, relative to the class nearest the number-mean length):")
for sp, g in (("saithe", "ben"), ("saithe", "plk"), ("spurdog", "ben"), ("hake", "ben"), ("anglerfish", "ben"),
              ("whiting", "plk"), ("whiting", "ben"), ("mackerel", "plk")):
    col = "q3" if sp == "mackerel" else "pooled"
    L5, Lm, b = plen_pred(sp, col)
    sel = b > 0.01
    a = np.array([a_pe(sp, l5, lm, g)[0] for l5, lm in zip(L5, Lm)])
    m = np.array([a_pe(sp, l5, lm, g)[1] for l5, lm in zip(L5, Lm)])
    Ls, As = Lm[sel & (m > 0)], a[sel & (m > 0)]
    k = np.polyfit(np.log(Ls), np.log(As), 1)[0] if len(Ls) > 2 else float("nan")
    print(f"  {sp:12s} -> {g}: fitted exponent of a(L) over the bulk of the mix {k:.2f} "
          f"(L {Ls.min():.1f}-{Ls.max():.1f} cm)")
for sp in ("cod", "haddock"):
    L5, Lm, b = plen_benth(sp)
    a = np.array([a_benth(sp, x) for x in Lm])
    sel = b > 0.005
    k = np.polyfit(np.log(Lm[sel]), np.log(a[sel]), 1)[0]
    print(f"  {sp:12s} -> fish (benth_encounter form, R = 2 L): fitted exponent {k:.2f}; LW b {LW.loc[APH[sp], 'b']:.2f}")
r_ad = a_pe("saithe", 40, 43.0, "ben")[0] / a_pe("saithe", 70, 70.0, "ben")[0]
print(f"  saithe benth_fish a(43 cm)/a(70 cm) = {r_ad:.2f} (per-stage table 0.08048/0.03158 = 2.55)")

# ------------------------------------------------------------------ the fish pairs
fish_pair("water column -> benthivores", "ben", WC_BEN)
fish_pair("water column -> planktivores", "plk", WC_PLK, lib_members=WC_PLK_OLD)
fish_pair("water column -> whiting", "wht", WC_WHT)
# mackerel: library = stage_access.py with the field a-shares (Makrillens bytesstorlek, field run)
fish_pair("mackerel -> planktivores", "plk", MACK)
fish_pair("whiting -> planktivores", "plk", [("whiting", 1.0, None)], thr_matters=True)
fish_pair("whiting -> benthivores", "ben", [("whiting", 1.0, None)], thr_matters=True)

# ------------------------------------------------------------------ cod pairs
def benth_pair(name, group, members, enc=None, phi=None, extra=()):
    steps = [("lib (reproduction)", pair_access(group, members, "lib", "bio"))]
    s_norm = pair_access(group, members, "norm", "bio")
    steps.append(("+ norm (stage biomass, whole group)", s_norm))
    if enc is not None:
        s_enc = pair_access(group, members, "norm", "bio", enc=enc)
        steps.append(("+ ENC (encounter species weights, current F)", s_enc))
    s_aw = pair_access(group, members, "norm", "aw", enc=enc)
    steps.append(("+ aw (biomass x a(L), membership 1) = central", s_aw))
    lo = pair_access(group, members, "norm", "aw", kexp=-0.5, enc=enc)
    hi = pair_access(group, members, "norm", "aw", kexp=+0.5, enc=enc)
    steps.append(("  range: a(L) x L^-0.5", lo))
    steps.append(("  range: a(L) x L^+0.5", hi))
    steps.append(("  sens.: a(L) ~ L^-2 (audit's assumption)", pair_access(group, members, "norm", "pow", enc=enc)))
    if phi is not None:
        steps.append(("  sens.: window weighted by the library a's phi", pair_access(group, members, "norm", "aw", enc=enc,
                                                                                score_tab=phi)))
    for lab, s in extra:
        steps.append((lab, s))
    report(name, group, steps, s_aw[:2], (lo[:2], hi[:2]))
    return s_aw


COD = [("cod", 1.0, None)]
sa.WINDOWS["haddock"] = ("ratio", 0.30, "haddock")


def cod_had(c_h, enc):
    """cod and haddock (>= 20 cm) members, shares of the a = biomass(>= 20) x membership x mean a per tonne."""
    sh = {}
    for sp, c in (("cod", 1.0), ("haddock", c_h)):
        L5, Lm, b = plen_benth(sp)
        t = SL[(SL.species == sp) & (SL.period == "2001-25")]
        frac = t[t.L_cm >= 20][["J", "A"]].sum().sum() / t[["J", "A"]].sum().sum()
        abar = float((b * np.array([a_benth(sp, x) for x in Lm])).sum())
        sh[sp] = BEN_W[sp] * frac * c * abar
    tot = sum(sh.values())
    return [(sp, sh[sp] / tot, None) for sp in sh]


had = []
for c_h in (0.3, 0.5, 1.0):
    mem = cod_had(c_h, ENC_BEN)
    had.append((f"  sens.: + haddock member c_h {c_h} (a-share {mem[1][1]:.2f})",
                pair_access("plk", mem, "norm", "aw", enc=ENC_BEN)))
benth_pair("benthivores -> planktivores", "plk", COD, enc=ENC_BEN, phi=phi_plk, extra=had)
benth_pair("benthivores -> whiting", "wht", COD, phi=phi_wht)

# ------------------------------------------------------------------ mammals: no predator-length axis
for name, group, m in (("seals -> benthivores", "ben", "seal"), ("seals -> whiting", "wht", "seal"),
                       ("porpoises -> benthivores", "ben", "porpoise"), ("porpoises -> whiting", "wht", "porpoise")):
    mem = [(m, 1.0, None)]
    s_lib = pair_access(group, mem, "lib", "bio")
    s_norm = pair_access(group, mem, "norm", "bio")
    report(name, group, [("lib (reproduction)", s_lib), ("+ norm (stage biomass, whole group) = central", s_norm)],
           s_norm[:2], (s_norm[:2], s_norm[:2]), note="(aw n/a: fixed prey-length range, a not from predator length)")

# ------------------------------------------------------------------ checks
print("\nChecks:")
ZERO_UNKEYED = True
for name, group, mem, mode in (("water column -> benthivores", "ben", WC_BEN, "aw"),
                               ("whiting -> benthivores", "ben", [("whiting", 1.0, None)], "aw"),
                               ("seals -> benthivores", "ben", [("seal", 1.0, None)], "bio"),
                               ("porpoises -> benthivores", "ben", [("porpoise", 1.0, None)], "bio")):
    s_ = pair_access(group, mem, "norm", mode)
    print(f"  {name}: unkeyed benthivores (long rough dab, dab, lemon sole, grey gurnard, starry ray) counted 0 inside "
          f"(denominator only): [{s_[0]:.3f}, {s_[1]:.4f}], f_eff {eff(s_[0], s_[1], group):.4f}")
ZERO_UNKEYED = False
for name, m in (("seals -> planktivores", "seal"), ("porpoises -> planktivores", "porpoise")):
    s = pair_access("plk", [(m, 1.0, None)], "norm", "bio")
    print(f"  {name} (no stage_access in the library): window share incl. blue whiting [{s[0]:.3f}, {s[1]:.3f}]")
R = pd.DataFrame(rows)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print("\nSummary:")
    print(R.round(4).to_string(index=False))
