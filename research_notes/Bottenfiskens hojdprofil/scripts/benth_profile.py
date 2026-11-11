"""Benthivores' vertical SEARCH profile f_m(z) per member, and every benthivore pair on the band overlap
(section 242 concurrent search: one f_m per member and period, shared by all of its prey).

Usage: python3 -I benth_profile.py <repo_root> <scratch_dir> <out_json>

Nothing is fitted and no model output is read. No value is chosen against a booked or field share.

Machinery reused unchanged (exec'd heads / runpy):
  research_notes/"En synregel for fiskparen"/scripts/one_rule_pairs.py up to its run block: the krill pairs
      (krill_pairs_wc.py --variant field, the section 253 measured light, the section 262 krill layer and
      presence), rule A (R_sat = min(R0(15 cm fish, |C0| 0.7), 2 L), R_nv fish 0.3 m, Ke 0.02), gmat (light
      factor g(t, z) at every depth), the benthivore length classes and night_day.
  research_notes/"Bottenfiskens bytesbalans"/scripts/pout_vs_bed.py (runpy, quiet): the section 243.3
      planktivore-pair structure (members cod >= 20 cm at 1 and haddock >= 20 cm at c_h 0.65, species
      windows, pool shares) and the benthic per-member weights (a_enc x exposure e: infauna, epifauna).

Theorem (research_notes/"Overlapp i fiskparen" summary.md "Theory"): a_j = sum_m W_m sum_L c_L
  sum_z f_m(z) phi_j(z) R_L^2(E(z, t)), light at z inside the integral. Fish and krill pairs: the ratio of
  this integral with the field profile f_m to the same integral with the library's f (uniform in the 10 m
  bottom band, benth_encounter.py / krill_pairs_wc.py band("H_benth")) - the "profile factor" P. Krill: the
  whole chain (rule A + profile) against the reproduced library krill derivation, as band_pairs.py does.
  Benthic pairs (2-D swept area on the bed, prey phi = the bed): a_j x tau_m, tau_m = the member's share of
  SEARCH time in bed contact (the 0-2 m component; the finest class the tag data resolve).

Member profiles (height h above the bed; three components: bed 0-2 m, near-bed 2-10 m uniform, off-bed
h > 10 m exponential with scale H_off, truncated at the surface), day / night at the 0 deg switch:
  see PROF below; every number carries its source in the comment. "central", "deep" (Barents analogue for
  every gadoid), "bedbound" (Jutland Bank cod, Boudreau 1992 haddock), "rest50" (search != occupancy: half of
  the gadoids' on-bed occupancy counted as rest, Hobson et al. 2007's SB class), flatfish / LRD brackets, and
  a prey-side variant "pout_albert" (Albert 1994: pout within ~1.5 m of the bed by day, 5-10 m at night).
"""
import contextlib
import io
import json
import math
import os
import runpy
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
OUTJ = os.path.abspath(sys.argv[3])
RN = os.path.join(ROOT, "research_notes")
ORP = os.path.join(RN, "En synregel for fiskparen", "scripts", "one_rule_pairs.py")
src = open(ORP).read()
head = src[:src.index("# =============================================================== run")]
ARGV = list(sys.argv)
sys.argv = [ORP, ROOT, SCR, os.path.join(SCR, "_unused.json")]
O = {"__name__": "one_rule_pairs_head", "__file__": ORP}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, ORP, "exec"), O)
PVB = os.path.join(RN, "Bottenfiskens bytesbalans", "scripts", "pout_vs_bed.py")
sys.argv = [PVB, os.path.join(RN, "Attackrater ur mötesteori", "scripts", "benth_encounter.py"),
            os.path.join(RN, "Fiskgrupper i rutan", "benthivores", "tables", "length_biomass_longrun.csv"),
            os.path.join(ROOT, "maps", "depth.npz"), os.path.join(ROOT, "fgconfig", "fg_library.yaml")]
with contextlib.redirect_stdout(io.StringIO()):
    PV = runpy.run_path(PVB, run_name="pout_vs_bed")
sys.argv = ARGV

K = O["K"]
ZG, BEDS, WCELL, CLS_IDX, ELEV, MONTH = O["ZG"], O["BEDS"], O["WCELL"], O["CLS_IDX"], O["ELEV"], O["MONTH"]
IS_NIGHT, night_day, gmat, r0_c, qk0 = O["IS_NIGHT"], O["night_day"], O["gmat"], O["r0_c"], O["qk0"]
RULE, run_krill, KRILL_LIB = O["RULE"], O["run_krill"], O["KRILL_LIB"]
uniform = K["uniform"]
LB = O["LB"]
NB = len(BEDS)
FDAY = float((~IS_NIGHT).mean())
R0_FISH = r0_c(qk0, O["L_FISH_PREY"], O["C_FISH"])
RNV = 0.3


# ------------------------------------------------------------------ profile components
def expo_off(bed, hn, H):
    """density ~ exp(-(h - hn) / H) for heights h >= hn above the bed, z >= 0 (per metre on ZG)."""
    h = bed - ZG
    out = np.where((h >= hn) & (ZG >= 0.0), np.exp(-(h - hn) / max(H, 0.5)), 0.0)
    s = out.sum()
    return out / s if s > 0 else out


def mix(bed, sb, sn, so, H, alpha=1.0, hb=2.0, hn=10.0):
    """bed 0-2 m (weight sb x alpha: alpha < 1 discounts on-bed rest), near-bed 2-10 m, off-bed exp(H)."""
    hn_eff = min(hn, bed - 1.0)
    ub = uniform(bed - hb, bed)
    un = uniform(bed - hn_eff, bed - hb) if hn_eff > hb else ub
    uo = expo_off(bed, hn_eff, H) if so > 0 else np.zeros_like(ZG)
    if uo.sum() <= 0:
        uo = un
    w = np.array([sb * alpha, sn, so])
    w = w / w.sum()
    return w[0] * ub + w[1] * un + w[2] * uo


# (sb, sn, so, H_off) for day / night. Heights: sb = 0-2 m, sn = 2-10 m, so = > 10 m.
COD_NS = ((0.70, 0.205, 0.095, 20.0),) * 2     # Hobson et al. 2007 (92 DSTs, North Sea stock incl. the
                                               # Skagerrak): SB 70 %, SW 20.5 %, MW + CS 9.5 %; > 55 % within
                                               # 5 m; height up to 35.8 m d-1 mean, 67.9 m max -> H 20 [J]
COD_SW = ((0.905, 0.0, 0.095, 20.0),) * 2   # upper bed-contact reading [J]: Hobson's SW (seabed with ascents)
                                               # counted as bed contact too (SB + SW 90.5 %)
COD_BAR = ((0.19, 0.27, 0.54, 50.0),           # Stensholt & Stensholt 2004: 18.7 % day within 2 m (winter,
           (0.34, 0.23, 0.43, 30.0))           # beds > 200 m); Hjellvik et al. 2003 Table 6 winter large cod
                                               # 0.46 day / 0.57 night within 10 m (acoustic: lower bound);
                                               # H 50 from Hjellvik Table 4 (>10 m part: 53 % below 50 m,
                                               # 85 % below 100 m) [D]; night H 30 [J]
COD_JUT = ((0.639, 0.359, 0.002, 20.0),) * 2   # Hobson et al. 2007 Jutland Bank (56.7 N): SB 63.9 %,
                                               # SW 35.9 %, MW 0.2 %
HAD_L_C = ((0.30, 0.20, 0.50, 35.0),           # haddock >= 35 cm, central [J]: midway between the Barents
           (0.50, 0.30, 0.20, 15.0))           # analogue and the bed-bound reading (no box data)
HAD_L_B = ((0.15, 0.15, 0.70, 40.0),           # Aglen et al. 1997: large and medium haddock from the bottom
           (0.41, 0.34, 0.25, 15.0))           # up to 100 m above it by day, down at night; Engas et al.
                                               # 1988: haddock up to ~150 m by day, 0-2 m channel 0.24 day /
                                               # 0.41 night of the lowest 22 m (350 m bed)
BOUD = ((0.45, 0.40, 0.15, 10.0),) * 2         # Boudreau 1992 (abstract): 50 % within 2.5 m, density
                                               # falling with height (spawning haddock, Scotian Shelf)
HAD_M_B = ((0.40, 0.35, 0.25, 20.0),) * 2      # Aglen 1997: medium groups 'fairly stable' in bottom catches [J]
HAD_S = ((0.60, 0.35, 0.05, 10.0),             # Aglen 1997: small haddock close to the bottom by day,
         (0.30, 0.30, 0.40, 15.0))             # lifted up at night (records dense near the bottom,
                                               # decreasing to 30-40 m)
FLAT_C = ((1.0, 0.0, 0.0, 1.0),) * 2           # Hunter et al. 2009 (139 DST plaice): rarely leave the sea-bed in
                                               # summer, < 5 h/d mid-water only on migration (not search)
FLAT_B = ((0.90, 0.10, 0.0, 1.0),) * 2         # bracket [J]
LRD_B = ((0.70, 0.30, 0.0, 1.0),) * 2          # long rough dab bracket [J]: takes euphausiids / fish near bed

VARIANTS = {
    "central": dict(cod=COD_NS, had_s=HAD_S, had_m=COD_NS, had_l=HAD_L_C, flat=FLAT_C, lrd=FLAT_C),
    "deep": dict(cod=COD_BAR, had_s=HAD_S, had_m=HAD_M_B, had_l=HAD_L_B, flat=FLAT_C, lrd=FLAT_C),
    "bedbound": dict(cod=COD_JUT, had_s=BOUD, had_m=BOUD, had_l=BOUD, flat=FLAT_C, lrd=FLAT_C),
    "rest50": dict(cod=COD_NS, had_s=HAD_S, had_m=COD_NS, had_l=HAD_L_C, flat=FLAT_C, lrd=FLAT_C, alpha=0.5),
    "deep_rest50": dict(cod=COD_BAR, had_s=HAD_S, had_m=HAD_M_B, had_l=HAD_L_B, flat=FLAT_C, lrd=FLAT_C,
                        alpha=0.5),
    "flat_lrd_off": dict(cod=COD_NS, had_s=HAD_S, had_m=COD_NS, had_l=HAD_L_C, flat=FLAT_B, lrd=LRD_B),
    "had_barents": dict(cod=COD_NS, had_s=HAD_S, had_m=HAD_M_B, had_l=HAD_L_B, flat=FLAT_C, lrd=FLAT_C),
    "cod_barents": dict(cod=COD_BAR, had_s=HAD_S, had_m=COD_NS, had_l=HAD_L_C, flat=FLAT_C, lrd=FLAT_C),
    "sw_contact": dict(cod=COD_SW, had_s=HAD_S, had_m=COD_SW, had_l=HAD_L_C, flat=FLAT_C, lrd=FLAT_C),
    "pout_albert": dict(cod=COD_NS, had_s=HAD_S, had_m=COD_NS, had_l=HAD_L_C, flat=FLAT_C, lrd=FLAT_C,
                        pout="albert"),
}
GADOID = {"cod", "had_s", "had_m", "had_l"}
MEMBER_OF = {"cod": "cod", "grey gurnard": "cod", "long rough dab": "lrd", "dab": "flat", "plaice": "flat",
             "witch": "flat", "lemon sole": "flat", "starry ray": "flat"}


def prof_fn(var, mem):
    v = VARIANTS[var]
    p = v[mem]
    alpha = v.get("alpha", 1.0) if mem in GADOID else 1.0

    def f(bed, q, night):
        sb, sn, so, H = p[1] if night else p[0]
        return mix(bed, sb, sn, so, H, alpha)
    return f


def band10(bed, q, night):
    return uniform(bed - 10.0, bed)


PC = {}


def pmat(fn, key, night):
    k = (key, night)
    if k not in PC:
        PC[k] = np.array([fn(b, qk0, night) for b in BEDS]).T          # (n_z, n_b)
    return PC[k]


# ------------------------------------------------------------------ length classes per member subset
def classes(sp, lo=0.0, hi=1e9):
    d = LB[(LB.species == sp) & (LB.share_2001_25 > 0)].copy()
    d["s"] = d.share_2001_25 / d.share_2001_25.sum()
    sel = d[(d.L_cm >= lo) & (d.L_cm < hi)]
    share = float(sel.s.sum())
    s = sel.s / share
    n = (s * 1e6 / sel.w_g).values
    L = (sel.L_cm.values + 0.5) / 100.0
    return n, L, share


SUBSETS = {"had_s": ("haddock", 0, 20), "had_m": ("haddock", 20, 35), "had_l": ("haddock", 35, 1e9)}
G_LIGHT = gmat("gadoid", 0.02, 0.06)                                    # (n_t, n_z), measured light


def member_A(sp, lo, hi, fn, key, phi_fn, phi_key):
    """(n_t, n_b): sum_L n v_rel [R_nv^2 (O - G) + R_sat^2 G] per tonne of the subset (constants dropped)."""
    n, L, _ = classes(sp, lo, hi)
    vr = O["vr_s"](0.4 * L, 0.10)
    rs = np.maximum(np.minimum(R0_FISH, 2 * L), RNV)
    S1 = float((n * vr).sum())
    S3 = float((n * vr * rs ** 2).sum())
    out = np.zeros((len(ELEV), NB))
    for night in (False, True):
        rows = IS_NIGHT if night else ~IS_NIGHT
        W = pmat(fn, key, night) * pmat(phi_fn, phi_key, night)       # (n_z, n_b)
        o = W.sum(0)
        g = G_LIGHT[rows] @ W
        out[rows] = S1 * RNV ** 2 * (o[None, :] - g) + S3 * g
    return out


def stats(ser):
    s = ser @ WCELL
    cls = np.array([float((ser[:, idx] @ WCELL[idx]).mean() / WCELL[idx].sum()) for idx in CLS_IDX])
    return float(s.mean()), cls, s


# ------------------------------------------------------------------ prey profiles
small, large = K["bw_prof"](False), K["bw_prof"](True)


def pout_albert(bed, q, night):
    """Albert 1994: by day below the 1.2-1.5 m detection height (within ~1.5 m of the bed), at night the
    lowest 5-10 m; the library's 0.7 / 0.4 bottom share and its upper part kept [J sensitivity]."""
    if not night:
        return 0.7 * uniform(bed - 1.5, bed) + 0.3 * uniform(bed - 30, bed - 10)
    return 0.4 * uniform(bed - 10, bed - 5) + 0.6 * uniform(25.0, max(26.0, bed - 10))


def whiting_prey(bed, q, night):
    """whiting as prey: section 251 field share in the bottom 10 m 0.7 day / 0.5 night (no diel GOV change,
    Adlerstein & Ehrich 2000); the rest 10-40 m above the bed (Onsrud et al. 2004: 20-45 m) [J]."""
    f = 0.5 if night else 0.7
    return f * uniform(bed - 10, bed) + (1 - f) * uniform(max(0.0, bed - 40), bed - 10)


PREY = {"herring": K["herring_prof"], "sprat": K["herring_prof"], "Norway pout": K["pout_prof"],
        "blue whiting": lambda b, q, n: 0.787 * small(b, q, n) + 0.213 * large(b, q, n),
        "whiting": whiting_prey, "pout_albert": pout_albert}
POOL_S = {"herring": 0.554, "blue whiting": 0.075, "Norway pout": 0.350, "sprat": 0.021}   # 243.3
W_COD, W_HAD, C_H = PV["W_COD"], PV["W_HAD"], PV["C_H"]
win, WIN_COD, WIN_HAD = PV["win"], PV["WIN_COD"], PV["WIN_HAD"]
# haddock >= 20 cm subset shares (within >= 20)
_, _, sh_m = classes("haddock", 20, 35)
_, _, sh_l = classes("haddock", 35, 1e9)
HM, HL = sh_m / (sh_m + sh_l), sh_l / (sh_m + sh_l)


def plk_pair(var, ref=False):
    """series a(t, b) of the planktivore pair (relative units); ref = the library band (uniform 10 m)."""
    pout_key = "pout_albert" if VARIANTS[var].get("pout") == "albert" else "Norway pout"
    tot = np.zeros((len(ELEV), NB))
    contrib = {}
    for sp, s in POOL_S.items():
        pk = pout_key if sp == "Norway pout" else sp
        if ref:
            fc = fh_m = fh_l = (band10, "band10")
        else:
            fc, fh_m, fh_l = ((prof_fn(var, "cod"), (var, "cod")), (prof_fn(var, "had_m"), (var, "had_m")),
                              (prof_fn(var, "had_l"), (var, "had_l")))
        ac = W_COD * win(WIN_COD, sp) * member_A("cod", 20, 1e9, fc[0], fc[1], PREY[pk], pk)
        ah = W_HAD * C_H * win(WIN_HAD, sp) * (HM * member_A("haddock", 20, 35, fh_m[0], fh_m[1], PREY[pk], pk)
                                               + HL * member_A("haddock", 35, 1e9, fh_l[0], fh_l[1], PREY[pk], pk))
        tot += s * (ac + ah)
        contrib[sp] = (float((s * ac @ WCELL).mean()), float((s * ah @ WCELL).mean()))
    return tot, contrib


def wht_pair(var, ref=False):
    """cod >= 20 cm membership 1 (0.308), rays and gurnard 0.5 (0.012) (sections 249 / 251)."""
    mem = [("cod", 20, 1e9, "cod", 1.0), ("grey gurnard", 0, 1e9, "cod", 0.5), ("starry ray", 0, 1e9, "flat", 0.5)]
    tot = np.zeros((len(ELEV), NB))
    contrib = {}
    for sp, lo, hi, m, memb in mem:
        _, _, share = classes(sp, lo, hi)
        W = O["GROUP_B"][sp] / 100.0 * share * memb
        fn, key = (band10, "band10") if ref else (prof_fn(var, m), (var, m))
        a = W * member_A(sp, lo, hi, fn, key, PREY["whiting"], "whiting")
        tot += a
        contrib[sp] = float((a @ WCELL).mean())
    return tot, contrib


# ------------------------------------------------------------------ krill pair: swap the member profiles
GB = K["GROUPS"]["benthivorous_fish"]
ORIG_MEMBERS = list(GB["members"])


def krill_members(var):
    out = []
    for mem in ORIG_MEMBERS:
        nm = mem["name"]
        if nm == "haddock":
            for sub, (sp, lo, hi) in SUBSETS.items():
                n, L, share = classes(sp, lo, hi)
                out.append(dict(mem, name=f"haddock {sub}", W=mem["W"] * share, bm=(n, L), prof=prof_fn(var, sub)))
        elif nm in ("cod", "grey gurnard"):
            out.append(dict(mem, prof=prof_fn(var, "cod")))
        elif nm == "long rough dab":
            out.append(dict(mem, prof=prof_fn(var, "lrd")))
        else:
            out.append(mem)
    return out


RULE.update(rule="lib", RD=1.2, Lk=None)
GB["members"] = ORIG_MEMBERS
KR_REPRO = run_krill(dict(K["level"](1)))["benthivorous_fish"]
RULE.update(rule="A", RD=1.2, Lk=None)
KR_A_BAND = run_krill(dict(K["level"](1)))["benthivorous_fish"]


def krill_pair(var):
    GB["members"] = krill_members(var)
    RULE.update(rule="A", RD=1.2, Lk=None)
    r = run_krill(dict(K["level"](1)))["benthivorous_fish"]
    GB["members"] = ORIG_MEMBERS
    lib = KRILL_LIB["benthivorous_fish"][0]
    return dict(ratio=r["eq"] / KR_REPRO["eq"], a_new=lib * r["eq"] / KR_REPRO["eq"],
                classes_lib_units=[c * lib / KR_REPRO["eq"] for c in r["classes"]], nd=r["nd"],
                profile_factor=r["eq"] / KR_A_BAND["eq"])


# ------------------------------------------------------------------ occupancy / search summaries per member
def member_summary(var, mem):
    fn = prof_fn(var, mem)
    out = {}
    for night in (False, True):
        P = pmat(fn, (var, mem), night)
        h = BEDS[None, :] - ZG[:, None]
        w2 = float(((P * (h <= 2.0)).sum(0)) @ WCELL)
        w10 = float(((P * (h <= 10.0)).sum(0)) @ WCELL)
        mh = float(((P * np.clip(h, 0, None)).sum(0)) @ WCELL)
        out["night" if night else "day"] = dict(within2=w2, within10=w10, mean_h=mh)
    out["tau_bed"] = FDAY * out["day"]["within2"] + (1 - FDAY) * out["night"]["within2"]
    return out


def tau_of(var, sp):
    """search-time share in bed contact of a benthivore species (haddock: size classes by area moment)."""
    if sp == "haddock":
        num = den = 0.0
        for sub, (s_, lo, hi) in SUBSETS.items():
            n, L, share = classes(s_, lo, hi)
            wgt = share * float((n * L ** 2).sum())
            num += wgt * member_summary(var, sub)["tau_bed"]
            den += wgt
        return num / den, None
    m = MEMBER_OF[sp]
    ms = member_summary(var, m)
    return ms["tau_bed"], ms


def tau_nd(var, sp):
    if sp == "haddock":
        d = n_ = den = 0.0
        for sub, (s_, lo, hi) in SUBSETS.items():
            nn, L, share = classes(s_, lo, hi)
            wgt = share * float((nn * L ** 2).sum())
            ms = member_summary(var, sub)
            d += wgt * ms["day"]["within2"]
            n_ += wgt * ms["night"]["within2"]
            den += wgt
        return d / den, n_ / den
    ms = member_summary(var, MEMBER_OF[sp])
    return ms["day"]["within2"], ms["night"]["within2"]


# benthic pair member weights (pout_vs_bed section D: per-species a_enc x exposure e, central)
WTS, INF_RAW, EPI_RAW = PV["wts"], PV["inf_raw"], PV["epi_raw"]
BW_INF = {sp: WTS[sp] * INF_RAW[sp] for sp in WTS}
BW_EPI = {sp: WTS[sp] * EPI_RAW[sp] for sp in WTS}
BW_NEP = {"cod": 1.0}                    # E1 (223.15): cod on emerged Nephrops (few haddock, LRD) [J]
BW_NEP_B = {"cod": 0.8, "haddock": 0.15, "long rough dab": 0.05}      # bracket [J]
BENTHIC = {"benthic_infauna": (0.036, BW_INF), "mobile_epifauna": (0.15, BW_EPI),
           "norway_lobster": (0.44, BW_NEP)}


def benthic(var, weights):
    tot = sum(weights.values())
    f = sum(w / tot * tau_of(var, sp)[0] for sp, w in weights.items())
    d = sum(w / tot * tau_nd(var, sp)[0] for sp, w in weights.items())
    n = sum(w / tot * tau_nd(var, sp)[1] for sp, w in weights.items())
    return f, n / d


# ================================================================== run
RES = {"variants": {}, "repro": dict(krill_lib=KR_REPRO, krill_A_band=KR_A_BAND)}
print("=" * 112)
print("0. Reproduction")
print("=" * 112)
print(f"  krill (library rule, band 10 m, measured light, 262 layer): eq {KR_REPRO['eq']:.5g} -> library 0.013; classes "
      + " / ".join(f"{c * 0.013 / KR_REPRO['eq']:.3g}" for c in KR_REPRO["classes"]) + " (library 0.046 / 0.011 / 0.00059 / 0)"
      f"; n/d {KR_REPRO['nd']:.3f} (library 0.03)")
print(f"  krill, rule A on the old band: x{KR_A_BAND['eq'] / KR_REPRO['eq']:.3f} (Overlapp i fiskparen: x0.727)")
print(f"  rule A R0 on 15 cm fish (|C0| 0.7): {R0_FISH:.3f} m; R_nv 0.3 m; Ke 0.02 lx; measured light (K_deep 0.06)")
print(f"  planktivore pair: members cod >= 20 cm W {W_COD:.3f}, haddock >= 20 cm W {W_HAD:.3f} x c_h {C_H} (20-34 cm "
      f"{HM:.2f}, >= 35 cm {HL:.2f}); pool shares {POOL_S}")
r_ref, c_ref = plk_pair("central", ref=True)
tot_ref = sum(sum(v) for v in c_ref.values())
print("  planktivore pair, reference (uniform 10 m band, rule A, light at depth): species shares of a "
      + ", ".join(f"{k} {sum(v) / tot_ref:.2f}" for k, v in c_ref.items())
      + f"; cod / haddock {sum(v[0] for v in c_ref.values()) / tot_ref:.2f} / {sum(v[1] for v in c_ref.values()) / tot_ref:.2f}"
      "  (243.3 lambda: pout 0.87, herring 0.11; pout_vs_bed cod / haddock 0.67 / 0.33 on pout)")
ref_eq, ref_cls, ref_s = stats(r_ref)
w_ref, wc_ref = stats(wht_pair("central", ref=True)[0])[0], wht_pair("central", ref=True)[1]
print(f"  whiting pair, reference member shares: " + ", ".join(f"{k} {v / sum(wc_ref.values()):.3f}" for k, v in wc_ref.items()))
wr_eq, wr_cls, wr_s = stats(wht_pair("central", ref=True)[0])
print(f"  reference night/day (0 deg): planktivores {night_day(ref_s, 0.0):.3f} (library 0.06, band A 0.090); "
      f"whiting {night_day(wr_s, 0.0):.3f} (library 0.15, band A 0.180)")
BAND = json.load(open(os.path.join(RN, "Overlapp i fiskparen", "tables", "band_pairs.json")))["variants"]["A_band"]["fish"]
V_A = {"benthivorous_fish->planktivorous_fish": BAND["benthivorous_fish->planktivorous_fish"]["ratio"],
       "benthivorous_fish->whiting": BAND["benthivorous_fish->whiting"]["ratio"]}
print(f"  vision factor rule A at the bed (Overlapp i fiskparen A_band): planktivores x{V_A['benthivorous_fish->planktivorous_fish']:.3f}, "
      f"whiting x{V_A['benthivorous_fish->whiting']:.3f}")

print("\n" + "=" * 112)
print("1. Member profiles: share of time within 2 m / within 10 m of the bed, mean height (m), day | night; tau_bed")
print("   = annual search-time share in bed contact (0-2 m), equal cells (16 x 16 bins of maps/depth.npz)")
print("=" * 112)
for var in VARIANTS:
    if var == "pout_albert":
        continue
    line = []
    for mem in ("cod", "had_s", "had_m", "had_l", "lrd", "flat"):
        ms = member_summary(var, mem)
        line.append(f"{mem} {ms['day']['within2']:.2f}/{ms['day']['within10']:.2f}/{ms['day']['mean_h']:.0f} | "
                    f"{ms['night']['within2']:.2f}/{ms['night']['within10']:.2f}/{ms['night']['mean_h']:.0f}; tau {ms['tau_bed']:.2f}")
    print(f"  {var}:\n    " + "\n    ".join(line))
    RES["variants"].setdefault(var, {})["members"] = {m: member_summary(var, m) for m in
                                                       ("cod", "had_s", "had_m", "had_l", "lrd", "flat")}

print("\n" + "=" * 112)
print("2. Pairs: a_new = a_lib x vision (rule A at the bed, band) x profile factor P (fish), krill as one chain;")
print("   benthic a x tau. Class a's < 100 / 100-150 / 150-250 / >= 250 m; night/day at 0 deg (the pairs' threshold)")
print("=" * 112)
A_LIB = {"benthivorous_fish->planktivorous_fish": 0.0198, "benthivorous_fish->whiting": 0.021}
for var in VARIANTS:
    R = RES["variants"].setdefault(var, {})
    kr = krill_pair(var)
    R["krill"] = kr
    rp, cp = plk_pair(var)
    eq, cls, s = stats(rp)
    tot = sum(sum(v) for v in cp.values())
    Pp = eq / ref_eq
    name = "benthivorous_fish->planktivorous_fish"
    a_new = A_LIB[name] * V_A[name] * Pp
    R[name] = dict(P=Pp, ratio=V_A[name] * Pp, a_new=a_new, cls_rel=(cls / eq).tolist(),
                   classes=(a_new * cls / eq).tolist(), nd=night_day(s, 0.0),
                   species={k: sum(v) / tot for k, v in cp.items()},
                   cod_had=(sum(v[0] for v in cp.values()) / tot, sum(v[1] for v in cp.values()) / tot))
    rw, cw = wht_pair(var)
    eqw, clsw, sw = stats(rw)
    Pw = eqw / wr_eq
    name = "benthivorous_fish->whiting"
    a_w = A_LIB[name] * V_A[name] * Pw
    R[name] = dict(P=Pw, ratio=V_A[name] * Pw, a_new=a_w, cls_rel=(clsw / eqw).tolist(),
                   classes=(a_w * clsw / eqw).tolist(), nd=night_day(sw, 0.0))
    for prey, (alib, wts) in BENTHIC.items():
        f, nd = benthic(var, wts)
        R[f"benthivorous_fish->{prey}"] = dict(tau=f, a_new=alib * f, nd=nd)
    fb, _ = benthic(var, BW_NEP_B)
    R["benthivorous_fish->norway_lobster"]["tau_bracket_weights"] = fb
    print(f"  {var}:")
    print(f"    krill        0.013 -> {kr['a_new']:.4g} (x{kr['ratio']:.3f}; profile alone x{kr['profile_factor']:.3f} vs A on the"
          f" band); classes " + " / ".join(f"{c:.3g}" for c in kr["classes_lib_units"]) + f"; n/d {kr['nd']:.3f}")
    r1 = R["benthivorous_fish->planktivorous_fish"]
    print(f"    planktivores 0.0198 -> {r1['a_new']:.4g} (P x{Pp:.3f}, total x{r1['ratio']:.3f}); classes "
          + " / ".join(f"{c:.3g}" for c in r1["classes"]) + f"; n/d {r1['nd']:.3f}; species "
          + ", ".join(f"{k} {v:.2f}" for k, v in r1["species"].items()) + f"; cod/had {r1['cod_had'][0]:.2f}/{r1['cod_had'][1]:.2f}")
    r2 = R["benthivorous_fish->whiting"]
    print(f"    whiting      0.021 -> {r2['a_new']:.4g} (P x{Pw:.3f}, total x{r2['ratio']:.3f}); classes "
          + " / ".join(f"{c:.3g}" for c in r2["classes"]) + f"; n/d {r2['nd']:.3f}")
    for prey in BENTHIC:
        rb = R[f"benthivorous_fish->{prey}"]
        print(f"    {prey:16s} a {BENTHIC[prey][0]} -> {rb['a_new']:.4g} (tau x{rb['tau']:.3f}; implied night/day {rb['nd']:.2f})"
              + (f"; Nephrops weights cod 0.8 / haddock 0.15 / LRD 0.05: x{rb['tau_bracket_weights']:.3f}"
                 if prey == "norway_lobster" else ""))
print("\n  benthic pair member weights (a_enc x e, pout_vs_bed section D): infauna "
      + ", ".join(f"{k} {v / sum(BW_INF.values()):.3f}" for k, v in BW_INF.items()) + "; epifauna "
      + ", ".join(f"{k} {v / sum(BW_EPI.values()):.3f}" for k, v in BW_EPI.items()))
RES["benthic_weights"] = dict(infauna={k: v / sum(BW_INF.values()) for k, v in BW_INF.items()},
                              epifauna={k: v / sum(BW_EPI.values()) for k, v in BW_EPI.items()})
json.dump(RES, open(OUTJ, "w"), indent=1, default=float)
print(f"\nwritten {os.path.basename(OUTJ)}")
