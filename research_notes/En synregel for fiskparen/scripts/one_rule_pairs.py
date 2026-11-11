"""One visual-encounter rule on every visual fish-predator pair (krill and fish prey): a_new / a_library.

Usage: python3 -I one_rule_pairs.py <repo_root> <scratch_dir> <out_json>

Nothing is fitted and no model output is read. Every derivation the library a's came from is exec'd
unchanged and only its VISION term is replaced; every other factor of each pair (scan, speeds, numbers
per tonne, membership, overlap / in-layer O, band, prey shares, depth geometry, capture p) is kept.

Rule (research_notes/"Synregel i vattenpelaren"/summary.md, "Proposed single rule"):
    R_j^2(E) = R_nv,j^2 (1 - g) + R_sat,j^2 g,   g = E / (E + Ke)  (the predator's own eye)
    rule A: R_sat,j = min(R0_j, cap_L L_pred), R0_j^2 exp(c R0_j) = |C0_j| A_j k_v^2, A_j = 0.75 L_j (0.2 L_j)
            |C0| krill 0.4, fish 0.7 [J]; k_v 100, c 0.25 /m, cap_L 2 (library)
    rule B: R_sat(fish) = min(RD_max 1.2 m (1.0-1.9), cap_L L_pred) (salmonid RD_max on >= 50 mm fish);
            R_sat(krill) = R_sat(fish) (L_krill / 50 mm)^b sqrt(0.4 / 0.7), b = geometric mid of 0.415 / 0.709
    R_nv: krill 0.1 m (library gadoid floor; clupeid eye keeps its library 0.05 m), fish 0.3 m [J];
          non-visual members (spurdog, velvet belly) keep R_nv on every prey (fish 0.3 = R_fish_elasmo);
          when R_sat < R_nv (predators < 15 cm on fish) R_sat is raised to R_nv (meas_pairs.py convention; flagged)
    Ke: gadoid eye 0.02 lx (bracket 2e-4 .. 0.1); clupeid eye (herring, mackerel) kept at its library
        Ke_clup 1 lx with the biting threshold 0.016 lx (the rule's Ke bracket is a gadoid bracket).
    Light: the measured box light of section 253 (K_deep 0.06; 0.05 / 0.09 as a sensitivity).

Pairs and where the library derivation lives:
  krill pairs (all five groups): research_notes/"Krill i Skagerrak"/wc_depth/scripts/krill_pairs_wc.py
      --variant field, with the 253 light and the 262 layer / presence weights (copied from
      research_notes/"Krill over grunda bottnar" (o-umlaut)/rederive/scripts/krill_geometry.py, as
      research_notes/"Vattenpelarens par per djupklass"/scripts/wc_terms.py does). Band convention kept.
  in-layer fish pairs (water column, whiting, mackerel): research_notes/"Rovfiskens uppdelning"/
      split_encounter_223.py head (section 220 pred_encounter_a.py), dispersed prey, layer membership,
      243.3 pool shares. The in-layer O = 1/H is KEPT; the light is the light at the prey's own vertical
      profile (krill_pairs_wc.py prey profiles), i.e. "band light, library overlap". This is NOT the
      sibling note's V_one (which also moved the fish pairs to the band overlap).
  benthivore fish pairs: research_notes/"Attackrater ur motesteori" (o-umlaut)/scripts/benth_encounter.py
      (already light-driven: R_sat 2 L, R_nv 0.6 L, Ke 0.02, light at the bed under the old central
      profile, Kd_top 0.25). The ratio replaces R_nv by the rule's 0.3 m, the light by the measured light,
      and (rule B) R_sat by min(RD_max, 2 L). Day / night weights from the library dark_ratio (0 deg).
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
OUTJ = sys.argv[3]
RN = os.path.join(ROOT, "research_notes")
ARGV0 = list(sys.argv)
LBCSV = os.path.join(RN, "Fiskgrupper i rutan", "benthivores", "tables", "length_biomass_longrun.csv")

# =============================================================== krill machinery (exec, unchanged head)
WC = os.path.join(RN, "Krill i Skagerrak", "wc_depth", "scripts", "krill_pairs_wc.py")
src = open(WC).read()
head = src[:src.index("\n# ================================================================ central\n")]
sys.argv = [WC, os.path.join(ROOT, "maps", "depth.npz"), LBCSV, "10", "--variant", "field"]
K = {"__name__": "krill_pairs_wc_head", "__file__": WC}
exec(compile(head, WC, "exec"), K)
sys.argv = ARGV0
ZG, ELEV, MONTH, E0 = K["ZG"], K["ELEV"], K["MONTH"], K["E0"]
BEDS, WCELL, CLS_IDX, CLASSES = K["BEDS"], K["WCELL"], K["CLS_IDX"], K["CLASSES"]
T = K["T"]
NB = len(BEDS)

# ---- section 253 measured light and section 262 layer / presence (as krill_geometry.py / wc_terms.py)
SECCHI = np.array([5.0, 7.0, 7.0, 8.5, 8.0, 7.0, 10.0, 9.5, 8.0, 9.0, 9.0, 11.5])
K["P"].update({"k2_mult": (1.23, 1.0, 0.83), "k20_50": (0.138, 0.109, 0.101), "k50_65": (0.091, 0.091, 0.091),
               "k_deep": (0.09, 0.06, 0.05)})
TRM = {}


def trans_month(q):
    z = ZG
    out = np.empty((12, len(z)))
    for m in range(12):
        k2 = 1.56 / SECCHI[m] * q["k2_mult"]
        tau = (0.30 * np.minimum(z, 2) + k2 * np.clip(np.minimum(z, 20) - 2, 0, None)
               + q["k20_50"] * np.clip(np.minimum(z, 50) - 20, 0, None)
               + q["k50_65"] * np.clip(np.minimum(z, 65) - 50, 0, None) + q["k_deep"] * np.clip(z - 65, 0, None))
        out[m] = np.exp(-tau)
    return out


_lux = K["lux"]
K["lux"] = lambda prof: E0[:, None] * TRM[prof][MONTH] if prof in TRM else _lux(prof)


def meas(q):
    q = dict(q)
    TRM["meas"] = trans_month(q)
    q["prof"] = "meas"
    return q


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


K["krill_day_layer"] = new_layer
WPRES = np.full(12, 0.43)
WPRES[[4, 5, 6, 7]] = 0.05
WT = WPRES[MONTH]
IS_NIGHT = ELEV <= K["SUN_KRILL"]          # 0 deg: the prey / speed day-night switch of every derivation
FDAY = float((~IS_NIGHT).mean())

# ---- rule constants
C_KRILL, C_FISH = 0.4, 0.7                  # |C0| [J]
L_FISH_PREY = 0.15                          # m, fish prey (meas_pairs.py central)
RNV_FISH = 0.3                              # m [J]
B_LO, B_HI = 0.415, 0.709                   # Hansen et al. 2013 prey-size exponents
B_MID = 0.5 * (B_LO + B_HI)                 # geometric mid of the two factors
RULE = dict(rule="lib", RD=1.2, Lk=None)


def r0_c(q, L, C0):
    """R at saturating light: R^2 exp(c R) = C0 k_v^2 A, A = 0.75 L x 0.2 L (krill_pairs_wc convention)."""
    r00 = math.sqrt(C0) * q["k_v"] * math.sqrt(0.75 * L * 0.2 * L)
    r = r00
    for _ in range(80):
        r = r00 * math.exp(-q["c_beam"] * r / 2.0)
    return r


def fb_krill(q):
    Lk = RULE["Lk"] or q["L_krill"]
    return (Lk / 0.05) ** B_MID * math.sqrt(C_KRILL / C_FISH)


def rsat_krill(q, Lpred):
    Lk = RULE["Lk"] or q["L_krill"]
    cap = q["cap_L"] * np.asarray(Lpred, float)
    if RULE["rule"] == "lib":
        return np.minimum(r0_c(q, Lk, 1.0), cap)
    if RULE["rule"] == "A":
        return np.minimum(r0_c(q, Lk, C_KRILL), cap)
    return np.minimum(RULE["RD"], cap) * fb_krill(q)


def rsat_fish(q, Lpred):
    cap = q["cap_L"] * np.asarray(Lpred, float)
    if RULE["rule"] == "A":
        return np.minimum(r0_c(q, L_FISH_PREY, C_FISH), cap)
    return np.minimum(RULE["RD"], cap)


def member_moments(mem, q, night):
    """krill_pairs_wc.member_moments with R_sat from the active rule (only the R_sat line differs)."""
    vb = q[mem["vkey"]] * (q[mem["night_v"]] if night and "night_v" in mem else 1.0)
    if "bm" in mem:
        n, L = mem["bm"]
        vr = K["vrel"](vb * L, q["u_krill"])
        rs = rsat_krill(q, L)
        return float((n * vr).sum()), float((n * vr * rs ** 2).sum())
    if "m1" in mem:
        if mem["m1"] <= 0:
            return 0.0, 0.0
        Lnum = mem["m1"] / mem["nt"]
        vr = float(K["vrel"](vb * Lnum, q["u_krill"]))
        rs = float(rsat_krill(q, Lnum))
        S1 = mem["nt"] * vr
        return S1, S1 * rs ** 2
    if mem["nt"] <= 0:
        return 0.0, 0.0
    vr = float(K["vrel"](vb * mem["L"], q["u_krill"]))
    rs = float(rsat_krill(q, mem["L"]))
    S1 = mem["nt"] * vr
    return S1, S1 * rs ** 2


K["member_moments"] = member_moments


def krill_classes(a_t_b):
    """class means; class 0 presence-weighted (krill_geometry.py / section 262)."""
    out = []
    for k, idx in enumerate(CLS_IDX):
        s = a_t_b[:, idx] @ WCELL[idx] / WCELL[idx].sum()
        out.append(float((s * WT).sum() / WT.sum()) if k == 0 else float(s.mean()))
    return np.array(out)


SHARE_CLS = np.array([WCELL[idx].sum() for idx in CLS_IDX])


def night_day(series, thr):
    d = series[ELEV > thr].mean()
    return float(series[ELEV <= thr].mean() / d) if d > 0 else float("nan")


KRILL_LIB = {"predatory_fish": (0.026, [0.068, 0.031, 0.0058, 0.00014], 0.02, 0.0),
             "whiting": (0.043, [0.148, 0.035, 0.002, 0.0], None, 0.0),
             "planktivorous_fish": (0.023, [0.069, 0.021, 0.0037, 0.0017], 0.57, 0.0),
             "benthivorous_fish": (0.013, [0.046, 0.011, 0.00059, 0.0], 0.03, 0.0),
             "mackerel": (0.0039, [0.0053, 0.0043, 0.0031, 0.0028], 0.42, -5.0)}


def run_krill(q):
    q = meas(q)
    ev = K["evaluate"](q)
    out = {}
    for g, r in ev.items():
        cls = krill_classes(r["a"])
        s = r["a"] @ WCELL / WCELL.sum()
        out[g] = dict(classes=cls.tolist(), eq=float(SHARE_CLS @ cls), nd=night_day(s, KRILL_LIB[g][3]))
    return out


# =============================================================== in-layer fish machinery (section 220 head)
p2 = os.path.join(RN, "Rovfiskens uppdelning", "scripts", "split_encounter_223.py")
src2 = open(p2).read()
src2 = src2[:src2.index("ALL = [")]
os.makedirs(SCR, exist_ok=True)
sys.argv = [p2, os.path.join(RN, "Attackrater ur mötesteori"), os.path.join(RN, "Fiskgrupper i rutan", "predators"), SCR]
GE = {"__name__": "split_head", "__file__": p2}
exec(compile(src2, p2, "exec"), GE)
sys.argv = ARGV0
GP = GE["G"]
GP["SCHOOLS"] = False
PS = {"herring": 0.554, "blue whiting": 0.075, "pout": 0.350, "sprat": 0.021}   # 243.3 pool shares
GP["PLANK_SHARE"].clear()
GP["PLANK_SHARE"].update(PS)
STG_T, stage_pair, sc = GE["STG_T"], GE["stage_pair"], GE["sc"]
a3d, area_in_slab, TYPE, SPEED, NIGHTV = GP["a3d"], GP["area_in_slab"], GP["TYPE"], GP["SPEED"], GP["NIGHTV"]
qf = GE["level"](1)
REST = ["saithe", "spurdog", "hake", "anglerfish", "ling", "velvet belly", "pollack"]
EATERS = ["spurdog", "hake", "anglerfish", "ling", "velvet belly"]
OVL_BENTH = {"saithe": 0.05, "pollack": 0.05, "spurdog": 0.15, "hake": 0.25, "velvet belly": 0.10, "anglerfish": 1.0,
             "ling": 0.25}
DENS = {"mackerel": 0.80, "horse mackerel": 0.05}
WMUL = {sp: DENS[sp] / sum(r.W for r in STG_T if r.sp == sp) for sp in DENS}
EYE = {"cruiser": "gadoid", "midambush": "gadoid", "mackerel": "clupeid", "elasmo": "none", "bedambush": "fixed"}


def v_of(r, q, night):
    v = q[SPEED[r.sp]] * r.L
    if night and r.sp in NIGHTV:
        v *= q[NIGHTV[r.sp]]
    return v


def terms(r, prey_type, q, subset=None):
    """the section 220 stage a split into (period, prey sub-type) terms; mirrors stage_a / whiting_prey_a."""
    typ = TYPE[r.sp]
    out = []
    p = q["p_fish"]
    if prey_type == "plank_fish":
        if typ == "bedambush":
            for prey, H in (("pout", q["H_pout"]), ("blue whiting", q["H_bluewhiting"])):
                a = a3d(q["R_strike_angler"], q["u_prey_angler"], 0.0, H, r.Nt, p, 1.0, PS[prey])
                for night in (False, True):
                    out.append(dict(night=night, prey=prey, R=q["R_strike_angler"], H=H, a=a, eye="fixed"))
            return [t for t in out if subset is None or t["prey"] in subset]
        if typ in ("cruiser", "mackerel"):
            R_day = min(q["R_fish_day_cruiser"], 2 * r.L)
        elif typ == "elasmo":
            R_day = q["R_fish_elasmo"]
        else:
            R_day = q["R_fish_hake_day"]
        R_dark = q["R_fish_elasmo"] if typ == "elasmo" else q["R_fish_dark"]
        for night in (False, True):
            v = v_of(r, q, night)
            for prey in ("herring", "sprat"):
                if night:
                    a = a3d(R_dark, v, q["u_herring"], q["H_herring_night"], r.Nt, p, q["scan"], PS[prey], q["f_ov_night"])
                    out.append(dict(night=True, prey=prey, R=R_dark, H=q["H_herring_night"], a=a, eye=EYE[typ]))
                else:
                    a = a3d(R_day, v, q["u_herring"], q["H_herring_day"], r.Nt, p, q["scan"], PS[prey])
                    out.append(dict(night=False, prey=prey, R=R_day, H=q["H_herring_day"], a=a, eye=EYE[typ]))
            for prey, H in (("blue whiting", q["H_bluewhiting"]), ("pout", q["H_pout"])):
                a = a3d(R_dark, v, q["u_smallfish"], H, r.Nt, p, q["scan"], PS[prey])
                out.append(dict(night=night, prey=prey, R=R_dark, H=H, a=a, eye=EYE[typ]))
        return [t for t in out if subset is None or t["prey"] in subset]
    if prey_type in ("benth_fish", "whiting_prey"):
        H = q["H_juv"] if prey_type == "benth_fish" else q["H_whiting"]
        prey = "juvenile benthivores" if prey_type == "benth_fish" else "whiting"
        if typ == "bedambush":
            a = a3d(q["R_strike_angler"], q["u_prey_angler"], 0.0, H, r.Nt, p, 1.0, 1.0)
            return [dict(night=n, prey=prey, R=q["R_strike_angler"], H=H, a=a, eye="fixed") for n in (False, True)]
        for night in (False, True):
            v = v_of(r, q, night)
            if typ == "elasmo":
                R = q["R_fish_elasmo"]
            else:
                R = q["R_fish_dark"] if night else max(q["R_bedfish_day"], q["R_fish_dark"])
            out.append(dict(night=night, prey=prey, R=R, H=H, a=a3d(R, v, q["u_smallfish"], H, r.Nt, p, q["scan"], 1.0),
                            eye=EYE[typ]))
        return out
    raise ValueError(prey_type)


# reproduction of the 220 stage values through the term split
for r in STG_T:
    for pt in ("plank_fish", "benth_fish", "whiting_prey"):
        ref = 0.5 * (stage_pair(r, pt, qf, False) + stage_pair(r, pt, qf, True))
        mine = 0.5 * sum(t["a"] for t in terms(r, pt, qf))
        assert abs(ref - mine) <= 1e-9 + 1e-6 * abs(ref), (r.sp, r.st, pt, ref, mine)

# fish pair definitions: (members, prey_type, prey sub-type subset, light profile override, overlap, field weights)
FISH_PAIRS = {
    "predatory_fish->planktivorous_fish": (REST, "plank_fish", None, None, None, False, 0.0060, 0.41, -5.0),
    "predatory_fish->benthivorous_fish": (REST, "benth_fish", None, None, OVL_BENTH, False, 0.0055, 0.48, -5.0),
    "predatory_fish->whiting": (REST, "whiting_prey", None, None, None, False, 0.013, 0.48, -5.0),
    "predatory_fish->mackerel": (EATERS, "plank_fish", None, "mackerel", None, False, 0.00088, 0.92, -5.0),
    "whiting->planktivorous_fish": (["whiting"], "plank_fish", None, None, None, False, 0.0212, 0.48, -5.0),
    "whiting->benthivorous_fish": (["whiting"], "benth_fish", None, None, None, False, 0.21, 0.64, -5.0),
    "mackerel->planktivorous_fish": (["mackerel", "horse mackerel"], "plank_fish", ("herring", "sprat"), None, None,
                                     True, 0.0266, 0.16, -5.0),
}

# ---- prey vertical profiles (light sampling) on the bed bins: phi[(prey, night)] (n_z, n_b)
qk0 = K["level"](1)
small, large = K["bw_prof"](False), K["bw_prof"](True)
PROF = {
    "herring": K["herring_prof"], "sprat": K["herring_prof"], "pout": K["pout_prof"],
    "blue whiting": lambda b, qq, n: 0.787 * small(b, qq, n) + 0.213 * large(b, qq, n),
    "juvenile benthivores": lambda b, qq, n: K["uniform"](b - qf["H_juv"], b),
    "whiting": lambda b, qq, n: K["uniform"](b - qf["H_whiting"], b),
    "mackerel": K["mack_prof"],
}
PHI = {}
for pn, f in PROF.items():
    for night in (False, True):
        PHI[(pn, night)] = np.array([f(b, qk0, night) for b in BEDS]).T


def area_vec(R, H):
    h = H / 2.0
    R = np.asarray(R, float)
    out = np.pi * R * R
    big = R > h
    Rb = R[big]
    out[big] = 2.0 * (Rb * Rb * np.arcsin(h / Rb) + h * np.sqrt(Rb * Rb - h * h))
    return out


GCACHE = {}


def gmat(eye, Ke, kdeep):
    key = (eye, Ke, kdeep)
    if key not in GCACHE:
        q = dict(qk0, k_deep=kdeep)
        E = E0[:, None] * trans_month(q)[MONTH]
        if eye == "gadoid":
            g = E / (E + Ke)
        elif eye == "clupeid":
            g = np.where(E >= qk0["thr_clup"], E / (E + qk0["Ke_clup"]), 0.0)
        else:
            g = np.zeros_like(E)
        GCACHE[key] = g
    return GCACHE[key]


ACACHE = {}


def a_eff(eye, rsat, rnv, H, prey, night, Ke, kdeep):
    """(n_rows_period, n_b): sum_z phi_prey(z) A(R(z, t), H) over the period's time rows."""
    key = (eye, round(rsat, 6), round(rnv, 6), H, prey, night, Ke, kdeep)
    if key not in ACACHE:
        rows = IS_NIGHT if night else ~IS_NIGHT
        g = gmat(eye, Ke, kdeep)[rows]
        rs = max(rsat, rnv)
        R = np.sqrt(rnv ** 2 * (1 - g) + rs ** 2 * g)
        ACACHE[key] = area_vec(R, H) @ PHI[(prey, night)]
    return ACACHE[key]


def fish_pair(name, Ke=0.02, kdeep=0.06, library=False):
    members, pt, subset, lightprey, ovl, field, a_lib, nd_lib, thr = FISH_PAIRS[name]
    den = 0.0
    DEC = {}
    ser = np.zeros((len(ELEV), NB))
    a_enc_lib = 0.0
    contrib = {}
    for r in STG_T:
        if r.sp not in members:
            continue
        w = r.W * (WMUL.get(r.sp, 1.0) if field else 1.0)
        den += w
        if sc(r, pt) < 0.1:
            continue
        m = (ovl or {}).get(r.sp, 1.0) if ovl is not None else 1.0
        for t in terms(r, pt, qf, subset):
            wa = w * m * t["a"]
            a_enc_lib += 0.5 * wa
            rows = IS_NIGHT if t["night"] else ~IS_NIGHT
            if library or t["eye"] in ("fixed",):
                kk = (t["prey"], "night" if t["night"] else "day")
                DEC.setdefault(kk, [0.0, 0.0, np.zeros(4)])
                DEC[kk][0] += 0.5 * wa
                DEC[kk][1] += 0.5 * wa
                DEC[kk][2] += 0.5 * wa
                ser[rows] += wa
                contrib[(r.sp, r.st)] = contrib.get((r.sp, r.st), 0.0) + 0.5 * wa
                continue
            if t["eye"] == "none":            # non-visual: the library R_fish_elasmo 0.3 m on every fish prey
                A = np.full((rows.sum(), NB), area_in_slab(qf["R_fish_elasmo"], t["H"]))
            else:
                rs = float(rsat_fish(qk0, r.L))
                rn = RNV_FISH
                A = a_eff(t["eye"], rs, rn, t["H"], lightprey or t["prey"], t["night"], Ke, kdeep)
            fac = A / area_in_slab(t["R"], t["H"])
            ser[rows] += wa * fac
            kk = (t["prey"], "night" if t["night"] else "day")
            DEC.setdefault(kk, [0.0, 0.0, np.zeros(4)])
            DEC[kk][0] += 0.5 * wa
            fe = (wa * fac) @ WCELL
            DEC[kk][1] += 0.5 * float(fe.mean())
            DEC[kk][2] += 0.5 * np.array([float(((wa * fac)[:, idx] @ WCELL[idx]).mean() / WCELL[idx].sum())
                                          for idx in CLS_IDX])
            contrib[(r.sp, r.st)] = contrib.get((r.sp, r.st), 0.0) + 0.5 * float((wa * fac * WCELL[None, :]).sum(1).mean())
    if members is EATERS:                     # per tonne of the whole rest water column (wc_terms.py)
        den = sum(r.W for r in STG_T if r.sp in REST)
    ser /= den
    a_enc_lib /= den
    s = ser @ WCELL
    cls = np.array([float((ser[:, idx] @ WCELL[idx]).mean() / WCELL[idx].sum()) for idx in CLS_IDX])
    eq = float(s.mean())
    tot = sum(contrib.values())
    return dict(a_enc_lib=a_enc_lib, eq=eq, ratio=eq / a_enc_lib, classes=cls.tolist(),
                cls_rel=(cls / eq).tolist(), nd=night_day(s, thr), a_lib=a_lib, nd_lib=nd_lib,
                members={f"{k[0]}|{k[1]}": v / tot for k, v in contrib.items() if v / tot >= 0.005},
                decomp={f"{k[0]}|{k[1]}": dict(lib=v[0] / den, new=v[1] / den, new_cls=(v[2] / den).tolist())
                        for k, v in DEC.items()})


# =============================================================== benthivore fish pairs (benth_encounter.py)
LB = pd.read_csv(LBCSV)
GROUP_B = {"haddock": 38.62, "cod": 31.38, "long rough dab": 12.63, "dab": 4.49, "plaice": 3.80,
           "witch": 1.67, "starry ray": 1.47, "grey gurnard": 0.89, "lemon sole": 0.87}
CUTS = {"cod": [30, 50], "haddock": [40], "long rough dab": [20], "dab": [20], "plaice": [25],
        "witch": [30], "lemon sole": [25], "grey gurnard": [25], "starry ray": [30]}
SCORE_P = {"cod": [.3, .7, .9], "haddock": [.1, .25], "long rough dab": [.05, .05], "starry ray": [.4, .4],
           "grey gurnard": [.15, .15]}
FLAT = {"dab", "plaice", "witch", "lemon sole"}


def bclasses(sp):
    d = LB[(LB.species == sp) & (LB.share_2001_25 > 0)].copy()
    d["s"] = d.share_2001_25 / d.share_2001_25.sum()
    d["L"] = (d.L_cm + 0.5) / 100.0
    edges = CUTS[sp]
    d["stage"] = np.searchsorted(np.array(edges), d.L_cm.values, side="right")
    d["n"] = d.s * 1e6 / d.w_g
    return d


def vr_s(v, u):
    return np.where(v >= u, (u * u + 3 * v * v) / (3 * v), (v * v + 3 * u * u) / (3 * u))


def benth_weights(pair):
    """(L array, weight array): benth_encounter pair_weights x n x v_rel per 1-cm class (R^2 multiplies)."""
    Ls, ws = [], []
    rows = []
    for sp in GROUP_B:
        d = bclasses(sp)
        for st, dd in d.groupby("stage"):
            if pair == "P":
                s = SCORE_P.get(sp, [None] * 3)[st] if sp in SCORE_P else (0.0 if sp in FLAT else 0.05)
                wt = GROUP_B[sp] * dd.s.sum() * s
                sel = dd
            else:   # section 249/251 membership on whiting: cod >= 20 cm 1 (0.982 of cod), rays and gurnard 0.5
                memb = {"cod": 1.0, "starry ray": 0.5, "grey gurnard": 0.5}.get(sp, 0.0)
                sel = dd[dd.L_cm >= 20] if sp == "cod" else dd
                wt = GROUP_B[sp] * sel.s.sum() * memb
            if wt <= 0 or len(sel) == 0:
                continue
            rows.append((wt, sel))
    tot = sum(w for w, _ in rows)
    for wt, sel in rows:
        s = sel.s / sel.s.sum()
        n = (s * 1e6 / sel.w_g).values
        L = sel.L.values
        ws.append(wt / tot * n * vr_s(0.4 * L, 0.10))
        Ls.append(L)
    return np.concatenate(Ls), np.concatenate(ws)


BW = {"P": benth_weights("P"), "W": benth_weights("W")}
TR_LIB = np.exp(-0.25 * np.minimum(ZG, 20) - 0.12 * np.clip(ZG - 20, 0, 30) - 0.06 * np.clip(ZG - 50, 0, None))
IBED = np.clip(np.round(BEDS - 0.5).astype(int), 0, len(ZG) - 1)
E_BED_LIB = E0[:, None] * TR_LIB[IBED][None, :]


def benth_V(pair, rule, Ke, kdeep):
    """(n_t, n_b) sum_L w_L R_L^2(E_bed): library (R_sat 2 L, R_nv 0.6 L, old light) or the rule."""
    L, w = BW[pair]
    if rule == "lib":
        Eb = E_BED_LIB
        G = Eb / (Eb + 0.02)
        rs2, rn2 = (2 * L) ** 2, (0.6 * L) ** 2
    else:
        Eb = E0[:, None] * trans_month(dict(qk0, k_deep=kdeep))[MONTH][:, IBED]
        G = Eb / (Eb + Ke)
        rs = np.minimum(r0_c(qk0, L_FISH_PREY, C_FISH), 2 * L) if rule == "A" else np.minimum(RULE["RD"], 2 * L)
        rn = np.full_like(L, RNV_FISH)
        rs = np.maximum(rs, rn)
        rs2, rn2 = rs ** 2, rn ** 2
    return (w * rn2).sum() + ((w * (rs2 - rn2)).sum()) * G


BENTH_PAIRS = {"benthivorous_fish->planktivorous_fish": ("P", 0.0198, 0.06, 0.0),
               "benthivorous_fish->whiting": ("W", 0.021, 0.15, 0.0)}


def benth_pair(name, rule, Ke=0.02, kdeep=0.06):
    pair, a_lib, rho, thr = BENTH_PAIRS[name]
    Vl = benth_V(pair, "lib", 0.02, 0.06)
    Vn = benth_V(pair, rule, Ke, kdeep)
    ld = float((Vl[~IS_NIGHT] @ WCELL).mean())
    ln = float((Vl[IS_NIGHT] @ WCELL).mean())
    wd, wn = 1.0 / (1.0 + rho), rho / (1.0 + rho)
    # a(t, b) / a_lib, with 2 x (period weight) per row (day share 0.5)
    ser = np.where(IS_NIGHT[:, None], 2 * wn * Vn / ln, 2 * wd * Vn / ld)
    s = ser @ WCELL
    cls = np.array([float((ser[:, idx] @ WCELL[idx]).mean() / WCELL[idx].sum()) for idx in CLS_IDX])
    eq = float(s.mean())
    return dict(ratio=eq, classes=(cls * a_lib).tolist(), cls_rel=(cls / eq).tolist(), nd=night_day(s, thr),
                a_lib=a_lib, nd_lib=rho, R_eff_lib_day=math.sqrt(ld / BW[pair][1].sum()),
                R_eff_new_day=math.sqrt(float((Vn[~IS_NIGHT] @ WCELL).mean()) / BW[pair][1].sum()),
                R_eff_lib_night=math.sqrt(ln / BW[pair][1].sum()),
                R_eff_new_night=math.sqrt(float((Vn[IS_NIGHT] @ WCELL).mean()) / BW[pair][1].sum()))


# =============================================================== run
VARIANTS = {
    "A": dict(rule="A", Ke=0.02, kdeep=0.06),
    "A_Ke2e-4": dict(rule="A", Ke=2e-4, kdeep=0.06),
    "A_Ke0.1": dict(rule="A", Ke=0.1, kdeep=0.06),
    "A_kdeep0.05": dict(rule="A", Ke=0.02, kdeep=0.05),
    "A_kdeep0.09": dict(rule="A", Ke=0.02, kdeep=0.09),
    "A_Lk30": dict(rule="A", Ke=0.02, kdeep=0.06, Lk=0.030),
    "B": dict(rule="B", Ke=0.02, kdeep=0.06),
    "B_Ke2e-4": dict(rule="B", Ke=2e-4, kdeep=0.06),
    "B_Ke0.1": dict(rule="B", Ke=0.1, kdeep=0.06),
    "B_RD1.0": dict(rule="B", Ke=0.02, kdeep=0.06, RD=1.0),
    "B_RD1.9": dict(rule="B", Ke=0.02, kdeep=0.06, RD=1.9),
    "A_Rnv0.2": dict(rule="A", Ke=0.02, kdeep=0.06, rnv=0.2),     # the library's own fish dark floor [J]
    "A_Rnv0.5": dict(rule="A", Ke=0.02, kdeep=0.06, rnv=0.5),     # Chinook low-light asymptote [S analogue]
}
qk = K["level"](1)
print("=" * 112)
print("0. Reproduction of the library derivations (library rule, measured light, 262 layer)")
print("=" * 112)
RULE.update(rule="lib", RD=1.2, Lk=None)
KR_REPRO = run_krill(dict(qk))
for g, r in KR_REPRO.items():
    lib = KRILL_LIB[g]
    print(f"  krill {g:18s} eq {r['eq']:.5g} (library {lib[0]}); classes " + " / ".join(f"{x:.4g}" for x in r["classes"])
          + "   library " + " / ".join(str(x) for x in lib[1]) + f"; night/day {r['nd']:.3f} (library {lib[2]})")
FISH_REPRO = {}
for name in FISH_PAIRS:
    r = fish_pair(name, library=True)
    FISH_REPRO[name] = r
    print(f"  {name:38s} 220 encounter a {r['a_enc_lib']:.5g} (library a {r['a_lib']}); equal-cell series check {r['eq']:.5g}")
print("  benthivore fish pairs: library a 0.0198 (243.3) / 0.021 (251) are re-derivations of benth_encounter.py; only")
print("  the ratio of its vision term is used (R_sat 2 L, R_nv 0.6 L, Ke 0.02, central light Kd_top 0.25 at the bed).")

print("\n" + "=" * 112)
print("1. R_sat per member (m): library / rule A / rule B (RD 1.2); krill L 25 mm; fish prey 15 cm")
print("=" * 112)
RS_TAB = []
for g, d in K["GROUPS"].items():
    for mem in d["members"]:
        if mem["vis"] in ("none",):
            continue
        if "bm" in mem:
            n, L = mem["bm"]
            Lm = float((n * L).sum() / n.sum())
        elif "m1" in mem:
            Lm = mem["m1"] / mem["nt"] if mem["m1"] > 0 else float("nan")
        else:
            Lm = mem["L"]
        if not np.isfinite(Lm):
            continue
        vals = []
        for rule in ("lib", "A", "B"):
            RULE.update(rule=rule, RD=1.2, Lk=None)
            rk = float(rsat_krill(qk, Lm))
            rf = 0.2 if rule == "lib" else max(float(rsat_fish(qk, Lm)), RNV_FISH)
            vals.append((rk, rf))
        eye = "non-visual" if mem["vis"] == "none_vis" else mem["vis"]
        RS_TAB.append(dict(group=g, member=mem["name"], L=Lm, eye=eye, vals=vals))
        print(f"  {g:18s} {mem['name']:28s} L {Lm:.3f} m ({eye:10s}): krill {vals[0][0]:.2f} / {vals[1][0]:.2f} / "
              f"{vals[2][0]:.2f};  fish (lib fixed dark 0.2-0.5) -> A {vals[1][1]:.2f} / B {vals[2][1]:.2f}"
              f"{'  [2L < R_nv fish 0.3: floor used]' if 2 * Lm < RNV_FISH else ''}")
RULE.update(rule="A", RD=1.2, Lk=None)
print(f"  rule A R0 on krill 25 mm |C0| 0.4: {r0_c(qk, 0.025, C_KRILL):.3f} m (30 mm {r0_c(qk, 0.030, C_KRILL):.3f}); "
      f"library |C0| 1: {r0_c(qk, 0.025, 1.0):.3f}; fish 15 cm |C0| 0.7: {r0_c(qk, 0.15, C_FISH):.2f} m")
print(f"  rule B krill factor (L/50 mm)^{B_MID:.3f} sqrt(0.4/0.7): 25 mm {(0.5) ** B_MID * math.sqrt(C_KRILL / C_FISH):.3f}, "
      f"30 mm {(0.6) ** B_MID * math.sqrt(C_KRILL / C_FISH):.3f}")

RES = {"repro": dict(krill=KR_REPRO, fish={k: dict(a_enc_lib=v["a_enc_lib"]) for k, v in FISH_REPRO.items()}),
       "variants": {}, "rsat": RS_TAB}
for vname, v in VARIANTS.items():
    RULE.update(rule=v["rule"], RD=v.get("RD", 1.2), Lk=v.get("Lk"))
    RNV_FISH = v.get("rnv", 0.3)
    q = dict(qk, Ke_gad=v["Ke"], k_deep=v["kdeep"])
    kr = run_krill(q)
    out = {"krill": {}, "fish": {}}
    for g, r in kr.items():
        rep = KR_REPRO[g]
        lib = KRILL_LIB[g]
        out["krill"][g] = dict(ratio=r["eq"] / rep["eq"], classes_lib_units=[c * lib[0] / rep["eq"] for c in r["classes"]],
                               class_ratio=[(c / c0 if c0 > 0 else float("nan")) for c, c0 in zip(r["classes"], rep["classes"])],
                               nd=r["nd"], nd_repro=rep["nd"], a_lib=lib[0], a_new=lib[0] * r["eq"] / rep["eq"])
    for name in FISH_PAIRS:
        r = fish_pair(name, Ke=v["Ke"], kdeep=v["kdeep"])
        r["a_new"] = r["a_lib"] * r["ratio"]
        out["fish"][name] = r
    for name in BENTH_PAIRS:
        r = benth_pair(name, v["rule"], v["Ke"], v["kdeep"])
        r["a_new"] = r["a_lib"] * r["ratio"]
        out["fish"][name] = r
    RES["variants"][vname] = out

print("\n" + "=" * 112)
print("2. a_new / a_library (equal-cell annual mean; library p). Krill: class ratios < 100 / 100-150 / 150-250 / > 250 m")
print("=" * 112)
hdr = list(VARIANTS)
print(f"  {'pair':40s} " + " ".join(f"{h:>11s}" for h in hdr))
for g in KRILL_LIB:
    print(f"  {g + '->krill':40s} " + " ".join(f"{RES['variants'][h]['krill'][g]['ratio']:11.3f}" for h in hdr))
for name in list(FISH_PAIRS) + list(BENTH_PAIRS):
    print(f"  {name:40s} " + " ".join(f"{RES['variants'][h]['fish'][name]['ratio']:11.3f}" for h in hdr))
print("\n  krill class ratios (new / library class a), rule A and B central:")
for g in KRILL_LIB:
    for h in ("A", "B"):
        cr = RES["variants"][h]["krill"][g]["class_ratio"]
        print(f"    {g:18s} {h}: " + " / ".join(f"{x:.3f}" for x in cr) + f"  night/day {RES['variants'][h]['krill'][g]['nd']:.3f}"
              f" (repro {RES['variants'][h]['krill'][g]['nd_repro']:.3f})")
print("\n  fish pairs: class a relative to the new equal-cell a (rule A / B central), new night/day vs library dark_ratio:")
for name in list(FISH_PAIRS) + list(BENTH_PAIRS):
    for h in ("A", "B"):
        r = RES["variants"][h]["fish"][name]
        print(f"    {name:38s} {h}: classes/eq " + " / ".join(f"{x:.2f}" for x in r["cls_rel"])
              + f"; night/day {r['nd']:.3f} (library {r['nd_lib']})")
print("\n  member shares of the new a (rule A central):")
for name in FISH_PAIRS:
    m = RES["variants"]["A"]["fish"][name]["members"]
    print(f"    {name:38s} " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(m.items(), key=lambda kv: -kv[1])))
for name in BENTH_PAIRS:
    r = RES["variants"]["A"]["fish"][name]
    print(f"    {name:38s} R_eff day lib {r['R_eff_lib_day']:.2f} -> A {r['R_eff_new_day']:.2f} m; night "
          f"{r['R_eff_lib_night']:.2f} -> {r['R_eff_new_night']:.2f} m (weighted over length classes)")

print("\n  decomposition of the in-layer fish pairs by prey sub-type and period (rule A central): share of the library"
      " a -> share of the new a (equal cells); x = new / library for that term; class 150-250 m term ratio")
for name in FISH_PAIRS:
    d = RES["variants"]["A"]["fish"][name]["decomp"]
    tl = sum(v["lib"] for v in d.values())
    tn = sum(v["new"] for v in d.values())
    print(f"    {name}")
    for k, v in sorted(d.items(), key=lambda kv: -kv[1]["new"]):
        if v["new"] / tn < 0.005 and v["lib"] / tl < 0.005:
            continue
        print(f"      {k:28s} lib {v['lib'] / tl:.3f} -> new {v['new'] / tn:.3f}  x{v['new'] / v['lib']:.2f}"
              f"  (150-250 m x{v['new_cls'][2] / v['lib']:.2f})")

# daytime light at the prey layers (central variant) for the record
print("\n3. Daytime-median light (lx) at each prey's layer centre, by class median bed (measured light, K_deep 0.06)")
Em = E0[:, None] * trans_month(dict(qk0, k_deep=0.06))[MONTH]
for pn in PROF:
    vals = []
    for (lab, lo, hi), idx in zip(CLASSES, CLS_IDX):
        b_idx = np.where(idx)[0]
        bmed = b_idx[np.argmin(np.abs(BEDS[b_idx] - np.median(K["DEPTH"][(K["DEPTH"] >= lo) & (K["DEPTH"] < hi)])))]
        ph = PHI[(pn, False)][:, bmed]
        e = (Em[~IS_NIGHT] * ph[None, :]).sum(1)
        vals.append(f"{lab} {np.median(e):.3g}")
    print(f"  {pn:22s} " + "; ".join(vals))
json.dump(RES, open(OUTJ, "w"), indent=1, default=float)
print(f"\nwritten {os.path.basename(OUTJ)}")
