"""Term-by-term decomposition of every water-column (predatory_fish) pair as the library has it.

Usage: python3 -I wc_terms.py <repo_root> <scratch_dir> <out_json>

Nothing is tuned and no model output is read. Two existing derivations are exec'd unchanged and their
factors are only read out:

  KRILL  research_notes/"Krill i Skagerrak"/wc_depth/scripts/krill_pairs_wc.py --variant field, with the
         section 253 measured light and the section 262 day layer / presence weighting copied from
         research_notes/"Krill over grunda bottnar" (o-umlaut)/rederive/scripts/krill_geometry.py (the source
         of the library class table 0.068 / 0.031 / 0.0058 / 0.00014). Band convention: a predator meets the
         krill concentration at its own depth; R^2 = R_nv^2 (1 - g) + R_sat^2 g, g = E/(E + Ke).
  FISH   research_notes/"Rovfiskens uppdelning"/scripts/split_encounter_223.py head, which exec's
         research_notes/"Attackrater ur motesteori" (o-umlaut)/scripts/pred_encounter_a.py (section 220):
         in-layer a3d = scan A(R, H) v_rel T N_t p f / H / 1e6 s, D_DAY 0.5, members whose capability score
         is >= 0.1 ("layer"), per tonne of the WHOLE rest water column (G3_rest_wc). Planktivores on
         dispersed prey with the 243.3 pool shares; benthivores x the 243 member overlaps O_m; whiting
         (H 10 m, no overlap factor); mackerel = the planktivore geometry over the eaters only.

Factorisation used for every pair, per member k (and per bed class c for the krill pair):
    a_k = [scan pi p T / 1e6] x [N_t v_rel]_day x O_eff x R2_eff   (x s for a pool share)
  O_eff  = time-mean overlap density (1/m) = sum_z p_pred(z) phi_prey(z) (band) or 1/H (in-layer),
           night rows weighted by the night speed ratio;
  R2_eff = the overlap- and time-weighted mean of R^2 (m^2).
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
OUTJ = sys.argv[3]
RN = os.path.join(ROOT, "research_notes")
ARGV0 = list(sys.argv)

# =============================================================== KRILL machinery (exec, unchanged)
WC = os.path.join(RN, "Krill i Skagerrak", "wc_depth", "scripts", "krill_pairs_wc.py")
src = open(WC).read()
head = src[:src.index("\n# ================================================================ central\n")]
sys.argv = [WC, os.path.join(ROOT, "maps", "depth.npz"),
            os.path.join(RN, "Fiskgrupper i rutan", "benthivores", "tables", "length_biomass_longrun.csv"),
            "10", "--variant", "field"]
K = {"__name__": "krill_pairs_wc_head", "__file__": WC}
exec(compile(head, WC, "exec"), K)
sys.argv = ARGV0
ZG, ELEV, MONTH, E0 = K["ZG"], K["ELEV"], K["MONTH"], K["E0"]
BEDS, WCELL, CLS_IDX, CLASSES = K["BEDS"], K["WCELL"], K["CLS_IDX"], K["CLASSES"]
T = K["T"]

# ---- section 253 measured light and section 262 layer / presence (copied from krill_geometry.py)
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
WT = WPRES[MONTH]                       # presence weight over < 100 m beds, by time row
qc = meas(K["level"](1))
LIB_KRILL = [0.068, 0.031, 0.0058, 0.00014]
is_night = ELEV <= K["SUN_KRILL"]
FDAY = float((~is_night).mean())


def class_mean(x_t_b, k, presence=True):
    """annual mean over class k of a (time, bed) array; class 0 presence-weighted as krill_geometry."""
    idx = CLS_IDX[k]
    s = x_t_b[:, idx] @ WCELL[idx] / WCELL[idx].sum()
    if k == 0 and presence:
        return float((s * WT).sum() / WT.sum())
    return float(s.mean())


# ---- generic band evaluator (the krill_pairs_wc evaluate loop, per member, components kept)
def band_member(mem, q, prey_prof, R_sat, R_nv, u_prey, p, vis="gadoid"):
    """returns dict of (time, bed) arrays: a, base S1_day O (R^2 = 1 m^2 normaliser), O_day, O_night."""
    gmat = K["light_g"](q, vis)
    seasons = [(np.ones(len(ELEV), bool), mem["prof"])]
    if mem.get("prof_summer") is not None:
        summer = (MONTH >= 5) & (MONTH <= 8)
        seasons = [(~summer, mem["prof"]), (summer, mem["prof_summer"])]
    vb = q[mem["vkey"]]
    S1d = mem["nt"] * float(K["vrel"](vb * mem["L"], u_prey))
    vbn = vb * (q[mem["night_v"]] if "night_v" in mem else 1.0)
    S1n = mem["nt"] * float(K["vrel"](vbn * mem["L"], u_prey))
    base = q["scan"] * math.pi * p * T / 1e6
    a = np.zeros((len(ELEV), len(BEDS)))
    norm = np.zeros_like(a)
    Od = np.zeros_like(a)
    for smask, prof in seasons:
        for night in (False, True):
            pp = np.array([prof(b, q, night) for b in BEDS]).T
            ph = np.array([prey_prof(b, q, night) for b in BEDS]).T
            w = pp * ph
            o = w.sum(axis=0)
            rows = (is_night if night else ~is_night) & smask
            S1 = S1n if night else S1d
            if vis == "none":
                og = np.zeros((rows.sum(), len(BEDS)))
            else:
                og = gmat[rows] @ w
            a[rows] = base * S1 * (R_nv ** 2 * (o[None, :] - og) + R_sat ** 2 * og)
            norm[rows] = base * S1d * o[None, :] * (S1 / S1d)
            Od[rows] = o[None, :]
    return dict(a=a, norm=norm, O=Od, S1d=S1d, S1n=S1n, base=base)


def r0(q, L):
    r00 = q["k_v"] * math.sqrt(0.75 * L * 0.2 * L)
    r = r00
    for _ in range(60):
        r = r00 * math.exp(-q["c_beam"] * r / 2.0)
    return r


WCM = [m for m in K["GROUPS"]["predatory_fish"]["members"]]
print("=" * 112)
print("A. Krill pair (water column), library derivation (section 245 / 253 / 262), central")
print("=" * 112)
krill_prof = K["krill_profile"]
R0K = r0(qc, qc["L_krill"])
print(f"R_sat on krill = min(R0 {R0K:.3f} m (k_v {qc['k_v']}, L {qc['L_krill'] * 1000:.0f} mm, beam c {qc['c_beam']}), "
      f"cap {qc['cap_L']} L_pred); R_nv {qc['R_nv']} m; p {qc['p_cap']}; scan {qc['scan']}; u_krill {qc['u_krill']} m/s; "
      f"Ke_gad {qc['Ke_gad']} lx; day share of the year (sun > {K['SUN_KRILL']} deg) {FDAY:.3f}")
KM = {}
tot = np.zeros((len(ELEV), len(BEDS)))
for mem in WCM:
    if mem["vis"] == "none" or mem.get("nt", 0) <= 0:
        continue
    vis = "none" if mem["vis"] == "none_vis" else "gadoid"
    rs = min(R0K, qc["cap_L"] * mem["L"])
    d = band_member(mem, qc, krill_prof, rs, qc["R_nv"], qc["u_krill"], qc["p_cap"], vis)
    d["R_sat"] = rs
    KM[mem["name"]] = (mem, d)
    tot += mem["W"] * d["a"]
cls_k = [class_mean(tot, k) for k in range(4)]
print("reproduction, class a: " + " / ".join(f"{x:.4g}" for x in cls_k) + "   library " +
      " / ".join(str(x) for x in LIB_KRILL))
shares_cells = np.array([WCELL[i].sum() for i in CLS_IDX])
print(f"equal-cell a {float(shares_cells @ np.array(cls_k)):.4g} (library 0.026)")

print("\nPer member, per class: W | share of the class a | O_eff (1/m, x1000) | R_eff (m) | night share of a")
KREC = {}
for name, (mem, d) in KM.items():
    rec = []
    for k in range(4):
        am = class_mean(d["a"], k)
        nm = class_mean(d["norm"], k)
        om = class_mean(d["O"] * np.where(is_night, d["S1n"] / d["S1d"], 1.0)[:, None], k)
        an = class_mean(np.where(is_night[:, None], d["a"], 0.0), k)
        r2 = am / nm if nm > 0 else 0.0
        rec.append(dict(a=am, share=mem["W"] * am / cls_k[k] if cls_k[k] > 0 else 0.0, O=om, R2=r2,
                        night=an / am if am > 0 else 0.0))
    KREC[name] = dict(W=mem["W"], R_sat=d["R_sat"], S1d=d["S1d"], S1n=d["S1n"], classes=rec)
    print(f"  {name:22s} W {mem['W']:.3f} R_sat {d['R_sat']:.2f} N_t v_rel(day) {d['S1d']:8.1f}: " +
          " | ".join(f"{r['share']:.2f} {1000 * r['O']:6.2f} {math.sqrt(max(r['R2'], 0)):.2f} {r['night']:.2f}"
                     for r in rec))

# krill layer and light at each class's median bed
print("\nKrill day layer and daytime light at the layer, at each class's median bed (measured light):")
light = K["lux"]("meas")
LAY = []
for (lab, lo, hi), idx in zip(CLASSES, CLS_IDX):
    dep = K["DEPTH"]
    b = float(np.median(dep[(dep >= lo) & (dep < hi)]))
    t_, b_ = new_layer(b, qc)
    zm = int(0.5 * (t_ + b_))
    dl = np.median(light[~is_night, zm])
    g = np.median(light[~is_night, zm] / (light[~is_night, zm] + qc["Ke_gad"]))
    LAY.append(dict(cls=lab, bed=b, top=t_, bot=b_, above_bed=b - b_, lux_mid=float(dl), g_mid=float(g)))
    print(f"  {lab:10s} median bed {b:5.0f} m: layer {t_:5.1f}-{b_:5.1f} m ({b - b_:4.0f} m above bed); "
          f"daytime median light at its centre {dl:.3g} lx, gadoid g {g:.2f}")

# =============================================================== FISH machinery (exec, unchanged)
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
GP["PLANK_SHARE"].update({"herring": 0.554, "blue whiting": 0.075, "pout": 0.350, "sprat": 0.021})
STG_T, stage_pair, sc = GE["STG_T"], GE["stage_pair"], GE["sc"]
qf = GE["level"](1)
REST = ["saithe", "spurdog", "hake", "anglerfish", "ling", "velvet belly", "pollack"]
EATERS = ["spurdog", "hake", "anglerfish", "ling", "velvet belly"]
OVL_BENTH = {"saithe": 0.05, "pollack": 0.05, "spurdog": 0.15, "hake": 0.25, "velvet belly": 0.10, "anglerfish": 1.0,
             "ling": 0.25}
W_rest = sum(r.W for r in STG_T if r.sp in REST)


def fish_group(prey_type, members, overlap=None):
    """group a (annual, 0.5 day + 0.5 night), per tonne of the whole rest group, layer membership."""
    num = 0.0
    parts = {}
    for r in STG_T:
        if r.sp not in members:
            continue
        s = sc(r, prey_type)
        if s < 0.1 and prey_type != "whiting_prey":
            continue
        if prey_type == "whiting_prey" and s < 0.1:
            continue
        ad, an = stage_pair(r, prey_type, qf, False), stage_pair(r, prey_type, qf, True)
        a = 0.5 * ad + 0.5 * an
        if overlap is not None:
            a *= overlap.get(r.sp, 0.0)
        num += r.W * a
        parts[(r.sp, r.st)] = r.W * a
    return num / W_rest, {k: v / W_rest for k, v in parts.items()}


a_plk, parts_plk = fish_group("plank_fish", REST)
a_ben_inlayer, parts_bi = fish_group("benth_fish", REST)
a_ben, parts_ben = fish_group("benth_fish", REST, OVL_BENTH)
a_whi, parts_whi = fish_group("whiting_prey", REST)
a_mac_raw, parts_mac = fish_group("plank_fish", EATERS)
print("\n" + "=" * 112)
print("B. Fish pairs (section 220 in-layer encounter, rest water column, per whole-group tonne), central")
print("=" * 112)
print(f"  planktivores (243.3 pool shares, dispersed prey): {a_plk:.4g}  (library 0.0060)")
print(f"  benthivores in-layer {a_ben_inlayer:.4g} (243: 0.0651) x member overlaps -> {a_ben:.4g} (library 0.0055)")
print(f"  whiting (H 10 m, no overlap factor): {a_whi:.4g}  (library 0.013)")
print(f"  mackerel prey (planktivore geometry, eaters only; per rest tonne): {a_mac_raw:.4g} (library 0.00088 after"
      f" the 228 split-script's own pool shares; geometry only, not re-derived here)")

# saithe adult: the per-prey-type in-layer terms
r_sa = [r for r in STG_T if r.sp == "saithe" and r.st == "adult"][0]
L = r_sa.L
v = qf["v_saithe"] * L
vn = v * qf["nv_saithe"]
vrel = GP["vrel"]
print(f"\nSaithe < 60 cm (L {L:.3f} m, N/t {r_sa.Nt:.0f}, v {v:.3f} m/s day, {vn:.3f} night), in-layer terms per prey type:")
FT = {}
rows = [
    ("herring+sprat day", min(qf["R_fish_day_cruiser"], 2 * L), qf["H_herring_day"], qf["u_herring"], v, "day"),
    ("herring+sprat night", qf["R_fish_dark"], qf["H_herring_night"], qf["u_herring"], vn, "night"),
    ("pout (day and night)", qf["R_fish_dark"], qf["H_pout"], qf["u_smallfish"], None, "both"),
    ("blue whiting (day and night)", qf["R_fish_dark"], qf["H_bluewhiting"], qf["u_smallfish"], None, "both"),
    ("juvenile benthivores day", max(qf["R_bedfish_day"], qf["R_fish_dark"]), qf["H_juv"], qf["u_smallfish"], v, "day"),
    ("juvenile benthivores night", qf["R_fish_dark"], qf["H_juv"], qf["u_smallfish"], vn, "night"),
    ("whiting day", max(qf["R_bedfish_day"], qf["R_fish_dark"]), qf["H_whiting"], qf["u_smallfish"], v, "day"),
    ("whiting night", qf["R_fish_dark"], qf["H_whiting"], qf["u_smallfish"], vn, "night"),
]
for name, R, H, u, vv, when in rows:
    if vv is None:
        vals = [(R, H, u, v), (R, H, u, vn)]
    else:
        vals = [(R, H, u, vv)]
    for R_, H_, u_, v_ in vals:
        A = GP["area_in_slab"](R_, H_)
        FT.setdefault(name, []).append(dict(R=R_, H=H_, O=1.0 / H_, Reff2=A / math.pi, vrel=vrel(v_, u_)))
    d = FT[name][0]
    print(f"  {name:30s} R {d['R']:.2f} m  H {d['H']:5.1f} m  O = 1/H {1000 / d['H']:6.2f} e-3/m  "
          f"A/pi {d['Reff2']:.3f} m2  v_rel {d['vrel']:.3f} m/s  p {qf['p_fish']}")

# =============================================================== band O and one-vision R for the fish prey (saithe)
print("\n" + "=" * 112)
print("C. The same saithe < 60 cm met in the krill pair's convention (band O, one vision function, measured light)")
print("=" * 112)
mem_sa = [m for m in WCM if m["name"] == "saithe < 60 cm"][0]
PREY = {
    "krill": (krill_prof, R0K, qc["R_nv"], qc["u_krill"]),
    "pout": (K["pout_prof"], r0(qc, 0.15), 0.2, 0.1),
    "herring": (K["herring_prof"], r0(qc, 0.15), 0.2, 0.1),
    "blue whiting": (lambda b, qq, n: 0.787 * K["bw_prof"](False)(b, qq, n) + 0.213 * K["bw_prof"](True)(b, qq, n),
                     r0(qc, 0.15), 0.2, 0.1),
    "juvenile benthivores (bottom 2 m)": (lambda b, qq, n: K["uniform"](b - 2.0, b), r0(qc, 0.15), 0.2, 0.1),
    "whiting (bottom 10 m)": (lambda b, qq, n: K["uniform"](b - 10.0, b), r0(qc, 0.15), 0.2, 0.1),
}
CREC = {}
for pn, (pf, R0p, Rnv, u) in PREY.items():
    rs = min(R0p, qc["cap_L"] * mem_sa["L"])
    d = band_member(mem_sa, qc, pf, rs, Rnv, u, 1.0)
    rec = []
    for k in range(4):
        pres = pn == "krill"
        am = class_mean(d["a"], k, pres)
        nm = class_mean(d["norm"], k, pres)
        om = class_mean(d["O"] * np.where(is_night, d["S1n"] / d["S1d"], 1.0)[:, None], k, pres)
        od = class_mean(np.where(is_night[:, None], 0.0, d["O"]), k, pres) / FDAY
        onn = class_mean(np.where(is_night[:, None], d["O"], 0.0), k, pres) / (1 - FDAY)
        rec.append(dict(a_p1=am, O=om, O_day=od, O_night=onn, R2=am / nm if nm > 0 else 0.0))
    CREC[pn] = dict(R_sat=rs, R_nv=Rnv, classes=rec)
    print(f"  {pn:34s} R_sat {rs:.2f} R_nv {Rnv:.2f}: O day/night x1000 by class " +
          " | ".join(f"{1000 * r['O_day']:.2f}/{1000 * r['O_night']:.2f}" for r in rec) +
          "; R_eff " + " / ".join(f"{math.sqrt(max(r['R2'], 0)):.2f}" for r in rec))

json.dump(dict(krill_classes=cls_k, krill_members=KREC, layer=LAY, fday=FDAY,
               fish=dict(planktivores=a_plk, benth_inlayer=a_ben_inlayer, benth=a_ben, whiting=a_whi,
                         mackerel_geom=a_mac_raw,
                         parts_plk={f"{k[0]}|{k[1]}": v for k, v in parts_plk.items()},
                         parts_ben={f"{k[0]}|{k[1]}": v for k, v in parts_ben.items()},
                         parts_whi={f"{k[0]}|{k[1]}": v for k, v in parts_whi.items()}),
               fish_terms_saithe=FT, saithe_band=CREC,
               saithe_adult=dict(L=L, Nt=r_sa.Nt, v=v, vn=vn), qf={k: qf[k] for k in qf},
               qk={k: (qc[k] if not isinstance(qc[k], str) else qc[k]) for k in qc}),
          open(OUTJ, "w"), indent=1, default=float)
