"""The fish-predator pairs on the BAND overlap (one time-at-depth density per predator member, the same one its
krill pair uses), with or without the one vision rule (rule A): a_new / a_library per pair, class a's,
night/day, and the overlap geometry (credited time per prey layer).

Usage: python3 -I band_pairs.py <repo_root> <scratch_dir> <out_json>

Nothing is fitted and no model output is read.

Base: research_notes/"En synregel for fiskparen"/scripts/one_rule_pairs.py is exec'd up to its own "run" block,
so every factor of every section 220 / 243.3 / 251 derivation (members, stages, N/t, speeds, scan, capture p,
capability membership, pool shares, R by predator type, the krill machinery with the section 253 light and the
section 262 krill layer) is the one the library a's came from. ONE thing changes per in-layer fish term:

    library (in-layer)  a_term = k_term x A(R, H) / H                     (a3d: predator credited with all of its
                                                                          time inside prey layer H)
    band                a_term = k_term x sum_z f_m(z) phi_j(z) A_R(z, t)  (f_m: the member's time-at-depth
                                                                          density, sum 1; phi_j: prey profile)
    k_term = scan v_rel T N_t p s / 1e6 (unchanged), A_R = pi R^2 (point form) or, for prey in a slab at the bed
    (juvenile benthivores, whiting as prey), the disc area inside the slab averaged over the predator's height
    (slab clipping, the band analogue of area_in_slab).

f_m: the krill pairs' member profiles (research_notes/"Krill i Skagerrak"/wc_depth/scripts/krill_pairs_wc.py,
--variant field): saithe and pollack exp(15 m) above the bed by day, 0.07 in 0-30 m + exp(10.5 m) at night;
hake exp(8 m) by day, bottom 80 m (not above 30 m) at night; spurdog Nov-May 162-231 m (bottom 20 m over
beds < 182 m) by day, 37.5-88 m at night, Jun-Sep 0-25 m; velvet belly bottom 30 m over beds >= 150 m;
whiting uniform bottom 10 m; mackerel 0-40 m day / 0-50 m night; horse mackerel bottom 30 m day / 0-50 m
night. Two members take no krill and had no krill profile [J here, tagged]: anglerfish on the bed (bottom 1 m;
Ofstad et al. 2022: 95 % of the time inactive on the bed), ling exp(5 m) above the bed (demersal).
phi_j: the prey profiles of one_rule_pairs.py (krill_pairs_wc prey profiles; juvenile benthivores uniform
bottom H_juv 2 m, whiting uniform bottom H_whiting 10 m, mackerel as the mackerel predator profile).

Vision: "lib" = each term's own library R (fixed); "A" = the one rule (rule A of one_rule_pairs.py: R^2 =
R_nv^2 (1 - g) + R_sat^2 g with the light AT THE OVERLAP DEPTH, R_sat = min(R0(15 cm fish, |C0| 0.7), 2 L),
R_nv fish 0.3 m; clupeid eye Ke 1 lx + biting threshold; non-visual members R 0.3 m; anglerfish strike 0.4 m).
The section 243 time-near-bed factors (OVL_BENTH) are DROPPED on the band: the profile carries them.

Benthivore fish pairs: benth_encounter.py is already the band form (pi R^2 v p F_j / H_band: predator uniform
in its 10 m band, F_j = prey share in the band); only rule A's vision ratio applies (one_rule_pairs.benth_pair).
Krill pairs: already band; rule A via one_rule_pairs.run_krill with the same q (so profile variants move the
krill pair and the fish pairs together).
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
OUTJ = os.path.abspath(sys.argv[3])
ORP = os.path.join(ROOT, "research_notes", "En synregel for fiskparen", "scripts", "one_rule_pairs.py")
src = open(ORP).read()
head = src[:src.index("# =============================================================== run")]
ARGV_SAVE = list(sys.argv)
sys.argv = [ORP, ROOT, SCR, os.path.join(SCR, "_unused.json")]
O = {"__name__": "one_rule_pairs_head", "__file__": ORP}
exec(compile(head, ORP, "exec"), O)
sys.argv = ARGV_SAVE

K, GE, GP = O["K"], O["GE"], O["GP"]
ZG, ELEV, MONTH, BEDS, WCELL, CLS_IDX = O["ZG"], O["ELEV"], O["MONTH"], O["BEDS"], O["WCELL"], O["CLS_IDX"]
IS_NIGHT, STG_T, terms, sc = O["IS_NIGHT"], O["STG_T"], O["terms"], O["sc"]
area_in_slab, gmat, rsat_fish, RULE = O["area_in_slab"], O["gmat"], O["rsat_fish"], O["RULE"]
FISH_PAIRS, WMUL, REST, EATERS, qf, qk0 = O["FISH_PAIRS"], O["WMUL"], O["REST"], O["EATERS"], O["qf"], O["qk0"]
night_day, run_krill, benth_pair, KRILL_LIB = O["night_day"], O["run_krill"], O["benth_pair"], O["KRILL_LIB"]
uniform, expbed = K["uniform"], K["expbed"]
NB = len(BEDS)
SUMMER = (MONTH >= 5) & (MONTH <= 8)
RNV_FISH = 0.3


# ------------------------------------------------------------------ predator time-at-depth densities f_m(z)
def angler(b, q, n):
    return uniform(b - 1.0, b)                      # [J] Ofstad et al. 2022: inactive on the bed


def ling(b, q, n):
    return expbed(b, 5.0)                           # [J] demersal; no field profile found


PRED_PROF = {"saithe": K["saithe_field"], "pollack": K["saithe_field"], "hake": K["hake_field"],
             "spurdog": K["spurdog_field"], "velvet belly": K["deep_band"], "anglerfish": angler, "ling": ling,
             "whiting": K["whiting_prof"], "mackerel": K["mack_prof"], "horse mackerel": K["hmack_prof"]}
SEASONAL = {"spurdog": lambda b, q, n: K["spurdog_field"](b, q, n, True)}


def prey_prof(name, q):
    small, large = K["bw_prof"](False), K["bw_prof"](True)
    return {"herring": K["herring_prof"], "sprat": K["herring_prof"], "pout": K["pout_prof"],
            "blue whiting": lambda b, qq, n: 0.787 * small(b, qq, n) + 0.213 * large(b, qq, n),
            "juvenile benthivores": lambda b, qq, n: uniform(b - qf["H_juv"], b),
            "whiting": lambda b, qq, n: uniform(b - qq["H_whiting"], b),
            "mackerel": K["mack_prof"]}[name]


SLAB = {"juvenile benthivores": lambda q: qf["H_juv"], "whiting": lambda q: q["H_whiting"]}
PCACHE = {}


def pmat(kind, name, q, night, season="all"):
    key = (kind, name, night, season, q.get("Hs_saithe"), q.get("f_up_saithe"), q.get("Hn_ratio"), q.get("H_whiting"))
    if key not in PCACHE:
        if kind == "pred":
            fn = SEASONAL[name] if season == "summer" else PRED_PROF[name]
        else:
            fn = prey_prof(name, q)
        PCACHE[key] = np.array([fn(b, q, night) for b in BEDS]).T          # (n_z, n_b), columns sum to 1 (or 0)
    return PCACHE[key]


# ------------------------------------------------------------------ disc area inside a bed slab
def seg(R, y0, y1):
    """area of a disc of radius R (array) between y0 and y1 (offsets from the centre), clipped to [-R, R]."""
    a = np.clip(y0, -R, R)
    b = np.clip(y1, -R, R)

    def F(y):
        return y * np.sqrt(np.maximum(R * R - y * y, 0.0)) + R * R * np.arcsin(np.clip(y / np.maximum(R, 1e-12), -1, 1))
    return np.maximum(F(b) - F(a), 0.0)


NSUB = 10
IZ_BED = {}


def slab_series(pp, H, R_of_iz, rows_n):
    """sum over bins near the bed of f(bin) x mean_h A(R, slab [0, H]) / H; R_of_iz(iz) -> (rows, n_b) or scalar."""
    out = np.zeros((rows_n, NB))
    nbin = int(math.ceil(H + 2.0))
    for j in range(nbin):                            # bin j spans heights [j, j + 1] above the bed
        iz = np.clip(np.round(BEDS - j - 0.5 - 0.5).astype(int), 0, len(ZG) - 1)    # ZG[iz] = bed - j - 0.5
        f = pp[iz, np.arange(NB)]                    # (n_b,)
        if not np.any(f > 0):
            continue
        R = R_of_iz(iz)
        acc = 0.0
        for s in range(NSUB):
            h = j + (s + 0.5) / NSUB
            acc = acc + seg(R, -h, H - h)
        out += f[None, :] * (acc / NSUB) / H
    return out


def point_series(W, rn2, rs2, g_rows):
    """pi sum_z W(z) R^2(z, t): W = f phi (n_z, n_b); g_rows (rows, n_z) or None (fixed R: rn2 = rs2)."""
    o = W.sum(axis=0)
    if g_rows is None:
        return np.pi * rn2 * np.broadcast_to(o, (1, NB))
    og = g_rows @ W
    return np.pi * (rn2 * (o[None, :] - og) + rs2 * og)


def band_fish_pair(name, q, vision, Ke=0.02, kdeep=0.06, clip=True):
    members, pt, subset, lightprey, ovl, field, a_lib, nd_lib, thr = FISH_PAIRS[name]
    den = 0.0
    a_enc_lib = 0.0
    ser = np.zeros((len(ELEV), NB))
    contrib = {}
    for r in STG_T:
        if r.sp not in members:
            continue
        w = r.W * (WMUL.get(r.sp, 1.0) if field else 1.0)
        den += w
        if sc(r, pt) < 0.1:
            continue
        m_lib = (ovl or {}).get(r.sp, 1.0) if ovl is not None else 1.0
        seasons = [("all", np.ones(len(ELEV), bool))]
        if r.sp in SEASONAL:
            seasons = [("all", ~SUMMER), ("summer", SUMMER)]
        for t in terms(r, pt, qf, subset):
            a_enc_lib += 0.5 * w * m_lib * t["a"]
            k = t["a"] / (area_in_slab(t["R"], t["H"]) / t["H"])           # scan v_rel T N_t p f s / 1e6
            prey = lightprey or t["prey"]
            for sname, smask in seasons:
                rows = (IS_NIGHT if t["night"] else ~IS_NIGHT) & smask
                if not rows.any():
                    continue
                pp = pmat("pred", r.sp, q, t["night"], sname)
                ph = pmat("prey", prey, q, t["night"])
                fixed = vision == "lib" or t["eye"] in ("fixed", "none")
                if fixed:
                    Rfix = t["R"] if (vision == "lib" or t["eye"] == "fixed") else qf["R_fish_elasmo"]
                    rn2 = rs2 = Rfix ** 2
                    g_rows = None
                else:
                    rs = max(float(rsat_fish(qk0, r.L)), RNV_FISH)
                    rn2, rs2 = RNV_FISH ** 2, rs ** 2
                    g_rows = gmat(t["eye"], Ke, kdeep)[rows]
                if clip and prey in SLAB:
                    H = SLAB[prey](q)
                    if g_rows is None:
                        val = slab_series(pp, H, lambda iz: np.full((1, NB), math.sqrt(rn2)), 1)
                    else:
                        G = g_rows

                        def R_of(iz, G=G, rn2=rn2, rs2=rs2):
                            gg = G[:, iz]
                            return np.sqrt(rn2 * (1 - gg) + rs2 * gg)
                        val = slab_series(pp, H, R_of, int(rows.sum()))
                else:
                    val = point_series(pp * ph, rn2, rs2, g_rows)
                ser[rows] += w * k * val
                contrib[(r.sp, r.st)] = contrib.get((r.sp, r.st), 0.0) + float((w * k * val @ WCELL).mean()) * rows.mean()
    if members is EATERS:
        den = sum(r.W for r in STG_T if r.sp in REST)
    ser /= den
    a_enc_lib /= den
    s = ser @ WCELL
    cls = np.array([float((ser[:, idx] @ WCELL[idx]).mean() / WCELL[idx].sum()) for idx in CLS_IDX])
    eq = float(s.mean())
    tot = sum(contrib.values()) or 1.0
    mon = [float(s[MONTH == m].mean()) for m in range(12)]
    return dict(a_enc_lib=a_enc_lib, eq=eq, ratio=eq / a_enc_lib, classes=cls.tolist(), cls_rel=(cls / eq).tolist(),
                nd=night_day(s, thr), nd0=night_day(s, 0.0), a_lib=a_lib, nd_lib=nd_lib, month_rel=[x / eq for x in mon],
                members={f"{k[0]}|{k[1]}": v / tot for k, v in contrib.items() if v / tot >= 0.005})


# ================================================================== 0. overlap geometry
print("=" * 112)
print("0. Overlap geometry (no vision): tau_j = O_band / O_in-layer = H_j sum_z f(z) phi_j(z) day/night, and in brackets")
print("   the member's time share inside prey j's support (phi_j > 0) day/night; in-layer credits 1 to EVERY layer.")
print("   Equal-cell means; day / night (sun 0 deg switch). H_j = the section 220 in-layer thickness.")
print("=" * 112)
q0 = dict(qk0)
H_INLAYER = {"herring": (qf["H_herring_day"], qf["H_herring_night"]), "pout": (qf["H_pout"],) * 2,
             "blue whiting": (qf["H_bluewhiting"],) * 2, "juvenile benthivores": (qf["H_juv"],) * 2,
             "whiting": (qf["H_whiting"],) * 2, "mackerel": (40.0, 50.0)}
GEO = {}
for sp in ["saithe", "hake", "spurdog", "velvet belly", "anglerfish", "ling", "whiting", "mackerel", "horse mackerel"]:
    row = {}
    for prey in H_INLAYER:
        vals = []
        for night in (False, True):
            if sp in SEASONAL:
                o = (~SUMMER).mean() * (pmat("pred", sp, q0, night) * pmat("prey", prey, q0, night)).sum(0) \
                    + SUMMER.mean() * (pmat("pred", sp, q0, night, "summer") * pmat("prey", prey, q0, night)).sum(0)
            else:
                o = (pmat("pred", sp, q0, night) * pmat("prey", prey, q0, night)).sum(0)
            vals.append(float(o @ WCELL) * H_INLAYER[prey][int(night)])
            # time share of the member inside the prey's support (phi > 0)
            def share(season):
                f = pmat("pred", sp, q0, night, season)
                ph = pmat("prey", prey, q0, night)
                return float(((f * (ph > 0)).sum(0)) @ WCELL)
            vals.append(share("all") if sp not in SEASONAL else
                        (~SUMMER).mean() * share("all") + SUMMER.mean() * share("summer"))
        row[prey] = vals
    GEO[sp] = row
    print(f"  {sp:15s} " + "; ".join(f"{p[:12]} {v[0]:.2f}/{v[2]:.2f} [{v[1]:.2f}/{v[3]:.2f}]" for p, v in row.items())
          + f";  sum of time shares day {sum(v[1] for v in row.values()):.2f} / night {sum(v[3] for v in row.values()):.2f}"
          f" (in-layer credits {len(row)})")
for sp, hs in (("saithe", 5.0), ("saithe", 15.0), ("saithe", 40.0)):
    qq = dict(q0, Hs_saithe=hs)
    PCACHE.clear()
    f = pmat("pred", sp, qq, False)
    near2 = []
    near10 = []
    for ib, b in enumerate(BEDS):
        zb = ZG > b - 2.0
        near2.append(f[zb & (ZG < b), ib].sum())
        near10.append(f[(ZG > b - 10.0) & (ZG < b), ib].sum())
    print(f"  saithe day profile Hs {hs:4.0f} m: time within 2 m of the bed {np.array(near2) @ WCELL:.3f}, within 10 m "
          f"{np.array(near10) @ WCELL:.3f} (243 judged 'foraging time within ~2 m' 0.05, 0.02-0.15)")
PCACHE.clear()

# ================================================================== variants
VAR = {
    "lib_band": dict(vision="lib"),
    "A_band": dict(vision="A"),
    "A_band_noclip": dict(vision="A", clip=False),
    "A_band_Hs5": dict(vision="A", q=dict(Hs_saithe=5.0)),
    "A_band_Hs40": dict(vision="A", q=dict(Hs_saithe=40.0)),
    "A_band_fup0.25": dict(vision="A", q=dict(f_up_saithe=0.25)),
    "A_band_Hn1.0": dict(vision="A", q=dict(Hn_ratio=1.0)),
    "A_band_Hw5": dict(vision="A", q=dict(H_whiting=5.0)),
    "A_band_Hw30": dict(vision="A", q=dict(H_whiting=30.0)),
    "A_band_Ke2e-4": dict(vision="A", Ke=2e-4),
    "A_band_Ke0.1": dict(vision="A", Ke=0.1),
}
KR_REPRO = None
RULE.update(rule="lib", RD=1.2, Lk=None)
KR_REPRO = run_krill(dict(K["level"](1)))
RES = {"variants": {}, "geometry": GEO}
for vn, v in VAR.items():
    Ke = v.get("Ke", 0.02)
    kdeep = v.get("kdeep", 0.06)
    qv = dict(qk0, **v.get("q", {}))
    out = {"krill": {}, "fish": {}}
    rule = "A" if v["vision"] == "A" else "lib"
    RULE.update(rule=rule, RD=1.2, Lk=None)
    kr = run_krill(dict(K["level"](1), Ke_gad=Ke, k_deep=kdeep, **v.get("q", {})))
    for g, r in kr.items():
        rep = KR_REPRO[g]
        lib = KRILL_LIB[g]
        out["krill"][g] = dict(ratio=r["eq"] / rep["eq"], classes_lib_units=[c * lib[0] / rep["eq"] for c in r["classes"]],
                               nd=r["nd"], a_lib=lib[0],
                               a_new=lib[0] * r["eq"] / rep["eq"])
    for name in FISH_PAIRS:
        r = band_fish_pair(name, qv, v["vision"], Ke, kdeep, v.get("clip", True))
        r["a_new"] = r["a_lib"] * r["ratio"]
        out["fish"][name] = r
    for name in O["BENTH_PAIRS"]:
        r = benth_pair(name, "A" if v["vision"] == "A" else "lib", Ke, kdeep)
        r["a_new"] = r["a_lib"] * r["ratio"]
        out["fish"][name] = r
    RES["variants"][vn] = out
    print(f"{vn}: done")

print("\n" + "=" * 112)
print("1. a_new / a_library (equal-cell annual mean, library p); in-layer -> band. Krill pairs: rule A on the band"
      " (lib_band: library vision)")
print("=" * 112)
hdr = list(VAR)
print(f"  {'pair':40s} " + " ".join(f"{h[:13]:>13s}" for h in hdr))
for g in KRILL_LIB:
    print(f"  {g + '->krill':40s} " + " ".join(f"{RES['variants'][h]['krill'][g]['ratio']:13.3f}" for h in hdr))
for name in list(FISH_PAIRS) + list(O["BENTH_PAIRS"]):
    print(f"  {name:40s} " + " ".join(f"{RES['variants'][h]['fish'][name]['ratio']:13.3f}" for h in hdr))

print("\n2. A_band central: new a, class a / equal-cell a (< 100 / 100-150 / 150-250 / > 250 m), night/day at the pair's"
      " threshold (and at 0 deg), member shares of the new a")
for name in FISH_PAIRS:
    r = RES["variants"]["A_band"]["fish"][name]
    rl = RES["variants"]["lib_band"]["fish"][name]
    print(f"  {name:38s} a_lib {r['a_lib']:.4g} -> lib_band {rl['a_new']:.4g}, A_band {r['a_new']:.4g}; classes/eq "
          + " / ".join(f"{x:.2f}" for x in r["cls_rel"]) + f"; n/d {r['nd']:.3f} (0 deg {r['nd0']:.3f}; lib dark_ratio"
          f" {r['nd_lib']}; lib_band n/d {rl['nd']:.3f})")
    print("      members: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(r["members"].items(), key=lambda kv: -kv[1])))
for name in O["BENTH_PAIRS"]:
    r = RES["variants"]["A_band"]["fish"][name]
    print(f"  {name:38s} a_lib {r['a_lib']:.4g} -> A {r['a_new']:.4g} (already band; vision only); classes/eq "
          + " / ".join(f"{x:.2f}" for x in r["cls_rel"]) + f"; n/d {r['nd']:.3f} (lib {r['nd_lib']})")
for g in KRILL_LIB:
    r = RES["variants"]["A_band"]["krill"][g]
    print(f"  {g + '->krill':38s} a_lib {r['a_lib']:.4g} -> A {r['a_new']:.4g}; classes "
          + " / ".join(f"{x:.4g}" for x in r["classes_lib_units"]) + f"; n/d {r['nd']:.3f}")
json.dump(RES, open(OUTJ, "w"), indent=1, default=float)
print(f"\nwritten {os.path.basename(OUTJ)}")
