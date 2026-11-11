"""Seasonal herring / sprat profile in the fish -> planktivore pairs: band overlap + vision rule A (section 269 form)
with a WINTER day profile for herring and sprat, the schooling factor F recomputed at the combined derivation's new
capture p, and budget_gate on library copies (combined central vs seasonal readings).

Usage: python3 seasonal_pairs.py <repo_root> <scratch_dir> <out_dir>   (~3-5 min)

Nothing is fitted; no model output is read except budget_gate's own encounter arithmetic, which is a CHECK, never a
target; nothing in the repository is written (library copies only in <scratch_dir>).

Sources of the machinery (exec'd, unchanged):
  band_pairs_copy.py      = research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py, head up to its run block
                            (it execs "En synregel for fiskparen"/scripts/one_rule_pairs.py from its original place)
  school_factor_pout_copy.py = research_notes/"Vitlinglojans stim"/scripts/school_factor_pout.py, head up to VARIANTS
                            (S, sigma, pair_factor, PAR_NEW), as combined.py uses it
  research_notes/"Fangstsannolikhet for fiskbyten"/scripts/p_ppr.json (central p per pair) and gate_p.py (run_gate,
                            parse), as combined.py uses them
  research_notes/"Stimskydd i motes-a"/scripts/pair_terms.json (the combined central's term shares, for the check)

ONE thing changes: the herring and sprat prey profile phi_j(z) is seasonal. band_fish_pair is re-written here
(band_seasonal) with the prey-season split; with the winter profile set equal to the July one it reproduces
band_pairs.json A_band and pair_terms.json exactly (printed check).

  July / rest of year (library, krill_pairs_wc.herring_prof): day f_b(bed) uniform in the bottom 10 m (HERAS July
      bottom-10-m share 0.66 at 70 m ... 0.04-0.06 over >= 120 m) + (1 - f_b) uniform 10-100 m; night uniform 0-50 m.
  Winter (months WIN): day min(cap, k f_b(bed)) in the bottom 10 m + the rest uniform in [z_top, min(z_bot, bed - 10)];
      night uniform 0-50 m (unchanged).
"""
import contextlib
import io
import json
import math
import os
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
OUTD = os.path.abspath(sys.argv[3])
os.makedirs(SCR, exist_ok=True)
os.makedirs(os.path.join(OUTD, "gates"), exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.join(ROOT, "research_notes")

# ------------------------------------------------------------------ 1. band machinery (band_pairs.py head)
BP = os.path.join(HERE, "band_pairs_copy.py")
src = open(BP).read()
MARK = "# ================================================================== 0. overlap geometry"
head = src[:src.index(MARK)]
save = list(sys.argv)
sys.argv = [os.path.join(NOTES, "Overlapp i fiskparen", "scripts", "band_pairs.py"), ROOT, SCR,
            os.path.join(SCR, "_unused_band.json")]
G = {"__name__": "band_pairs_head", "__file__": sys.argv[0]}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, BP, "exec"), G)
sys.argv = save
K = G["K"]
ZG, ELEV, MONTH, BEDS, WCELL, CLS_IDX = G["ZG"], G["ELEV"], G["MONTH"], G["BEDS"], G["WCELL"], G["CLS_IDX"]
IS_NIGHT, STG_T, terms, sc, qf = G["IS_NIGHT"], G["STG_T"], G["terms"], G["sc"], G["qf"]
area_in_slab, gmat, rsat_fish, FISH_PAIRS = G["area_in_slab"], G["gmat"], G["rsat_fish"], G["FISH_PAIRS"]
WMUL, REST, EATERS, SLAB, SUMMER = G["WMUL"], G["REST"], G["EATERS"], G["SLAB"], G["SUMMER"]
uniform, NB, RNV_FISH = K["uniform"], G["NB"], G["RNV_FISH"]
slab_series, point_series, night_day = G["slab_series"], G["point_series"], G["night_day"]
PRED_PROF, SEASONAL, prey_prof = G["PRED_PROF"], G["SEASONAL"], G["prey_prof"]
qk0 = dict(G["qk0"])

# same state as band_pairs.py's A_band run (krill machinery run once under each rule; it sets RULE)
with contextlib.redirect_stdout(io.StringIO()):
    G["RULE"].update(rule="lib", RD=1.2, Lk=None)
    G["run_krill"](dict(K["level"](1)))
    G["RULE"].update(rule="A", RD=1.2, Lk=None)
    G["run_krill"](dict(K["level"](1), Ke_gad=0.02, k_deep=0.06))

# ------------------------------------------------------------------ 2. seasonal herring / sprat profile
SEAS = dict(k=1.0, z_top=10.0, z_bot=100.0, cap=0.9, months=(10, 11, 0, 1, 2))   # set per reading


def fb_july(bed, q):
    fb = float(np.interp(bed, [70, 95, 120, 140, 160], [0.66, 0.10, 0.04, 0.06, 0.05])) * q["herring_fb"]
    return min(fb, 0.9)


def herring_winter(bed, q, night):
    if night:
        return uniform(0.0, min(50.0, bed))                    # Onsrud 2004 Nov / Mar: 0-50 m at night [F analogue]
    fb = min(SEAS["cap"], SEAS["k"] * fb_july(bed, q))
    hi = min(SEAS["z_bot"], bed - 10.0)
    lo = min(SEAS["z_top"], max(10.0, hi - 20.0))
    if hi - lo < 1.0:
        lo = max(0.0, hi - 1.0)
    return fb * uniform(bed - 10, bed) + (1 - fb) * uniform(lo, hi)


PCACHE = {}


def pmat(kind, name, q, night, season="all", pseason="rest"):
    key = (kind, name, night, season, pseason, tuple(sorted((k, str(v)) for k, v in SEAS.items())),
           q.get("Hs_saithe"), q.get("f_up_saithe"), q.get("Hn_ratio"), q.get("H_whiting"))
    if key not in PCACHE:
        if kind == "pred":
            fn = SEASONAL[name] if season == "summer" else PRED_PROF[name]
        elif pseason == "winter" and name in ("herring", "sprat"):
            fn = herring_winter
        else:
            fn = prey_prof(name, q)
        PCACHE[key] = np.array([fn(b, q, night) for b in BEDS]).T
    return PCACHE[key]


def win_mask():
    return np.isin(MONTH, SEAS["months"])


# ------------------------------------------------------------------ 3. the band pair with the prey-season split
def band_seasonal(name, q, Ke=0.02, kdeep=0.06, clip=True):
    """band_pairs.band_fish_pair (vision 'A') with the prey profile split by season; also returns term shares and
    R_eff per (prey, night, prey season) as pair_terms.py does, and the per-time series."""
    members, pt, subset, lightprey, ovl, field, a_lib, nd_lib, thr = FISH_PAIRS[name]
    WIN = win_mask()
    den = 0.0
    a_enc_lib = 0.0
    ser = np.zeros((len(ELEV), NB))
    contrib, RW, OW, OVL = {}, {}, {}, {}
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
            k = t["a"] / (area_in_slab(t["R"], t["H"]) / t["H"])
            prey = lightprey or t["prey"]
            pseas = [("rest", ~WIN), ("winter", WIN)] if prey in ("herring", "sprat") else [("rest", np.ones(len(ELEV), bool))]
            for sname, smask in seasons:
                for psn, pmask in pseas:
                    rows = (IS_NIGHT if t["night"] else ~IS_NIGHT) & smask & pmask
                    if not rows.any():
                        continue
                    pp = pmat("pred", r.sp, q, t["night"], sname)
                    ph = pmat("prey", prey, q, t["night"], pseason=psn)
                    fixed = t["eye"] in ("fixed", "none")
                    if fixed:
                        Rfix = t["R"] if t["eye"] == "fixed" else qf["R_fish_elasmo"]
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
                            Gm = g_rows

                            def R_of(iz, Gm=Gm, rn2=rn2, rs2=rs2):
                                gg = Gm[:, iz]
                                return np.sqrt(rn2 * (1 - gg) + rs2 * gg)
                            val = slab_series(pp, H, R_of, int(rows.sum()))
                    else:
                        val = point_series(pp * ph, rn2, rs2, g_rows)
                    ser[rows] += w * k * val
                    key = (t["prey"], bool(t["night"]), psn)
                    fr = rows.mean()
                    contrib[key] = contrib.get(key, 0.0) + float((w * k * val @ WCELL).mean()) * fr
                    if not (clip and prey in SLAB):
                        o = (pp * ph).sum(axis=0)
                        RW[key] = RW.get(key, 0.0) + float(((w * k * val).mean(axis=0) @ WCELL)) * fr
                        OW[key] = OW.get(key, 0.0) + float(((w * k * np.pi * o) @ WCELL)) * fr
                        OVL[key] = OVL.get(key, 0.0) + float(w * k * (o @ WCELL)) * fr
    if members is EATERS:
        den = sum(r.W for r in STG_T if r.sp in REST)
    ser /= den
    a_enc_lib /= den
    s = ser @ WCELL
    eq = float(s.mean())
    cls = np.array([float((ser[:, idx] @ WCELL[idx]).mean() / WCELL[idx].sum()) for idx in CLS_IDX])
    tot = sum(contrib.values()) or 1.0
    mon = np.array([float(s[MONTH == m].mean()) for m in range(12)])
    return dict(eq=eq, s=s, classes=cls, cls_rel=cls / eq, nd=night_day(s, thr), a_lib=a_lib, mon=mon,
                shares={k: v / tot for k, v in contrib.items()},
                reff={k: math.sqrt(RW[k] / OW[k]) for k in RW if OW[k] > 0},
                ovl_per_time={k: OVL[k] / den for k in OVL})


# ------------------------------------------------------------------ 4. schooling factor (school_factor_pout head)
SFP = os.path.join(HERE, "school_factor_pout_copy.py")
ssrc = open(SFP).read()
shead = ssrc[:ssrc.index("VARIANTS = {")]
TERMS0 = os.path.join(NOTES, "Stimskydd i motes-a", "scripts", "pair_terms.json")
sys.argv = ["school_factor_pout.py", TERMS0, os.path.join(SCR, "_unused_sf.json")]
NS = {"__name__": "pout_head"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(shead, SFP, "exec"), NS)
sys.argv = save
PAR = NS["PAR_NEW"]
PREYMAP = NS["PREYMAP"]
PP = json.load(open(os.path.join(NOTES, "Fangstsannolikhet for fiskbyten", "scripts", "p_ppr.json")))["pairs"]

PLK = ["predatory_fish->planktivorous_fish", "whiting->planktivorous_fish", "mackerel->planktivorous_fish"]
BEN = "benthivorous_fish->planktivorous_fish"
STOMACH_ND = {"mackerel->planktivorous_fish"}


def school_F(name, res):
    rows = []
    for (prey, night, psn), share in res["shares"].items():
        R = res["reff"].get((prey, night, psn))
        if R is None:
            R = 0.3 if night else 0.6
        rows.append((PREYMAP[prey], "night" if night else "day", share, R))
    NS["P_I"] = PP[name]["central"]["p"]
    return NS["pair_factor"](dict(rows=rows, nd=res["nd"]), PAR, 1, None)


# ------------------------------------------------------------------ 5. benthivores -> planktivores (benth_encounter path)
VBEN = G["O"]["benth_V"]("P", "A", 0.02, 0.06) if "O" in G else None


def benth_herring_factor():
    """time-weighted factor on the herring DAY term of benthivores -> planktivores: sum over day rows and cells of
    V_A(t, b) f_b,season(t, b) / the same with the July f_b (V_A = rule-A visual term at the bed, one_rule_pairs.benth_V)."""
    V = VBEN
    WIN = win_mask()
    fbs = np.array([fb_july(b, qk0) for b in BEDS])
    fbw = np.minimum(SEAS["cap"], SEAS["k"] * fbs)
    day = ~IS_NIGHT
    num = (V[day & ~WIN] * fbs).sum(0) + (V[day & WIN] * fbw).sum(0)
    den = (V[day] * fbs).sum(0)
    tot = float((num @ WCELL) / (den @ WCELL))
    cls = [float((num[idx] @ WCELL[idx]) / (den[idx] @ WCELL[idx])) for idx in CLS_IDX]
    # monthly herring day-term ratio (winter months only differ)
    return tot, cls, float(fbs @ WCELL), float(fbw @ WCELL)


def benth_pair_seasonal():
    nd = 0.141
    dsh = 1.0 / (1.0 + nd)
    hf, hcls, fbs_mean, fbw_mean = benth_herring_factor()
    rows0 = [("pout", "day", 0.85 * dsh), ("pout", "night", 0.85 * (1 - dsh)),
             ("herring", "day", 0.15 * dsh), ("herring", "night", 0.15 * (1 - dsh))]
    rows = [(sp, per, sh * (hf if (sp, per) == ("herring", "day") else 1.0)) for sp, per, sh in rows0]
    tot = sum(r[2] for r in rows)
    a_rel = tot / 1.0
    dsh_new = sum(r[2] for r in rows if r[1] == "day") / tot
    nd_new = (1 - dsh_new) / dsh_new
    rr = [(sp, per, sh / tot, 0.6 if per == "day" else 0.3) for sp, per, sh in rows]
    NS["P_I"] = PP[BEN]["central"]["p"]
    f = NS["pair_factor"](dict(rows=rr, nd=nd_new), PAR, 1, None)
    cls_rel = [(0.85 + 0.15 * dsh * c + 0.15 * (1 - dsh)) / a_rel for c in hcls]
    return dict(a_rel=a_rel, F=f, nd=nd_new, herring_day_factor=hf, herring_cls=hcls, cls_rel_change=cls_rel,
                fb_box_july=fbs_mean, fb_box_winter=fbw_mean)


# ------------------------------------------------------------------ 6. readings
LIB0 = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8-sig"))
INTER = LIB0["interaction_definitions"]
BANDJ = json.load(open(os.path.join(NOTES, "Overlapp i fiskparen", "tables", "band_pairs.json")))
TERMSJ = json.load(open(TERMS0))
COMBJ = json.load(open(os.path.join(NOTES, "Fangst och stim i fiskparen", "tables", "gates", "combined.json")))

# monthly presence, fraction of August (research_notes/"Bestandsreglering" migration.md section 4, WP2) [D/J there]
PRES = {"herring": [0.60, 0.60, 0.60, 0.65, 0.80, 0.95, 1, 1, 0.85, 0.70, 0.60, 0.60],
        "predatory_fish": [0.12, 0.09, 0.13, 0.29, 0.54, 0.83, 1, 1, 0.99, 0.85, 0.54, 0.29],
        "mackerel": [0.01, 0.01, 0.02, 0.03, 0.10, 0.50, 1, 1, 0.90, 0.50, 0.10, 0.03],
        "whiting": [1.0] * 12,
        "benthivorous_fish": [0.73, 0.71, 0.71, 0.78, 0.91, 0.96, 1, 1, 1, 0.96, 0.88, 0.81]}


def weighted(s, name, mode):
    pred = name.split("->")[0]
    if mode == "equal":
        w = np.ones_like(s)
    elif mode == "prey":
        w = np.array(PRES["herring"])[MONTH]
    else:
        w = np.array(PRES["herring"])[MONTH] * np.array(PRES[pred])[MONTH]
    return float((s * w).sum() / w.sum())


READ = {
    "null (winter = July profile)": dict(k=1.0, z_top=10.0, z_bot=100.0, months=(10, 11, 0, 1, 2)),
    "central": dict(k=2.0, z_top=50.0, z_bot=120.0, months=(10, 11, 0, 1, 2)),
    "k1.5": dict(k=1.5, z_top=50.0, z_bot=120.0, months=(10, 11, 0, 1, 2)),
    "k3": dict(k=3.0, z_top=50.0, z_bot=120.0, months=(10, 11, 0, 1, 2)),
    "k2_july_layer": dict(k=2.0, z_top=10.0, z_bot=100.0, months=(10, 11, 0, 1, 2)),
    "k1_deep_layer": dict(k=1.0, z_top=50.0, z_bot=120.0, months=(10, 11, 0, 1, 2)),
    "k2_layer70_bed": dict(k=2.0, z_top=70.0, z_bot=1000.0, months=(10, 11, 0, 1, 2)),
    "central_DecFeb": dict(k=2.0, z_top=50.0, z_bot=120.0, months=(11, 0, 1)),
    "central_OctApr": dict(k=2.0, z_top=50.0, z_bot=120.0, months=(9, 10, 11, 0, 1, 2, 3)),
}

RES = {}
print("=" * 118)
print("Seasonal herring / sprat profile. Readings (k on the July bottom-10-m share, winter day midwater layer, months):")
for rn, p in READ.items():
    print(f"  {rn:30s} k {p['k']}, day layer {p['z_top']:.0f}-{min(p['z_bot'], 999):.0f} m (bed - 10 at most), months "
          f"{[m + 1 for m in p['months']]}")
print("=" * 118)
for rn, p in READ.items():
    SEAS.update(p)
    out = {}
    for name in PLK:
        r = band_seasonal(name, qk0)
        lib_eq = BANDJ["variants"]["A_band"]["fish"][name]["eq"]
        r["ratio_vs_annual"] = r["eq"] / lib_eq
        r["F"] = school_F(name, r)
        out[name] = r
    out[BEN] = benth_pair_seasonal()
    RES[rn] = out
    print(f"{rn}: done", flush=True)

# ------------------------------------------------------------------ 7. reproduction check
print("\n1. Reproduction (null reading = July profile all year) vs band_pairs.json A_band and the combined central F")
for name in PLK:
    r = RES["null (winter = July profile)"][name]
    ref = BANDJ["variants"]["A_band"]["fish"][name]["eq"]
    cf = COMBJ["pairs"][name]["readings"]["central"]
    print(f"  {name:38s} eq {r['eq']:.6g} vs {ref:.6g} (x{r['eq'] / ref:.6f}); F {r['F']['F']:.4f} vs combined "
          f"{cf['F']:.4f} (day {r['F']['F_day']:.4f} vs {cf['F_day']:.4f})")
rb = RES["null (winter = July profile)"][BEN]
cf = COMBJ["pairs"][BEN]["readings"]["central"]
print(f"  {BEN:38s} a_rel {rb['a_rel']:.4f}; F {rb['F']['F']:.4f} vs combined {cf['F']:.4f}")
print(f"  herring July bottom-10-m share, box equal-cell mean {rb['fb_box_july']:.3f} (benth_encounter herring F_day"
      " central 0.15; 'Bottenfiskens sillbyte' reconcile 0.16)")

# ------------------------------------------------------------------ 8. geometry of the herring day term
print("\n2. Herring day term per pair: overlap per unit time (sum f phi, rel. to the July rest-of-year value) and the")
print("   light-weighted detection radius R_eff (m), winter vs rest of year, central reading")
for rn in ("central", "k2_july_layer", "k1_deep_layer", "k3"):
    print(f"  [{rn}]")
    for name in PLK:
        r = RES[rn][name]
        o, re_ = r["ovl_per_time"], r["reff"]
        kr = ("herring", False, "rest")
        kw = ("herring", False, "winter")
        WINm = np.isin(MONTH, READ[rn]["months"])
        orest = o.get(kr, 0) / max(1e-12, float((~IS_NIGHT & ~WINm).mean()))     # per unit DAY time of the season
        owin = o.get(kw, 0) / max(1e-12, float((~IS_NIGHT & WINm).mean()))
        print(f"    {name:38s} herring|day overlap winter/rest {owin / orest if orest else float('nan'):5.2f};"
              f" R_eff rest {re_.get(kr, float('nan')):.2f} m, winter {re_.get(kw, float('nan')):.2f} m")
    b = RES[rn][BEN]
    print(f"    {BEN:38s} herring bottom-10-m share box mean July {b['fb_box_july']:.3f}, winter {b['fb_box_winter']:.3f};"
          f" light-weighted herring|day term x{b['herring_day_factor']:.3f} (annual)")

# ------------------------------------------------------------------ 9. pair table
print("\n3. Annual-mean a by pair (central p and schooling F at the new p, as the combined derivation). a_band = band a,"
      " A rule,\n   equal-time; factor = p factor x F; 'comb' = the combined central (July profile all year).")
TAB = {}
for rn in READ:
    TAB[rn] = {}
    for name in PLK + [BEN]:
        a0 = INTER[f"{name.split('->')[0]}_preys_on_{name.split('->')[1]}"]["max_intake_rate"]
        pf = PP[name]["central"]["p"] / PP[name]["p_lib"]
        r = RES[rn][name]
        if name == BEN:
            band_rel = r["a_rel"]
            F = r["F"]
            nd = r["nd"]
        else:
            band_rel = r["ratio_vs_annual"]
            F = r["F"]
            nd = r["nd"]
        a_new = a0 * band_rel * pf * F["F"]
        nd_new = INTER[f"{name.split('->')[0]}_preys_on_{name.split('->')[1]}"].get("dark_ratio") if name in STOMACH_ND \
            else (nd if name == BEN else r["nd"]) * F["F_night"] / F["F_day"]
        TAB[rn][name] = dict(a_lib=a0, band_rel=band_rel, p_factor=pf, F=F["F"], F_day=F["F_day"], F_night=F["F_night"],
                             a_new=a_new, nd_new=nd_new)
for rn in READ:
    print(f"  [{rn}]")
    for name in PLK + [BEN]:
        t = TAB[rn][name]
        comb = COMBJ["pairs"][name]["readings"]["central"]["a_new"]
        print(f"    {name:38s} a_lib {t['a_lib']:.4g}  band x{t['band_rel']:.3f}  p x{t['p_factor']:.2f}  F {t['F']:.3f}"
              f" (day {t['F_day']:.3f})  a {t['a_new']:.4g}  (comb {comb:.4g}, x{t['a_new'] / comb:.3f})  night/day "
              f"{t['nd_new']:.3f}")

# ------------------------------------------------------------------ 10. monthly swing and time weighting
print("\n4. Monthly a (band x vision, before p and F) relative to the pair's equal-time annual mean, central vs null;")
print("   and the annual mean under three weightings (equal time = engine convention; prey presence; prey x predator")
print("   presence = the flow-preserving mean), relative to equal time")
MN = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
for name in PLK:
    for rn in ("null (winter = July profile)", "central", "k3"):
        r = RES[rn][name]
        print(f"  {name:38s} {rn[:12]:12s} " + " ".join(f"{MN[m]} {r['mon'][m] / r['eq']:.2f}" for m in range(12)))
    for rn in ("null (winter = July profile)", "central", "k3"):
        r = RES[rn][name]
        e = weighted(r["s"], name, "equal")
        print(f"      weights [{rn[:12]}]: prey-presence x{weighted(r['s'], name, 'prey') / e:.3f}, flow x"
              f"{weighted(r['s'], name, 'flow') / e:.3f}; winter(Nov-Mar)/rest-of-year a "
              f"{r['s'][np.isin(MONTH, (10, 11, 0, 1, 2))].mean() / r['s'][~np.isin(MONTH, (10, 11, 0, 1, 2))].mean():.2f}")
for rn in ("null (winter = July profile)", "central", "k3"):
    SEAS.update(READ[rn])
    # benthivores: monthly herring-term factor
    V = VBEN
    fbs = np.array([fb_july(b, qk0) for b in BEDS])
    fbw = np.minimum(SEAS["cap"], SEAS["k"] * fbs)
    WIN = win_mask()
    nd, dsh = 0.141, 1 / 1.141
    her_day = np.where(WIN[:, None], (V * fbw[None, :]), (V * fbs[None, :])) @ WCELL
    her_day = np.where(IS_NIGHT, 0.0, her_day)
    ref = np.where(IS_NIGHT, 0.0, (V * fbs[None, :]) @ WCELL)
    mon = [her_day[MONTH == m].sum() / ref[MONTH == m].sum() for m in range(12)]
    print(f"  {BEN:38s} {rn[:12]:12s} herring|day term, month / July-profile month: " + " ".join(
        f"{MN[m]} {mon[m]:.2f}" for m in range(12)))

# ------------------------------------------------------------------ 10b. derived monthly a vs the engine's daylight lever
sys.path.insert(0, ROOT)
from lib.world import daylight as DL                     # noqa: E402  (read-only import of the engine's m(t))
sys.path.pop(0)
NT = DL.ticks_per_year(6)
TDAY = (np.arange(NT) * 6.0 + 3.0) / 24.0
TMON = np.minimum((TDAY / (365.0 / 12.0)).astype(int), 11)
print("\n4b. Derived monthly a AFTER the schooling factor (day rows x F_day, night rows x F_night; p is a constant) relative")
print("    to its annual mean, against the engine's only seasonal lever m(t) (lib/world/daylight.attack_multiplier at the")
print("    pair's new night/day and light_threshold_deg; annual mean 1). residual = derived / engine, per month.")
SWING = {}
for name in PLK:
    thr = FISH_PAIRS[name][8]
    for rn in ("null (winter = July profile)", "central", "k3"):
        r = RES[rn][name]
        F = TAB[rn][name]
        sF = np.where(IS_NIGHT, r["s"] * F["F_night"], r["s"] * F["F_day"])
        der = np.array([sF[MONTH == m].mean() for m in range(12)])
        der = der / sF.mean()
        nd = TAB[rn][name]["nd_new"]
        m = DL.attack_multiplier(58.15, 6, nd, thr)
        eng = np.array([m[TMON == k].mean() for k in range(12)])
        res_ = der / eng
        SWING[(name, rn)] = dict(derived=der.tolist(), engine=eng.tolist(), residual=res_.tolist())
        print(f"  {name:36s} {rn[:12]:12s} nd {nd:.3f} thr {thr}")
        print("      derived  " + " ".join(f"{x:5.2f}" for x in der))
        print("      engine   " + " ".join(f"{x:5.2f}" for x in eng))
        print("      residual " + " ".join(f"{x:5.2f}" for x in res_) +
              f"   (Nov-Mar mean {res_[[10, 11, 0, 1, 2]].mean():.2f}, Apr-Oct {res_[3:10].mean():.2f})")
print("    (MN: " + " ".join(MN) + ")")

# ------------------------------------------------------------------ 11. gate
gsrc = open(os.path.join(NOTES, "Fangstsannolikhet for fiskbyten", "scripts", "gate_p.py")).read()
GNS = {"__name__": "gate_head", "ROOT": ROOT, "os": os, "sys": sys}
BG = os.path.join(ROOT, "tools", "probes", "budget_gate.py")
GNS["BG"] = BG
GNS["BG_SRC"] = open(BG).read()
GNS["OLD"] = "apply_natural_mortality=True, migration=True)"
assert GNS["BG_SRC"].count(GNS["OLD"]) == 1
exec(compile("import contextlib, io, re\n" + gsrc[gsrc.index("def run_gate"):gsrc.index("summary = {}")],
             "gate_p.py", "exec"), GNS)
run_gate, parse = GNS["run_gate"], GNS["parse"]

CF = COMBJ["factors"]["central"]          # the combined central factor per pair (all nine fish pairs)
EDGES = [(99.9, 0), (100.0, 1), (149.9, 1), (150.0, 2), (249.9, 2), (250.0, 3)]


def key_of(name):
    p, q = name.split("->")
    return f"{p}_preys_on_{q}"


def make_lib(rn, tables=False, weight="equal"):
    lib = json.loads(json.dumps(LIB0))
    inter = lib["interaction_definitions"]
    fixes = {}
    for name, f in CF.items():
        k = key_of(name)
        inter[k]["max_intake_rate"] = float(f"{INTER[k]['max_intake_rate'] * f:.6g}")
    if rn is not None:
        for name in PLK + [BEN]:
            k = key_of(name)
            t = TAB[rn][name]
            a = t["a_new"]
            if weight != "equal" and name != BEN:
                r = RES[rn][name]
                a *= weighted(r["s"], name, weight) / weighted(r["s"], name, "equal")
            inter[k]["max_intake_rate"] = float(f"{a:.6g}")
            fixes[name] = inter[k]["max_intake_rate"]
            if tables and "depth_access_table" in inter[k]:
                if name == BEN:
                    old = [row["factor"] for row in inter[k]["depth_access_table"]]
                    cr = RES[rn][BEN]["cls_rel_change"]
                    vals = [old[i] * cr[c] for i, (_, c) in enumerate(EDGES)]
                else:
                    cr = RES[rn][name]["cls_rel"]
                    vals = [a * cr[c] for _, c in EDGES]
                inter[k]["depth_access_table"] = [{"depth_m": d, "factor": float(f"{v:.4g}")} for (d, _), v in
                                                  zip(EDGES, vals)]
    return lib, fixes


GRUNS = [("combined_central", None, False, "equal"), ("seasonal_central", "central", False, "equal"),
         ("seasonal_central_T", "central", True, "equal"), ("seasonal_k1.5", "k1.5", False, "equal"),
         ("seasonal_k3", "k3", False, "equal"), ("seasonal_k2_july_layer", "k2_july_layer", False, "equal"),
         ("seasonal_k1_deep_layer", "k1_deep_layer", False, "equal"),
         ("seasonal_k2_layer70_bed", "k2_layer70_bed", False, "equal"),
         ("seasonal_DecFeb", "central_DecFeb", False, "equal"), ("seasonal_OctApr", "central_OctApr", False, "equal"),
         ("seasonal_central_flow", "central", False, "flow"), ("null_flow", "null (winter = July profile)", False, "flow")]
GS = {}
for lab, rn, tables, wt in GRUNS:
    lib, fixes = make_lib(rn, tables, wt)
    lp = os.path.join(SCR, f"fg_library_seasonal_{lab}.yaml")
    with open(lp, "w") as fh:
        yaml.safe_dump(lib, fh, sort_keys=False, allow_unicode=True)
    txt = run_gate(lp)
    with open(os.path.join(OUTD, "gates", f"budget_gate_{lab}.txt"), "w") as fh:
        fh.write(f"# budget_gate.py on a temporary library copy ({lab}); combined central factors on the nine fish pairs"
                 + (", then the four planktivore pairs: " + ", ".join(f"{k} {v:.4g}" for k, v in fixes.items())
                    if fixes else "") + "\n" + txt)
    GS[lab] = parse(txt)
    print(f"gate {lab}: done", flush=True)

print("\n5. budget_gate (implied / booked M2; * = OFF)")
LABS = [g[0] for g in GRUNS]
print("  " + " " * 40 + " ".join(f"{l.replace('seasonal_', 's_')[:13]:>13s}" for l in LABS))
for p in ("planktivorous_fish", "krill", "benthivorous_fish", "whiting", "zooplankton"):
    print(f"  {'prey total ' + p:40s}" + " ".join(f"{GS[l]['prey_total'].get(p, float('nan')):13.2f}" for l in LABS))
for k in ("predatory_fish->planktivorous_fish", "whiting->planktivorous_fish", "mackerel->planktivorous_fish",
          "benthivorous_fish->planktivorous_fish", "predatory_fish->krill"):
    row = []
    for l in LABS:
        r = GS[l]["prey_pair"].get(k)
        row.append(f"{r[0]:12.2f}{'*' if r[1] else ' '}" if r else f"{'-':>13s}")
    print(f"  {'prey side ' + k:40s}" + " ".join(row))
for key, lab in (("diet", "diet equal cells"), ("diet_field", "diet field spread")):
    for k in ("predatory_fish->krill", "predatory_fish->planktivorous_fish", "whiting->planktivorous_fish",
              "benthivorous_fish->planktivorous_fish", "mackerel->planktivorous_fish", "whiting->krill"):
        row = []
        for l in LABS:
            r = GS[l][key].get(k)
            row.append(f"{r[0]:12.2f}{'*' if r[1] else ' '}" if r else f"{'-':>13s}")
        print(f"  {lab + ' ' + k:40s}"[:42] + " ".join(row))
print("\nOFF pairs per run:")
for l in LABS:
    offs = sorted({k for key in ("prey_pair", "diet", "diet_field") for k, r in GS[l][key].items() if r[1]})
    print(f"  {l:26s} {', '.join(offs)}")

json.dump(dict(readings=READ, table=TAB, gate=GS, swing={f"{k[0]}|{k[1]}": v for k, v in SWING.items()},
               months={rn: {n: (RES[rn][n]["mon"] / RES[rn][n]["eq"]).tolist() for n in PLK} for rn in RES},
               cls_rel={rn: {n: RES[rn][n]["cls_rel"].tolist() for n in PLK} for rn in RES}),
          open(os.path.join(OUTD, "seasonal_pairs.json"), "w"), indent=1, default=float)
print("\nwritten seasonal_pairs.json")
