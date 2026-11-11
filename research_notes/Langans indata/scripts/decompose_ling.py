"""COPY of research_notes/"Vattenpelarens vuxna bottenfisk"/scripts/decompose_members.py (2026-11-11). Unchanged
except for (a) a ling hook in detail_pair (LING_MODE, default None = the original run; the baseline is computed
first and must still agree with band_fish_pair), (b) the ling variants run after the baseline, (c) ling's own
windows on planktivores and whiting, and (d) extra json keys (ling_variants, ling_windows). The ling variants
(research_notes/"Langans indata" evidence.md):
  act    duty cycle: active cruising f_act of the time at v = v_hake x L (0.5 BL/s [J]), the rest sit-and-wait
         (v = 0: prey moves past at its own speed u, the same branch the bed ambusher uses; vrel(0, u) = u), visual
         R unchanged (rule A). k_term is linear in vrel and in scan, so each term is multiplied by
         (f_act vrel(v, u) + (1 - f_act) u sw_scan / scan) / vrel(v, u)   [Lokkeborg et al. 2000: inactive ~65 %]
         sw_scan: the stationary part's scan (default = the cruiser's 0.5; 1.0 as the bed-ambush branch [J]).
         'idle' variant: the inactive time does not forage, factor f_act.
  night  ling night terms x nf (0 = no night foraging; Lokkeborg et al. 2000: no reaction to bait at night).
  prof   ling profile exp(H m) above the bed instead of exp(5 m) (bracket 2 m: closer to the bed).
Original docstring follows.

Member decomposition of the water column (WC) -> benthivores pair (and WC -> whiting) at the section 269 central
(band overlap + rule A, exp Hs 15 m), with the stage windows per member and a stage- and length-specific capture p.

Usage: python3 -I decompose_members.py <repo_root> <scratch_dir> <out_json>

Nothing is fitted and no model output is read. Two derivations are exec'd UNCHANGED (heads only), as the sibling
scripts research_notes/"Sejens hojd over botten"/scripts/band_hs.py and window_hs.py do:
  * research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py up to its section 0 block: band_fish_pair and every
    factor behind the library a 0.0418 (members, N/t, speeds, scan, R by rule A, light, slab clipping, p 0.2).
    band_fish_pair drops members below 0.005 of the a; its loop is re-stated here (same arithmetic, checked against
    band_fish_pair's eq) to keep every member and to record the factors of each term.
  * research_notes/"Fonsterviktning"/scripts/aw_stage_access.py up to its run block: the stage windows (p95 prey /
    predator length ratio over each member's box length mix, a(L)-weighted, normalised over all benthivore tonnes).

Capture p per encounter (research_notes/"Fangstsannolikhet for fiskbyten", central): p = q p_s / (1 - r (1 - p_s)),
p_s = 0.973 - 1.230 PPR (bluefish line), q 0.8, r 0.3; 0.592 at PPR 0.25 (the pair value of the combined
derivation). Here p is averaged over the encountered prey tonnes INSIDE each member's window, per stage, at each
prey length / predator length pair (the window's own a(L) weights): the same definition, resolved by member and
stage instead of one central PPR.
"""
import contextlib
import io
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
OUTJ = os.path.abspath(sys.argv[3])
os.makedirs(SCR, exist_ok=True)
RN = os.path.join(ROOT, "research_notes")
ARGV = list(sys.argv)

# ------------------------------------------------------------------ band_pairs head (unchanged)
BP = os.path.join(RN, "Overlapp i fiskparen", "scripts", "band_pairs.py")
src = open(BP).read()
head = src[:src.index("# ================================================================== 0. overlap geometry")]
sys.argv = [BP, ROOT, SCR, os.path.join(SCR, "_unused_band.json")]
B = {"__name__": "band_pairs_head", "__file__": BP}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, BP, "exec"), B)
sys.argv = ARGV
qk0, qf, STG_T, WCELL, ELEV = B["qk0"], B["qf"], B["STG_T"], B["WCELL"], B["ELEV"]
IS_NIGHT, NB, BEDS, ZG = B["IS_NIGHT"], B["NB"], B["BEDS"], B["ZG"]
B["RULE"].update(rule="A", RD=1.2, Lk=None)


# Adult benthivores' share in the bottom 2 m (research_notes/"Bottenfiskens hojdprofil" central profiles: cod and
# haddock 20-34 cm 0.70, haddock >= 35 cm 0.30 day / 0.50 night, flatfish and LRD 1.0), weighted by the A biomass:
# haddock 27-34 cm 0.077, haddock >= 35 cm 0.174, cod 0.294, flatfish + others 0.270 of B (J/B 0.155, J 0.87
# haddock; haddock size shares 0.086 / 0.462 / 0.452 of < 20 / 20-34 / >= 35 cm).
_AW = dict(h27=0.077, h35=0.174, cod=0.294, flat=0.270)
SA_DAY = (_AW["h27"] * 0.70 + _AW["h35"] * 0.30 + _AW["cod"] * 0.70 + _AW["flat"]) / sum(_AW.values())
SA_NIGHT = (_AW["h27"] * 0.70 + _AW["h35"] * 0.50 + _AW["cod"] * 0.70 + _AW["flat"]) / sum(_AW.values())
print(f"adult benthivores in the bottom 2 m: day {SA_DAY:.3f}, night {SA_NIGHT:.3f} (off-bed part uniform 2-20 m [J])")


LING_MODE = None      # dict(f_act=, idle=bool, nf=) or None


def ling_factor(t, r, q):
    """multiplier on one ling term (see the docstring); 1 when LING_MODE is None or the member is not ling."""
    if LING_MODE is None or r.sp != "ling":
        return 1.0
    f = 1.0
    if "f_act" in LING_MODE:
        v = q["v_hake"] * r.L
        prey = t["prey"]
        u = q["u_herring"] if prey in ("herring", "sprat") else q["u_smallfish"]
        vr = B["GP"]["vrel"](v, u)
        fa = LING_MODE["f_act"]
        sw = LING_MODE.get("sw_scan", q["scan"]) / q["scan"]   # scan of the stationary part relative to the cruising scan
        f *= fa if LING_MODE.get("idle") else (fa * vr + (1.0 - fa) * u * sw) / vr
    if t["night"]:
        f *= LING_MODE.get("nf", 1.0)
    return f


def detail_pair(name, q):
    """band_fish_pair's loop (vision A, clip on) with per-member records."""
    members, pt, subset, lightprey, ovl, field, a_lib, nd_lib, thr = B["FISH_PAIRS"][name]
    den = 0.0
    rec = {}
    ser_tot = np.zeros((len(ELEV), NB))
    for r in STG_T:
        if r.sp not in members:
            continue
        w = r.W
        den += w
        key = (r.sp, r.st)
        d = rec.setdefault(key, dict(W=w, Nt=r.Nt, L=r.L, score=float(B["sc"](r, pt)), contrib=0.0, day=0.0, night=0.0,
                                     terms=[]))
        if B["sc"](r, pt) < 0.1:
            continue
        seasons = [("all", np.ones(len(ELEV), bool))]
        if r.sp in B["SEASONAL"]:
            seasons = [("all", ~B["SUMMER"]), ("summer", B["SUMMER"])]
        for t in B["terms"](r, pt, qf, subset):
            k = t["a"] / (B["area_in_slab"](t["R"], t["H"]) / t["H"]) * ling_factor(t, r, qf)
            prey = lightprey or t["prey"]
            for sname, smask in seasons:
                rows = (IS_NIGHT if t["night"] else ~IS_NIGHT) & smask
                if not rows.any():
                    continue
                pp = B["pmat"]("pred", r.sp, q, t["night"], sname)
                fixed = t["eye"] in ("fixed", "none")
                slab = prey in B["SLAB"]
                if fixed:
                    Rfix = t["R"] if t["eye"] == "fixed" else qf["R_fish_elasmo"]
                    rn2 = rs2 = Rfix ** 2
                    rs = Rfix
                    G = None
                else:
                    rs = max(float(B["rsat_fish"](qk0, r.L)), B["RNV_FISH"])
                    rn2, rs2 = B["RNV_FISH"] ** 2, rs ** 2
                    G = B["gmat"](t["eye"], 0.02, 0.06)[rows]
                if slab:
                    H = B["SLAB"][prey](q)
                    if G is None:
                        val = B["slab_series"](pp, H, lambda iz: np.full((1, NB), math.sqrt(rn2)), 1)
                    else:
                        def R_of(iz, G=G, rn2=rn2, rs2=rs2):
                            gg = G[:, iz]
                            return np.sqrt(rn2 * (1 - gg) + rs2 * gg)
                        val = B["slab_series"](pp, H, R_of, int(rows.sum()))
                else:
                    H = 0.0
                    ph = B["pmat"]("prey", prey, q, t["night"])
                    val = B["point_series"](pp * ph, rn2, rs2, G)
                ser_tot[rows] += w * k * val
                if slab and prey == "juvenile benthivores":
                    # adults: share sA in the bottom 2 m slab, the rest uniform 2-20 m above the bed [S/J]
                    sA = SA_NIGHT if t["night"] else SA_DAY
                    ph_off = np.array([B["uniform"](b - 20.0, b - 2.0) for b in BEDS]).T
                    voff = B["point_series"](pp * ph_off, rn2, rs2, G)
                    vA = sA * val + (1 - sA) * voff
                    d["contrib_Aprof"] = d.get("contrib_Aprof", 0.0) + float((w * k * vA @ WCELL).mean()) * rows.mean()
                c = float((w * k * val @ WCELL).mean()) * rows.mean()
                d["contrib"] += c
                d["night" if t["night"] else "day"] += c
                # geometry: the member's time share within H of the bed in this period (equal-cell mean)
                near = 0.0
                for ib, b in enumerate(BEDS):
                    near += WCELL[ib] * (pp[(ZG > b - H) & (ZG < b), ib].sum() if slab else (pp[:, ib] * ph[:, ib]).sum())
                d["terms"].append(dict(night=t["night"], season=sname, eye=t["eye"], R_lib=t["R"], R_sat=rs,
                                       k_per_t=k / r.Nt if r.Nt else 0.0, near_bed=near / WCELL.sum(),
                                       rows=float(rows.mean()), c=c))
    if members is B["EATERS"]:
        den = sum(r.W for r in STG_T if r.sp in B["REST"])
    eq = float((ser_tot @ WCELL).mean() / den)
    tot = sum(d["contrib"] for d in rec.values())
    return eq, den, rec, tot


q0 = dict(qk0)
out = {}
for name in ("predatory_fish->benthivorous_fish", "predatory_fish->whiting", "predatory_fish->planktivorous_fish",
             "predatory_fish->mackerel"):
    eq, den, rec, tot = detail_pair(name, q0)
    with contextlib.redirect_stdout(io.StringIO()):
        ref = B["band_fish_pair"](name, q0, "A", 0.02, 0.06, True)
    print(f"\n{name}: eq here {eq:.6g}, band_fish_pair {ref['eq']:.6g} (must agree); a_new = a_lib x ratio "
          f"{ref['a_lib'] * ref['ratio']:.4g}")
    sp_w = {}
    for (sp, st), d in rec.items():
        sp_w[sp] = sp_w.get(sp, 0.0) + d["W"]
    tw = sum(sp_w.values())
    print("  biomass weights in the group (renormalised):", {k: round(v / tw, 3) for k, v in sp_w.items()})
    print(f"  {'member|stage':20s} {'W':>6s} {'score':>5s} {'L m':>5s} {'N/t':>8s} {'share a':>8s} {'a/t rel':>8s}"
          f" {'day/ngt':>7s}  terms: period, eye, R_sat (m), k per fish, time share within the slab")
    per_sp = {}
    for (sp, st), d in sorted(rec.items(), key=lambda kv: -kv[1]["contrib"]):
        sh = d["contrib"] / tot
        per_sp.setdefault(sp, [0.0, 0.0])
        per_sp[sp][0] += sh
        per_sp[sp][1] += d["W"]
        apt = (d["contrib"] / d["W"]) if d["W"] else 0.0
        out.setdefault(name, {})[f"{sp}|{st}"] = dict(Aprof=(d.get("contrib_Aprof", 0.0) / d["contrib"]) if d["contrib"] else 0.0,
                                                     share=sh, W=d["W"] / tw, a_per_t=apt / (tot / tw),
                                                     day=d["day"] / tot, night=d["night"] / tot, L=d["L"], Nt=d["Nt"],
                                                     score=d["score"])
        tl = "; ".join(f"{'n' if t['night'] else 'd'}{'(s)' if t['season'] == 'summer' else ''} {t['eye']} "
                       f"{t['R_sat']:.2f} {t['k_per_t'] * 1e6:.3g} {t['near_bed']:.3f}" for t in d["terms"][:4])
        print(f"  {sp + '|' + st:20s} {d['W'] / tw:6.3f} {d['score']:5.2f} {d['L']:5.2f} {d['Nt']:8.1f} {sh:8.4f}"
              f" {apt / (tot / tw):8.2f} {d['day'] / max(d['contrib'], 1e-30):7.2f}  {tl}")
    print("  per species: share of the a / biomass share / a per tonne relative to the group mean")
    for sp, (sh, w) in sorted(per_sp.items(), key=lambda kv: -kv[1][0]):
        print(f"    {sp:13s} {sh:.4f} / {w / tw:.3f} / {sh / (w / tw) if w else 0:.2f}")
    out[name + "|eq"] = eq
    if name == "predatory_fish->benthivorous_fish":
        print("  adult-prey vertical profile: member a on A relative to the J slab a (x factor on the member's A flow)")
        for k, v in out[name].items():
            if v["share"] > 0:
                print(f"    {k:20s} x{v['Aprof']:.3f}")

# ------------------------------------------------------------------ ling variants (this note)
PAIRS4 = ("predatory_fish->benthivorous_fish", "predatory_fish->whiting", "predatory_fish->planktivorous_fish",
          "predatory_fish->mackerel")
LING_PROF0 = B["PRED_PROF"]["ling"]


def ling_run(mode, prof_h=None):
    """per pair: (g = ling a under the variant / ling a at baseline, ling share of the pair a, ling A-profile factor,
    ling day share, pair eq / baseline eq)."""
    global LING_MODE
    LING_MODE = mode
    if prof_h is not None:
        B["PRED_PROF"]["ling"] = (lambda b, q, n, H=prof_h: B["expbed"](b, H))
    B["PCACHE"].clear()
    res = {}
    for name in PAIRS4:
        eq, den, rec, tot = detail_pair(name, q0)
        lc = sum(d["contrib"] for (sp, st), d in rec.items() if sp == "ling")
        lA = sum(d.get("contrib_Aprof", 0.0) for (sp, st), d in rec.items() if sp == "ling")
        ld = sum(d["day"] for (sp, st), d in rec.items() if sp == "ling")
        l0 = sum(v["share"] for k, v in out[name].items() if k.startswith("ling|")) * BASE_TOT[name]
        res[name] = dict(g=lc / l0 if l0 else float("nan"), share=lc / tot, Aprof=(lA / lc) if lc else float("nan"),
                         day=ld / lc if lc else float("nan"), eq_ratio=eq / out[name + "|eq"])
    LING_MODE = None
    B["PRED_PROF"]["ling"] = LING_PROF0
    B["PCACHE"].clear()
    return res


BASE_TOT = {}
for name in PAIRS4:
    _eq, _den, _rec, _tot = detail_pair(name, q0)
    assert abs(_eq / out[name + "|eq"] - 1) < 1e-12
    BASE_TOT[name] = _tot
V0 = qf["v_hake"] * 0.629
print(f"\nLing variants (ling L number-mean 0.629 m, v_hake {qf['v_hake']} BL/s -> {V0:.3f} m/s; u small fish "
      f"{qf['u_smallfish']}, herring {qf['u_herring']} m/s; vrel(v, 0.1) {B['GP']['vrel'](V0, qf['u_smallfish']):.3f})")
LVAR = {
    "base": (None, None),
    "act0.35": (dict(f_act=0.35), None),            # central: 0.35 active cruising, 0.65 sit-and-wait
    "act0.50": (dict(f_act=0.50), None),            # bracket: more active
    "act0.25": (dict(f_act=0.25), None),            # bracket: less active
    "idle0.35": (dict(f_act=0.35, idle=True), None),  # bracket: inactive time does not forage
    "night0": (dict(nf=0.0), None),                 # bracket: no night foraging, cruising otherwise
    "act0.35+night0": (dict(f_act=0.35, nf=0.0), None),
    "prof2": (None, 2.0),                           # bracket: exp(2 m) profile (closer to the bed)
    "act0.35+prof2": (dict(f_act=0.35), 2.0),
    # stationary part scanning the whole disc (scan 1, as the derivation's bed-ambush branch) instead of the
    # cruiser's 0.5 [J]: the central of this note
    "act0.35s1": (dict(f_act=0.35, sw_scan=1.0), None),
    "act0.50s1": (dict(f_act=0.50, sw_scan=1.0), None),
    "act0.35s1+night0": (dict(f_act=0.35, sw_scan=1.0, nf=0.0), None),
    "act0.35s1+prof2": (dict(f_act=0.35, sw_scan=1.0), 2.0),
}
LRES = {}
for tag, (mode, ph) in LVAR.items():
    LRES[tag] = ling_run(mode, ph)
print(f"  {'variant':16s} " + " ".join(f"{n.split('->')[1][:12]:>30s}" for n in PAIRS4))
print(f"  {'':16s} " + " ".join(f"{'g / share / Aprof / day':>30s}" for n in PAIRS4))
for tag, r in LRES.items():
    print(f"  {tag:16s} " + " ".join(f"{r[n]['g']:7.3f} {r[n]['share']:7.4f} {r[n]['Aprof']:6.3f} {r[n]['day']:6.3f}"
                                    for n in PAIRS4))

# ------------------------------------------------------------------ windows (aw head, unchanged)
AW = os.path.join(RN, "Fönsterviktning", "scripts", "aw_stage_access.py")
src = open(AW).read()
head = src[:src.index('print("a(L) shape check')]
sys.argv = [AW, RN, SCR]
NS = {"__name__": "aw_head", "__file__": AW}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, AW, "exec"), NS)
sys.argv = ARGV
sa, pair_access, lweights, prey_dist, GROUP_W = NS["sa"], NS["pair_access"], NS["lweights"], NS["prey_dist"], NS["GROUP_W"]
IDENT = {m: ident for m, _, ident in NS["WC_BEN"]}
KEXP = 1.73          # window_hs.py central (band-A a(L) exponent)


def p_enc(ppr, q=0.8, r=0.3):
    ps = np.clip(0.973 - 1.230 * ppr, 0.0, 1.0)
    return q * ps / (1.0 - r * (1.0 - ps))


def window_p(member, ident, stage, kexp, group="ben"):
    """(f, f x mean p) for one member and stage, quarters averaged as pair_access does; prey tonnes inside the
    window weighted by p at each (prey length, predator length)."""
    window = sa.WINDOWS[member]
    Ls, ws = lweights(member, group, "aw", kexp)
    fq, fpq = [], []
    for qq in (1, 2, 3, 4):
        num = nump = den = 0.0
        for sp, w in GROUP_W[group](qq).items():
            if w <= 0:
                continue
            dist, share = prey_dist(qq, sp, stage)
            k = w * share
            if k <= 0:
                continue
            den += k
            if dist is None or (ident is not None and sp not in ident):
                continue
            L, v = dist
            for Lp, wp in zip(Ls, ws):
                m = L <= window[1] * Lp
                num += k * wp * v[m].sum()
                nump += k * wp * (v[m] * p_enc(L[m] / Lp)).sum()
        fq.append(num / den)
        fpq.append(nump / den)
    return float(np.mean(fq)), float(np.mean(fpq))


print("\nWindows per member (norm + aw, kexp +1.73) and the mean capture p of the encountered tonnes inside the window")
print("  (p at PPR 0.25 = 0.592 is the combined derivation's pair value)")
WIN = {}
for m in ("saithe", "hake", "anglerfish", "spurdog", "velvet belly"):
    fj, fa, _ = pair_access("ben", [(m, 1.0, IDENT[m])], "norm", "aw", kexp=KEXP)
    if sa.WINDOWS[m][0] == "ratio":
        fj2, fpj = window_p(m, IDENT[m], "J", KEXP)
        fa2, fpa = window_p(m, IDENT[m], "A", KEXP)
    else:
        fj2, fa2 = fj, fa
        fpj, fpa = fj * float(p_enc(0.25)), fa * float(p_enc(0.25))
    WIN[m] = dict(fJ=fj, fA=fa, pJ=fpj / fj2 if fj2 else float("nan"), pA=fpa / fa2 if fa2 else float("nan"))
    print(f"  {m:13s} window {sa.WINDOWS[m][:2]}  [f_J, f_A] [{fj:.3f}, {fa:.4f}] (check {fj2:.3f}, {fa2:.4f});"
          f" mean p inside: J {WIN[m]['pJ']:.3f}, A {WIN[m]['pA']:.3f}")
# ling: the library maps ling to anglerfish's window [J]; ling's own length mix with a gadoid ratio as variants
for rr in (0.30, 0.40, 0.50, 0.57):
    sa.WINDOWS["ling"] = ("ratio", rr, "ling")
    fj, fpj = window_p("ling", None, "J", KEXP)
    fa, fpa = window_p("ling", None, "A", KEXP)
    WIN[f"ling@{rr}"] = dict(fJ=fj, fA=fa, pJ=fpj / fj if fj else float("nan"), pA=fpa / fa if fa else float("nan"))
    print(f"  ling own length mix, ratio {rr:.2f}: [{fj:.3f}, {fa:.4f}]; mean p inside: J {fpj / fj:.3f}, A {fpa / fa:.3f}")
WIN["ling"] = dict(WIN["anglerfish"])
WIN["pollack"] = dict(WIN["saithe"])

# ------------------------------------------------------------------ flows by member and stage
JS = 0.155            # benthivore J/B, annual mean of the library start_share (decompose_wc_pair.py)
name = "predatory_fish->benthivorous_fish"
sh_sp = {}
for k, v in out[name].items():
    sp = k.split("|")[0]
    sh_sp[sp] = sh_sp.get(sp, 0.0) + v["share"]
rows = []
for sp, s in sh_sp.items():
    w = WIN.get(sp, dict(fJ=0, fA=0, pJ=0, pA=0))
    rows.append((sp, s, w["fJ"], w["fA"], JS * s * w["fJ"], (1 - JS) * s * w["fA"], w["pJ"], w["pA"]))
FJ = sum(r[4] for r in rows)
FA = sum(r[5] for r in rows)
print(f"\nPair window from the member shares: [{FJ / JS:.3f}, {FA / (1 - JS):.4f}] (library [0.392, 0.0388]); "
      f"J share of the encounter flow {FJ / (FJ + FA):.3f}")
print(f"  {'member':13s} {'a share':>7s} {'f_J':>6s} {'f_A':>7s} {'J flow':>7s} {'A flow':>7s} {'p J':>6s} {'p A':>6s}"
      "   (flows as shares of the pair's J and A encounter flows, library p)")
res_rows = {}
for sp, s, fj, fa, fJ, fA, pj, pa in sorted(rows, key=lambda r: -r[1]):
    res_rows[sp] = dict(a_share=s, fJ=fj, fA=fa, J_flow=fJ / FJ, A_flow=fA / FA, pJ=pj, pA=pa)
    print(f"  {sp:13s} {s:7.4f} {fj:6.3f} {fa:7.4f} {fJ / FJ:7.3f} {fA / FA:7.3f} {pj:6.3f} {pa:6.3f}")
# with the member/stage p instead of one 0.592
P0 = float(p_enc(0.25))
FJp = sum(r[4] * (r[6] if r[6] == r[6] else 0) / P0 for r in rows)
FAp = sum(r[5] * (r[7] if r[7] == r[7] else 0) / P0 for r in rows)
print(f"\nWith p per member and stage (relative to the combined pair p {P0:.3f}): J flow x{FJp / FJ:.3f}, A flow x{FAp / FA:.3f},"
      f" pair x{(FJp + FAp) / (FJ + FA):.3f}; J share {FJp / (FJp + FAp):.3f}")
# ------------------------------------------------------------------ member-level fish diet (encounter-implied)
# Each member as if it were its own predator: encounter on each fish prey = a (per member tonne, section 269 central)
# x combined factor (capture p x schooling F, research_notes/"Fangst och stim i fiskparen", central) x the member's
# own stage window (planktivores, whiting: the pair's member windows of the aw method; benthivores: as above, with
# the member/stage p) x prey biomass. Krill and zooplankton are left out on BOTH sides (no member split of the krill
# a here); the field diets are renormalised over the same fish prey. Prey biomass t/km2: planktivores 4.72 (J/B
# 0.38), benthivores 1.53 (J/B 0.155), whiting 0.446 (J/B 0.165), mackerel 0.85 (f 0.14 flat; a member window for
# mackerel is not derived, the pair value is used for every member that has the pair).
ALIB = {"predatory_fish->benthivorous_fish": 0.0418, "predatory_fish->whiting": 0.0359,
        "predatory_fish->planktivorous_fish": 0.0174, "predatory_fish->mackerel": 0.000137}
FAC = {"predatory_fish->benthivorous_fish": 2.96, "predatory_fish->whiting": 2.92,
       "predatory_fish->planktivorous_fish": 1.61, "predatory_fish->mackerel": 0.24}
BPREY = {"plk": (4.72, 0.38), "ben": (1.53, JS), "wht": (0.446, 0.165)}
MEMW = {}
for m in ("saithe", "hake", "anglerfish", "spurdog", "velvet belly"):
    for g in ("plk", "wht"):
        fj, fa, _ = pair_access(g, [(m, 1.0, None)], "norm", "aw", kexp=KEXP)
        MEMW[(m, g)] = (fj, fa)
for g in ("plk", "wht"):
    MEMW[("ling", g)] = MEMW[("anglerfish", g)]
    MEMW[("pollack", g)] = MEMW[("saithe", g)]
print("\nMember windows on planktivores and whiting (norm + aw, kexp 1.73; ling = anglerfish [J], pollack = saithe [J]):")
for (m, g), v in sorted(MEMW.items()):
    print(f"  {m:13s} {g}: [{v[0]:.3f}, {v[1]:.4f}]")
PAIRG = {"predatory_fish->planktivorous_fish": "plk", "predatory_fish->whiting": "wht"}
diet = {}
for sp in ("saithe", "hake", "ling", "anglerfish", "spurdog", "pollack", "velvet belly"):
    d = {}
    for name, a_lib in ALIB.items():
        mem = out[name]
        sh = sum(v["share"] for k, v in mem.items() if k.split("|")[0] == sp)
        w = sum(v["W"] for k, v in mem.items() if k.split("|")[0] == sp)
        if w <= 0 or sh <= 0:
            continue
        a_t = a_lib * FAC[name] * sh / w            # per member tonne, group-equivalent units
        if name == "predatory_fish->benthivorous_fish":
            W_ = WIN[sp]
            B_, js = BPREY["ben"]
            d["ben J"] = a_t * B_ * js * W_["fJ"] * (W_["pJ"] / P0 if W_["pJ"] == W_["pJ"] else 1)
            d["ben A"] = a_t * B_ * (1 - js) * W_["fA"] * (W_["pA"] / P0 if W_["pA"] == W_["pA"] else 1)
        elif name in PAIRG:
            g = PAIRG[name]
            B_, js = BPREY[g]
            fj, fa = MEMW[(sp, g)]
            d[g] = a_t * B_ * (js * fj + (1 - js) * fa)
        else:
            d["mackerel"] = a_t * 0.85 * 0.14
    diet[sp] = d
print("\nEncounter-implied FISH diet per member (shares over planktivores, whiting, benthivores J / A, mackerel), and the")
print("member's benthivore-A encounter rate per tonne relative to saithe's planktivore rate:")
for sp, d in diet.items():
    t = sum(d.values()) or 1.0
    print(f"  {sp:13s} " + ", ".join(f"{k} {v / t:.3f}" for k, v in d.items()))
# ------------------------------------------------------------------ ling's own windows on planktivores and whiting
LWIN = {}
for rr in (0.30, 0.40, 0.57):
    sa.WINDOWS["ling"] = ("ratio", rr, "ling")
    for g in ("plk", "wht"):
        fj, fa, _ = pair_access(g, [("ling", 1.0, None)], "norm", "aw", kexp=KEXP)
        LWIN[f"ling@{rr}|{g}"] = (fj, fa)
print("\nLing's own length mix on planktivores and whiting (norm + aw, kexp 1.73):")
for k, v in LWIN.items():
    print(f"  {k:14s} [{v[0]:.3f}, {v[1]:.4f}]")
print("  (anglerfish's, used for ling so far: plk [%.3f, %.4f], wht [%.3f, %.4f])"
      % (MEMW[("anglerfish", "plk")] + MEMW[("anglerfish", "wht")]))

# ling's encounter-implied fish diet under the variants (as the member diet above; own window at 0.30 / 0.40)
print("\nLing encounter-implied FISH diet (shares) per variant x window:")
LDIET = {}
for tag in ("base", "act0.35", "act0.35s1", "idle0.35", "act0.35s1+night0", "act0.35s1+prof2"):
    for wkey in ("anglerfish", "ling@0.3", "ling@0.4"):
        d = {}
        for name, a_lib in ALIB.items():
            mem = out[name]
            sh = sum(v["share"] for k, v in mem.items() if k.startswith("ling|"))
            w = sum(v["W"] for k, v in mem.items() if k.startswith("ling|"))
            a_t = a_lib * FAC[name] * sh / w * LRES[tag][name]["g"]
            if name == "predatory_fish->benthivorous_fish":
                W_ = WIN[wkey]
                B_, js = BPREY["ben"]
                d["ben J"] = a_t * B_ * js * W_["fJ"] * W_["pJ"] / P0
                d["ben A"] = a_t * B_ * (1 - js) * W_["fA"] * W_["pA"] / P0
            elif name in PAIRG:
                g = PAIRG[name]
                B_, js = BPREY[g]
                fj, fa = MEMW[("anglerfish", g)] if wkey == "anglerfish" else LWIN[f"{wkey}|{g}"]
                d[g] = a_t * B_ * (js * fj + (1 - js) * fa)
            else:
                d["mackerel"] = a_t * 0.85 * 0.14
        t = sum(d.values())
        LDIET[f"{tag}|{wkey}"] = d
        print(f"  {tag:16s} {wkey:10s} " + ", ".join(f"{k} {v / t:.3f}" for k, v in d.items())
              + f"   (sum of encounter rates relative to base|anglerfish: {t / sum(LDIET['base|anglerfish'].values()):.3f})")

json.dump(dict(members=out, windows=WIN, flows=res_rows, JS=JS, p0=P0, FJ=FJ, FA=FA, FJp=FJp, FAp=FAp, member_fish_diet=diet,
               member_windows={f'{m}|{g}': v for (m, g), v in MEMW.items()}, ling_variants=LRES,
               ling_windows=LWIN, ling_diet=LDIET),
          open(OUTJ, "w"), indent=1, default=float)
print(f"written {os.path.basename(OUTJ)}")
