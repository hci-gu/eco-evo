"""Scope A (gadoids only) against scope B (every member on the chain's gadoid eye) at the band stage (step 1).

The chain carries Ke per EYE KEY, not per member: one_rule_pairs.gmat(eye, Ke, kdeep) and krill_pairs_wc.light_g
give every member on the "gadoid" key the same Ke. Members on that key that are not gadoids (Gadiformes):
  * horse mackerel (Carangidae; TYPE "cruiser" in the fish pairs, vis "gadoid" in the krill pairs): mackerel ->
    planktivores, mackerel -> krill;
  * long rough dab (Pleuronectidae) and grey gurnard (Triglidae): benthivores -> krill;
  * long rough dab, dab, plaice, witch, lemon sole, starry ray, grey gurnard: benth_V of the benthivore fish pairs.
Hake and ling are Gadiformes (Merlucciidae, Lotidae) and count as gadoids. Herring and mackerel are on the clupeid
key (Ke_clup 1 lx), spurdog and velvet belly are non-visual, anglerfish is a fixed strike radius: Ke never reaches
them, in either scope.

Every term is linear in its members' contributions, so the band-stage a of scope A is the gadoid members at Ke 2e-3
plus the non-gadoid members at Ke 0.02. This script prints, per pair, the non-gadoid share of the equal-cell annual
a and a_B / a_live-Ke and a_A / a_live-Ke at the band stage (K_deep 0.09, rule A). It does not run steps 2-8.

Usage: python3 scope_a_shares.py <repo_root> <scratch_dir> [Ke_new, default 0.002]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
KE_NEW = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002
KE0, KD = 0.02, 0.09
os.makedirs(SCR, exist_ok=True)
BP = os.path.join(ROOT, "research_notes", "Overlapp i fiskparen", "scripts", "band_pairs.py")
src = open(BP).read()
MARK = "# ================================================================== 0. overlap geometry"
head = src[:src.index(MARK)]
save = list(sys.argv)
sys.argv = [BP, ROOT, SCR, os.path.join(SCR, "_unused_band.json")]
G = {"__name__": "band_pairs_head", "__file__": BP}
exec(compile(head, BP, "exec"), G)
sys.argv = save
O, K = G["O"], G["K"]
NONGAD_KRILL = {"horse mackerel", "long rough dab", "grey gurnard"}
NONGAD_FISH = {"horse mackerel"}
GADOID_B = {"cod", "haddock"}

# ------------------------------------------------------------------ krill pairs (rule A, band, member resolved)
G["RULE"].update(rule="A", RD=1.2, Lk=None)
ev = {}
for ke in (KE0, KE_NEW):
    q = O["meas"](dict(K["level"](1), Ke_gad=ke, k_deep=KD))
    ev[ke] = K["evaluate"](q, keep_members=True)[1]


def eq_of(a_tb):
    return float(O["SHARE_CLS"] @ O["krill_classes"](a_tb))


res = {"krill": {}, "fish": {}, "benth": {}}
print(f"Band stage (step 1), K_deep {KD}, rule A; Ke {KE0} -> {KE_NEW} lx. a ratios are to the band a at Ke {KE0}.")
print("\nKrill pairs: non-gadoid share of the a at Ke 0.02 / at the new Ke; a_B / a_0, a_A / a_0")
for gname, g in K["GROUPS"].items():
    W = {m["name"]: m["W"] for m in g["members"]}
    parts = {ke: {nm: W[nm] * r for (gg, nm), r in ev[ke].items() if gg == gname} for ke in ev}
    tot0 = sum(eq_of(a) for a in parts[KE0].values())
    totB = sum(eq_of(a) for a in parts[KE_NEW].values())
    ng0 = sum(eq_of(a) for nm, a in parts[KE0].items() if any(n in nm for n in NONGAD_KRILL))
    ngB = sum(eq_of(a) for nm, a in parts[KE_NEW].items() if any(n in nm for n in NONGAD_KRILL))
    totA = totB - ngB + ng0
    res["krill"][gname] = dict(ng_share0=ng0 / tot0, ng_shareB=ngB / totB, B=totB / tot0, A=totA / tot0)
    print(f"  {gname + ' -> krill':36s} non-gadoid {ng0 / tot0:6.3f} / {ngB / totB:6.3f};  B x{totB / tot0:.4f}  "
          f"A x{totA / tot0:.4f}  (A/B {totA / totB:.4f})")

# ------------------------------------------------------------------ fish pairs (band_fish_pair member shares)
print("\nFish pairs (band_fish_pair members >= 0.005 of the a): non-gadoid share; a_B / a_0, a_A / a_0")
q0 = dict(G["qk0"])
for name in G["FISH_PAIRS"]:
    r0 = G["band_fish_pair"](name, q0, "A", KE0, KD)
    rB = G["band_fish_pair"](name, q0, "A", KE_NEW, KD)
    ng0 = sum(v for k, v in r0["members"].items() if k.split("|")[0] in NONGAD_FISH)
    ngB = sum(v for k, v in rB["members"].items() if k.split("|")[0] in NONGAD_FISH)
    A = (rB["eq"] * (1 - ngB) + r0["eq"] * ng0) / r0["eq"]
    res["fish"][name] = dict(ng_share0=ng0, ng_shareB=ngB, B=rB["eq"] / r0["eq"], A=A)
    print(f"  {name:40s} non-gadoid {ng0:6.3f} / {ngB:6.3f};  B x{rB['eq'] / r0['eq']:.4f}  A x{A:.4f}")

# ------------------------------------------------------------------ benthivore fish pairs (benth_V by species)
print("\nBenthivore fish pairs (one_rule_pairs.benth_V by species): non-gadoid share of V; a_B / a_0, a_A / a_0")
LB = O["LB"]


def species_weights(pair):
    """benth_weights(pair) kept per species: {sp: (L, w)} with the same normalisation over all species."""
    rows = []
    for sp in O["GROUP_B"]:
        d = O["bclasses"](sp)
        for st, dd in d.groupby("stage"):
            if pair == "P":
                s = O["SCORE_P"].get(sp, [None] * 3)[st] if sp in O["SCORE_P"] else (0.0 if sp in O["FLAT"] else 0.05)
                wt = O["GROUP_B"][sp] * dd.s.sum() * s
                sel = dd
            else:
                memb = {"cod": 1.0, "starry ray": 0.5, "grey gurnard": 0.5}.get(sp, 0.0)
                sel = dd[dd.L_cm >= 20] if sp == "cod" else dd
                wt = O["GROUP_B"][sp] * sel.s.sum() * memb
            if wt <= 0 or len(sel) == 0:
                continue
            rows.append((sp, wt, sel))
    tot = sum(w for _, w, _ in rows)
    out = {}
    for sp, wt, sel in rows:
        s = sel.s / sel.s.sum()
        n = (s * 1e6 / sel.w_g).values
        L = sel.L.values
        w = wt / tot * n * O["vr_s"](0.4 * L, 0.10)
        Lo, wo = out.get(sp, (np.zeros(0), np.zeros(0)))
        out[sp] = (np.concatenate([Lo, L]), np.concatenate([wo, w]))
    return out


def V_sp(L, w, Ke):
    Eb = O["E0"][:, None] * O["trans_month"](dict(O["qk0"], k_deep=KD))[O["MONTH"]][:, O["IBED"]]
    Gm = Eb / (Eb + Ke)
    rs = np.minimum(O["r0_c"](O["qk0"], O["L_FISH_PREY"], O["C_FISH"]), 2 * L)
    rn = np.full_like(L, O["RNV_FISH"] if "RNV_FISH" in O else 0.3)
    rs = np.maximum(rs, rn)
    return (w * rn ** 2).sum() + (w * (rs ** 2 - rn ** 2)).sum() * Gm


IS_NIGHT, WCELL = O["IS_NIGHT"], O["WCELL"]
for name, (pair, a_lib, rho, thr) in O["BENTH_PAIRS"].items():
    SW = species_weights(pair)
    # check: the species split reproduces benth_V at Ke 0.02
    Vfull = O["benth_V"](pair, "A", KE0, KD)
    Vsum = sum(V_sp(L, w, KE0) for L, w in SW.values())
    assert np.allclose(Vfull, Vsum, rtol=1e-10), name
    Vl = O["benth_V"](pair, "lib", 0.02, 0.06)
    ld, ln = float((Vl[~IS_NIGHT] @ WCELL).mean()), float((Vl[IS_NIGHT] @ WCELL).mean())
    wd, wn = 1.0 / (1.0 + rho), rho / (1.0 + rho)

    def ratio(V):
        ser = np.where(IS_NIGHT[:, None], 2 * wn * V / ln, 2 * wd * V / ld)
        return float((ser @ WCELL).mean())
    V0 = sum(V_sp(L, w, KE0) for L, w in SW.values())
    VB = sum(V_sp(L, w, KE_NEW) for L, w in SW.values())
    VA = sum(V_sp(L, w, KE_NEW if sp in GADOID_B else KE0) for sp, (L, w) in SW.items())
    ng0 = ratio(sum(V_sp(L, w, KE0) for sp, (L, w) in SW.items() if sp not in GADOID_B)) / ratio(V0)
    ngB = ratio(sum(V_sp(L, w, KE_NEW) for sp, (L, w) in SW.items() if sp not in GADOID_B)) / ratio(VB)
    res["benth"][name] = dict(ng_share0=ng0, ng_shareB=ngB, B=ratio(VB) / ratio(V0), A=ratio(VA) / ratio(V0),
                              species=sorted(SW))
    print(f"  {name:40s} non-gadoid {ng0:6.3f} / {ngB:6.3f};  B x{ratio(VB) / ratio(V0):.4f}  "
          f"A x{ratio(VA) / ratio(V0):.4f}   (species: {', '.join(sorted(SW))})")
json.dump(res, open(os.path.join(SCR, "scope_a_shares.json"), "w"), indent=1)
