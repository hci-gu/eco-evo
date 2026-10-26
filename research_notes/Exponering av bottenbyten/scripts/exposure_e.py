"""Exposure / detectability factor e on the benthic pairs, built step by step from biology.

Usage: python3 -I exposure_e.py <length_biomass_longrun.csv> [n_mc] [variant]
  <length_biomass_longrun.csv>: research_notes/"Fiskgrupper i rutan"/benthivores/tables/ (NS-IBTS GOV Q1+Q3
      2001-25, 1-cm classes).
  variant: "central" (default) or "skold90" (infauna composition of Skold & Gunnarsson 1996, 90 m).

What the encounter a of sections 220/221 assumes (scripts/benth_encounter_split.py in
research_notes/"Infauna och rorlig epifauna", reproduced exactly here as a_enc):
    a_enc = sum_members wt x sum_L n_L v(L) w(L) p 21 600 s   [km2 per t of predator per 6 h tick]
    intake = a_enc x A, A = the accessible biomass (refuge exposed share phi x B, or f x B).
Implicitly every tonne of A inside the swept strip is (1) detectable at the moment of passage,
(2) taken with probability p, (3) taken WHOLE (the A tonne leaves the pool as eaten mass), and
(4) the flatfish see the strip at their daylight visual width.
A = phi B mixes item classes the encounter treats alike but biology does not
(research_notes/"Bentos atkomlighet for torskfiskar"/evidence_A_accessibility.md: phi = biomass in the
0-2 cm reach + exposed parts x time exposed):
    X1 exposed parts counted AS PARTS (Amphiura arm tips): detection on contact is assumed, but tips
       retract at disturbance and a strike may get part of the tip;
    X2 whole bodies buried in 0-2 cm (bivalves, polychaetes): "Food covered by sand was not found"
       (Brawn 1969 [A]); detectable only while a siphon / palp / head is at the surface, and a strike
       may take only that part (siphon nip, tail or head crop) unless the animal is pulled out;
    X3 small crustaceans (amphipods, cumaceans): whole, mobile, detectable when on/above the surface.
So a_eff = sum_members wt x sum_L n_L v w_eff 21 600 x sum_k s_k t_k, where s_k is the share of A in
class k and t_k the TAKEABLE share of class k's A-mass per unit swept area (detection x capture x
yield). e = a_eff / a_enc. p of a_enc is replaced by the class capture terms (not multiplied on top).
Fish-side: flatfish width at the dim bottom (step f) is kept separate from e (factor L_w).

Not in e (kept where they live): phi / f (reach, emergence, time out of burrow), eat share (foraging
time), Nephrops size window (already in its f 0.02 = 0.07 x 0.29), the predators' bed access by
non-bed members (saithe, mackerel; a foraging-mode share, AGENTS.md says do not reintroduce it).

Every factor is (low, central, high). MC: the a_enc factors are log-uniform as in the original script;
the new e factors are triangular with the mode at the central value (in log space when low > 0). Tags in the comments: [F]/[A]/[S] as in tables/*_evidence.txt; J judgement.
"""
import math
import sys

import numpy as np
import pandas as pd

LB = pd.read_csv(sys.argv[1])
NMC = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
VARIANT = sys.argv[3] if len(sys.argv) > 3 else "central"
RNG = np.random.default_rng(20261026)
TICK = 21600.0

# ---------------------------------------------------------------------------------------------
# 1. a_enc, reproduced from benth_encounter_split.py (same members, modes, factors, seed logic)
# ---------------------------------------------------------------------------------------------
GROUP = {"haddock": 38.62, "cod": 31.38, "long rough dab": 12.63, "dab": 4.49, "plaice": 3.80,
         "witch": 1.67, "starry ray": 1.47, "grey gurnard": 0.89, "lemon sole": 0.87}
SP_MODE = {"haddock": "gadoid", "cod": "gadoid", "dab": "flat", "plaice": "flat", "witch": "flat",
           "lemon sole": "flat", "long rough dab": "lrd", "grey gurnard": "gurnard", "starry ray": "ray"}
MODE = {
    "gadoid": dict(v=(0.2, 0.4, 0.8), wi=(0.07, 0.15, 0.4), pi=(0.25, 0.5, 0.9),
                   we=(0.3, 0.6, 1.2), pe=(0.1, 0.3, 0.7)),
    "flat": dict(v=(0.04, 0.12, 0.3), wi=(0.25, 0.6, 1.2), pi=(0.4, 0.7, 1.0),
                 we=(0.3, 0.8, 1.5), pe=(0.15, 0.4, 0.8)),
    "lrd": dict(v=(0.04, 0.12, 0.3), wi=(0.25, 0.6, 1.2), pi=(0.25, 0.5, 0.9),
                we=(0.3, 0.8, 1.5), pe=(0.15, 0.4, 0.8)),
    "gurnard": dict(v=(0.1, 0.2, 0.4), wi=(0.15, 0.3, 0.6), pi=(0.25, 0.5, 0.9),
                    we=(0.3, 0.6, 1.2), pe=(0.1, 0.3, 0.7)),
    "ray": dict(v=(0.05, 0.15, 0.3), wi=(0.2, 0.4, 0.8), pi=(0.25, 0.5, 0.9),
                we=(0.3, 0.6, 1.2), pe=(0.1, 0.3, 0.7)),
}


def classes(sp):
    d = LB[(LB.species == sp) & (LB.share_2001_25 > 0)].copy()
    d["s"] = d.share_2001_25 / d.share_2001_25.sum()
    d["L"] = (d.L_cm + 0.5) / 100.0
    return d


CL = {sp: classes(sp) for sp in GROUP}
WT = {sp: GROUP[sp] / sum(GROUP.values()) for sp in GROUP}
M2L = {sp: float((CL[sp].s * 1e6 / CL[sp].w_g * CL[sp].L ** 2).sum()) for sp in GROUP}


def logu(lo, hi, n=NMC):
    if lo == hi:
        return np.full(n, float(lo))
    if lo <= 0:
        return RNG.uniform(lo, hi, n)
    return np.exp(RNG.uniform(math.log(lo), math.log(hi), n))


def draw(spec):
    return {k: logu(v[0], v[2]) for k, v in spec.items()}


DRAW = {m: draw(s) for m, s in MODE.items()}


def a_enc(pair):
    wk, pk = ("wi", "pi") if pair == "inf" else ("we", "pe")
    c, mc, per = 0.0, np.zeros(NMC), {}
    for sp, wt in WT.items():
        md, dd = MODE[SP_MODE[sp]], DRAW[SP_MODE[sp]]
        per[sp] = md["v"][1] * md[wk][1] * md[pk][1] * TICK * M2L[sp] / 1e6
        c += wt * per[sp]
        mc += wt * dd["v"] * dd[wk] * dd[pk] * TICK * M2L[sp] / 1e6
    return c, mc, per


# ---------------------------------------------------------------------------------------------
# 2. Fish side: flatfish width at the dim bottom (step f), kept OUT of e
# ---------------------------------------------------------------------------------------------
# Daytime bottom light, median cell ~3e-3 lx central profile (benth_encounter stdout), 1e-10 at night.
# Pacific flatfish visual foraging "decreased rapidly below 1e-4 umol photons m-2 s-1" (Hurst et al.
# 2007 [A]) ~5e-3 lx at 500 nm (J conversion): the bottom is at or below the knee by day, dark at night.
# Flatfish still crop arms and siphons with little light (Kaiser & Ramsay 1997 [F], dab at 40 m "with
# little if any light penetration"; plaice siphon attacks "independent of light", Burrows 1994b [A];
# plaice localise a chemical jet from 5 cm, de Groot 1969 [F]). Chemo-tactile width 0.1-0.3 L (J, B).
FLAT_VIS_SHARE = (0.0, 0.1, 0.3)       # share of foraging time at the visual width (J)
FLAT_W_DARK_INF = (0.1, 0.2, 0.3)      # x L, chemo-tactile width on infauna (J from B)
FLAT_W_DARK_EPI = (0.15, 0.3, 0.5)     # x L, lateral line / smell on MOVING epifauna (J)

# ---------------------------------------------------------------------------------------------
# 3. Infauna: composition of A and takeable share per class
# ---------------------------------------------------------------------------------------------
# Taxon weights in B and accessible share f_taxon (evidence_A_accessibility.md table, central; the
# library phi 0.2 is their weighted result). Two depths, grid = 0.6 x 100 m + 0.4 x 300 m (J: the
# infauna biomass sits more on the shallower half; K 132 vs 66 t/km2).
COMP = {
    "central": {100: {"amphiura": (0.50, 0.06), "polychaete": (0.30, 0.25), "bivalve": (0.15, 0.30),
                      "crustacean": (0.05, 0.70)},
                300: {"amphiura": (0.15, 0.08), "polychaete": (0.55, 0.25), "bivalve": (0.25, 0.30),
                      "crustacean": (0.05, 0.70)}},
    # Skold & Gunnarsson 1996 [F], 90 m Skagerrak slope: A. filiformis + A. chiajei 58 % of WW,
    # molluscs 3.6 %, polychaetes 7 %, small crustaceans 2 %; the remaining ~29 % (other echinoderms,
    # large burrowers) taken as out of reach (J). Used at both depths as a sensitivity variant.
    "skold90": {100: {"amphiura": (0.58, 0.06), "polychaete": (0.07, 0.25), "bivalve": (0.036, 0.30),
                      "crustacean": (0.02, 0.70)},
                300: {"amphiura": (0.58, 0.06), "polychaete": (0.07, 0.25), "bivalve": (0.036, 0.30),
                      "crustacean": (0.02, 0.70)}},
}[VARIANT]
GRIDW = {100: 0.6, 300: 0.4}

# Takeable-share terms (J unless tagged). "g" = share of successful strikes that take the WHOLE
# animal (pulled out / sucked out of the top 2 cm); otherwise only the exposed part ("tip").
INF = {
    # X1 Amphiura arm tips (the A-mass IS the exposed tips)
    "amphiura": dict(
        pre=(0.2, 0.5, 1.0),   # tips still raised when the strip reaches them: "When subjected to a
                               # disturbance from above, individuals immediately retracted ... for
                               # extended periods" (Solan & Kennedy 2002 [F]); arms "dropped toward the
                               # sediment" when dab fed upstream (Rosenberg & Selander 2000 [S]) (J)
        det=(0.7, 0.9, 1.0),   # a raised tip in the strip is touched / seen (J)
        cap=(0.3, 0.6, 1.0),   # strike gets (part of) the tip before it withdraws; unmeasured; 96 % of
                               # individuals regenerating (Duineveld & Van Noort 1986 [A]) says strikes
                               # often succeed (J)
        g=(0.0, 0.0, 0.0),     # discs at 4-6 cm are out of reach unless the bed is disturbed: dab
                               # "graze the exposed arms, while leaving the oral disc intact" (Kaiser &
                               # Ramsay 1997 [F])
        tip=(0.5, 0.8, 1.0)),  # share of the raised tip mass taken per success (J)
    # X2 shallow bivalves (Abra nitida, Thyasira; whole bodies in 0-2 cm)
    "bivalve": dict(
        pre=(0.08, 0.15, 0.25),  # siphon out: Abra 25 % of time at 10 C, 55 % at 15 C (Gremare 2004 [F]);
                                 # ~13 % at 6 C by extrapolation (J); thyasirids show no siphon (J low)
        det=(0.5, 0.8, 1.0),     # a siphon tip in the strip is detected (J)
        cap=(0.3, 0.6, 1.0),     # strike before retraction (J; bivalves "withdrew their foot and
                                 # siphon(s)", Zwarts & Wanink 1993 [F])
        g_gad=(0.2, 0.5, 0.8),   # gadoids suck prey "on and below the bottom surface" (Mattson 1990 [A]);
                                 # haddock aim at "the entire fraction of small to medium-sized
                                 # endobenthos", feeding depth rising with size (Mattson 1992 [A]) (J)
        g_flat=(0.05, 0.2, 0.5), # plaice nips usually crop the siphon, sometimes pull the clam out (Ansell
                                 # et al. 1999 [A]); siphon tips only from adult bivalves (de Vlas 1985 via
                                 # van der Veer 2011 [F/S]) (J)
        tip=(0.003, 0.01, 0.03)),  # siphon tip share of body mass: 0.6 %/cm (Zwarts et al. 1994 [A]),
                                   # "Only siphon tips" = 3 % of biomass (de Vlas 1985 via van der Veer
                                   # 2011 [S]); Abra siphon ~0.5-1 cm above the surface (J)
    # X2 polychaetes in 0-2 cm (cirratulids, spionids, small errant worms)
    "polychaete": dict(
        pre=(0.1, 0.25, 0.5),    # a part at the surface: spionids out of the tube 24-79 % of time at
                                 # 13-14 C (Lindsay & Woodin 1995 [F]); cirratulid palps roam the surface
                                 # [S]; cold deep mud lower (J)
        det=(0.5, 0.8, 1.0),
        cap=(0.3, 0.6, 1.0),     # haddock did not take tube spionids, "ability to withdraw below the
                                 # feeding depths" (Schuckel et al. 2010 [A]) (J)
        g_gad=(0.2, 0.5, 0.8),
        g_flat=(0.1, 0.3, 0.6),  # Wadden Sea flatfish: Pygospio heads and tentacles cropped, other
                                 # polychaetes eaten whole (van der Veer 2011 [F]) (J)
        tip=(0.05, 0.1, 0.15)),  # head / palps / tail tip share (Arenicola tail tips 15 % of biomass,
                                 # de Vlas 1979b via van der Veer 2011 [S]) (J)
    # X3 small crustaceans (amphipods, cumaceans): whole animals, mobile
    "crustacean": dict(
        pre=(0.3, 0.5, 0.8),     # on/above the surface: Diastylis "never leaves the sediment while light
                                 # remains", swims at night (Anger & Valentin 1976 [S]); cod specialise on
                                 # "nocturnally swarming males of cumaceans and amphipods" (Mattson 1990
                                 # [A]) (J)
        det=(0.7, 0.9, 1.0),
        cap=(0.3, 0.6, 0.9),     # free crustaceans: plaice/flounder 99-100 % on Corophium, turbot 84-89 %
                                 # on mysids (Holmes 1979 [F-internal]); gadoid lower (J)
        g=(1.0, 1.0, 1.0),
        tip=(1.0, 1.0, 1.0)),
}
# Size of a takeable item (mg wet weight) and strike + handling time (s), for the behavioural
# handling check (step e). Arm tip: regenerated arm DW = 0.01056 L^1.288 mg (Skold et al. 1994 [F]) ->
# 0.21-0.50 mg DW for 10-20 mm; DW/WW 0.318 (Skold 1994 [F]) -> 0.7-1.6 mg WW. Abra / small worms:
# Dauwe 1998 mean individual 0.5 mg AFDW at 280 m [F via evidence_A] -> ~4 mg WW at AFDW/WW 0.13 (J).
# Strike: plaice strike "after one or two seconds" in the arched posture; median 8 s between attacks
# on surface worms, search included (Holmes 1979 [F-internal]); cod ~1 s per 13 mm mysid (Persson 2012 [F],
# h 0.026 s/J, J per item).
ITEM = {"arm_tip": (0.7, 1.0, 1.6), "siphon_tip": (0.02, 0.05, 0.2), "bivalve_whole": (2.0, 5.0, 20.0),
        "worm_whole": (1.0, 4.0, 10.0), "worm_part": (0.1, 0.4, 1.0), "crustacean": (1.0, 2.0, 5.0)}
T_STRIKE = (1.0, 2.0, 8.0)

# ---------------------------------------------------------------------------------------------
# 4. Epifauna (E1 pools): composition of A and takeable share per taxon and member
# ---------------------------------------------------------------------------------------------
# Exposed pool B 3.06 t/km2 (central E1): crabs/hermits/Munida 0.65, northern shrimp 0.18, small
# shrimps 0.10, mysids 0.07 ("Havskrafta och exponerad epifauna"/summary.md). Within the crab member
# brachyurans 0.50, hermits 0.28, Munida 0.22 (evidence_B_other_epifauna.md central 100/300 m, J).
# f per taxon as the E1 build (split_e1.py): benthivores crabs 0.6, Pandalus 0.75, small 0.6;
# predators crabs 0.6 x 0.25 (small crabs only; a size window already IN f), Pandalus 0.75, small 0.6.
EPI_B = {"brachyuran": 0.65 * 0.50, "hermit": 0.65 * 0.28, "munida": 0.65 * 0.22, "pandalus": 0.18,
         "small_shrimp": 0.10, "mysid": 0.07}
F_BEN = {"brachyuran": 0.6, "hermit": 0.6, "munida": 0.6, "pandalus": 0.75, "small_shrimp": 0.6, "mysid": 0.6}
F_PRED = {"brachyuran": 0.15, "hermit": 0.15, "munida": 0.15, "pandalus": 0.75, "small_shrimp": 0.6,
          "mysid": 0.6}
# Takeable share relative to an exposed reference prey (D_epifauna_evidence.txt):
#   E  = share of the time out during which a passing fish can detect it;
#   pr = capture relative to an exposed reference prey (a_enc's p_e is the reference);
#   S  = biomass share inside the member's size window (member-specific, so it belongs in a, not f).
EPI = {
    "brachyuran": dict(E=(0.7, 0.9, 1.0), pr=(0.7, 1.0, 1.0)),   # Glass 1985 [F]; Arnott & Pihl 2000 [F]
    "hermit": dict(E=(0.8, 1.0, 1.0), pr=(0.7, 1.0, 1.0)),       # shell no refuge from cod/haddock (diets, D)
    "munida": dict(E=(0.6, 0.75, 0.9), pr=(0.7, 1.0, 1.0)),      # at burrow mouths (refuge note, J)
    "pandalus": dict(E=(0.9, 1.0, 1.0), pr=(0.8, 1.0, 1.0)),     # schools on the bottom, no burial (J)
    # small shrimps: the lab burier/exposed ratio (Crangon / Palaemon 0.086 on sand, Persson 2012 [F];
    # 0.20-0.32 per capita, Ljungberg 2012 [F]) is a PRODUCT of time out x E x capture, so it replaces
    # f 0.6 x 1: (time out x E x pr) = 0.35 (0.1-0.6) J -> relative to f 0.6:
    "small_shrimp": dict(E=(0.1 / 0.6, 0.35 / 0.6, 1.0), pr=(1.0, 1.0, 1.0)),
    "mysid": dict(E=(0.8, 1.0, 1.0), pr=(1.0, 1.0, 1.0)),        # Neomysis a 57 vs Palaemon 21 (Persson [F])
}
# Size windows S (biomass share swallowable), D_crab_window.py: brachyurans CW <= 0.06-0.09 TL (Arnott &
# Pihl 2000 [F]); cod 0.45 (0.2-0.7), haddock (k 0.7 x cod) and whiting ~0.03 (0-0.1); flatfish,
# gurnard, ray J. Hermits and Munida not covered by k (narrow, small at 100-200 m): J.
S_WIN = {
    "cod": {"brachyuran": (0.2, 0.45, 0.7), "hermit": (0.7, 1.0, 1.0), "munida": (0.6, 0.8, 1.0)},
    "haddock": {"brachyuran": (0.0, 0.03, 0.1), "hermit": (0.4, 0.7, 1.0), "munida": (0.3, 0.5, 0.8),
                "pandalus": (0.5, 0.8, 1.0)},
    "flat": {"brachyuran": (0.0, 0.03, 0.1), "hermit": (0.2, 0.5, 0.8), "munida": (0.05, 0.2, 0.4),
             "pandalus": (0.3, 0.5, 0.8)},
    "gurnard": {"brachyuran": (0.02, 0.1, 0.3), "hermit": (0.5, 0.8, 1.0), "munida": (0.3, 0.5, 0.8)},
    "ray": {"brachyuran": (0.1, 0.3, 0.6), "hermit": (0.5, 0.8, 1.0), "munida": (0.3, 0.6, 0.9)},
}
S_KEY = {"cod": "cod", "haddock": "haddock", "dab": "flat", "plaice": "flat", "witch": "flat",
         "lemon sole": "flat", "long rough dab": "flat", "grey gurnard": "gurnard", "starry ray": "ray"}
# Predators -> exposed: whiting is ~92 % of the pooled bed consumption ("Rovfiskens uppdelning").
# crabs: whiting S 0.03 (0-0.1) against the 0.25 already in f_p -> 0.12; Pandalus: adult shrimp 0.35-0.7
# of a median whiting's length, beyond Staniland's tested range (J) -> S 0.4 (0.2-0.8).
PRED_S = {"brachyuran": (0.0, 0.12, 0.4), "hermit": (0.0, 0.12, 0.4), "munida": (0.0, 0.12, 0.4),
          "pandalus": (0.2, 0.4, 0.8), "small_shrimp": (1.0, 1.0, 1.0), "mysid": (1.0, 1.0, 1.0)}
# Nephrops (benthivores -> norway_lobster): emergence and the 0.29 size window are in f 0.02; e is the
# capture of an emerged lobster at its burrow mouth (tail-flip retreat), J 0.5 (0.2-1.0) vs p_e.
NEP_CAP = (0.2, 0.5, 1.0)


def c(t):
    return t[1]


def mcv(t):
    """New factors: triangular with the mode at the central value (log space when all > 0), so a
    central near one end of its range is not pulled to the middle as a log-uniform draw would."""
    lo, md, hi = t
    if lo == hi:
        return np.full(NMC, float(lo))
    if lo > 0:
        return np.exp(RNG.triangular(math.log(lo), math.log(md), math.log(hi), NMC))
    return RNG.triangular(lo, md, hi, NMC)


# ---------------------------------------------------------------------------------------------
# 5. Engine state for the tests (today's pools; h at the revised ceiling 26.18 MJ/t/tick,
#    tables/C_cmax_bottom_stdout.txt CENTRAL; old 27.1)
# ---------------------------------------------------------------------------------------------
CEIL_NEW, CEIL_OLD, CEIL_PRED = 26.18, 27.1, 25.5
A_INF = 0.2 * 93.6            # infauna accessible today (equilibrium mean ~93.6 t/km2, phi 0.2)
A_EPI_B = 3.06 * sum(EPI_B[k] * F_BEN[k] for k in EPI_B) / sum(EPI_B.values())
A_EPI_P = 3.06 * sum(EPI_B[k] * F_PRED[k] for k in EPI_B) / sum(EPI_B.values())
A_NEP = 4.57 * 0.02
H = {"inf": 2600 / CEIL_NEW, "epi": 3750 / CEIL_NEW, "nep": 3600 / CEIL_NEW, "plk": 7125 / CEIL_NEW,
     "prd": 5252 / CEIL_NEW, "pred_epi": 3750 / CEIL_PRED}

out = []


def say(s=""):
    print(s)


say(f"variant {VARIANT}; n_mc {NMC}")
say("HEADLINE VALUES = MC medians (5-95 %). 'product of modes' = every factor at its central value; it is")
say("upper-biased where a mode sits at its range bound (several pr, E and S factors), so it is not the headline.")
say("=" * 100)
ai_c, ai_mc, ai_per = a_enc("inf")
ae_c, ae_mc, ae_per = a_enc("epi")
say(f"a_enc reproduced: infauna {ai_c:.4f} (MC 5-95 % {np.percentile(ai_mc, 5):.3f}-{np.percentile(ai_mc, 95):.3f}),"
    f" epifauna {ae_c:.4f} (MC {np.percentile(ae_mc, 5):.3f}-{np.percentile(ae_mc, 95):.3f}); library 0.25 / 0.44")

# ---- infauna composition of A ----
say("\n1. INFAUNA: composition of A (share of the accessible tonnes by item class)")
share = {}
for z in (100, 300):
    tot = sum(w * f for w, f in COMP[z].values())
    share[z] = {k: w * f / tot for k, (w, f) in COMP[z].items()}
    say(f"  {z} m: phi = {tot:.3f}; " + ", ".join(f"{k} {v:.2f}" for k, v in share[z].items()))
shG = {k: sum(GRIDW[z] * share[z][k] * sum(w * f for w, f in COMP[z].values()) for z in GRIDW) /
       sum(GRIDW[z] * sum(w * f for w, f in COMP[z].values()) for z in GRIDW) for k in INF}
say("  grid (0.6 x 100 m + 0.4 x 300 m, by accessible tonnes): " + ", ".join(f"{k} {v:.2f}" for k, v in shG.items()))


def takeable(k, mode, mc=False):
    """Takeable share of class k's A-mass per unit swept area (replaces p of a_enc)."""
    f = mcv if mc else c
    d = INF[k]
    gk = "g" if "g" in d else ("g_gad" if mode in ("gadoid", "gurnard", "ray") else "g_flat")
    g = f(d[gk])
    return f(d["pre"]) * f(d["det"]) * f(d["cap"]) * (g + (1 - g) * f(d["tip"]))


say("\n   takeable share t_k per class (central), gadoid / flatfish mode:")
for k in INF:
    say(f"     {k:<11} {takeable(k, 'gadoid'):.4f} / {takeable(k, 'flat'):.4f}")


def width_factor(sp, pair, mc=False):
    """Flatfish width at the dim bottom relative to the a_enc width (step f; 1 for non-flatfish)."""
    mode = SP_MODE[sp]
    if mode not in ("flat", "lrd"):
        return 1.0
    f = mcv if mc else c
    vis = f(FLAT_VIS_SHARE)
    w_light = MODE[mode]["wi" if pair == "inf" else "we"]
    w_dark = f(FLAT_W_DARK_INF if pair == "inf" else FLAT_W_DARK_EPI)
    # the ratio uses the central light width of a_enc (J simplification)
    return (vis * w_light[1] + (1 - vis) * w_dark) / w_light[1]


def a_eff_inf(mc=False, with_width=True):
    tot = np.zeros(NMC) if mc else 0.0
    tot_noW = np.zeros(NMC) if mc else 0.0
    for sp, wt in WT.items():
        mode = SP_MODE[sp]
        md = MODE[mode]
        if mc:
            dd = DRAW[mode]
            base = dd["v"] * dd["wi"] * TICK * M2L[sp] / 1e6
        else:
            base = md["v"][1] * md["wi"][1] * TICK * M2L[sp] / 1e6
        tk = sum(shG[k] * takeable(k, mode, mc) for k in INF)
        wf = width_factor(sp, "inf", mc)
        tot = tot + wt * base * tk * wf
        tot_noW = tot_noW + wt * base * tk
    return tot, tot_noW


aI_c, aI_noW_c = a_eff_inf()
aI_mc, aI_noW_mc = a_eff_inf(mc=True)
p5, p50, p95 = np.percentile(aI_mc, [5, 50, 95])
eI_mc = aI_noW_mc / ai_mc
say(f"\n   a_eff (benthivores -> infauna) product of modes {aI_c:.4f} km2/t/tick; MC 5/50/95 % {p5:.4f} {p50:.4f} {p95:.4f}")
say(f"   e = a_eff(no width) / a_enc: product of modes {aI_noW_c / ai_c:.3f}; MC 5/50/95 % "
    f"{np.percentile(eI_mc, 5):.3f} {np.percentile(eI_mc, 50):.3f} {np.percentile(eI_mc, 95):.3f}")
say(f"   flatfish width factor (step f) on the pair: x{aI_c / aI_noW_c:.3f}; combined a_eff / a_enc {aI_c / ai_c:.3f}")
say("   contribution to a_eff by class (central): " + ", ".join(
    f"{k} {sum(WT[sp] * MODE[SP_MODE[sp]]['v'][1] * MODE[SP_MODE[sp]]['wi'][1] * TICK * M2L[sp] / 1e6 * shG[k] * takeable(k, SP_MODE[sp]) * width_factor(sp, 'inf') for sp in WT) / aI_c:.2f}"
    for k in INF))

# ---- behavioural handling (step e): is b comparable with the digestion h? ----
say("\n2. BEHAVIOURAL HANDLING per item (step e) vs digestion h (Jeschke SSS: binds only if b ~ h or above)")
say("   b = W / ((m_item / t_strike) x 21 600 s), ticks per t prey per t predator; h_infauna = "
    f"{H['inf']:.1f} at the revised ceiling")
for W in (100.0, 300.0, 1000.0):
    row = []
    for it, m in ITEM.items():
        b = W / ((m[1] / 1000.0 / T_STRIKE[1]) * TICK)
        b_hi = W / ((m[0] / 1000.0 / T_STRIKE[2]) * TICK)
        row.append(f"{it} {b:.0f} ({b_hi:.0f})")
    say(f"   W {W:5.0f} g: " + "; ".join(row) + "   [central m, t 2 s (low m, t 8 s)]")
# intake-mass-weighted b of the infauna mix for a 300 g gadoid and flatfish
for mode in ("gadoid", "flat"):
    num = den = 0.0
    for k in INF:
        d = INF[k]
        g = c(d["g"]) if "g" in d else c(d["g_gad" if mode == "gadoid" else "g_flat"])
        mass_share = shG[k] * c(d["pre"]) * c(d["det"]) * c(d["cap"])
        parts = {"amphiura": [("arm_tip", 1.0)], "crustacean": [("crustacean", 1.0)],
                 "bivalve": [("bivalve_whole", g), ("siphon_tip", 1 - g)],
                 "polychaete": [("worm_whole", g), ("worm_part", 1 - g)]}[k]
        for it, pr in parts:
            y = 1.0 if it in ("arm_tip", "crustacean", "bivalve_whole", "worm_whole") else c(d["tip"])
            m_eaten = mass_share * pr * y              # mass eaten per unit area swept via this item type
            n_items = m_eaten / ITEM[it][1]            # items per unit area (relative)
            num += n_items * T_STRIKE[1]
            den += m_eaten
    b_mix = 300.0 / ((den / num) / 1000.0 * TICK)
    say(f"   infauna mix, 300 g {mode}: mean mass per strike {den / num * 1000 / T_STRIKE[1] * T_STRIKE[1] / 1000:.2f} mg,"
        f" b_mix {b_mix:.1f} ticks = {b_mix / H['inf']:.2f} x h")

# ---- epifauna ----
say("\n3. EXPOSED EPIFAUNA (E1 pool): takeable share relative to an exposed reference prey")
shB = {k: EPI_B[k] * F_BEN[k] for k in EPI_B}
s = sum(shB.values())
shB = {k: v / s for k, v in shB.items()}
shP = {k: EPI_B[k] * F_PRED[k] for k in EPI_B}
s = sum(shP.values())
shP = {k: v / s for k, v in shP.items()}
say("   benthivore A composition: " + ", ".join(f"{k} {v:.2f}" for k, v in shB.items()) + f"; A today {A_EPI_B:.2f} t/km2")
say("   predator A composition:   " + ", ".join(f"{k} {v:.2f}" for k, v in shP.items()) + f"; A today {A_EPI_P:.2f} t/km2")


def e_epi_member(sp, mc=False):
    f = mcv if mc else c
    tot = 0.0
    for k, w in shB.items():
        win = S_WIN[S_KEY[sp]].get(k, (1.0, 1.0, 1.0))
        tot = tot + w * f(EPI[k]["E"]) * f(EPI[k]["pr"]) * f(win)
    return tot


def a_eff_epi(mc=False):
    tot = np.zeros(NMC) if mc else 0.0
    tot_noW = np.zeros(NMC) if mc else 0.0
    for sp, wt in WT.items():
        mode = SP_MODE[sp]
        md = MODE[mode]
        if mc:
            dd = DRAW[mode]
            base = dd["v"] * dd["we"] * dd["pe"] * TICK * M2L[sp] / 1e6
        else:
            base = md["v"][1] * md["we"][1] * md["pe"][1] * TICK * M2L[sp] / 1e6
        ee = e_epi_member(sp, mc)
        wf = width_factor(sp, "epi", mc)
        tot = tot + wt * base * ee * wf
        tot_noW = tot_noW + wt * base * ee
    return tot, tot_noW


say("   e per member (central): " + ", ".join(f"{sp} {e_epi_member(sp):.2f}" for sp in WT))
aE_c, aE_noW_c = a_eff_epi()
aE_mc, aE_noW_mc = a_eff_epi(mc=True)
eE_mc = aE_noW_mc / ae_mc
say(f"   benthivores -> exposed epifauna: a_enc {ae_c:.3f}; a_eff product of modes {aE_c:.4f}, MC 5/50/95 % "
    f"{np.percentile(aE_mc, 5):.4f} {np.percentile(aE_mc, 50):.4f} {np.percentile(aE_mc, 95):.4f}")
say(f"   e (no width) product of modes {aE_noW_c / ae_c:.3f}, MC 5/50/95 % {np.percentile(eE_mc, 5):.3f} "
    f"{np.percentile(eE_mc, 50):.3f} {np.percentile(eE_mc, 95):.3f}; flatfish width x{aE_c / aE_noW_c:.3f}; "
    f"combined {aE_c / ae_c:.3f}")

ePc = sum(w * c(EPI[k]["E"]) * c(EPI[k]["pr"]) * c(PRED_S[k]) for k, w in shP.items())
ePmc = sum(w * mcv(EPI[k]["E"]) * mcv(EPI[k]["pr"]) * mcv(PRED_S[k]) for k, w in shP.items())
a_pred_enc = 0.33
say(f"   predators -> exposed epifauna (whiting-dominated; a_enc 0.33 per whole-group tonne): e product of modes {ePc:.3f},"
    f" MC 5/50/95 % {np.percentile(ePmc, 5):.3f} {np.percentile(ePmc, 50):.3f} {np.percentile(ePmc, 95):.3f};"
    f" a_eff (modes) {a_pred_enc * ePc:.3f}, MC median {a_pred_enc * np.percentile(ePmc, 50):.3f}"
    f" ({a_pred_enc * np.percentile(ePmc, 5):.3f}-{a_pred_enc * np.percentile(ePmc, 95):.3f}; a_enc's own spread"
    " 0.10-1.5 not included)")
say("     (the bed-access gap of non-bed members, ~2.7x in 'Fodoniva och mattnad i falt', is NOT in e)")
say("   benthivores -> norway_lobster: emergence and the 0.29 size window are in f 0.02; an emerged lobster's"
    " tail-flip retreat into the burrow is what a_enc's p_e 0.3 already stands for -> e 1.0 (0.4-1.0, J);"
    " no measurement of fish capture success on emerged Nephrops was found")

# ---- 4. tests ----
say("\n4. INDEPENDENT TESTS (none of these numbers entered e)")
aI_med, aE_med = float(np.percentile(aI_mc, 50)), float(np.percentile(aE_mc, 50))
aP_med = a_pred_enc * float(np.percentile(ePmc, 50))
say(f"   a_eff used below = MC medians: infauna {aI_med:.4f}, exposed epi (benthivores) {aE_med:.4f},"
    f" exposed epi (predators) {aP_med:.4f}")
pairs = {
    "ben->infauna": (0.25, aI_med, H["inf"], A_INF, 93.6),
    "ben->exposed epi": (0.44, aE_med, H["epi"], A_EPI_B, 3.06),
    "pred->exposed epi": (0.33, aP_med, H["pred_epi"], A_EPI_P, 3.06),
}
say("   x = a h A today and the intake share of the ceiling while foraging x/(1+x); half-saturation 1/(a h)")
for nm, (alib, aeff, h, A, Btot) in pairs.items():
    for lab, a in (("library a", alib), ("a_eff", aeff)):
        x = a * h * A
        say(f"   {nm:<18} {lab:<9} a {a:.4f}  x {x:7.1f}  x/(1+x) {x / (1 + x):.3f}  1/(a h) {1 / (a * h):.4f} t/km2 A"
            f"  ({Btot / A / (a * h):.3f} in pool units)")

say("\n   T1 Hiddink et al. 2016 [A] (food in plaice/LRD stomachs and condition fall with prey per fish above"
    " ~5 passes/yr; q ~0.5-0.7 of the prey, J): intake ratio R(q) = q (1 + x)/(1 + q x)")
for nm, (alib, aeff, h, A, Btot) in pairs.items():
    for lab, a in (("library", alib), ("a_eff", aeff)):
        x = a * h * A
        say(f"     {nm:<18} {lab:<8}: q 0.7 -> {0.7 * (1 + x) / (1 + 0.7 * x):.3f}; q 0.5 -> {0.5 * (1 + x) / (1 + 0.5 * x):.3f};"
            f" q 0.3 -> {0.3 * (1 + x) / (1 + 0.3 * x):.3f}")
say("     Field: a measurable fall (Hiddink), but flat stomach fullness along an Irish Sea gradient where the prey/"
    "fish ratio did not change (Johnson et al. 2015 [F]).")

say("\n   T2 Diet composition vs availability, benthivores (a fish searching all pairs at once, unsaturated"
    " relative intake a_i h-free A_i; field central infauna 0.544, epifauna 0.314, planktivores 0.120, predators 0.023):")
fish = {"planktivores": (0.019, 3.3), "predators": (0.026, 1.2)}
for lab, ai, ae in (("library", 0.25, 0.44), ("a_eff", aI_med, aE_med)):
    r = {"infauna": ai * A_INF, "exposed epi": ae * A_EPI_B, "nephrops": ae * A_NEP,
         "planktivores": fish["planktivores"][0] * fish["planktivores"][1],
         "predators": fish["predators"][0] * fish["predators"][1]}
    t = sum(r.values())
    say(f"     {lab:<8}: " + ", ".join(f"{k} {v / t:.3f}" for k, v in r.items()))
say("   Predators (field: epifauna 0.072 of the diet; planktivores 0.48, zooplankton 0.36, benthivores 0.08 by"
    " mass in 'Fodoniva och mattnad i falt'/field_implied_a):")
for lab, ap in (("library", 0.33), ("a_eff", aP_med)):
    r = {"planktivores": 0.012 * 3.3, "zooplankton": 0.002 * 7.0, "exposed epi": ap * A_EPI_P,
         "benthivores": 0.18 * 1.53}
    t = sum(r.values())
    say(f"     {lab:<8}: " + ", ".join(f"{k} {v / t:.3f}" for k, v in r.items()) +
        "   (zooplankton pool taken as 7 t/km2, J; relative intake ignores h, J)")

say("\n   T3 Share of infauna production eaten by fish (field 1.3-6.5 %, Dauvin 1988 via the refuge note;"
    " 2.9-5.5 % macrobenthos, Mackinson & Daskalov 2007): flow = feeding level x ceiling x diet share / E")
for F in (0.36, 0.54, 1.0):
    flow = F * CEIL_NEW * 1460 * 0.544 / 2600 * 1.53
    say(f"     feeding level {F:.2f}: {flow:.2f} t/km2/yr = {flow / (1.0 * 93.6) * 100:.1f} % of P (P/B 1.0 x 93.6)")
say("   Contamination note: cod's p_rel ~1 on crabs and hermits rests partly on diet grams (within the epifauna),")
say("   not on the infauna/epifauna split that T2 scores.")
say("     This tests the FEEDING LEVEL (eat x hunger x x/(1+x)), not a: with x >> 1 the level is the policy's.")
