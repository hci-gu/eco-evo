"""Exposed share phi of benthic_infauna from reach depth x vertical biomass profiles.

phi_member(R) = sum_t w_t [F_t(R) + c_t (1 - F_t(R))]
  F_t(R): share of taxon t's biomass above reach depth R (cm), piecewise-linear cumulative profile
  c_t:    share of the deeper animals' body mass exposed above the surface as croppable parts
Purely geometric (no time-active, detection or capture factor: those sit in the pair's exposure e,
research_notes/"Exponering av bottenbyten").

Composition: 100 m = SHARK SK3 standing stock without heart urchins (research_notes/"Bentos och
torskfiskar"/evidence_A_standing_stock.md, D); 300 m = Dauwe et al. 1998 station SK (280 m) whole-
community profile (F) plus croppable parts. Cells blended linearly in depth between 100 and 300 m and
weighted by the library K(depth) (132.35 -> 66.17 t/km2, an input table, not model output) over
maps/depth.npz. Members weighted by NS-IBTS 2001-25 biomass shares (library member_species), the six
named members renormalised (others assigned the group mean).

Run from the repo root:  python3 -I "research_notes/Räckdjup i mjukbotten/scripts/phi_reach.py"
"""
import itertools
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ---------------------------------------------------------------- vertical profiles (cm, cum. share)
def cum(knots):
    z = np.array([k[0] for k in knots], float)
    f = np.array([k[1] for k in knots], float)
    return lambda r: float(np.interp(r, z, f))


def uniform(a, b):
    return cum([(0, 0), (a, 0), (b, 1), (100, 1)])


# Dauwe et al. 1998 SK 280 m, digitized Fig. 2 (+-5 pp): 0-2 27 %, 0-5 51 %, 5-10 30 %, 10-20 19 %.
# Linear inside a layer (J). Low / high variants move the 0-2 and 0-5 shares by -/+5 pp.
SK = {
    "low": cum([(0, 0), (2, 0.22), (5, 0.46), (10, 0.79), (20, 1.0)]),
    "central": cum([(0, 0), (2, 0.27), (5, 0.51), (10, 0.81), (20, 1.0)]),
    "high": cum([(0, 0), (2, 0.32), (5, 0.56), (10, 0.83), (20, 1.0)]),
}

# Amphiura filiformis disc depth: distribution of individuals (biomass sits in disc + buried arms).
# Filled from tables/vertical_rows.csv evidence (see summary.md); scenario bounds below.
AMPHIURA_DISC = {
    "low": uniform(4.0, 9.0),       # deeper discs (Solan & Kennedy 2002 Galway: chamber 9.2 cm, n 1)
    "central": uniform(3.0, 6.0),   # Rosenberg et al. 1997 [F] "about 4 cm", "3 to 4 cm"; MarLIN 4-8 [S]
    "high": uniform(2.0, 5.0),      # Gullmarsfjord individuals from 1.5 cm [A]
}
# Arms are 0.85-0.90 of the body (Dobson et al. 1991, amphiurid analogue [F]). Above a buried disc a
# fish biting R cm into the sediment can take the raised arms' segments inside 0-R cm: ~2 raised arms
# (Loo et al. 1996 [F]) x R / (5 arms x ~8 cm arm) x 0.875 arm share x 0.7 (distal half thinner, J)
AMPHIURA_ARM_PER_CM = 2.0 / (5 * 8.0) * 0.875 * 0.7   # 0.031 of body mass per cm of reach (J)

BIVALVE_PARTS = {  # SK3: Abra nitida 40.2 of 46.9 g/m2; rest thyasirids / Nucula (J split)
    "abra": (40.2 / 46.9, uniform(0.0, 2.0)),              # Dauwe 1998: confined to 0-2 cm [F]
    "thyasirid": (0.5 * 6.7 / 46.9, uniform(1.2, 6.4)),    # Zanzerl & Dufour 2017 12-64 mm [F]
    "nucula": (0.5 * 6.7 / 46.9, uniform(4.0, 11.0)),      # Dauwe 1998 peak 5-10 cm [F]
}

# Croppable exposed share of the body mass of animals below the reach (low, central, high).
CROP = {
    "amphiura": (0.03, 0.05, 0.09),   # arm tips above the surface while feeding (vertical_rows J)
    "bivalve": (0.005, 0.01, 0.02),   # siphon 0.6 % body mass per cm (Zwarts et al. 1994 [A]) x 1-3 cm
    "polychaete": (0.0, 0.01, 0.03),  # palps, tentacles, tail ends (J)
    "deep_other": (0.0, 0.0, 0.01),
}
CRUST_IN_REACH = (0.5, 0.7, 0.9)      # amphipods / cumaceans near the surface (J, earlier note)
GASTRO_IN_REACH = (0.6, 0.8, 1.0)     # J

# SK3 100 m standing stock without heart urchins (g WW/m2): D
SK3 = {"amphiura": 252.6, "polychaete": 63.9, "bivalve": 46.9, "deep_other": 15.3,
       "crustacean": 4.0, "gastropod": 3.2}
# 300 m: community profile of Dauwe SK; croppable parts on the deeper biomass, share of polychaetes
# (LYSE 8 numbers 44 % polychaetes, 44 % bivalves, 4 % ophiuroids; J by biomass)
W300_CROP = {"polychaete": 0.55, "bivalve": 0.30, "amphiura": 0.10, "deep_other": 0.05}

IDX = {"low": 0, "central": 1, "high": 2}


def phi100(r, s, discs=True):
    i = IDX[s]
    tot = sum(SK3.values())
    fa = AMPHIURA_DISC[s](r) if discs else 0.0
    arm = min(CROP["amphiura"][i] + AMPHIURA_ARM_PER_CM * min(r, 4.0), 1.0)
    amph = fa + arm * (1 - fa)
    fb = sum(w * f(r) for w, f in BIVALVE_PARTS.values())
    biv = fb + CROP["bivalve"][i] * (1 - fb)
    fp = SK[s](r)  # polychaetes at 100 m: Dauwe SK community profile as proxy (J)
    pol = fp + CROP["polychaete"][i] * (1 - fp)
    deep = 0.25 * SK[s](r) + CROP["deep_other"][i]  # priapulids / nemerteans: deep (J)
    parts = {"amphiura": amph, "polychaete": pol, "bivalve": biv, "deep_other": deep,
             "crustacean": CRUST_IN_REACH[i], "gastropod": GASTRO_IN_REACH[i]}
    return sum(SK3[k] * parts[k] for k in SK3) / tot


def phi300(r, s):
    i = IDX[s]
    f = SK[s](r)
    crop = sum(w * CROP[k][i] for k, w in W300_CROP.items())
    return f + crop * (1 - f)


# ---------------------------------------------------------------- grid blend
def grid_weights():
    d = np.load(os.path.join(REPO, "maps", "depth.npz"))["depth"].ravel()
    k = np.interp(d, [100.0, 300.0], [132.35, 66.17])
    w100 = np.clip((300.0 - d) / 200.0, 0.0, 1.0)
    return float(np.sum(k * w100) / np.sum(k)), d


def phi_grid(r, s, w100, discs=True):
    return w100 * phi100(r, s, discs) + (1 - w100) * phi300(r, s)


# ---------------------------------------------------------------- members
MEMBERS = {  # biomass share, reach low / central / high (cm), tag
    "haddock": (0.386, (1.0, 2.0, 4.0), "J on S/A bounds"),
    "cod": (0.314, (0.0, 0.5, 1.0), "J on A"),
    "long_rough_dab": (0.126, (0.0, 0.5, 1.0), "J"),
    "dab": (0.045, (0.0, 0.5, 1.0), "J on F bound"),
    "plaice": (0.038, (1.0, 2.0, 4.0), "J on F mechanism"),
    "witch": (0.017, (0.5, 1.0, 2.0), "J on F diet"),
}
# Infauna share of each member's diet (sensitivity weights only): ICES DB box/Skagerrak/NNS (D),
# witch Bergstad 2003 (F), dab J.
INFAUNA_DIET = {"haddock": 0.61, "cod": 0.10, "long_rough_dab": 0.24, "dab": 0.5,
                "plaice": 0.99, "witch": 0.80}
QB = {"haddock": 3.8, "cod": 3.2, "long_rough_dab": 3.4, "dab": 3.5, "plaice": 3.42, "witch": 3.0}


# Branch "arms only": members that crop Amphiura arms and leave the buried disc (dab: Kaiser & Ramsay
# 1997 [F]; dab, haddock: Skold et al. 1994 [F, citing Mattson 1992 = S for haddock]) take no discs at
# any reach. Branch "discs": a disc above the reach depth is taken whole.
ARMS_ONLY = {"haddock", "dab"}
BRANCH = {"value": "discs"}


def group(scen_reach, scen_prof, w100, weights="biomass"):
    num = den = 0.0
    for m, (b, reach, _) in MEMBERS.items():
        wt = b if weights == "biomass" else b * QB[m] * INFAUNA_DIET[m]
        discs = not (BRANCH["value"] == "arms_only" and m in ARMS_ONLY)
        num += wt * phi_grid(reach[IDX[scen_reach]], scen_prof, w100, discs)
        den += wt
    return num / den


def main():
    w100, d = grid_weights()
    print(f"grid: {d.size} cells, depth median {np.median(d):.0f} m; K-weighted 100 m composition "
          f"weight {w100:.3f} (300 m {1 - w100:.3f})")
    print("\nphi by reach depth R (cm) and profile scenario: 100 m | 300 m | grid")
    for r in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        row = []
        for s in ("low", "central", "high"):
            row.append(f"{s[0]}: {phi100(r, s):.3f} {phi300(r, s):.3f} {phi_grid(r, s, w100):.3f}")
        print(f"  R {r:3.1f}  " + "  |  ".join(row))
    print("\nper member, grid phi at its reach (low / central / high reach), central profile:")
    for m, (b, reach, tag) in MEMBERS.items():
        vals = [phi_grid(x, "central", w100) for x in reach]
        print(f"  {m:15s} b {b:.3f} reach {reach} cm [{tag}] -> phi "
              + " / ".join(f"{v:.3f}" for v in vals))
    print("\ngroup phi (biomass-weighted), reach scenario x profile scenario:")
    for sr, sp in itertools.product(("low", "central", "high"), ("low", "central", "high")):
        print(f"  reach {sr:7s} profile {sp:7s}: {group(sr, sp, w100):.3f}")
    print("\nheadline: low = reach low + profile low; high = reach high + profile high")
    lo, ce, hi = group("low", "low", w100), group("central", "central", w100), group("high", "high", w100)
    print(f"  biomass-weighted          : {lo:.3f} / {ce:.3f} / {hi:.3f}")
    lo2, ce2, hi2 = (group(s, s, w100, "infauna") for s in ("low", "central", "high"))
    print(f"  infauna-consumption-weighted: {lo2:.3f} / {ce2:.3f} / {hi2:.3f}")
    # one-at-a-time spread: which input drives it
    print("\none-at-a-time swing of the central group phi (biomass-weighted):")
    base = ce
    saved = dict(MEMBERS)
    for m in MEMBERS:
        for k, lab in ((0, "low"), (2, "high")):
            b, reach, tag = saved[m]
            MEMBERS[m] = (b, (reach[k], reach[k], reach[k]), tag)
            v = group("central", "central", w100)
            MEMBERS[m] = saved[m]
            print(f"  reach {m:15s} {lab:4s}: {v:.3f} ({v - base:+.3f})")
    for s in ("low", "high"):
        v = group("central", s, w100)
        print(f"  profile+crop {s:4s}       : {v:.3f} ({v - base:+.3f})")
    # the two branches on Amphiura discs
    for br in ("discs", "arms_only"):
        BRANCH["value"] = br
        g = [group(x, x, w100) for x in ("low", "central", "high")]
        gi = [group(x, x, w100, "infauna") for x in ("low", "central", "high")]
        hr = [group("high", "central", w100), group("central", "high", w100)]
        print(f"\nBRANCH {br:9s}: biomass-weighted {g[0]:.3f} / {g[1]:.3f} / {g[2]:.3f}; "
              f"infauna-weighted {gi[0]:.3f} / {gi[1]:.3f} / {gi[2]:.3f}; "
              f"reach high x profile central {hr[0]:.3f}; reach central x profile high {hr[1]:.3f}")
        for r in (3.0, 4.0):
            saved_h = MEMBERS["haddock"]
            MEMBERS["haddock"] = (saved_h[0], (r, r, r), "")
            print(f"   haddock reach {r:.0f} cm, others central, central profile: "
                  f"{group('central', 'central', w100):.3f}")
            MEMBERS["haddock"] = saved_h
    BRANCH["value"] = "discs"
    # comparison with the earlier note's uniform reach (every member 1 / 2 / 5 cm)
    saved = dict(MEMBERS)
    for m in MEMBERS:
        MEMBERS[m] = (saved[m][0], (1.0, 2.0, 5.0), "earlier note J")
    old = [group(s, s, w100) for s in ("low", "central", "high")]
    MEMBERS.update(saved)
    print("\nearlier note's uniform reach 1 / 2 / 5 cm on today's profiles: "
          + " / ".join(f"{v:.3f}" for v in old))
    # haddock reach that gives the library's 0.20 with the other members at central
    for r in np.arange(1.0, 4.01, 0.1):
        MEMBERS["haddock"] = (saved["haddock"][0], (r, r, r), "")
        if group("central", "central", w100) >= 0.20:
            print(f"haddock reach giving group phi 0.20 (others central): {r:.1f} cm")
            break
    MEMBERS.update(saved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
