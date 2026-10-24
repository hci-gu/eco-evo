"""Biomass-weighted benthivorous_fish parameters from the per-species values of evidence A-C.

Usage: python3 -I group_params.py <size_factors_csv> <stage_shares_csv> <out_dir>

Inputs typed below are the per-species central / low / high values chosen in
../params_benthivorous_fish.md from evidence A (bioenergetics), B (population, fishing) and C (diets,
Q/B); the size factors and stage shares come from length_structure.py (long-run 2001-25 survey
length mix). Weights: long-run 2001-25 biomass shares in the group (../../tables/
long_run_species_by_group.csv), others = the remaining 2.5 %. Writes group_params.csv and prints.
"""
import math
import os
import sys

import pandas as pd

sizef, stages, out = sys.argv[1:4]
SF = pd.read_csv(sizef).set_index("species")
ST = pd.read_csv(stages)

W = {"haddock": 38.6, "cod": 31.4, "long rough dab": 12.6, "dab": 4.5, "plaice": 3.8,
     "witch": 1.7, "starry ray": 1.5, "hagfish": 1.3, "grey gurnard": 0.9, "lemon sole": 0.9,
     "greater weever": 0.8, "others": 2.5}
tot = sum(W.values())
W = {k: v / tot for k, v in W.items()}

T50 = [7.7, 7.2, 6.6, 6.6, 6.8, 7.2, 8.0, 8.5, 10.2, 11.7, 10.6, 9.3]


def kitchell(T, CQ=2.41, CTO=13.7, CTM=21.0):
    V = (CTM - T) / (CTM - CTO)
    Z = math.log(CQ) * (CTM - CTO)
    Y = math.log(CQ) * (CTM - CTO + 2)
    X = Z * Z * (1 + (1 + 40 / Y) ** 0.5) ** 2 / 400
    return V ** X * math.exp(X * (1 - V))


f_ann = sum(kitchell(t) for t in T50) / 12
# Hansson et al. 1996 cod Cmax, g/g/d at 4184 J/g, averaged over the stock's biomass (E[W^-0.195])
cod_fn = lambda sp: 0.099 * SF.loc[sp, "E_W^-0.195"] * f_ann * 4184  # kJ/kg/d


def at_size(val_ref, w_ref, sp, b=-0.22):
    """Scale a per-kg rate measured at w_ref (g) to the species' biomass-weighted W^b."""
    return val_ref * SF.loc[sp, "E_W^-0.2"] / w_ref ** b if sp in SF.index else val_ref


# Cmax, kJ per kg fish per day, annual mean over the 50 m series (central, low, high)
plaice_c = at_size(113.0, 150, "plaice")  # mean of Fonds 1985 table (135) and fit (92) routes at 150 g
CMAX = {
    "cod": (cod_fn("cod"), cod_fn("cod") * 0.85, cod_fn("cod") * 1.15),
    "haddock": (1.3 * cod_fn("haddock"), 1.0 * cod_fn("haddock"), 1.6 * cod_fn("haddock")),
    "plaice": (plaice_c, at_size(80.0, 150, "plaice"), at_size(135.0, 150, "plaice")),
    "dab": (0.7 * at_size(113.0, 150, "dab"), at_size(54.0, 150, "dab"), at_size(91.0, 150, "dab")),
    "long rough dab": (0.6 * at_size(113.0, 150, "long rough dab"), 50.0, at_size(91.0, 150, "long rough dab")),
    "witch": (0.6 * at_size(113.0, 150, "witch"), 40.0, at_size(91.0, 150, "witch")),
    "lemon sole": (at_size(113.0, 150, "lemon sole"), at_size(80.0, 150, "lemon sole"),
                   at_size(135.0, 150, "lemon sole")),
    "grey gurnard": (cod_fn("grey gurnard"), 0.8 * cod_fn("grey gurnard"), 1.3 * cod_fn("grey gurnard")),
    "starry ray": (50.0, 30.0, 80.0),
    "hagfish": (20.0, 10.0, 40.0),
    "greater weever": (90.0, 60.0, 120.0),
    "others": (90.0, 60.0, 120.0),
}
# SMR annual mean (50 m series), MJ/t/tick (evidence A; size-adjusted where noted in the md)
SMR = {"haddock": (3.0, 2.2, 3.8), "cod": (3.7, 3.3, 4.1), "plaice": (3.1, 2.6, 3.3),
       "dab": (3.3, 2.6, 4.2), "long rough dab": (1.6, 1.0, 2.0), "witch": (1.5, 1.1, 2.0),
       "lemon sole": (3.1, 2.6, 3.3), "grey gurnard": (3.0, 2.2, 3.8), "starry ray": (2.0, 1.2, 3.0),
       "hagfish": (1.2, 0.8, 2.3), "greater weever": (3.0, 2.2, 3.8), "others": (3.0, 2.2, 3.8)}
Q10 = {"haddock": 2.5, "cod": 2.2, "plaice": 2.0, "dab": 1.8, "long rough dab": 2.0, "witch": 2.2,
       "lemon sole": 2.0, "grey gurnard": 2.2, "starry ray": 2.0, "hagfish": 2.0,
       "greater weever": 2.0, "others": 2.0}
# Whole-body energy density as eaten, kJ/g wet (central, low, high), and lean body
EWHOLE = {"haddock": (4.6, 4.2, 4.75), "cod": (4.75, 4.2, 5.0), "plaice": (4.6, 3.0, 5.8),
          "dab": (4.5, 3.5, 5.5), "long rough dab": (3.2, 2.8, 3.6), "witch": (5.6, 4.5, 5.6),
          "lemon sole": (5.8, 4.5, 5.8), "grey gurnard": (4.4, 4.0, 4.8), "starry ray": (4.0, 3.5, 4.5),
          "hagfish": (4.5, 3.5, 5.5), "greater weever": (5.3, 4.5, 5.5), "others": (4.5, 3.5, 5.5)}
ELEAN = {"haddock": 3.8, "cod": 3.8, "plaice": 3.6, "dab": 3.6, "long rough dab": 2.6, "witch": 4.0,
         "lemon sole": 4.0, "grey gurnard": 3.8, "starry ray": 3.6, "hagfish": 3.5,
         "greater weever": 4.0, "others": 3.6}
ME = {"haddock": 2000, "cod": 2000, "plaice": 1850, "dab": 1500, "long rough dab": 1000,
      "witch": 1500, "lemon sole": 1850, "grey gurnard": 1500, "starry ray": 1500, "hagfish": 1500,
      "greater weever": 1500, "others": 1500}
# Population (evidence B): r_now, M0, non-model predation, F
POP = {"haddock": (0.24, 0.20, 0.11, 0.06), "cod": (0.35, 0.20, 0.11, 0.39),
       "long rough dab": (0.30, 0.20, 0.03, 0.10), "dab": (0.55, 0.20, 0.05, 0.18),
       "plaice": (0.40, 0.20, 0.05, 0.04), "witch": (0.28, 0.17, 0.02, 0.12),
       "starry ray": (0.15, 0.15, 0.03, 0.03), "hagfish": (0.10, 0.13, 0.02, 0.01),
       "grey gurnard": (0.65, 0.20, 0.05, 0.03), "lemon sole": (0.50, 0.20, 0.03, 0.09),
       "greater weever": (0.50, 0.20, 0.05, 0.02), "others": (0.40, 0.20, 0.05, 0.10)}
SPEED = {"haddock": 0.25, "cod": 0.25, "grey gurnard": 0.2}  # others sedentary 0.1
VISF = {"haddock": 0.6, "cod": 0.6, "grey gurnard": 0.6, "hagfish": 0.4}  # flatfish/rays 0.4, else 0.5
FLAT = {"long rough dab", "dab", "plaice", "witch", "lemon sole", "starry ray"}


def wavg(d, i=None):
    return sum(W[k] * (d[k][i] if i is not None else d[k]) for k in W)


rows = []
def put(field, c, lo=None, hi=None, note=""):
    rows.append(dict(field=field, central=c, low=lo, high=hi, note=note))
    print(f"{field:42s} {c:10.4g}  {'' if lo is None else f'{lo:.4g}':>9s}  {'' if hi is None else f'{hi:.4g}':>9s}  {note}")


print("annual-mean Kitchell factor over the 50 m series:", round(f_ann, 3))
for sp in CMAX:
    print(f"  Cmax {sp:15s} {CMAX[sp][0]:6.1f} ({CMAX[sp][1]:.0f}-{CMAX[sp][2]:.0f}) kJ/kg/d")
cm = [wavg(CMAX, i) for i in range(3)]
put("Cmax group (kJ/kg/d, annual mean)", *cm)
for i, lab in enumerate(["central", "low", "high"]):
    pass
CE = [c / 4 for c in cm]  # MJ per t per 6 h tick
put("energy ceiling (MJ/t/tick)", *CE, note="old gadoid 34.6")
E_PREY = {"benthic_community": 3000, "zooplankton": 4500, "planktivorous_fish": 7500,
          "predatory_fish": 4750}
for p, e in E_PREY.items():
    put(f"h {p} (E {e})", e / CE[0], e / CE[2], e / CE[1], note="h = E_prey / ceiling")
smr = [wavg(SMR, i) for i in range(3)]
put("SMR group (MJ/t/tick, annual mean)", *smr)
put("resting_metabolism = 1.2 x SMR", 1.2 * smr[0], smr[1], 1.2 * smr[2], note="routine at rest")
put("metabolism_q10", wavg(Q10))
ew = [wavg(EWHOLE, i) * 1000 for i in range(3)]
put("energy_content whole body (MJ/t)", *ew)
put("energy_content lean (MJ/t)", wavg(ELEAN) * 1000)
me = wavg(ME)
put("max_energy_reserve (MJ/t)", me)
put("reference fill (whole-lean)/ME", (ew[0] - wavg(ELEAN) * 1000) / me)
r = sum(W[k] * POP[k][0] for k in W)
m0 = sum(W[k] * POP[k][1] for k in W)
mp = sum(W[k] * POP[k][2] for k in W)
F = sum(W[k] * POP[k][3] for k in W)
put("r_max (r_now)", r, 0.22, 0.41)
put("M0", m0, 0.15, 0.30)
put("non-model predation", mp, 0.04, 0.16)
put("M1 /yr", m0 + mp, 0.19, 0.46)
put("M1 per tick", 1 - math.exp(-(m0 + mp) / 1460), 1 - math.exp(-0.19 / 1460), 1 - math.exp(-0.46 / 1460))
put("F total /yr", F, 0.14, 0.24)
K = lambda B, F, r: B / (1 - F / r) if F < r else float("inf")
put("K = 1.53/(1-F/r)", K(1.53, F, r), K(1.53, 0.14, 0.41), K(1.53, 0.24, 0.22))
put("K, 2021-25 weights (F 0.09, r 0.26)", K(1.53, 0.09, 0.26))
put("movement_speed", sum(W[k] * SPEED.get(k, 0.1) for k in W))
put("visibility_floor", sum(W[k] * VISF.get(k, 0.4 if k in FLAT else 0.5) for k in W))

# predator_share per pair from stage scores (long-run stage shares)
s = {(a, b): v for a, b, v in ST[["species", "stage", "share_2001_25"]].itertuples(index=False)}
cod_st = [s[("cod", "0-30")], s[("cod", "30-50")], s[("cod", "50-")]]
had_st = [s[("haddock", "0-40")], s[("haddock", "40-")]]
print("cod stages", [round(x, 3) for x in cod_st], "haddock stages", [round(x, 3) for x in had_st])
SC = {  # scores central, low, high per stage: benthos, planktivorous fish, predatory fish, copepod-sized zoo
    "B": {"cod": [(1, .72, 1), (.72, .42, 1), (.64, .30, 1)], "haddock": [(1, .72, 1), (1, .72, 1)],
          "flat": (1, .9, 1), "grey gurnard": (1, .7, 1), "hagfish": (1, .7, 1), "greater weever": (1, .7, 1),
          "others": (.8, .5, 1)},
    "P": {"cod": [(.3, .15, .5), (.7, .4, .9), (.9, .6, 1)], "haddock": [(.1, .02, .25), (.25, .1, .5)],
          "long rough dab": (.05, 0, .1), "starry ray": (.4, .2, .6), "grey gurnard": (.15, .05, .3),
          "greater weever": (.2, .05, .4), "others": (.05, 0, .1)},
    "W": {"cod": [(0, 0, .05), (.15, .05, .3), (.4, .2, .6)], "haddock": [(0, 0, 0), (0, 0, .05)],
          "starry ray": (.3, .1, .5), "grey gurnard": (.3, .1, .5)},
    "Zc": {"cod": [(.12, .05, .2), (.03, 0, .1), (.01, 0, .05)], "haddock": [(.05, 0, .1), (.03, 0, .08)],
           "dab": (.05, 0, .1), "long rough dab": (.05, 0, .15)},
}
for pair, sc in SC.items():
    res = []
    for i in range(3):
        v = W["cod"] * sum(w * t[i] for w, t in zip(cod_st, sc["cod"]))
        v += W["haddock"] * sum(w * t[i] for w, t in zip(had_st, sc["haddock"]))
        for k in W:
            if k in ("cod", "haddock"):
                continue
            t = sc.get(k, sc.get("flat") if k in FLAT else None)
            if t is not None:
                v += W[k] * t[i]
        res.append(v)
    put(f"predator_share {pair}", *res)

# consumption-based M2 on the prey at the target biomasses (evidence C per-tonne rates, Q/B 3.48)
B_PRED = 1.53
PREY = {"benthic_community": (95.0, 2.506, 2.167, 2.544),
        "zooplankton": (35000 / 3600, 0.087, 0.020, 0.175),
        "planktivorous_fish": (3.3, 0.293, 0.226, 0.511),
        "predatory_fish": (1.2, 0.067, 0.060, 0.127)}
for p, (bp, c, lo, hi) in PREY.items():
    put(f"M2 on {p} (/yr)", c * B_PRED / bp, lo * B_PRED / bp, hi * B_PRED / bp,
        note=f"cons {c} t/t/yr x {B_PRED} / {bp:.2f}")
# Q/B scaling: realised ration vs the ceiling
put("realised Q/B / ceiling Q/B (benthos)", 3.48 / (1460 / (3000 / CE[0])))


def t_total(rm, me, u, sr, frac=0.05, ticks=4000):
    B, R = 1.0, me * u
    for t in range(1, ticks):
        R = max(0.0, R - B * rm)
        s_ = R / (B * me)
        if s_ < u:
            loss = B * sr * (u - s_)
            R *= (B - loss) / B
            B -= loss
        if B <= frac:
            return t / 4
    return None


rm = 1.2 * smr[0]
for sr in (0.008, 0.010, 0.012, 0.015):
    print(f"starve_rate {sr}: T_total (95 % loss from s = u, rest only) = {t_total(rm, me, 0.5, sr)} d;"
          f" 40 % loss = {t_total(rm, me, 0.5, sr, frac=0.6)} d")
pd.DataFrame(rows).to_csv(os.path.join(out, "group_params.csv"), index=False)
