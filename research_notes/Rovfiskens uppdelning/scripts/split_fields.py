"""Species fields, diets and pair M2 of the two candidate predator groups.

Usage: python3 -I split_fields.py <fg_predators_dir> <out_dir>

<fg_predators_dir> = research_notes/"Fiskgrupper i rutan"/predators (inputs/species_values.csv,
inputs/diet_consumption_central.csv, tables/pred_season.csv, tables/pred_length_biomass.csv,
tables/benthivore_prey_sizes_stdout.txt is not read; its cumulative shares are copied below).

Same construction as ../../Fiskgrupper i rutan/predators/scripts/weighted_fields.py (section 209):
biomass-weighted species values with the long-run (2001-25) NS-IBTS GOV shares, B* = 1.20 t/km2 x
the members' share (top of the long-run span, user's choice), ranges = corner bounds. No model output
is read. Two groupings are evaluated:

  G1 (diet criterion, recommended): bed = whiting; water column = the other nine members.
  G2 (habitat criterion, variant):  bed = whiting, hake, spurdog, anglerfish, ling, velvet belly;
                                     water column = saithe, mackerel, horse mackerel, pollack.

Writes <out_dir>/split_fields_<G>.json and prints every number used in params_split.md.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

d, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
TICKS = 1460.0
B_TOTAL = 1.20  # t/km2, predatory_fish B* (section 209; memory note "top of long-run span")
V = pd.read_csv(os.path.join(d, "inputs", "species_values.csv")).set_index("species")
S = pd.read_csv(os.path.join(d, "tables", "pred_season.csv")).set_index("common")
W = S.loc[V.index, "pooled"]
W = W / W.sum()
SPECIES = list(V.index)

# ------------------------------------------------------------------ diets (shares of total food by mass)
DC = pd.read_csv(os.path.join(d, "inputs", "diet_consumption_central.csv"))
DC["species"] = DC["species"].str.replace("_", " ")
DC = DC.set_index("species")
DIET = {sp: {k: float(DC.loc[sp, k]) for k in ("P", "Bf", "W", "Z", "B", "O")} for sp in DC.index}
# proxies for the three small members without a diet (1.6 % of the pooled biomass), J
DIET["ling"] = dict(DIET["anglerfish"])        # demersal piscivore (J)
DIET["pollack"] = dict(DIET["saithe"])         # saithe congener (J)
DIET["velvet belly"] = dict(DIET["horse mackerel"])  # deep shrimp/krill and small fish eater (J)
# epifauna share of each member's benthos (B) part: ../../Infauna och rorlig epifauna/tables/
# pred_benthos_split_stdout.txt (ICES stomach DB): whiting box+Skagerrak 0.61 (2022 0.74, all areas
# 2022-24 0.71) -> 0.65; hake 0.99 (Pagurus); horse mackerel 1.0 (small shrimps, mysids); saithe 1.0
# (one fish, Crangon); spurdog 0 (infauna, 2 stomachs); mackerel 0.5 (two samples, 0.51 and 1.0 / 0.0);
# anglerfish, ling 0.5 (J); pollack as saithe; velvet belly as horse mackerel.
EPI_FRAC = {"saithe": 1.0, "whiting": 0.65, "mackerel": 0.5, "spurdog": 0.0, "horse mackerel": 1.0,
            "hake": 0.99, "anglerfish": 0.5, "ling": 0.5, "velvet belly": 1.0, "pollack": 1.0}
# composition of each member's within-group (W) prey by prey species, J from the ICES taxa table
# (../../Fiskgrupper i rutan/predators/tables/ices_diet_pools_taxa.csv) and Ecopath 1991:
#  saithe: SMS 2023 saithe -> whiting (0.085 /yr on whiting 1+) is the only W flow;
#  whiting: Skagerrak/box 2022 all whiting; C/S North Sea 2022 hake 56 g and horse mackerel 6 g of 288 g;
#  mackerel: whiting (C/S North Sea 2022); spurdog: Ecopath whiting 0.12, mackerel 0.08 of W 0.24;
#  horse mackerel: horse mackerel 13 g, whiting 2 g (C/S North Sea 2023); hake: Skagerrak 2024 all
#  whiting, Ecopath 1991 (North Sea wide) horse mackerel and mackerel -> whiting 0.9; anglerfish:
#  ICES 2022-24 (n 9) hake 650 g, horse mackerel 314, whiting 158; Ecopath whiting 0.12 of W 0.21.
WCOMP = {"saithe": {"whiting": 1.0},
         "whiting": {"whiting": 0.95, "hake": 0.04, "horse mackerel": 0.01},
         "mackerel": {"whiting": 1.0},
         "spurdog": {"whiting": 0.5, "mackerel": 0.33, "horse mackerel": 0.17},
         "horse mackerel": {"whiting": 0.12, "horse mackerel": 0.88},
         "hake": {"whiting": 0.9, "horse mackerel": 0.05, "mackerel": 0.05},
         "anglerfish": {"whiting": 0.4, "hake": 0.4, "horse mackerel": 0.2},
         "ling": {"whiting": 0.4, "hake": 0.4, "horse mackerel": 0.2},
         "pollack": {"whiting": 1.0},
         "velvet belly": {"whiting": 0.12, "horse mackerel": 0.88}}
# gape: largest prey length (cm) at the typical predator size (../../Fiskgrupper i rutan/
# params_predatory_fish.md, accessible_fraction note): whiting ~12, saithe ~19, hake ~28,
# spurdog 25-30, anglerfish > 40; mackerel, horse mackerel, velvet belly ~10 (J); ling 40 (J).
GAPE = {"saithe": 19, "whiting": 12, "mackerel": 10, "spurdog": 27, "horse mackerel": 10, "hake": 28,
        "anglerfish": 45, "ling": 40, "velvet belly": 10, "pollack": 19}
# benthivore guild: biomass share below L cm, pooled Q1+Q3 (benthivore_prey_sizes_stdout.txt), plus
# the 0-group the GOV does not catch (~0.02-0.05, parent note) -> +0.03 (J)
BF_CUM = [(10, 0.002), (15, 0.053), (20, 0.146), (25, 0.264), (30, 0.385), (45, 0.70), (60, 0.90)]
# 45 and 60 cm: J extrapolation of the guild length biomass (cod and haddock 30-60 cm)


def cum(L, table):
    xs, ys = zip(*table)
    return float(np.interp(L, xs, ys))


LB = pd.read_csv(os.path.join(d, "tables", "pred_length_biomass.csv"))
lb_wh = LB[LB.common == "whiting"].sort_values("L5")
wh_tot = lb_wh.pooled.sum()
# whiting biomass share below L (class lower bound L5, 5-cm classes): upper edge = L5 + 5
WH_CUM = [(0, 0.0)] + [(float(r.L5) + 5.0, float(lb_wh[lb_wh.L5 <= r.L5].pooled.sum() / wh_tot))
                       for r in lb_wh.itertuples()]
# other members' length biomass (as prey of the bed group in G2, or of whiting): from the same table
LB_CUM = {}
for sp in SPECIES:
    t = LB[LB.common == sp].sort_values("L5")
    tot = t.pooled.sum()
    LB_CUM[sp] = [(0, 0.0)] + [(float(r.L5) + 5.0, float(t[t.L5 <= r.L5].pooled.sum() / tot)) for r in t.itertuples()]

PREY_B = {"planktivorous_fish": 3.3, "zooplankton": 35000 / 3600, "mobile_epifauna": 7.88,
          "benthic_infauna": 93.6, "benthivorous_fish": 1.53}
PREY_E = {"planktivorous_fish": 7125.0, "zooplankton": 4500.0, "mobile_epifauna": 3750.0,
          "benthic_infauna": 2600.0, "benthivorous_fish": 4500.0}
ASSIM = {"planktivorous_fish": 0.8, "zooplankton": 0.8, "mobile_epifauna": 0.8, "benthic_infauna": 0.7,
         "benthivorous_fish": 0.8, "other_predator_group": 0.8}
F_RANGE = {"saithe": (0.11, 0.38), "whiting": (0.035, 0.15)}  # evidence B; others held

GROUPINGS = {
    "G1": {"bed": ["whiting"],
           "wc": ["saithe", "mackerel", "spurdog", "horse mackerel", "hake", "anglerfish", "ling",
                  "velvet belly", "pollack"]},
    "G2": {"bed": ["whiting", "hake", "spurdog", "anglerfish", "ling", "velvet belly"],
           "wc": ["saithe", "mackerel", "horse mackerel", "pollack"]},
}


def fields(members):
    w = W[members] / W[members].sum()
    v = V.loc[members]
    wm = lambda c: float((w * v[c]).sum())
    share = float(W[members].sum())
    B = B_TOTAL * share
    r, rlo, rhi = wm("r_now"), wm("r_lo"), wm("r_hi")
    Fb, Fp = wm("F_bot"), wm("F_pel")
    F = Fb + Fp
    Flo, Fhi = F, F
    for sp, (lo, hi) in F_RANGE.items():
        if sp in members:
            Fc = float(V.loc[sp, "F_bot"] + V.loc[sp, "F_pel"])
            Flo -= w[sp] * (Fc - lo)
            Fhi += w[sp] * (hi - Fc)
    K = B / (1 - F / r)
    Klo = B / (1 - Flo / rhi)
    Khi = B / (1 - Fhi / rlo) if Fhi < rlo else float("inf")
    cm, clo, chi = wm("cmax_kJkgd"), wm("cmax_lo"), wm("cmax_hi")
    ceil, ceil_lo, ceil_hi = cm / 4.0, clo / 4.0, chi / 4.0
    smr, slo, shi = wm("smr"), wm("smr_lo"), wm("smr_hi")
    el = wm("e_lean")
    rmx, rmean = wm("res_max"), wm("res_mean")
    qb, qlo, qhi = wm("qb"), wm("qb_lo"), wm("qb_hi")
    u_lit = 0.15
    f = dict(members=members, weights={k: round(float(x), 4) for k, x in w.items()}, share_of_pooled=share,
             B=B, tonnes=B * 3600, r=r, r_lo=rlo, r_hi=rhi, F=F, F_bot=Fb, F_pel=Fp, F_lo=Flo, F_hi=Fhi,
             K=K, K_lo=Klo, K_hi=Khi, M1=wm("natmort"), M1_lo=wm("nm_lo"), M1_hi=wm("nm_hi"),
             cmax=cm, cmax_lo=clo, cmax_hi=chi, ceiling=ceil, ceiling_lo=ceil_lo, ceiling_hi=ceil_hi,
             smr=smr, smr_lo=slo, smr_hi=shi, q10=wm("q10"), e_lean=el, res_max=rmx, res_mean=rmean,
             fill=rmean / rmx, E_whole=el + rmean, qb=qb, qb_lo=qlo, qb_hi=qhi,
             starve_rate=1.5 * smr / el / u_lit, starve_lo=smr / el / u_lit, starve_hi=2.5 * smr / el / u_lit,
             feeding_cost=1.5 + 0.17 * 0.5 * ceil / smr,
             bf100=Fb / TICKS / 3.0142 * 100, pf10=Fp / TICKS / 0.02496 * 10,
             bf100_lo=Flo * (Fb / F) / TICKS / 3.0142 * 100, bf100_hi=Fhi * (Fb / F) / TICKS / 3.0142 * 100)
    return f


def diet(members, own, other):
    """Consumption-weighted diet over the model's prey groups; W split by where the prey sits."""
    Q = {sp: float(V.loc[sp, "qb"] * W[sp] * B_TOTAL) for sp in members}
    Qt = sum(Q.values())
    out = {k: 0.0 for k in ("planktivorous_fish", "zooplankton", "mobile_epifauna", "benthic_infauna",
                            "benthivorous_fish", "own_group", "other_group", "other")}
    per = {}
    for sp in members:
        dd = DIET[sp]
        wc = WCOMP[sp]
        w_own = sum(v for k, v in wc.items() if k in own)
        w_oth = sum(v for k, v in wc.items() if k in other)
        row = {"planktivorous_fish": dd["P"], "zooplankton": dd["Z"],
               "mobile_epifauna": dd["B"] * EPI_FRAC[sp], "benthic_infauna": dd["B"] * (1 - EPI_FRAC[sp]),
               "benthivorous_fish": dd["Bf"], "own_group": dd["W"] * w_own, "other_group": dd["W"] * w_oth,
               "other": dd["O"]}
        per[sp] = row
        for k, x in row.items():
            out[k] += Q[sp] * x / Qt
    return Qt, out, per, Q


def gape_access(members, Q, per, cumtab_fn, key):
    """Consumption-on-that-prey weighted accessible share of a prey pool below each member's gape."""
    num = den = 0.0
    for sp in members:
        c = Q[sp] * per[sp][key]
        num += c * cumtab_fn(GAPE[sp])
        den += c
    return num / den if den > 0 else float("nan")


res_all = {}
for gname, gp in GROUPINGS.items():
    print("=" * 100)
    print(f"Grouping {gname}: bed = {gp['bed']}; water column = {gp['wc']}")
    res = {}
    for role in ("bed", "wc"):
        res[role] = fields(gp[role])
    for role, other in (("bed", "wc"), ("wc", "bed")):
        Qt, dshare, per, Q = diet(gp[role], gp[role], gp[other])
        f = res[role]
        f["Q"] = Qt
        f["Q_over_B"] = Qt / f["B"]
        f["diet"] = dshare
        f["diet_by_member"] = per
        f["consumption_by_member"] = Q
        # M2 on prey
        m2 = {}
        for prey, B in PREY_B.items():
            m2[prey] = Qt * dshare[prey] / B
        m2["other_group"] = Qt * dshare["other_group"] / res[other]["B"]
        m2["own_group (unbooked)"] = Qt * dshare["own_group"] / f["B"]
        f["M2"] = m2
        # dark ratios, consumption-weighted diel-stomach values (evidence C section 4)
        dr = {}
        for prey, col in (("planktivorous_fish", "dark_fish"), ("benthivorous_fish", "dark_fish"),
                          ("zooplankton", "dark_zoo"), ("other_group", "dark_fish")):
            num = den = 0.0
            for sp in gp[role]:
                key = sp if sp in DC.index else {"ling": "anglerfish", "pollack": "saithe",
                                                 "velvet belly": "horse mackerel"}[sp]
                c = Q[sp] * per[sp][prey]
                num += c * float(DC.loc[key, col])
                den += c
            dr[prey] = num / den if den > 0 else float("nan")
        f["dark_ratio_stomach"] = dr
        # accessible fractions of fish prey by gape
        f["acc_benthivores"] = min(1.0, gape_access(gp[role], Q, per, lambda L: cum(L, BF_CUM) + 0.03,
                                                    "benthivorous_fish"))
        def oth_cum(L, mem=gp[other]):
            ww = W[mem] / W[mem].sum()
            return float(sum(ww[s] * cum(L, LB_CUM[s]) for s in mem))
        f["acc_other_group"] = gape_access(gp[role], Q, per, oth_cum, "other_group")
    res_all[gname] = res
    for role in ("bed", "wc"):
        f = res[role]
        print(f"\n--- {gname} {role}: {f['members']}")
        print(f"  weights in group {f['weights']}")
        print(f"  share of pooled biomass {f['share_of_pooled']:.4f}; B* {f['B']:.4f} t/km2 = {f['tonnes']:.0f} t")
        print(f"  r_max {f['r']:.3f} ({f['r_lo']:.3f}-{f['r_hi']:.3f}); F {f['F']:.4f} (bottom {f['F_bot']:.4f}, "
              f"pelagic {f['F_pel']:.4f}; {f['F_lo']:.3f}-{f['F_hi']:.3f}); F/r {f['F'] / f['r']:.3f}")
        print(f"  K = B*/(1 - F/r) = {f['K']:.3f} ({f['K_lo']:.3f}-{f['K_hi']:.3f}) t/km2")
        print(f"  M1 {f['M1']:.3f} /yr ({f['M1_lo']:.3f}-{f['M1_hi']:.3f}) = {f['M1'] / TICKS:.4e} /tick")
        print(f"  Cmax {f['cmax']:.1f} kJ/kg/d ({f['cmax_lo']:.1f}-{f['cmax_hi']:.1f}) -> ceiling {f['ceiling']:.2f} "
              f"MJ/t/tick ({f['ceiling_lo']:.2f}-{f['ceiling_hi']:.2f})")
        print(f"  SMR {f['smr']:.3f} ({f['smr_lo']:.2f}-{f['smr_hi']:.2f}); Q10 {f['q10']:.2f}; feeding_cost "
              f"{f['feeding_cost']:.2f} x RM")
        print(f"  energy lean {f['e_lean']:.0f}; max reserve {f['res_max']:.0f}; mean reserve {f['res_mean']:.0f}; "
              f"fill {f['fill']:.3f}; whole body as eaten {f['E_whole']:.0f} MJ/t")
        print(f"  starve_rate (u 0.15 convention) {f['starve_rate']:.4f} ({f['starve_lo']:.4f}-{f['starve_hi']:.4f})")
        print(f"  Q/B (member values) {f['qb']:.3f} ({f['qb_lo']:.2f}-{f['qb_hi']:.2f}); Q {f['Q']:.3f} t/km2/yr")
        print(f"  trawl tables: bottom biomass_factor at 100 {f['bf100']:.6f} ({f['bf100_lo']:.6f}-{f['bf100_hi']:.6f});"
              f" pelagic at 10 {f['pf10']:.6f}")
        print("  diet (consumption-weighted): " + ", ".join(f"{k} {x:.3f}" for k, x in f["diet"].items()))
        print("  M2 on prey (/yr): " + ", ".join(f"{k} {x:.4f}" for k, x in f["M2"].items()))
        print("  dark_ratio (diel stomach, consumption-weighted): " +
              ", ".join(f"{k} {x:.2f}" for k, x in f["dark_ratio_stomach"].items()))
        print(f"  accessible_fraction by gape: benthivores {f['acc_benthivores']:.3f}; other group "
              f"{f['acc_other_group']:.3f}")
        for sp, row in f["diet_by_member"].items():
            print(f"    {sp:15s} Q {f['consumption_by_member'][sp]:.4f}  " +
                  " ".join(f"{k[:6]} {x:.3f}" for k, x in row.items()))
    with open(os.path.join(out, f"split_fields_{gname}.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

# pooled check: the two groups must add back to the pooled numbers
print("\n" + "=" * 100)
fp = fields(SPECIES)
Qp, dp, perp, _ = diet(SPECIES, SPECIES, [])
print(f"pooled check: B {fp['B']:.3f}, r {fp['r']:.3f}, F {fp['F']:.4f}, K {fp['K']:.3f}, ceiling {fp['ceiling']:.2f},"
      f" Q {Qp:.3f}; diet " + ", ".join(f"{k} {x:.3f}" for k, x in dp.items()))
for gname, res in res_all.items():
    tot_epi = res["bed"]["Q"] * res["bed"]["diet"]["mobile_epifauna"] + res["wc"]["Q"] * res["wc"]["diet"]["mobile_epifauna"]
    tot_p = res["bed"]["Q"] * res["bed"]["diet"]["planktivorous_fish"] + res["wc"]["Q"] * res["wc"]["diet"]["planktivorous_fish"]
    print(f"{gname}: epifauna consumption bed {res['bed']['Q'] * res['bed']['diet']['mobile_epifauna']:.3f} + wc "
          f"{res['wc']['Q'] * res['wc']['diet']['mobile_epifauna']:.3f} = {tot_epi:.3f} t/km2/yr (pooled "
          f"{Qp * dp['mobile_epifauna']:.3f}); planktivores {tot_p:.3f} (pooled {Qp * dp['planktivorous_fish']:.3f})")
