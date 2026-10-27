"""Species fields, diets and pair M2 of the two candidate predator groups, on the 223.15 library.

Usage: python3 -I split_fields_223.py <fg_predators_dir> <out_dir>

<fg_predators_dir> = research_notes/"Fiskgrupper i rutan"/predators (inputs/species_values.csv,
inputs/diet_consumption_central.csv, tables/pred_season.csv, tables/pred_length_biomass.csv,
tables/benthivore_prey_sizes_stdout.txt is not read; its cumulative shares are copied below).

Copy of split_fields.py (written for the section-222 library), adapted to resume 223.10-223.15:
  * prey pools: mobile_epifauna = the EXPOSED taxa only (E1, B* 3.78 t/km2), norway_lobster a group
    of its own (4.57 t/km2; no member takes more than traces -> flow 0, J);
  * a reproduction gate: the ten members pooled must land on the CURRENT library pair M2
    (predators -> planktivores 0.50, -> zooplankton 0.13, -> mobile_epifauna 0.076,
    -> benthivorous_fish 0.18) before anything is split; the split M2 are then scaled per prey so
    the two groups add back to the library value exactly (J: conserves every prey's growth budget);
  * temperature layer per group (bed group on the `bottom` layer as the benthivores since 223.15,
    J; water-column group at_50m): the energy ceiling is the GER Cmax at the layer's annual mean
    (whiting evacuation e^0.078 T, Andersen's form used for the gadoids), and resting_metabolism
    re-read at that mean with the group Q10; the Bjornsson cod re-anchoring of 223.15 is printed as
    an alternative (the predator GER fill was calibrated on the FB4 cod Cmax at 500 g, 7 C).

Same construction as ../../Fiskgrupper i rutan/predators/scripts/weighted_fields.py (section 209):
biomass-weighted species values with the long-run (2001-25) NS-IBTS GOV shares, B* = 1.20 t/km2 x
the members' share (top of the long-run span, user's choice), ranges = corner bounds. No model output
is read. Two groupings are evaluated:

  G1 (diet criterion, recommended): bed = whiting; water column = the other nine members.
  G2 (habitat criterion, variant):  bed = whiting, hake, spurdog, anglerfish, ling, velvet belly;
                                     water column = saithe, mackerel, horse mackerel, pollack.

Writes <out_dir>/split_fields_223_<G>.json and prints every number used in params_split_223.md.
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

# Prey pools of the 223.15 library (grid means, t/km2): planktivores 3.3 (12 000 t), zooplankton
# 35 000 t, mobile_epifauna E1 exposed taxa 3.78 (engine B* mean, 13 600 t), norway_lobster 4.57,
# infauna 93.6, benthivores 1.53. Energies as eaten (static quality): planktivores 4000 + 0.5 x 6250,
# benthivores 3600 + 0.5 x 1800 = 4500 (223.10), epifauna 3750, Nephrops 3600, infauna 2600.
PREY_B = {"planktivorous_fish": 3.3, "zooplankton": 35000 / 3600, "mobile_epifauna": 3.7814,
          "norway_lobster": 4.5711, "benthic_infauna": 93.6, "benthivorous_fish": 1.53}
PREY_E = {"planktivorous_fish": 7125.0, "zooplankton": 4500.0, "mobile_epifauna": 3750.0,
          "norway_lobster": 3600.0, "benthic_infauna": 2600.0, "benthivorous_fish": 4500.0}
ASSIM = {"planktivorous_fish": 0.8, "zooplankton": 0.8, "mobile_epifauna": 0.8, "norway_lobster": 0.8,
         "benthic_infauna": 0.7, "benthivorous_fish": 0.8, "other_predator_group": 0.8}
# Current library pair M2 of the pooled group (fg_library.yaml, 223.15): the reproduction targets.
LIB_M2 = {"planktivorous_fish": 0.50, "zooplankton": 0.13, "mobile_epifauna": 0.076,
          "benthivorous_fish": 0.18}
# Nephrops share of each member's benthos part (research_notes/"Havskrafta i fiskdieter",
# species_nephrops_summary.csv): whiting <= 0.55 % of its diet in the northern data, 0 in every
# Skagerrak/box/Kattegat sample, 0.2 % W Irish Sea; saithe and hake none recorded. Spurdog,
# anglerfish and ling are occasional Nephrops predators elsewhere (J) but carry 6 % of the pooled
# biomass and B shares of 0.02-0.06: their Nephrops flow is < 0.002 t/km2/yr. Set to 0 (J).
NEPH_FRAC = {sp: 0.0 for sp in ("saithe", "whiting", "mackerel", "spurdog", "horse mackerel", "hake",
                                "anglerfish", "ling", "velvet belly", "pollack")}
# Temperature layers (mareld2.yaml simulation_settings.temperature; annual means of the monthly
# tables): at_50m 8.37 C (the 209 Cmax and SMR reference), bottom 7.40 C (SHARK A16/A17 at each
# cell's depth, 223.15).
T_LAYER = {"at_50m": 8.367, "bottom": 7.40}
GER_K = 0.078       # gadoid evacuation temperature coefficient (Andersen's whiting form, e^0.078 T)
BJORNSSON_X = 1.30  # cod Cmax Bjornsson et al. 2001 ad-lib (prey-normalised) / FB4 at 447-874 g, 7 C:
                    # x1.31 / x1.29 (Exponering av bottenbyten/tables/C_cmax_bottom_stdout.txt, exps F, G)
LAYER = {"bed": "bottom", "wc": "at_50m"}  # J (params_split_223.md 2.2)
F_RANGE = {"saithe": (0.11, 0.38), "whiting": (0.035, 0.15)}  # evidence B; others held

GROUPINGS = {
    "G1": {"bed": ["whiting"],
           "wc": ["saithe", "mackerel", "spurdog", "horse mackerel", "hake", "anglerfish", "ling",
                  "velvet belly", "pollack"]},
    "G2": {"bed": ["whiting", "hake", "spurdog", "anglerfish", "ling", "velvet belly"],
           "wc": ["saithe", "mackerel", "horse mackerel", "pollack"]},
}


def fields(members, layer="at_50m"):
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
    # Cmax and SMR in species_values.csv are at the at_50m annual mean (8.4 C, evidence A); move them
    # to the group's layer mean: Cmax with the gadoid evacuation term, SMR with the group Q10.
    dT = T_LAYER[layer] - T_LAYER["at_50m"]
    q10 = wm("q10")
    t_cmax = float(np.exp(GER_K * dT))
    t_smr = float(q10 ** (dT / 10.0))
    cm, clo, chi = wm("cmax_kJkgd") * t_cmax, wm("cmax_lo") * t_cmax, wm("cmax_hi") * t_cmax
    ceil, ceil_lo, ceil_hi = cm / 4.0, clo / 4.0, chi / 4.0
    smr50 = wm("smr")
    smr, slo, shi = smr50 * t_smr, wm("smr_lo") * t_smr, wm("smr_hi") * t_smr
    el = wm("e_lean")
    rmx, rmean = wm("res_max"), wm("res_mean")
    qb, qlo, qhi = wm("qb"), wm("qb_lo"), wm("qb_hi")
    u_lit = 0.15
    f = dict(members=members, layer=layer, T_layer=T_LAYER[layer], t_cmax=t_cmax, t_smr=t_smr, smr_at_50m=smr50,
             cmax_at_50m=wm("cmax_kJkgd"), cmax_bjornsson=cm * BJORNSSON_X, ceiling_bjornsson=cm * BJORNSSON_X / 4.0,
             weights={k: round(float(x), 4) for k, x in w.items()}, share_of_pooled=share,
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
    out = {k: 0.0 for k in ("planktivorous_fish", "zooplankton", "mobile_epifauna", "norway_lobster",
                            "benthic_infauna", "benthivorous_fish", "own_group", "other_group", "other")}
    per = {}
    for sp in members:
        dd = DIET[sp]
        wc = WCOMP[sp]
        w_own = sum(v for k, v in wc.items() if k in own)
        w_oth = sum(v for k, v in wc.items() if k in other)
        row = {"planktivorous_fish": dd["P"], "zooplankton": dd["Z"],
               "mobile_epifauna": dd["B"] * EPI_FRAC[sp] * (1 - NEPH_FRAC[sp]),
               "norway_lobster": dd["B"] * EPI_FRAC[sp] * NEPH_FRAC[sp],
               "benthic_infauna": dd["B"] * (1 - EPI_FRAC[sp]),
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


# ------------------------------------------------------------------ reproduction gate (pooled, 223.15)
fp = fields(SPECIES)
Qp, dp, perp, _ = diet(SPECIES, SPECIES, [])
M2P = {prey: Qp * dp[prey] / B for prey, B in PREY_B.items()}
SCALE = {prey: (LIB_M2[prey] / M2P[prey] if prey in LIB_M2 else 1.0) for prey in PREY_B}
print("=" * 100)
print(f"REPRODUCTION GATE (all ten members pooled; the 223.15 library is the target)")
print(f"  B {fp['B']:.3f}, r {fp['r']:.3f}, F {fp['F']:.4f} (bottom {fp['F_bot']:.4f}, pelagic {fp['F_pel']:.4f}), "
      f"K {fp['K']:.3f}, M1 {fp['M1']:.3f}, ceiling at_50m {fp['ceiling']:.2f}, SMR {fp['smr']:.2f}, Q {Qp:.3f} "
      f"(Q/B {Qp / fp['B']:.2f}); lean {fp['e_lean']:.0f}, max reserve {fp['res_max']:.0f}, fill {fp['fill']:.3f}")
print("  library: B 1.20, r 0.42, F 0.162 + 0.020, K 2.12, M1 0.257, ceiling 25.5, RM 3.64, Q/B 3.34, lean 4100, "
      "ME 2400, fill 0.48")
print("  pooled diet: " + ", ".join(f"{k} {x:.4f}" for k, x in dp.items()))
for prey in PREY_B:
    lib = LIB_M2.get(prey)
    print(f"  M2 on {prey:18s} reproduced {M2P[prey]:.4f}" + (f"  library {lib:.4f}  ratio {M2P[prey] / lib:.3f}  -> "
          f"scale applied to the split {SCALE[prey]:.4f}" if lib else "  (unchecked in the library)"))
print(f"  epifauna flow pooled {Qp * dp['mobile_epifauna']:.4f} t/km2/yr (E1 split_e1 massbal_neph: 0.2866)")

res_all = {}
for gname, gp in GROUPINGS.items():
    print("=" * 100)
    print(f"Grouping {gname}: bed = {gp['bed']}; water column = {gp['wc']}")
    res = {}
    for role in ("bed", "wc"):
        res[role] = fields(gp[role], LAYER[role])
    for role, other in (("bed", "wc"), ("wc", "bed")):
        Qt, dshare, per, Q = diet(gp[role], gp[role], gp[other])
        f = res[role]
        f["Q"] = Qt
        f["Q_over_B"] = Qt / f["B"]
        f["diet"] = dshare
        f["diet_by_member"] = per
        f["consumption_by_member"] = Q
        # M2 on prey
        m2, m2raw = {}, {}
        for prey, B in PREY_B.items():
            m2raw[prey] = Qt * dshare[prey] / B
            m2[prey] = m2raw[prey] * SCALE[prey]
        f["M2_raw"] = m2raw
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
        print(f"  layer {f['layer']} (annual mean {f['T_layer']:.2f} C): Cmax x{f['t_cmax']:.4f}, SMR x{f['t_smr']:.4f} "
              f"vs at_50m")
        print(f"  Cmax {f['cmax']:.1f} kJ/kg/d ({f['cmax_lo']:.1f}-{f['cmax_hi']:.1f}; at_50m {f['cmax_at_50m']:.1f}) -> "
              f"ceiling {f['ceiling']:.2f} MJ/t/tick ({f['ceiling_lo']:.2f}-{f['ceiling_hi']:.2f}); Bjornsson-anchored "
              f"alternative x{BJORNSSON_X}: {f['cmax_bjornsson']:.1f} kJ/kg/d = {f['ceiling_bjornsson']:.2f}")
        print(f"  SMR {f['smr']:.3f} ({f['smr_lo']:.2f}-{f['smr_hi']:.2f}; at_50m {f['smr_at_50m']:.3f}); Q10 {f['q10']:.2f}; "
              f"feeding_cost {f['feeding_cost']:.2f} x RM")
        print(f"  energy lean {f['e_lean']:.0f}; max reserve {f['res_max']:.0f}; mean reserve {f['res_mean']:.0f}; "
              f"fill {f['fill']:.3f}; whole body as eaten {f['E_whole']:.0f} MJ/t")
        print(f"  starve_rate (u 0.15 convention) {f['starve_rate']:.4f} ({f['starve_lo']:.4f}-{f['starve_hi']:.4f})")
        print(f"  Q/B (member values) {f['qb']:.3f} ({f['qb_lo']:.2f}-{f['qb_hi']:.2f}); Q {f['Q']:.3f} t/km2/yr")
        print(f"  trawl tables: bottom biomass_factor at 100 {f['bf100']:.6f} ({f['bf100_lo']:.6f}-{f['bf100_hi']:.6f});"
              f" pelagic at 10 {f['pf10']:.6f}")
        print("  diet (consumption-weighted): " + ", ".join(f"{k} {x:.3f}" for k, x in f["diet"].items()))
        print("  M2 on prey (/yr, scaled to the library sums): " + ", ".join(f"{k} {x:.4f}" for k, x in f["M2"].items()))
        print("  M2 on prey (/yr, raw member diets): " + ", ".join(f"{k} {x:.4f}" for k, x in f["M2_raw"].items()))
        print("  dark_ratio (diel stomach, consumption-weighted): " +
              ", ".join(f"{k} {x:.2f}" for k, x in f["dark_ratio_stomach"].items()))
        print(f"  accessible_fraction by gape: benthivores {f['acc_benthivores']:.3f}; other group "
              f"{f['acc_other_group']:.3f}")
        for sp, row in f["diet_by_member"].items():
            print(f"    {sp:15s} Q {f['consumption_by_member'][sp]:.4f}  " +
                  " ".join(f"{k[:6]} {x:.3f}" for k, x in row.items()))
    with open(os.path.join(out, f"split_fields_223_{gname}.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

# conservation check: the two groups must add back to the pooled numbers
print("\n" + "=" * 100)
for gname, res in res_all.items():
    print(f"{gname}: M2 sums bed + wc (scaled) vs library: " + ", ".join(
        f"{p} {res['bed']['M2'][p] + res['wc']['M2'][p]:.4f}/{LIB_M2[p]:.4f}" for p in LIB_M2))
    tot_epi = res["bed"]["Q"] * res["bed"]["diet"]["mobile_epifauna"] + res["wc"]["Q"] * res["wc"]["diet"]["mobile_epifauna"]
    tot_p = res["bed"]["Q"] * res["bed"]["diet"]["planktivorous_fish"] + res["wc"]["Q"] * res["wc"]["diet"]["planktivorous_fish"]
    print(f"{gname}: epifauna consumption bed {res['bed']['Q'] * res['bed']['diet']['mobile_epifauna']:.3f} + wc "
          f"{res['wc']['Q'] * res['wc']['diet']['mobile_epifauna']:.3f} = {tot_epi:.3f} t/km2/yr (pooled "
          f"{Qp * dp['mobile_epifauna']:.3f}); planktivores {tot_p:.3f} (pooled {Qp * dp['planktivorous_fish']:.3f})")
