"""Pair tables, cross-group pairs, derived growth and the diagnostic for the predator split, 223.15 library.

Usage: python3 -I split_pairs_diagnostic_223.py <tables_dir> <fg_dir> <attack_dir>

<tables_dir> holds split_fields_223_G1.json, split_fields_223_G2.json and split_encounter_223.json
(written by split_fields_223.py and split_encounter_223.py). <fg_dir> = research_notes/"Fiskgrupper i
rutan" (benthivores/tables/length_biomass_longrun.csv; tables/long_series_by_year.csv and
long_run_species_by_group.csv for each group's own 2001-25 span). <attack_dir> = research_notes/
"Attackrater ur motesteori" (Swedish letters; not read, cited only: the benthivore stage a below are
printed in its tables/benth_encounter_kd0.25_stdout.txt section 5).

Copy of split_pairs_diagnostic.py (section-222 library), with every library constant moved to 223.15
(benthivore ceiling 26.18 and Q/B 4.46 diet, E1 exposed epifauna, the benthivores' M2 on planktivores
0.245 and on predators 0.128) and three additions: the attack-rate convention table (section 5 of
params_split_223.md), the M2 conservation table on every prey, and the benthivore-access diagnostic.

DIAGNOSTIC ONLY for the saturation / demand / access parts: nothing there sets a value.
Engine identities (lib/environments/ecosystem_env/predation.py, interactions.holling_a_eff,
lib/world/growth_budget.py):
  demand per cell = B_pred x eat x hunger x a (f B) / (1 + a h (f B) + w B_pred)   (no phi: a is per
  tonne of the WHOLE group that the policy allocates to the prey)
  ceiling 1/h; h = E_prey / energy ceiling
  s* = (C/(1-u) - cost - P ec) / (C/(1-u) + P ME), C = assimilation x ceiling, cost = RM x feeding_cost,
  P = (r_max + M1 + sum M2)/1460, g = P / (s* - u)   (section 160; regional K loss not in g)
  budget gate: sat_min = cost / (assimilation x ceiling)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

td, fgd = sys.argv[1:3]
F = {g: json.load(open(os.path.join(td, f"split_fields_223_{g}.json"))) for g in ("G1", "G2")}
ENC = json.load(open(os.path.join(td, "split_encounter_223.json")))
TICKS = 1460.0
U = 0.5  # engine maintenance_level convention (resume 187.4)
ASSIM = 0.8

# ---------------------------------------------------------------- library constants (223.15)
LIB = dict(B=1.20, r=0.42, M1=0.000176012 * 1460, ceiling=25.5, RM=3.64, fc=2.1, ec=4100.0, ME=2400.0,
           g=0.0024482169414)
PREY = {  # B (t/km2 today / start), E as eaten (MJ/t), accessible f (pooled), visible range (prey floor)
    "planktivorous_fish": dict(B=3.3, E=7125.0, f=1.0, vis=(0.35, 1.0)),
    "zooplankton": dict(B=9.72, E=4500.0, f=1.0, vis=(0.5 / 9.72, 1.0)),
    "mobile_epifauna": dict(B=3.7814, E=3750.0, f=0.29, vis=(1.0, 1.0)),
    "benthivorous_fish": dict(B=1.53, E=4500.0, f=None, vis=(0.55, 1.0)),
}
# benthivores -> predators: library M2 0.128 on 1.20 t/km2 (223.10: Q/B 4.46 x 1.53 x share 0.023)
BENTH_ON_PRED = 0.128 * 1.20                      # t/km2/yr
BENTH_W_TO_WHITING = (1.0, 0.89, 1.0)             # central, low, high: cod's W prey = whiting (Skagerrak 3.7/3.7,
                                                  # box 10.3/10.3, Kattegat 10.1/11.3, NNS 10.8/11.6 g; cod_fish_prey_pools.csv)
BENTH_CEIL = 26.18
BENTH_A_CAPABLE = {"P": 0.019, "W": 0.026}       # library (220, Kd 0.25), per tonne inside phi
# per-stage a (km2/t/tick, central, Kd 0.25) from Attackrater ur motesteori/tables/benth_encounter_kd0.25_stdout.txt sec. 5
BENTH_STAGE_A = {
    "P": {("cod", 2): 0.0170, ("cod", 1): 0.0198, ("haddock", 0): 0.0244, ("haddock", 1): 0.0194,
          ("cod", 0): 0.0263, ("starry ray", 1): 0.0250, ("long rough dab", 0): 0.0236,
          ("long rough dab", 1): 0.0174, ("grey gurnard", 0): 0.0313, ("grey gurnard", 1): 0.0263,
          ("starry ray", 0): 0.0294},
    "W": {("cod", 2): 0.0243, ("cod", 1): 0.0282, ("starry ray", 1): 0.0356, ("grey gurnard", 0): 0.0447,
          ("grey gurnard", 1): 0.0375, ("starry ray", 0): 0.0420},
}
# benth_encounter.py constants (copied): group shares, stage cuts, central stage scores
BGROUP = {"haddock": 38.62, "cod": 31.38, "long rough dab": 12.63, "dab": 4.49, "plaice": 3.80,
          "witch": 1.67, "starry ray": 1.47, "grey gurnard": 0.89, "lemon sole": 0.87}
BCUTS = {"cod": [30, 50], "haddock": [40], "long rough dab": [20], "dab": [20], "plaice": [25],
         "witch": [30], "lemon sole": [25], "grey gurnard": [25], "starry ray": [30]}
BSCORE = {"P": {"cod": [.3, .7, .9], "haddock": [.1, .25], "long rough dab": [.05, .05],
                "starry ray": [.4, .4], "grey gurnard": [.15, .15]},
          "W": {"cod": [0, .15, .4], "haddock": [0, 0], "starry ray": [.3, .3], "grey gurnard": [.3, .3]}}
BFLAT = {"dab", "plaice", "witch", "lemon sole"}

# seals and porpoises (params_top_predators.md): predator share of the diet 0.15 / 0.20, library M2 0.011
# each on 1.20 t/km2 -> 47.5 t/yr each. Whiting share of that (J, as the 222 version):
#  seals: Koster whiting 6-8 %; saithe ~9 % -> 0.44 (0.3-0.6)
#  porpoises: Borjesson 2003 whiting 5.4, saithe 5.0 + Gadidae split -> 0.52; Aarefjord 1995 -> 1.0;
#    by stomach numbers -> 0.64 (0.52-1.0)
MAMMALS = {"seals": dict(t=0.011 * 1.2 * 3600, wh=0.44, wh_lo=0.3, wh_hi=0.6, a=0.05, ceil=212.5),
           "porpoises": dict(t=0.011 * 1.2 * 3600, wh=0.64, wh_lo=0.52, wh_hi=1.0, a=0.20, h=28.6)}
# E1 exposed epifauna (fg_library.yaml 223.15): r_max 1.14, M2 benthivores 0.54 + predators 0.076,
# trawl F 0.186, B* 3.78, equilibrium share 0.543 -> K mean 6.96
EPI = dict(r_max=1.14, M2_ben=0.54, M2_pred=0.076, F=0.186, B=3.7814)
EPI["r_gross"] = EPI["r_max"] + EPI["M2_ben"] + EPI["M2_pred"]
EPI["K"] = EPI["B"] / (1 - (EPI["F"] + EPI["M2_ben"] + EPI["M2_pred"]) / EPI["r_gross"])
# planktivores (209, 223.10): r_max 0.37, K 4.8, M1 0.26, B* 3.3; checked predation budget per pair
PLK = dict(r=0.37, K=4.8, M1=0.26, B=3.3, F=0.10, M2={"predatory_fish": 0.50, "benthivorous_fish": 0.245,
                                                       "seals": 0.0067, "porpoises": 0.0127})
# benthivores (209, 223.10): r 0.31, M1 0.29, F 0.17, K 3.39, B 1.53; booked M2 by predators 0.18,
# seals 0.035, porpoises 0.0064
BEN = dict(r=0.31, M1=0.29, F=0.17, K=3.39, B=1.53, M2={"predatory_fish": 0.18, "seals": 0.035,
                                                       "porpoises": 0.0064})

lines = []


def p(s=""):
    print(s)
    lines.append(s)


def sat(a, h, B):
    x = a * h * B
    return x, x / (1 + x)


def s_star_g(gross, ceiling, rm, fc, ec, me):
    C = ASSIM * ceiling
    cost = rm * fc
    P = gross / TICKS
    s = (C / (1 - U) - cost - P * ec) / (C / (1 - U) + P * me)
    return s, P / (s - U), cost / C


# ---------------------------------------------------------------- 0. engine check of the closed form
gross_lib = LIB["r"] + LIB["M1"] + 0.011 + 0.011 + 0.128
s0, g0, sm0 = s_star_g(gross_lib, LIB["ceiling"], LIB["RM"], LIB["fc"], LIB["ec"], LIB["ME"])
p(f"0. closed-form check on the pooled library group: gross {gross_lib:.4f}, s* {s0:.4f}, g {g0:.10f} "
  f"(library growth_rate {LIB['g']:.10f}); budget gate cost/C {sm0:.3f}")

# ---------------------------------------------------------------- 1. benthivore-side convention factors
LB = pd.read_csv(os.path.join(fgd, "benthivores", "tables", "length_biomass_longrun.csv"))
stage_share = {}
for sp in BGROUP:
    d = LB[(LB.species == sp) & (LB.share_2001_25 > 0)]
    s = d.share_2001_25 / d.share_2001_25.sum()
    edges = BCUTS[sp]
    st = np.searchsorted(np.array(edges), d.L_cm.values, side="right")
    for k in range(len(edges) + 1):
        stage_share[(sp, k)] = float(s[st == k].sum())
gtot = sum(BGROUP.values())


def bscore(pair, sp, k):
    sc = BSCORE[pair].get(sp)
    if sc is not None:
        return sc[k]
    return 0.0 if (pair == "W" or sp in BFLAT) else 0.05


p("\n1. Benthivore fish pairs: the same a under the per-whole-group conventions (220 stage table, Kd 0.25)")
BCONV = {}
for pair in ("P", "W"):
    wsum = wss = num_cap = num_layer = num_scored = 0.0
    for (sp, k), sh in stage_share.items():
        Wk = BGROUP[sp] / gtot * sh
        sc = bscore(pair, sp, k)
        a_k = BENTH_STAGE_A[pair].get((sp, k), 0.0)
        wsum += Wk
        wss += Wk * sc
        num_cap += Wk * sc * a_k
        num_scored += Wk * sc * a_k
        num_layer += Wk * (1.0 if sc >= 0.1 else 0.0) * a_k
    cap, scored, layer = num_cap / wss, num_scored / wsum, num_layer / wsum
    phi = wss / wsum
    BCONV[pair] = dict(capable=cap, scored=scored, layer=layer, phi=phi)
    p(f"  benthivores -> {'planktivores' if pair == 'P' else 'predators  '}: capable (220) {cap:.4f} (library "
      f"{BENTH_A_CAPABLE[pair]}); phi = sum W s / sum W {phi:.3f}; per whole group: scored {scored:.4f}, "
      f"layer (s >= 0.1) {layer:.4f}")

for g in ("G1", "G2"):
    bed, wc = F[g]["bed"], F[g]["wc"]
    p("\n" + "=" * 110)
    p(f"{g}: bed = {bed['members']} (B* {bed['B']:.3f}, layer {bed['layer']}); water column = {wc['members']} "
      f"(B* {wc['B']:.3f}, layer {wc['layer']})")
    E_grp = {"bed": bed["E_whole"], "wc": wc["E_whole"]}
    B_grp = {"bed": bed["B"], "wc": wc["B"]}
    # ------------------------------------------------ 2. predation ON each group
    p("\n2. Predation ON each group by the other model groups (M2 /yr)")
    onM2 = {"bed": {}, "wc": {}}
    for mam, m in MAMMALS.items():
        for role, sh, lo, hi in (("bed", m["wh"], m["wh_lo"], m["wh_hi"]),
                                 ("wc", 1 - m["wh"], 1 - m["wh_hi"], 1 - m["wh_lo"])):
            t = m["t"] * sh
            m2 = t / (B_grp[role] * 3600)
            onM2[role][mam] = m2
            h = E_grp[role] / m["ceil"] if mam == "seals" else m["h"]
            p(f"  {mam:9s} -> {role}: {t:5.1f} t/yr, M2 {m2:.4f} (range {m['t'] * lo / (B_grp[role] * 3600):.4f}-"
              f"{m['t'] * hi / (B_grp[role] * 3600):.4f}); h {h:.1f}; a {m['a']} (kept, J)")
    wsh, wlo, whi = BENTH_W_TO_WHITING
    # in G2 the bed group also holds hake/spurdog/anglerfish, still the cod W prey is whiting
    m2b = BENTH_ON_PRED * wsh / B_grp["bed"]
    m2b_wc = BENTH_ON_PRED * (1 - wsh) / B_grp["wc"]
    onM2["bed"]["benthivorous_fish"] = m2b
    onM2["bed"]["wc (cross pair)"] = wc["M2"]["other_group"]
    onM2["wc"]["bed (cross pair, if checked)"] = bed["M2"]["other_group"]
    h_bb = bed["E_whole"] / BENTH_CEIL
    p(f"  benthivores -> bed: {BENTH_ON_PRED * wsh:.4f} t/km2/yr -> M2 {m2b:.4f} (whiting share of cod's W 0.89-1.0: "
      f"{BENTH_ON_PRED * wlo / B_grp['bed']:.3f}-{BENTH_ON_PRED * whi / B_grp['bed']:.3f}); h = {bed['E_whole']:.0f}/"
      f"{BENTH_CEIL} = {h_bb:.1f}; -> wc: M2 {m2b_wc:.4f} at the central share (unchecked)")
    p(f"  wc -> bed (cross pair): M2 {wc['M2']['other_group']:.4f}; bed -> wc: M2 {bed['M2']['other_group']:.4f} "
      f"(diet share {bed['diet']['other_group']:.3f}; unchecked)")
    tot_bed = bed["M1"] + sum(v for k, v in onM2["bed"].items())
    p(f"  bed group booked M1 + M2 = {bed['M1']:.3f} + {tot_bed - bed['M1']:.3f} = {tot_bed:.3f} /yr; check (not a "
      f"target): SMS 2023 North Sea whiting 1+ total M 0.685 (M1 0.20, cod 0.05, saithe 0.085, porpoise 0.19, "
      f"cannibalism 0.094)")
    # ------------------------------------------------ 3. derived growth and budget gate per group
    p("\n3. Derived growth (closed form, section 160) and budget gate")
    for role in ("bed", "wc"):
        f = F[g][role]
        checked_on = {k: v for k, v in onM2[role].items() if "if checked" not in k}
        gross = f["r"] + f["M1"] + sum(checked_on.values())
        fc = round(f["feeding_cost"], 2)
        for lab, ceil, rm in (("central", f["ceiling"], f["smr"]),
                              ("RM kept at the at_50m value", f["ceiling"], f["smr_at_50m"]),
                              ("Bjornsson-anchored ceiling", f["ceiling_bjornsson"], f["smr"])):
            s, gr, sm = s_star_g(gross, ceil, rm, fc, f["e_lean"], f["res_max"])
            p(f"  {role:3s} [{lab}]: gross {gross:.3f} /yr (r {f['r']:.3f} + M1 {f['M1']:.3f} + M2 "
              f"{sum(checked_on.values()):.3f}); ceiling {ceil:.2f}, RM {rm:.3f} x fc {fc}; s* {s:.3f}; growth_rate "
              f"{gr:.6f} /tick (pooled library {LIB['g']:.6f}); cost/C {sm:.3f}")
            if lab == "central":
                f["_g"], f["_sat_min"], f["_gross"], f["_s"] = gr, sm, gross, s
        p(f"     booked M2 on {role}: " + ", ".join(f"{k} {v:.4f}" for k, v in checked_on.items()))
    # ------------------------------------------------ 4. pair tables
    p("\n4. Pairs (a = layer variant central, MC 5-95 %; h = E/ceiling; x = a h f B at today's / low visible)")
    pairs = {"bed": ["planktivorous_fish", "zooplankton", "mobile_epifauna", "benthivorous_fish"],
             "wc": ["planktivorous_fish", "zooplankton", "benthivorous_fish", "whiting_prey", "mobile_epifauna"]}
    diag = {}
    for role in ("bed", "wc"):
        f = F[g][role]
        diag[role] = {}
        for pr in pairs[role]:
            e = ENC[f"{g}_{role}"][pr]["layer"]
            a, a05, a95, nd = e["central"], e["mc05"], e["mc95"], e["night_day"]
            if pr == "whiting_prey":
                E, acc, Bp, vis = bed["E_whole"], f["acc_other_group"], bed["B"], (0.5, 1.0)
                m2 = f["M2"]["other_group"]
                share = f["diet"]["other_group"]
                name = "other group (bed)"
            else:
                q = PREY[pr]
                E, Bp, vis = q["E"], q["B"], q["vis"]
                acc = q["f"] if q["f"] is not None else round(f["acc_benthivores"], 2)
                m2 = f["M2"][pr]
                share = f["diet"][pr]
                name = pr
            h = E / f["ceiling"]
            B_hi, B_lo = acc * Bp * vis[1], acc * Bp * vis[0]
            x_hi, fr_hi = sat(a, h, B_hi)
            x_lo, fr_lo = sat(a, h, B_lo)
            half = 1.0 / (a * h * acc) if a > 0 else float("inf")
            ceil_dem = f["B"] * TICKS / h
            mean_dem = f["B"] * TICKS * a * B_hi / (1 + a * h * B_hi)
            diet_dem = f["Q"] * share
            diag[role][pr] = dict(a=a, h=h, acc=acc, x_hi=x_hi, fr_hi=fr_hi, fr_lo=fr_lo, half=half,
                                  ceil_dem=ceil_dem, mean_dem=mean_dem, diet_dem=diet_dem, m2=m2, nd=nd)
            p(f"  {role:3s} -> {name:20s} a {a:.4g} ({a05:.3g}-{a95:.3g}) n/d {nd:.2f} h {h:6.1f} f {acc:.3f} M2 {m2:.4f} | "
              f"x {x_hi:6.2f} -> {fr_hi:.2f} of ceiling (low visible {fr_lo:.2f}) | half-sat (whole pool) {half:.3g} "
              f"vs B {Bp:.3g} | demand t/km2/yr: ceiling {ceil_dem:.2f}, eat share 1 today {mean_dem:.2f}, diet "
              f"{diet_dem:.3f}")
    h_bb = bed["E_whole"] / BENTH_CEIL
    p(f"  benthivores -> bed: h {h_bb:.1f}; a 0.026 capable (220) = {BCONV['W']['layer']:.4f} layer / "
      f"{BCONV['W']['scored']:.4f} scored per whole benthivore tonne; x at B 0.446 (all visible) with the scored a "
      f"{BCONV['W']['scored'] * h_bb * bed['B']:.3f} -> {sat(BCONV['W']['scored'], h_bb, bed['B'])[1]:.2f} of ceiling")
    # ------------------------------------------------ 5. diagnostic: demand vs supply
    p("\n5. DIAGNOSTIC (not used to set anything)")
    surplus_now = EPI["r_gross"] * EPI["B"] * (1 - EPI["B"] / EPI["K"])
    p(f"  epifauna (E1): r_gross {EPI['r_gross']:.3f}, K mean {EPI['K']:.2f}, max surplus r K/4 "
      f"{EPI['r_gross'] * EPI['K'] / 4:.2f}; surplus at B* {EPI['B']:.2f}: {surplus_now:.2f} = benthivores "
      f"{EPI['M2_ben'] * EPI['B']:.2f} + trawl {EPI['F'] * EPI['B']:.2f} + predators (pooled budget) "
      f"{EPI['M2_pred'] * EPI['B']:.3f}")
    room_max = EPI["r_gross"] * EPI["K"] / 4 - EPI["M2_ben"] * (EPI["K"] / 2) - EPI["F"] * (EPI["K"] / 2)
    room_now = EPI["M2_pred"] * EPI["B"]
    d = diag["bed"]["mobile_epifauna"]
    p(f"  bed -> epifauna: {d['fr_hi']:.3f} of ceiling; half-saturation {d['half']:.3f} t/km2 of the pool; demand at "
      f"the ceiling {d['ceil_dem']:.2f} t/km2/yr (pooled library group 1.20 x 1460/147.1 = {1.20 * 1460 / 147.1:.2f}); "
      f"diet share {d['diet_dem']:.3f}; room for predators at today's B {room_now:.3f} (at K/2: {room_max:.2f})")
    for frac in (0.137, 0.3, 0.5, 0.6):
        p(f"    eat share {frac:.3f} at the ceiling -> {d['ceil_dem'] * frac:.2f} t/km2/yr = "
          f"{d['ceil_dem'] * frac / room_now:.1f} x the room at today's B")
    hg = (1 - bed["_s"]) / (1 - U)
    p(f"    hunger at the derived s* {bed['_s']:.3f}: (1 - s)/(1 - u) = {hg:.2f} -> fed bed group at most "
      f"{d['ceil_dem'] * hg:.2f} t/km2/yr on epifauna")
    gross_plk = (PLK["r"] * (1 - PLK["B"] / PLK["K"]) + PLK["M1"] + sum(PLK["M2"].values())) * PLK["B"]
    for role in ("bed", "wc"):
        d = diag[role]["planktivorous_fish"]
        p(f"  {role} -> planktivores: {d['fr_hi']:.2f} of ceiling all visible ({d['fr_lo']:.2f} at 35 %); half-sat "
          f"{d['half']:.3g}; demand ceiling {d['ceil_dem']:.2f}, eat share 1 {d['mean_dem']:.2f}, diet "
          f"{d['diet_dem']:.3f} (budget M2 {d['m2']:.3f} x 3.3 = {d['m2'] * 3.3:.3f})")
    p(f"  planktivore budgets: all predators 0.50 x 3.3 = 1.65; all checked predation "
      f"{sum(PLK['M2'].values()) * PLK['B']:.2f}; gross production at B* {gross_plk:.2f} t/km2/yr")
    for role in ("bed", "wc"):
        d = diag[role]["zooplankton"]
        p(f"  {role} -> zooplankton: {d['fr_hi']:.2f} of ceiling at 9.72, half-sat {d['half']:.3g}; demand ceiling "
          f"{d['ceil_dem']:.2f}, eat share 1 {d['mean_dem']:.2f}, diet {d['diet_dem']:.3f}")
    for role in ("bed", "wc"):
        cand = {k: dd for k, dd in diag[role].items() if not (role == "wc" and k == "mobile_epifauna")}
        best = max((dd["fr_hi"], k) for k, dd in cand.items())
        p(f"  budget gate {role}: needs cost/C = {F[g][role]['_sat_min']:.3f}; best pair today {best[1]} "
          f"{best[0]:.2f}; " + ", ".join(f"{k} {dd['fr_hi']:.2f}/{dd['fr_lo']:.2f}" for k, dd in cand.items()))
    # ------------------------------------------------ 6. benthivore access: pooled vs split
    p("\n6. Access to the benthivores (B 1.53), sum over predator groups")
    ben_pairs = {
        "pooled, library as is (fix77)": [(0.18, 0.15, 1.20, 4500 / 25.5)],
        "pooled, layer convention": [(ENC["pooled"]["benthivorous_fish"]["layer"]["central"], 0.15, 1.20,
                                      4500 / 25.5)],
        "split, layer convention": [(diag["bed"]["benthivorous_fish"]["a"], diag["bed"]["benthivorous_fish"]["acc"],
                                     bed["B"], diag["bed"]["benthivorous_fish"]["h"]),
                                    (diag["wc"]["benthivorous_fish"]["a"], diag["wc"]["benthivorous_fish"]["acc"],
                                     wc["B"], diag["wc"]["benthivorous_fish"]["h"])],
        "split, capable convention (220 as in the library)": [
            (ENC[f"{g}_bed"]["benthivorous_fish"]["capable"]["central"], diag["bed"]["benthivorous_fish"]["acc"],
             bed["B"], diag["bed"]["benthivorous_fish"]["h"]),
            (ENC[f"{g}_wc"]["benthivorous_fish"]["capable"]["central"], diag["wc"]["benthivorous_fish"]["acc"],
             wc["B"], diag["wc"]["benthivorous_fish"]["h"])],
    }
    for lab, lst in ben_pairs.items():
        afb = sum(a * f * Bp for a, f, Bp, h in lst)
        lin = afb * TICKS
        for vis in (1.0, 0.55):
            Bb = BEN["B"] * vis
            m2u = sum(Bp * TICKS * a * f * vis / (1 + a * h * f * Bb) for a, f, Bp, h in lst)
            dem = sum(Bp * TICKS / h * (a * h * f * Bb) / (1 + a * h * f * Bb) for a, f, Bp, h in lst)
            p(f"  {lab:50s} sum a f B_pred {afb:.5f} km2/tick (linear-limit M2 per unit eat {lin:6.1f} /yr); at "
              f"{vis:.2f} visible: M2 per unit eat share {m2u:5.2f} /yr, demand at eat 1 {dem:.2f} t/km2/yr")
    p(f"  benthivore budget: booked M2 by predators 0.18, all predators {sum(BEN['M2'].values()):.3f}; gross "
      f"production ceiling r + M1 + M2 = {BEN['r'] + BEN['M1'] + sum(BEN['M2'].values()):.2f} /yr; fix77 realised "
      f"predator M2 0.48-0.65 (resume 223.25, model output, context only)")
    # ------------------------------------------------ 7. M2 conservation per prey
    p("\n7. M2 conservation on every prey (pooled library -> split)")
    cons = [("planktivorous_fish", 0.50, bed["M2"]["planktivorous_fish"], wc["M2"]["planktivorous_fish"], "both"),
            ("zooplankton", 0.13, bed["M2"]["zooplankton"], wc["M2"]["zooplankton"], "both"),
            ("mobile_epifauna", 0.076, bed["M2"]["mobile_epifauna"], 0.0, "bed only; wc unchecked"),
            ("benthivorous_fish", 0.18, bed["M2"]["benthivorous_fish"], wc["M2"]["benthivorous_fish"], "both")]
    for prey, lib, mb, mw, note in cons:
        p(f"  {prey:18s} library {lib:.4f} -> bed {mb:.4f} + wc {mw:.4f} = {mb + mw:.4f} ({note}); "
          f"difference {mb + mw - lib:+.4f}")
    t_seal = MAMMALS["seals"]["t"] / 3600
    t_porp = MAMMALS["porpoises"]["t"] / 3600
    p(f"  as prey (t/km2/yr): seals {t_seal:.4f} = bed {onM2['bed']['seals'] * bed['B']:.4f} + wc "
      f"{onM2['wc']['seals'] * wc['B']:.4f}; porpoises {t_porp:.4f} = {onM2['bed']['porpoises'] * bed['B']:.4f} + "
      f"{onM2['wc']['porpoises'] * wc['B']:.4f}; benthivores {BENTH_ON_PRED:.4f} = "
      f"{onM2['bed']['benthivorous_fish'] * bed['B']:.4f} + 0; NEW booked flow wc -> bed "
      f"{wc['M2']['other_group'] * bed['B']:.4f} (was an unbooked within-group transfer, section 160)")

# ---------------------------------------------------------------- 8. per-group span 2001-25
p("\n" + "=" * 110)
LS = pd.read_csv(os.path.join(fgd, "tables", "long_series_by_year.csv"))
LR = pd.read_csv(os.path.join(fgd, "tables", "long_run_species_by_group.csv")).set_index("common")
ratio = float(LR.loc["whiting", "t_km2"] / LR.loc["whiting", "kgh_2001_25"])
LS = LS.rename(columns={"Year": "year"})
ls = LS[(LS.year >= 2001) & (LS.year <= 2025)].dropna(subset=["predatory_fish_kgh"]).copy()
ls["whiting_t"] = ls["whiting_kgh"] * ratio
pr = LR[LR.guild == "predatory_fish"]
ratio_pr = float(pr.t_km2.sum() / pr.kgh_2001_25.sum())
ls["rest_t"] = ls["predatory_fish_kgh"] * ratio_pr - ls["whiting_t"]
p(f"8. Each group's own 2001-25 span (GOV wing, q = 1; {len(ls)} years):")
for col, B in (("whiting_t", F["G1"]["bed"]["B"]), ("rest_t", F["G1"]["wc"]["B"])):
    v = ls[col]
    p(f"  {col:10s} mean {v.mean():.3f} median {v.median():.3f} min {v.min():.3f} max {v.max():.3f}; 2021-25 mean "
      f"{ls[ls.year >= 2021][col].mean():.3f}; B* used {B:.3f} = x{B / v.mean():.2f} on the mean")
with open(os.path.join(td, "split_pairs_diagnostic_223_stdout.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
