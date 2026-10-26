"""Pair tables, cross-group pairs, derived growth and the step-4 diagnostic for the predator split.

Usage: python3 -I split_pairs_diagnostic.py <tables_dir> <fg_dir>

<tables_dir> holds split_fields_G1.json, split_fields_G2.json and split_encounter.json (written by
split_fields.py and split_encounter.py). <fg_dir> = research_notes/"Fiskgrupper i rutan" (for
tables/long_series_by_year.csv and long_run_species_by_group.csv: each group's own 2001-25 span).

DIAGNOSTIC ONLY: nothing here is used to set a value; it reports what the derived values imply.
Engine identities used (section 5 of AGENTS.md, lib/world/growth_budget.py, tools/probes/budget_gate.py):
  intake per tonne predator per tick at eat share 1, hunger 1: a B / (1 + a h B), B = f x visible prey
  ceiling 1/h; h = E_prey / energy ceiling, so every pair's energy ceiling is the same (x assimilation)
  s* = (C/(1-u) - cost - P ec) / (C/(1-u) + P ME), C = assimilation x ceiling, cost = RM x feeding_cost,
  P = (r_max + M1 + sum M2)/1460, g = P / (s* - u)   (section 160; regional K loss not in g)
  budget gate: sat_min = cost / (assimilation x ceiling); need visible B = sat_min / ((1 - sat_min) a h)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

td, fgd = sys.argv[1:3]
F = {g: json.load(open(os.path.join(td, f"split_fields_{g}.json"))) for g in ("G1", "G2")}
ENC = json.load(open(os.path.join(td, "split_encounter.json")))
TICKS = 1460.0
U = 0.5  # engine maintenance_level convention (resume 187.4)

PREY = {  # B (t/km2, today's / start), E as eaten (MJ/t), assimilation, accessible f, visible range
    "planktivorous_fish": dict(B=3.3, E=7125.0, ae=0.8, f=1.0, vis=(0.35, 1.0)),
    "zooplankton": dict(B=9.72, E=4500.0, ae=0.8, f=1.0, vis=(0.5 / 9.72, 1.0)),
    "mobile_epifauna": dict(B=7.63, E=3750.0, ae=0.8, f=0.13, vis=(1.0, 1.0)),
    "benthivorous_fish": dict(B=1.53, E=4500.0, ae=0.8, f=None, vis=(0.6, 1.0)),
}
# seals and porpoises (params_top_predators.md): predator share of the diet 0.15 / 0.20, rations
# 320 / 234 t/yr -> 48 / 47 t/yr on predatory_fish. Whiting share of that (J):
#  seals: Koster whiting 6-8 % of the diet; saithe = half of "haddock/pollack/saithe" (Kappa 2020
#    18.7 % by biomass; Sorlie 2020 Qi 32.7 %) ~9 % -> whiting:saithe ~7:9 -> 0.44 (0.3-0.6)
#  porpoises: Borjesson 2003 whiting 5.4, saithe 5.0, half of Gadidae spp. 6.8 split as identified
#    -> 0.52; Aarefjord 1995 whiting 8.2 + half of Gadidae 11.1, no saithe -> 1.0; by stomach
#    numbers (112, 39) -> 0.64 (0.52-1.0). Hake, spurdog, anglerfish, mackerel not reported.
MAMMALS = {"seals": dict(t=48.0, wh=0.44, wh_lo=0.3, wh_hi=0.6, a=0.05, ceil=212.5),
           "porpoises": dict(t=47.0, wh=0.64, wh_lo=0.52, wh_hi=1.0, a=0.20, h=28.6)}
# benthivores -> predators: consumption 0.067 t/t/yr x 1.53 t/km2 (benthivores/tables/
# group_diet_consumption.csv, central); cod's fish prey in this pool is whiting (cod_fish_prey_pools.csv)
BENTH_ON_PRED = 0.067 * 1.53
BENTH_CEIL = 27.1
# epifauna (research_notes/"Infauna och rorlig epifauna"/params_split.md 0a; resume 223)
EPI = dict(r_gross=0.879, K=(11.94, 12.2), B=7.63, benth=1.112, trawl=0.136 * 7.63, pred_pooled=0.287)
# planktivores (resume 209.3, 220): r_max 0.37, K 4.8, M1 0.26, B* 3.3; predation budget per pair
PLK = dict(r=0.37, K=4.8, M1=0.26, B=3.3, M2={"predatory_fish": 0.50, "benthivorous_fish": 0.14,
                                             "seals": 0.0067, "porpoises": 0.0127})
ZOO = dict(B=9.72, B_mean=5.5, M2_pred=0.13, M2_plk=1.6)


def sat(a, h, B):
    x = a * h * B
    return x, x / (1 + x)


lines = []


def p(s=""):
    print(s)
    lines.append(s)


for g in ("G1", "G2"):
    bed, wc = F[g]["bed"], F[g]["wc"]
    p("=" * 110)
    p(f"{g}: bed = {bed['members']} (B* {bed['B']:.3f}); water column = {wc['members']} (B* {wc['B']:.3f})")
    E_grp = {"bed": bed["E_whole"], "wc": wc["E_whole"]}
    B_grp = {"bed": bed["B"], "wc": wc["B"]}
    # ------------------------------------------------ mammals and benthivores on each group
    p("\n-- predation ON each group by the other model groups (M2 /yr)")
    onM2 = {"bed": {}, "wc": {}}
    for mam, m in MAMMALS.items():
        for role, sh, lo, hi in (("bed", m["wh"], m["wh_lo"], m["wh_hi"]), ("wc", 1 - m["wh"], 1 - m["wh_hi"], 1 - m["wh_lo"])):
            t = m["t"] * sh
            m2 = t / (B_grp[role] * 3600)
            onM2[role][mam] = m2
            h = m.get("h", E_grp[role] / m.get("ceil", 1.0)) if mam == "seals" else m["h"]
            p(f"  {mam:9s} -> {role}: {t:5.1f} t/yr, M2 {m2:.4f} (range {m['t'] * lo / (B_grp[role] * 3600):.4f}-"
              f"{m['t'] * hi / (B_grp[role] * 3600):.4f}); h {h:.1f}; a {m['a']} (kept, J)")
    m2b = BENTH_ON_PRED / B_grp["bed"]
    onM2["bed"]["benthivorous_fish"] = m2b
    onM2["bed"]["wc (cross pair)"] = wc["M2"]["other_group"]
    onM2["wc"]["bed (cross pair, if checked)"] = bed["M2"]["other_group"]
    p(f"  benthivores -> bed: {BENTH_ON_PRED:.4f} t/km2/yr -> M2 {m2b:.4f}; h = {bed['E_whole']:.0f}/{BENTH_CEIL} = "
      f"{bed['E_whole'] / BENTH_CEIL:.1f}; a 0.026 (section 220 near-bottom band, kept); -> wc: none (cod's "
      f"predator prey is whiting)")
    p(f"  wc -> bed (cross pair): M2 {wc['M2']['other_group']:.4f}; bed -> wc: M2 {bed['M2']['other_group']:.4f} "
      f"(diet share {bed['diet']['other_group']:.3f})")
    p(f"  check, whiting 1+ (SMS 2023 North Sea): saithe M2 0.085, cod 0.05, porpoise 0.19, cannibalism 0.094, "
      f"M1 0.20, total M 0.685")
    # ------------------------------------------------ derived growth (s*, g) and budget gate per group
    for role in ("bed", "wc"):
        f = F[g][role]
        cost = f["smr"] * f["feeding_cost"]
        C = 0.8 * f["ceiling"]
        checked_on = {k: v for k, v in onM2[role].items() if "if checked" not in k}
        gross = f["r"] + f["M1"] + sum(checked_on.values())
        P = gross / TICKS
        s_star = (C / (1 - U) - cost - P * f["e_lean"]) / (C / (1 - U) + P * f["res_max"])
        gr = P / (s_star - U)
        sat_min = cost / C
        p(f"\n-- {role}: derived growth: gross {gross:.3f} /yr (r {f['r']:.3f} + M1 {f['M1']:.3f} + M2 "
          f"{sum(checked_on.values()):.3f} {', '.join(f'{k} {v:.3f}' for k, v in checked_on.items())}); "
          f"C {C:.2f}, cost {cost:.2f} MJ/t/tick; s* {s_star:.3f}; growth_rate {gr:.6f} /tick "
          f"(pooled library 0.0022805)")
        p(f"   budget gate: sat_min = cost / C = {sat_min:.3f} (share of the ceiling a pair must deliver at eat share 1)")
        f["_g"] = gr
        f["_sat_min"] = sat_min
        f["_gross"] = gross
    # ------------------------------------------------ pair tables with saturation
    p("\n-- pairs (a = layer variant central, MC 5-95 %; h = E/ceiling; x = a h f B at today's / low visible)")
    pairs = {"bed": ["planktivorous_fish", "zooplankton", "mobile_epifauna", "benthivorous_fish"],
             "wc": ["planktivorous_fish", "zooplankton", "benthivorous_fish", "whiting_prey", "mobile_epifauna"]}
    diag = {}
    for role in ("bed", "wc"):
        f = F[g][role]
        diag[role] = {}
        for pr in pairs[role]:
            e = ENC[f"{g}_{role}"][pr]["layer"]
            a, a05, a95 = e["central"], e["mc05"], e["mc95"]
            if pr == "whiting_prey":
                E, ae, acc, Bp = bed["E_whole"], 0.8, f["acc_other_group"], bed["B"]
                vis = (0.6, 1.0)
                m2 = f["M2"]["other_group"]
                share = f["diet"]["other_group"]
                name = "other group (bed)"
            else:
                q = PREY[pr]
                E, ae, Bp, vis = q["E"], q["ae"], q["B"], q["vis"]
                acc = q["f"] if q["f"] is not None else f["acc_benthivores"]
                m2 = f["M2"][pr]
                share = f["diet"][pr]
                name = pr
            h = E / f["ceiling"]
            B_hi, B_lo = acc * Bp * vis[1], acc * Bp * vis[0]
            x_hi, fr_hi = sat(a, h, B_hi)
            x_lo, fr_lo = sat(a, h, B_lo)
            half = 1.0 / (a * h * acc) if a > 0 else float("inf")
            need = f["_sat_min"] / ((1 - f["_sat_min"]) * a * h) / acc if a > 0 else float("inf")
            ceil_dem = f["B"] * TICKS / h
            mean_dem = f["B"] * TICKS * a * B_hi / (1 + a * h * B_hi)
            diet_dem = f["Q"] * share
            diag[role][pr] = dict(a=a, h=h, acc=acc, x_hi=x_hi, fr_hi=fr_hi, fr_lo=fr_lo, half=half, need=need,
                                  ceil_dem=ceil_dem, mean_dem=mean_dem, diet_dem=diet_dem, m2=m2)
            p(f"  {role:3s} -> {name:20s} a {a:.4g} ({a05:.3g}-{a95:.3g}) h {h:6.1f} f {acc:.3f} | x {x_hi:7.2f} -> "
              f"{fr_hi:.2f} of ceiling (low visible {fr_lo:.2f}) | half-sat (whole pool) {half:.3g} t/km2 vs B {Bp:.3g} | "
              f"break-even pool {need:.3g} | demand t/km2/yr: ceiling {ceil_dem:.2f}, eat share 1 at today's B "
              f"{mean_dem:.2f}, diet share {diet_dem:.3f} (M2 {m2:.4f})")
    # ------------------------------------------------ step 4: epifauna, planktivores, zooplankton budgets
    p("\n-- step 4 diagnostic")
    for K in EPI["K"]:
        p(f"  epifauna maximum surplus r_gross K/4 = {EPI['r_gross']} x {K} / 4 = {EPI['r_gross'] * K / 4:.2f} t/km2/yr")
    surplus_now = EPI["r_gross"] * EPI["B"] * (1 - EPI["B"] / 12.2)
    p(f"  epifauna surplus at today's B {EPI['B']} (logistic, K 12.2): {surplus_now:.2f}; taken at equilibrium by "
      f"benthivores {EPI['benth']:.2f} + trawl {EPI['trawl']:.2f} + predators (pooled budget) {EPI['pred_pooled']:.2f} = "
      f"{EPI['benth'] + EPI['trawl'] + EPI['pred_pooled']:.2f}")
    room = EPI["r_gross"] * 12.2 / 4 - EPI["benth"] - EPI["trawl"]
    d = diag["bed"]["mobile_epifauna"]
    p(f"  bed group on epifauna: saturated ({d['fr_hi']:.3f} of ceiling; half-saturation {d['half']:.3f} t/km2 of the "
      f"whole pool, pooled today 0.16); demand at the ceiling with the whole eat share {d['ceil_dem']:.2f} t/km2/yr "
      f"(pooled group 1.20 x 1460/147.1 = {1.20 * 1460 / 147.1:.2f}); at its diet share {d['diet_dem']:.3f}; "
      f"room left by benthivores + trawl at the maximum surplus {room:.2f}")
    for frac in (0.137, 0.3, 0.5, 0.82):
        p(f"    eat share on epifauna {frac:.3f} at the ceiling -> {d['ceil_dem'] * frac:.2f} t/km2/yr "
          f"({d['ceil_dem'] * frac / room:.2f} x the room)")
    plk_budget = sum(PLK["M2"].values()) * PLK["B"]
    gross_plk = (PLK["r"] * (1 - PLK["B"] / PLK["K"]) + PLK["M1"] + sum(PLK["M2"].values())) * PLK["B"]
    for role in ("bed", "wc"):
        d = diag[role]["planktivorous_fish"]
        p(f"  {role} on planktivores: {d['fr_hi']:.2f} of ceiling at all visible ({d['fr_lo']:.2f} at 35 % visible); "
          f"half-sat {d['half']:.3g}; demand ceiling {d['ceil_dem']:.2f}, eat share 1 today {d['mean_dem']:.2f}, diet "
          f"{d['diet_dem']:.3f} t/km2/yr")
    p(f"  planktivore budgets: predatory_fish M2 0.50 x 3.3 = {0.5 * 3.3:.2f}; all predators {plk_budget:.2f}; gross "
      f"production at B* (r(1-B/K) + M1 + sum M2) B = {gross_plk:.2f} t/km2/yr")
    for role in ("bed", "wc"):
        d = diag[role]["zooplankton"]
        p(f"  {role} on zooplankton: {d['fr_hi']:.2f} of ceiling at 9.72 t/km2, half-sat {d['half']:.3g} t/km2; demand "
          f"ceiling {d['ceil_dem']:.2f}, eat share 1 at 9.72 {d['mean_dem']:.2f}, diet {d['diet_dem']:.3f}; budget "
          f"predators 0.13 x 9.72 = {0.13 * 9.72:.2f}")
    # achievable fill of the water-column group without bed pairs
    for role in ("bed", "wc"):
        best = max((dd["fr_hi"], k) for k, dd in diag[role].items() if not (role == "wc" and k == "mobile_epifauna"))
        p(f"  {role}: best pair at today's mean density gives {best[0]:.2f} of the ceiling ({best[1]}); budget gate "
          f"needs {F[g][role]['_sat_min']:.2f} just to pay the eat cost")
    # ------------------------------------------------ per-group span 2001-25
p("\n" + "=" * 110)
LS = pd.read_csv(os.path.join(fgd, "tables", "long_series_by_year.csv"))
LR = pd.read_csv(os.path.join(fgd, "tables", "long_run_species_by_group.csv")).set_index("common")
ratio = float(LR.loc["whiting", "t_km2"] / LR.loc["whiting", "kgh_2001_25"])
LS = LS.rename(columns={"Year": "year"})
ls = LS[(LS.year >= 2001) & (LS.year <= 2025)].dropna(subset=["predatory_fish_kgh"]).copy()
ls["whiting_t"] = ls["whiting_kgh"] * ratio
pr = LR[LR.guild == "predatory_fish"]
ratio_pr = float(pr.t_km2.sum() / pr.kgh_2001_25.sum())
ratio_sa = float(LR.loc["saithe", "t_km2"] / LR.loc["saithe", "kgh_2001_25"])
ls["rest_t"] = ls["predatory_fish_kgh"] * ratio_pr - ls["whiting_t"]
ls["saithe_t"] = ls["saithe_kgh"] * ratio_sa
p(f"Each group's own 2001-25 span (GOV wing, q = 1, kg/h -> t/km2 ratio {ratio:.5f}; years with a full stratum set "
  f"only, {len(ls)} years):")
for col, B in (("whiting_t", F["G1"]["bed"]["B"]), ("rest_t", F["G1"]["wc"]["B"]), ("saithe_t", None)):
    v = ls[col]
    p(f"  {col:10s} mean {v.mean():.3f} median {v.median():.3f} min {v.min():.3f} max {v.max():.3f}; 2001-20 mean "
      f"{ls[ls.year <= 2020][col].mean():.3f}; 2021-25 mean {ls[ls.year >= 2021][col].mean():.3f}"
      + (f"; B* used {B:.3f} = x{B / v.mean():.2f} on the mean" if B else ""))
p("  per year: " + "; ".join(f"{int(r.year)} wh {r.whiting_t:.2f} rest {r.rest_t:.2f}" for r in ls.itertuples()))
with open(os.path.join(td, "split_pairs_diagnostic_stdout.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
