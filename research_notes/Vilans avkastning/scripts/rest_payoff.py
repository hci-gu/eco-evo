"""Rest payoff arithmetic (research_notes/"Vilans avkastning", Q3).

Usage: python3 -I rest_payoff.py <repo_root> > ../tables/rest_payoff_stdout.txt

Reads only fgconfig/fg_library.yaml (checked pairs on the five fish groups, floors, costs, energy
contents). No model run; no value is chosen from model output. The frozen fix105 decomposition
(research_notes/"Mattnad hos rovfisk"/model/tables/fill_decomp_all_s0_fix105.txt, a-umlaut) enters only
as a yardstick for the rest share a full fish could take (rho = 1 - need/potential); it is hard-coded below
with its source and never used to pick a parameter.

Definitions (per fish group X, per predator p on X):
  M2_p      booked predation mortality of the checked pair p -> X (/yr; library `predation_mortality`).
            Booked, not realised: realised M2 is lower for several pairs (e.g. whiting ~0.24 of booking,
            research_notes/"Bytessidans predation"), so every payoff below is an upper bound.
  floor_p   the pair's visibility_floor, else X's species floor: a resting share is exposed at floor_p.
  r_p       activity-risk ratio = risk(active) / risk(rest) = 1 / floor_p in the engine's form.
  A.  The rest payoff ALREADY in the model at today's floors, per unit of rest share:
        dM2 = sum_p M2_p (1 - floor_p)          (/yr of biomass saved per unit rest share)
  B.  The payoff a uniform r on a predator class would give, with the a's read two ways:
        enc   the a is the ACTIVE prey's encounter (fish pairs: prey cruising 0.10-0.20 m/s in v_rel):
              saving per unit rested time = mu_act (1 - 1/r), mu_act = M2 (the booking is matched to it)
        avg   the a is a field AVERAGE over activity share T_f (mammal pairs: a set from diets):
              mu_act = M2 / (T_f + (1 - T_f)/r);  saving = mu_act (1 - 1/r)
        Payoff at exact surplus = rho x saving  (%/yr of biomass), rho = 1 - need/potential.
  C.  At need: value of one foraging tick (MJ/t) = S x ceiling x (1 - sda) - (feeding_cost - 1) RM
      vs value of the rest tick's mortality saving = saving/1460 x (E_lean + fill x ME).
      Break-even r at need. At surplus the foraging tick is worth ~0 (the reserve-room cap refills
      any cost), so any r > 1 pays in theory there.
  D.  Gerritsen-Strickler encounter ratio r_enc = v_rel(v, u) / v_rel(v, 0) for each predator member's
      search speed v (the chain's own BL/s x L values, research_notes/"Attackrater ur motesteori"
      pred_encounter_a.py P and benth_encounter.py V_PRED, o-umlaut) and the prey's cruising speed u
      (0.10 m/s small fish and juvenile gadoids, 0.20 m/s herring); ambushers meet a motionless prey
      only if it is within strike range, i.e. r_enc -> infinity for a pure ambusher [D/J].
"""
import math
import os
import sys

import yaml

ROOT = os.path.abspath(sys.argv[1])
LIB = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml")))
S = LIB["species_definitions"]
I = LIB["interaction_definitions"]

FISH = ["planktivorous_fish", "benthivorous_fish", "whiting", "predatory_fish", "mackerel"]
SHORT = {"planktivorous_fish": "planktivores", "benthivorous_fish": "benthivores", "whiting": "whiting",
         "predatory_fish": "water column", "mackerel": "mackerel", "porpoises": "porpoises",
         "seals": "seals", "seabirds": "seabirds"}
MAMMALS = {"porpoises", "seals"}
MUTED = {"seabirds"}  # mareld2.yaml mutes seabirds

# ---------------------------------------------------------------- yardstick (fix105, model output, sizing only)
# T, uncapped (feeding level at T), realised: research_notes/"Mattnad hos rovfisk"/model/tables/
# fill_decomp_all_s0_fix105.txt (seed 0). S = uncapped / T = share of the ceiling while foraging.
FIX105 = {"planktivorous_fish": (0.93, 0.540, 0.261), "benthivorous_fish": (0.95, 0.930, 0.638),
          "whiting": (0.89, 0.735, 0.389), "predatory_fish": (0.97, 0.756, 0.397),
          "mackerel": (0.97, 0.555, 0.283)}
# ceiling (MJ/t/tick) of the main prey pair, E_prey / h: library handling times (krill E 5000; benthivores
# the 223.15 base 26.18, AGENTS.md section 5)
CEIL = {"planktivorous_fish": 5000.0 / I["planktivorous_fish_preys_on_krill"]["handling_time"],
        "benthivorous_fish": 26.18,
        "whiting": 5000.0 / I["whiting_preys_on_krill"]["handling_time"],
        "predatory_fish": 5000.0 / I["predatory_fish_preys_on_krill"]["handling_time"],
        "mackerel": 5000.0 / I["mackerel_preys_on_krill"]["handling_time"]}


def pairs_on(prey):
    out = []
    for k, v in I.items():
        if "_preys_on_" not in k or not isinstance(v, dict) or not v.get("preys_on"):
            continue
        pred, py = k.split("_preys_on_")
        if py != prey or pred in MUTED:
            continue
        floor = v.get("visibility_floor", S[prey].get("visibility_floor", 1.0))
        out.append(dict(pred=pred, m2=float(v.get("predation_mortality") or 0.0), floor=float(floor),
                        key="pair" if "visibility_floor" in v else "species", sa=v.get("stage_access")))
    return out


print("=" * 100)
print("A. Booked M2 on each fish group by predator, today's floors, and the rest payoff already in the model")
print("=" * 100)
TOT = {}
for X in FISH:
    ps = pairs_on(X)
    fish_m2 = sum(p["m2"] for p in ps if p["pred"] not in MAMMALS)
    mam_m2 = sum(p["m2"] for p in ps if p["pred"] in MAMMALS)
    dm2 = sum(p["m2"] * (1.0 - p["floor"]) for p in ps)
    TOT[X] = (fish_m2, mam_m2, ps)
    print(f"\n{SHORT[X]}: booked M2 fish {fish_m2:.4f}  mammals {mam_m2:.4f}  total {fish_m2 + mam_m2:.4f} /yr")
    print(f"  {'predator':14s} {'M2':>7s} {'share':>6s} {'floor':>6s} {'(key)':>8s} {'r=1/f':>6s} "
          f"{'M2(1-f)':>8s}  stage_access [J, A]")
    for p in sorted(ps, key=lambda d: -d["m2"]):
        sh = p["m2"] / max(fish_m2 + mam_m2, 1e-12)
        r = 1.0 / p["floor"] if p["floor"] > 0 else float("inf")
        print(f"  {SHORT[p['pred']]:14s} {p['m2']:7.4f} {sh:6.2f} {p['floor']:6.2f} {p['key']:>8s} {r:6.2f} "
              f"{p['m2'] * (1 - p['floor']):8.4f}  {p['sa']}")
    print(f"  rest payoff in the model today: {dm2:.4f} /yr of biomass per unit rest share "
          f"({100 * dm2:.2f} %/yr if the whole group rested all year)")

print("\nM2 on juveniles vs adults: the engine has ONE rest share per cell for J and A; stage_access above")
print("shows that most fish-pair predation falls on J (f_J >> f_A), while Q2's activity data are for adults.")

# ---------------------------------------------------------------- B
print("\n" + "=" * 100)
print("B. Payoff of a uniform activity-risk ratio r per predator class, at the yardstick rest share rho")
print("=" * 100)
print("rho = 1 - (foraging share at need) = 1 - realised / S, S = uncapped / T; fix105 yardstick (sizing only)")
RHO = {}
for X in FISH:
    T, unc, real = FIX105[X]
    Sx = unc / T
    t_need = real / Sx
    RHO[X] = 1.0 - t_need
    print(f"  {SHORT[X]:13s} T {T:.2f}  S (while foraging) {Sx:.3f}  realised {real:.3f}  "
          f"foraging share at need {t_need:.2f}  rho {RHO[X]:.2f}")

RS = [1.05, 1.1, 1.2, 1.33, 1.5, 2.0, 3.0]
TF = (0.5, 0.75, 0.9)
print("\nSaving per unit rested time (/yr) and payoff at rho (%/yr of biomass); class r applied to that class only,")
print("others at r = 1.  enc = a is the active prey's encounter (fish pairs); avg(T_f) = a is a field average")
print("over activity share T_f (mammal pairs; a set from diets), mu_act = M2/(T_f + (1-T_f)/r).")
for X in FISH:
    fish_m2, mam_m2, ps = TOT[X]
    print(f"\n{SHORT[X]} (fish M2 {fish_m2:.4f}, mammal M2 {mam_m2:.4f}, rho {RHO[X]:.2f})")
    hdr = "  r     " + "  fish enc: save  pay%  " + "  ".join(f"mam avg{t:.2f}: save  pay%" for t in TF)
    print(hdr)
    for r in RS:
        s_f = fish_m2 * (1 - 1 / r)
        row = f"  {r:4.2f}   {s_f:14.4f} {100 * RHO[X] * s_f:5.2f}  "
        for t in TF:
            s_m = mam_m2 / (t + (1 - t) / r) * (1 - 1 / r)
            row += f"  {s_m:17.4f} {100 * RHO[X] * s_m:5.2f}"
        print(row)

# ---------------------------------------------------------------- C
print("\n" + "=" * 100)
print("C. Energy: one foraging tick at need vs the mortality saving of one rested tick (MJ/t/tick)")
print("=" * 100)
print("forage value = S x ceiling x (1 - sda) - (feeding_cost - 1) RM  (upper bound: assimilation < 1 ignored)")
print("saving value = (M2_total (1 - 1/r)) / 1460 x (E_lean + 1.0 x ME)  (enc form, all predators at r)")
for X in FISH:
    sp = S[X]
    T, unc, real = FIX105[X]
    Sx = unc / T
    rm = float(sp["resting_metabolism"]); fc = float(sp["feeding_cost"]); sda = float(sp.get("sda_fraction") or 0)
    E = float(sp["energy_content"]); ME = float(sp["max_energy_reserve"])
    fv = Sx * CEIL[X] * (1 - sda) - (fc - 1) * rm
    m2 = TOT[X][0] + TOT[X][1]
    tonne = E + ME
    sv_inf = m2 / 1460 * tonne
    print(f"\n{SHORT[X]}: ceiling {CEIL[X]:.1f}, S {Sx:.2f}, RM {rm}, feeding_cost {fc}, sda {sda} -> forage value "
          f"{fv:.1f} MJ/t/tick; extra foraging cost (fc-1)RM {(fc - 1) * rm:.2f}")
    print(f"  tonne value E + ME = {tonne:.0f} MJ; total booked M2 {m2:.4f}: complete safety at rest (r = inf) "
          f"saves {sv_inf:.3f} MJ/t/tick = {sv_inf / fv:.4f} of the forage value")
    for r in (1.1, 1.5, 2.0):
        sv = m2 * (1 - 1 / r) / 1460 * tonne
        print(f"    r {r:4.2f}: saving {sv:.4f} MJ/t/tick ({sv / fv:.5f} of the forage value; "
              f"{sv / max((fc - 1) * rm, 1e-9):.3f} of the extra foraging cost)")
    print("  -> at need rest never pays by risk (saving << forage value); at exact surplus the forage value of the")
    print("     marginal tick is ~0 (reserve-room cap) and only the saving remains: any r > 1 pays in theory.")

# ---------------------------------------------------------------- D
print("\n" + "=" * 100)
print("D. Gerritsen-Strickler encounter ratio r_enc = v_rel(v, u) / v_rel(v, 0) (rest = motionless prey)")
print("=" * 100)


def vrel(v, u):
    if v >= u:
        return (u * u + 3 * v * v) / (3 * v)
    return (v * v + 3 * u * u) / (3 * u)


# (label, predator search speed m/s); the chain's central BL/s x a typical member length [S on the chain]
PRED = [("saithe day 1.0 BL/s x 0.50 m", 0.50), ("saithe night x0.6", 0.30), ("saithe low 0.5 BL/s x0.3 night", 0.075),
        ("whiting 0.8 BL/s x 0.28 m", 0.224), ("whiting 0.6 BL/s x 0.25 m (293 reading)", 0.15),
        ("cod 0.4 BL/s x 0.40 m (benthivores)", 0.16), ("cod low 0.2 BL/s x 0.40 m", 0.08),
        ("mackerel 1.8 BL/s x 0.32 m", 0.576), ("hake 0.5 BL/s x 0.50 m", 0.25), ("spurdog 0.5 BL/s x 0.70 m", 0.35),
        ("harbour seal search ~1.0 m/s [J]", 1.0), ("harbour porpoise search ~1.2 m/s [J]", 1.2)]
for u, ul in ((0.10, "u 0.10 m/s (pout, juvenile gadoids, whiting)"), (0.20, "u 0.20 m/s (herring)")):
    print(f"\n  prey {ul}")
    for lab, v in PRED:
        print(f"    {lab:42s} v {v:5.3f}: r_enc {vrel(v, u) / vrel(v, 0.0):.3f}  (floor-equivalent "
              f"{vrel(v, 0.0) / vrel(v, u):.3f})")
print("\n  bed ambusher (anglerfish; prey must swim into the strike zone, v -> 0): r_enc -> infinity for a")
print("  motionless prey; anglerfish is a small share of every water-column pair's a (0.044 of the take on")
print("  benthivores, research_notes/'Bottenfiskens och vattenpelarens golv' D1) [S/J].")
print("  obligate swimmers (herring schools 0.25-0.42 m/s even when not feeding; mackerel): 'rest' is not")
print("  motionless, r_enc = 1 by this route [S, research_notes/'Planktivorernas golv' E8, 'Makrillens")
print("  antipredatorbeteende'].")
