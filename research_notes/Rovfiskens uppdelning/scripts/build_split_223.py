"""Predator split G1 -> a library COPY and a manifest COPY (the live files are never touched here).

Usage (from the repository root):
  python3 -I "research_notes/Rovfiskens uppdelning/scripts/build_split_223.py" \
      fgconfig/fg_library.yaml mareld2.yaml "research_notes/Rovfiskens uppdelning/tables/split_fields_223_G1.json" \
      "research_notes/Rovfiskens uppdelning/data"

User decisions (2026-10-27, resume 223.26): G1 (whiting = bed group, new id `whiting`; the water-column
group keeps `predatory_fish`); benthivores -> whiting M2 0.345; the whole-group attack-rate convention
library-wide; the Bjornsson anchor (x1.30) on both predator ceilings; whiting on the `bottom` layer with
its resting metabolism re-read there, the benthivores' re-read in the same step; water-column K 2.14.

What it does
1. Text-edits the live library (comments kept) into <out>/fg_library_split.yaml: the two species blocks,
   every pair / observes / impact entry the pooled group had for both ids, the new cross pairs, the
   benthivores' resting metabolism and planktivore attack rate, the seals' spawn refs.
2. h of every predator pair = prey_energy / the group's ceiling (lib/world/energy_balance.prey_energy).
3. Derives every active group's growth_rate with lib/world/growth_budget.py (as the loader does) and
   writes the changed ones; scales mobile_epifauna's capacity_depth_table so the equilibrium start
   total is kept (its M2 lost the water-column 0.007).
4. Writes <out>/mareld2_split.yaml (group_layers, decision_makers, FollowDemersalFish refs).
5. Verifies: both copies parse; stored growth_rate == derived for every active group; habitat totals;
   load_project_config builds every FG; M2 sums on every prey vs the live library.
"""
import copy
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
from lib.config.config_loader import load_config, load_project_config  # noqa: E402
from lib.world import growth_budget as gb                               # noqa: E402
from lib.world import habitat                                           # noqa: E402
from lib.world import temperature                                       # noqa: E402
from lib.world.energy_balance import prey_energy                        # noqa: E402
from lib.world.impact_maps import project_baseline_maps                 # noqa: E402
from lib.world.tick_time import LIBRARY_TICK_HOURS, ticks_per_year      # noqa: E402

lib_path, project_path, g1_json, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
TPY = ticks_per_year(LIBRARY_TICK_HOURS)
GRID = (60, 60)
NOTE = 'research_notes/"Rovfiskens uppdelning" (Swedish letters)/params_split_223.md'
G = json.load(open(g1_json))
bed, wc = G["bed"], G["wc"]

old = load_config(lib_path)
project = load_config(project_path)
S_old = old["species_definitions"]

# ---- temperatures: layer annual means from the manifest ----
tset = temperature.parse_settings(project)
T_BOTTOM = float(np.mean(tset["layers"]["bottom"]))
T_50 = float(np.mean(tset["layers"]["at_50m"]))

# ---- fields (user's decisions) ----
BJ = 1.30                                                  # Bjornsson re-anchoring of the GER ceilings
CEIL = {"whiting": round(bed["ceiling"] * BJ, 2), "predatory_fish": round(wc["ceiling"] * BJ, 2)}
RM = {"whiting": round(bed["smr"], 2), "predatory_fish": round(wc["smr"], 2)}
# feeding_cost = routine 1.5 x SMR + SDA 0.17 of half the ceiling, per RM (params_split_223.md 2.1)
FC = {k: round(1.5 + 0.17 * 0.5 * CEIL[k] / RM[k], 2) for k in CEIL}
STARVE = {"whiting": round(1.5 * RM["whiting"] / 3900.0 / 0.15, 4),
          "predatory_fish": round(1.5 * RM["predatory_fish"] / 4240.0 / 0.15, 4)}
M1 = {"whiting": bed["M1"] / TPY, "predatory_fish": wc["M1"] / TPY}
BEN_Q10 = float(S_old["benthivorous_fish"]["metabolism_q10"])
BEN_RM_OLD = float(S_old["benthivorous_fish"]["resting_metabolism"])
BEN_RM = round(BEN_RM_OLD * BEN_Q10 ** ((T_BOTTOM - T_50) / 10.0), 2)
# trawl tables: F at today's mean effort of the baseline maps (tests/test_fishing_maps.py means)
MEAN_BOT, MEAN_PEL = 3.0141578935527487, 0.02496212180679044
BF100 = {"whiting": bed["F_bot"] / TPY / MEAN_BOT * 100.0, "predatory_fish": wc["F_bot"] / TPY / MEAN_BOT * 100.0}
PF10 = {"whiting": bed["F_pel"] / TPY / MEAN_PEL * 10.0, "predatory_fish": wc["F_pel"] / TPY / MEAN_PEL * 10.0}

# M2 (scaled to the library sums by split_fields_223.py), rounded so the sums stay exact
M2 = {("whiting", "planktivorous_fish"): 0.160, ("predatory_fish", "planktivorous_fish"): 0.340,
      ("whiting", "zooplankton"): 0.0544, ("predatory_fish", "zooplankton"): 0.0756,
      ("whiting", "mobile_epifauna"): 0.069,
      ("whiting", "benthivorous_fish"): 0.0967, ("predatory_fish", "benthivorous_fish"): 0.0833,
      ("predatory_fish", "whiting"): 0.157,
      ("benthivorous_fish", "whiting"): 0.345,
      ("seals", "whiting"): 0.0130, ("seals", "predatory_fish"): 0.0098,
      ("porpoises", "whiting"): 0.0190, ("porpoises", "predatory_fish"): 0.0063}
A = {("whiting", "planktivorous_fish"): 0.021, ("predatory_fish", "planktivorous_fish"): 0.0117,
     ("whiting", "zooplankton"): 0.0023, ("predatory_fish", "zooplankton"): 0.0006,
     ("whiting", "mobile_epifauna"): 0.27,
     ("whiting", "benthivorous_fish"): 0.21, ("predatory_fish", "benthivorous_fish"): 0.053,
     ("predatory_fish", "whiting"): 0.0106,
     ("benthivorous_fish", "whiting"): 0.0026, ("benthivorous_fish", "planktivorous_fish"): 0.0062,
     ("seals", "whiting"): 0.05, ("seals", "predatory_fish"): 0.05,          # judgements (209)
     ("porpoises", "whiting"): 0.2, ("porpoises", "predatory_fish"): 0.2}
F_ACC = {("whiting", "mobile_epifauna"): 0.29, ("whiting", "benthivorous_fish"): 0.05,
         ("predatory_fish", "benthivorous_fish"): 0.24, ("predatory_fish", "whiting"): 0.47}
DARK = {("whiting", "planktivorous_fish"): (0.33, -5.0), ("predatory_fish", "planktivorous_fish"): (0.39, -5.0),
        ("whiting", "zooplankton"): (0.78, -5.0), ("predatory_fish", "zooplankton"): (0.30, -5.0),
        ("whiting", "benthivorous_fish"): (0.64, -5.0), ("predatory_fish", "benthivorous_fish"): (0.48, -5.0),
        ("predatory_fish", "whiting"): (0.48, -5.0), ("benthivorous_fish", "whiting"): (0.10, 0.0)}

# ---- the new species definitions in memory (for prey_energy / growth_budget) ----
new = copy.deepcopy(old)
S, I = new["species_definitions"], new["interaction_definitions"]
wh = copy.deepcopy(S["predatory_fish"])
wh.update(display_name="Whiting", member_species=[{"name": "whiting", "share": 1.0}],
          max_energy_reserve=2000.0, resting_metabolism=RM["whiting"], feeding_cost=FC["whiting"],
          energy_content=3900.0, reserve_reference_fill=0.45, r_max=0.49,
          regional_carrying_capacity=0.533, natural_mortality=float(f"{M1['whiting']:.6g}"),
          starve_rate=STARVE["whiting"], visibility_floor=0.5)
S["whiting"] = wh
S["predatory_fish"].update(
    max_energy_reserve=2600.0, resting_metabolism=RM["predatory_fish"],
    feeding_cost=FC["predatory_fish"], energy_content=4240.0, reserve_reference_fill=0.5, r_max=0.374,
    regional_carrying_capacity=2.14, natural_mortality=float(f"{M1['predatory_fish']:.6g}"),
    starve_rate=STARVE["predatory_fish"], visibility_floor=0.7)
S["benthivorous_fish"]["resting_metabolism"] = BEN_RM
E = {fid: prey_energy(S[fid]) for fid in S}


def h_of(pred, prey):
    ceiling = CEIL.get(pred)
    if ceiling is None:
        raise KeyError(pred)
    return round(E[prey] / ceiling, 1)


H_BEN_CEIL = 26.18                                        # the benthivores' ceiling (223.15)
H_SEAL_CEIL = 212.5                                       # seals: h = E_prey / 212.5 (209)
H = {k: h_of(*k) for k in M2 if k[0] in CEIL}
H[("benthivorous_fish", "whiting")] = round(E["whiting"] / H_BEN_CEIL, 1)
H[("seals", "whiting")] = round(E["whiting"] / H_SEAL_CEIL, 1)
H[("seals", "predatory_fish")] = round(E["predatory_fish"] / H_SEAL_CEIL, 1)
H[("porpoises", "whiting")] = H[("porpoises", "predatory_fish")] = 28.6


def pair(pred, prey, *, assim=0.8, floor=None, fr=None, extra=""):
    k = (pred, prey)
    s = (f"  {pred}_preys_on_{prey}:\n    preys_on: true\n    predation_mortality: {M2[k]}\n"
         f"    assimilation_factor: {assim}\n    handling_time: {H[k]}\n")
    if k in F_ACC:
        s += f"    accessible_fraction: {F_ACC[k]}\n"
    if floor is not None:
        s += f"    visibility_floor: {floor}\n"
    if fr is not None:
        s += f"    functional_response: {fr}\n"
    s += extra
    s += f"    max_intake_rate: {A[k]}\n"
    if k in DARK:
        s += f"    dark_ratio: {DARK[k][0]}\n    light_threshold_deg: {DARK[k][1]}\n"
    return s


def off(pred, prey, m2=None, comment=""):
    s = comment + f"  {pred}_preys_on_{prey}:\n    preys_on: false\n"
    if m2 is not None:
        s += f"    predation_mortality: {m2}\n"
    return s + "    assimilation_factor: 1.0\n    handling_time: 0.0\n"


# ---------------------------------------------------------------------------------------------------
# text edits of the library (comments kept)
# ---------------------------------------------------------------------------------------------------
text = open(lib_path, encoding="utf-8-sig").read()


def block(text, name):
    m = re.compile(rf"^  {re.escape(name)}:\s*$", re.M).search(text)
    if m is None:
        raise KeyError(name)
    nxt = re.compile(r"^  \S|^\S", re.M).search(text, m.end())
    a = m.start()
    # an entry's own comment lines are the '  #' run directly above it
    lines_before = text[:a].split("\n")[:-1]
    i = len(lines_before)
    while i > 0 and lines_before[i - 1].startswith("  #"):
        i -= 1
    a0 = len("\n".join(lines_before[:i])) + (1 if i > 0 else 0)
    b = nxt.start() if nxt else len(text)
    # trailing '  #' lines inside [m.end(), b) belong to the NEXT entry
    body = text[a:b].split("\n")
    j = len(body)
    while j > 1 and (body[j - 1].startswith("  #") or body[j - 1] == ""):
        j -= 1
    b = a + len("\n".join(body[:j])) + 1
    return a0, a, b


def replace(text, name, new_text, keep_comments=False):
    a0, a, b = block(text, name)
    return text[:(a if keep_comments else a0)] + new_text + text[b:]


def after(text, name, new_text):
    _, _, b = block(text, name)
    return text[:b] + new_text + text[b:]


def entry(text, name):
    _, a, b = block(text, name)
    return text[a:b]


def renamed(text_entry, old_key, new_key):
    return text_entry.replace(f"  {old_key}:", f"  {new_key}:", 1)


def set_field(text, name, field, value):
    a0, a, b = block(text, name)
    seg = text[a:b]
    seg2, n = re.subn(rf"^(    {re.escape(field)}:)\s*.*$", rf"\g<1> {value}", seg, count=1, flags=re.M)
    if n != 1:
        raise KeyError(f"{name}.{field}")
    return text[:a] + seg2 + text[b:]


# ---- species blocks ----
mem_wc = "".join(f"      - {{name: {n}, share: {round(w / sum(wc['weights'].values()), 3)}}}\n"
                 for n, w in wc["weights"].items())
species = f"""  predatory_fish:
    display_name: "Predatory Fish"
    # Section 223.26 (predator split G1, user 2026-10-27): the WATER-COLUMN
    # predators. Whiting, the bed feeder (91 % of the pooled group's
    # epifauna flow), is its own group `whiting` below; this group keeps the
    # id but no longer feeds on the bed (benthos 0.3-1 % of every member's
    # diet). Display only (live viewer hover): the members and their
    # long-run (NS-IBTS 2001-25) biomass shares.
    # {NOTE}.
    member_species:
{mem_wc}    is_decision_maker: true
    # Section 209 / 223.26: demersal and pelagic fish that feed on fish and
    # krill: saithe 0.65, mackerel 0.13, spurdog 0.08, horse mackerel
    # 0.055, hake 0.04, anglerfish, ling, velvet belly, pollack. Saithe (Q3
    # on the trench slope) and mackerel (summer) are visitors; the closed
    # pool charges them stock-level r and F, as herring. Every value is
    # the biomass-weighted species value (weighted_fields convention, 209).
    # u 0.5: the engine convention (resume 187.4).
    maintenance_level: 0.5
    # Liver / fat reserve: saithe liver ~2000, mackerel ~6000, horse
    # mackerel ~3800, spurdog ~2000 -> 2600 (1950-3250).
    max_energy_reserve: 2600.0
    # Measured SMR at the at_50m annual mean {T_50:.2f} C: saithe 3.5, mackerel
    # 3.4, spurdog 2.1, anglerfish 1.2 -> {RM['predatory_fish']} (2.8-5.0).
    resting_metabolism: {RM['predatory_fish']}
    # Cod SMR 2.2-2.6, ICES gastric evacuation exponent 2.25, mackerel 2.0,
    # spurdog 2.5 -> 2.3; annual mean of the at_50m layer.
    metabolism_q10: 2.3
    metabolism_t_ref: annual_mean
    # Every member but anglerfish cruises 6-15 cells/tick (saithe Uopt
    # 1.4 BL/s, mackerel schools 0.44-0.72 m/s net); residence is
    # behaviour, not a speed limit.
    movement_speed: 1.0
    # Routine 1.5 x SMR + SDA 0.17 of half the ceiling {CEIL['predatory_fish']} (Bjornsson-
    # anchored, 223.26) = {FC['predatory_fish']} x RM; cruising at Uopt 2-2.5 x SMR = 2.0 x RM.
    feeding_cost: {FC['predatory_fish']}
    movement_cost: 2.0
    # Lean tissue as eaten (saithe 4.0, hake 3.7, mackerel 4.7, spurdog
    # 6.0 kJ/g) = 4240; the liver / fat is eaten with the fish: at the
    # reference fill 0.5 the static quality is 4240 + 0.5 x 2600 = {E['predatory_fish']:.0f} MJ/t.
    energy_content: 4240.0
    prey_includes_reserve: true
    reserve_reference_fill: 0.5
    min_split_biomass: 0.5
    # growth_rate is DERIVED by lib/world/growth_budget.py and recomputed
    # by the loader for every run; the number stored here is what the
    # loader computes for mareld2.yaml (seabirds muted) with
    # --mortality on; with --mortality off the loader leaves M1 out. Edit
    # r_max, natural_mortality or the predation matrix, not this value.
    # r_max: today's productivity (Euler-Lotka r_now): saithe 0.44, hake
    # 0.26, mackerel 0.25, horse mackerel 0.33, spurdog 0.07, anglerfish
    # 0.47, ling 0.55 -> 0.374 (0.32-0.42).
    r_max: 0.374
    # K = B* / (1 - F/r) = 0.754 / (1 - 0.242/0.374) = 2.14 t/km2 (1.05-11;
    # saithe F 0.11-0.38 drives it; the user chose the central value).
    regional_carrying_capacity: 2.14
    growth_rate: GROWTH_predatory_fish
    # M1 = M0 + predation by groups NOT in the model: saithe 0.29,
    # mackerel 0.15, spurdog 0.10, horse mackerel 0.30, hake 0.45,
    # anglerfish 0.25 -> {wc['M1']:.3f} /yr (0.18-0.30). Within-group predation
    # (~0.035) is a transfer inside the pool and left out.
    natural_mortality: {M1['predatory_fish']:.6g}
    max_intake_rate: 0.04
    spawn:
      mode: colony
      n_colonies: 25
      sigma_cells: 1.0
      anchor: free
      amplitude_mode: jitter
      amplitude_min: 0.0
      amplitude_max: 1.0
    # 1.5 x SMR / lean energy / 0.15 (the section 209 construction).
    starve_rate: {STARVE['predatory_fish']}
    # Judgement: midwater schools of saithe and mackerel hide badly.
    visibility_floor: 0.7
    # No field evidence of interference for saithe or mackerel (Rindorf et
    # al. 2006: predator density did not affect intake).
    interference: 0.1
    extinction_threshold_factor: 0.5
  whiting:
    display_name: "Whiting"
    # Section 223.26 (predator split G1, user 2026-10-27): the BED-feeding
    # predators = whiting alone (0.446 t/km2, 0.37 of the pooled group;
    # 92 % of its benthos consumption). Resident all year, within ~5-30 m
    # of the bed by day. {NOTE}.
    member_species:
      - {{name: whiting, share: 1.0}}
    is_decision_maker: true
    maintenance_level: 0.5
    # Gadoid liver ~2000 MJ/t (1500-2500).
    max_energy_reserve: 2000.0
    # SMR 4.2 (Steinhausen et al. 2005, attributed to 8.4 C) re-read at the
    # bottom layer's annual mean {T_BOTTOM:.2f} C with Q10 2.3: {RM['whiting']} (3.0-4.6).
    resting_metabolism: {RM['whiting']}
    metabolism_q10: 2.3
    metabolism_t_ref: annual_mean
    # Uopt 1.0 BL/s = 5.8 cells/tick at 27 cm; not binding.
    movement_speed: 1.0
    # Routine 1.5 x SMR + SDA 0.17 of half the ceiling {CEIL['whiting']} (GER Cmax 96.4
    # kJ/kg/d at {T_BOTTOM:.2f} C x 1.30 Bjornsson re-anchoring, 223.26) = {FC['whiting']} x RM;
    # cruising 2 x SMR.
    feeding_cost: {FC['whiting']}
    movement_cost: 2.0
    # Lean 3.9 kJ/g; the liver is eaten with the fish: at the reference
    # fill 0.45 (mean reserve 900 / 2000) the static quality is {E['whiting']:.0f} MJ/t.
    energy_content: 3900.0
    prey_includes_reserve: true
    reserve_reference_fill: 0.45
    min_split_biomass: 0.5
    # growth_rate is DERIVED by lib/world/growth_budget.py (see
    # predatory_fish). r_max: whiting Euler-Lotka r_now 0.49 (0.44-0.49).
    r_max: 0.49
    # K = B* / (1 - F/r) = 0.446 / (1 - 0.08/0.49) = 0.533 t/km2 (0.48-0.68).
    regional_carrying_capacity: 0.533
    growth_rate: GROWTH_whiting
    # M1 = SMS residual M 0.20 + grey seal and birds 0.05 = 0.25 /yr
    # (0.20-0.32). Booked on pairs: benthivores 0.345, predatory_fish
    # 0.157, porpoises 0.019, seals 0.013; total M 0.78 vs SMS 0.685.
    natural_mortality: {M1['whiting']:.6g}
    max_intake_rate: 0.04
    spawn:
      mode: colony
      n_colonies: 25
      sigma_cells: 1.0
      anchor: free
      amplitude_mode: jitter
      amplitude_min: 0.0
      amplitude_max: 1.0
    # 1.5 x SMR / lean energy / 0.15 (the section 209 construction).
    starve_rate: {STARVE['whiting']}
    # Judgement: whiting on the bed hide better than midwater schools.
    visibility_floor: 0.5
    # Rindorf et al. 2006: predator density not significant for whiting.
    interference: 0.1
    extinction_threshold_factor: 0.5
"""
text = replace(text, "predatory_fish", species)

# benthivores: resting metabolism re-read at the bottom layer
text = text.replace(
    "    # Weighted SMR 3.0 MJ/t/tick at the 50 m layer (haddock 3.0, cod 3.7,\n"
    "    # plaice 3.1, long rough dab 1.6, witch 1.5) x 1.2 routine at rest =\n"
    "    # 3.6 (2.4-4.3). The old gadoid 6.0 was 1.5-2x the measured SMR.\n"
    f"    resting_metabolism: {BEN_RM_OLD}\n",
    "    # Weighted SMR 3.0 MJ/t/tick at the 50 m layer (haddock 3.0, cod 3.7,\n"
    "    # plaice 3.1, long rough dab 1.6, witch 1.5) x 1.2 routine at rest =\n"
    "    # 3.6 (2.4-4.3). The old gadoid 6.0 was 1.5-2x the measured SMR.\n"
    f"    # Section 223.26: re-read at the bottom layer's annual mean ({T_BOTTOM:.2f} C\n"
    f"    # vs at_50m {T_50:.2f} C; the group moved there in 223.15) with Q10 {BEN_Q10}:\n"
    f"    # 3.6 x {BEN_Q10}^(({T_BOTTOM:.2f} - {T_50:.2f})/10) = {BEN_RM}.\n"
    f"    resting_metabolism: {BEN_RM}\n", 1)
assert f"resting_metabolism: {BEN_RM}\n" in text
text = text.replace("    # (Fonds 1985). Read as the annual mean of the at_50m layer.\n",
                    "    # (Fonds 1985). Read as the annual mean of the bottom layer (223.26).\n", 1)

# seals spawn on both predator groups
text = text.replace(
    "        - name: predatory_fish\n          weight: 1.0\n          transform: linear\n    # 1.5 % of biomass",
    "        - name: predatory_fish\n          weight: 1.0\n          transform: linear\n"
    "        - name: whiting\n          weight: 1.0\n          transform: linear\n    # 1.5 % of biomass", 1)
assert "        - name: whiting\n" in text

# ---- pairs ----
HDR_PL = (f"  # Section 223.26 (split G1): a per tonne of the WHOLE group (the engine has\n"
          f"  # no phi since 221; layer convention), h = E_prey / the group's ceiling\n"
          f"  # (Bjornsson-anchored: whiting {CEIL['whiting']}, water column {CEIL['predatory_fish']} MJ/t/tick).\n"
          f"  # M2 = Q x diet share / prey B, scaled so the two groups add to the\n"
          f"  # pooled pair. Type II (Rindorf et al. 2006). {NOTE}.\n")
text = replace(text, "predatory_fish_preys_on_planktivorous_fish",
               HDR_PL
               + "  # Water column on planktivores: diet share 0.52 (saithe on Norway pout\n"
                 "  # and blue whiting, mackerel, hake on herring); a set by mackerel 0.49,\n"
                 "  # saithe 0.33, horse mackerel 0.12. Night/day 0.39 derived.\n"
               + pair("predatory_fish", "planktivorous_fish", floor=0.25, fr=2)
               + "  # Whiting on planktivores (adults on sprat, Norway pout, herring): diet\n"
                 "  # share 0.28. Night/day 0.33 derived.\n"
               + pair("whiting", "planktivorous_fish", floor=0.25, fr=2))

text = replace(text, "benthivorous_fish_preys_on_planktivorous_fish",
               entry(text, "benthivorous_fish_preys_on_planktivorous_fish").replace(
                   "    max_intake_rate: 0.019\n",
                   "    # Section 223.26: per WHOLE benthivore tonne (the engine has no phi\n"
                   "    # since 221): 0.019 per capable tonne x the gape share 0.32 = 0.0062.\n"
                   f"    # {NOTE} section 5.\n"
                   f"    max_intake_rate: {A[('benthivorous_fish', 'planktivorous_fish')]}\n"), keep_comments=True)

text = replace(text, "seals_preys_on_predatory_fish",
               "  # Section 209: 0.15 of the seal diet; split by member (223.26, J: whiting\n"
               "  # 0.44 of the seals' predator share). h = E_prey / 212.5; a 0.05 and\n"
               "  # the visibility floors are judgements (whiting on the bed 0.85).\n"
               + pair("seals", "predatory_fish", assim=0.85, floor=0.75)
               + pair("seals", "whiting", assim=0.85, floor=0.85))

text = after(text, "benthic_infauna_preys_on_predatory_fish",
             renamed(entry(text, "benthic_infauna_preys_on_predatory_fish"),
                     "benthic_infauna_preys_on_predatory_fish", "benthic_infauna_preys_on_whiting"))
text = after(text, "predatory_fish_preys_on_benthic_infauna",
             off("whiting", "benthic_infauna", comment=(
                 "  # Whiting's infauna is 7.3 % of its diet (M2 0.0015): unchecked (221).\n")))
text = replace(text, "predatory_fish_preys_on_mobile_epifauna",
               off("predatory_fish", "mobile_epifauna", m2=0.007, comment=(
                   "  # Section 223.26: the water-column group's epifauna (horse mackerel,\n"
                   "  # mackerel, saithe traces; 1.2 % of its diet, M2 0.007) is UNCHECKED -\n"
                   "  # the point of the split: 63 % of the predator biomass lost a bed food it\n"
                   "  # does not use in the field. The flow becomes unresolved predation.\n"))
               + "  # E1 exposed taxa: whiting on northern shrimp, small shrimps, mysids and\n"
                 "  # small crabs: diet share 0.137, 0.256 t/km2/yr (91 % of the pooled\n"
                 "  # flow), M2 0.069 on B* 3.78; accessible 0.29 (223.15); a = a_enc 0.69\n"
                 "  # per edible tonne x e 0.39 (whiting's crab window and Pandalus size;\n"
                 "  # research_notes/\"Exponering av bottenbyten\" (Swedish letters)) = 0.27,\n"
                 f"  # per whole whiting tonne. No dark_ratio. {NOTE}.\n"
               + pair("whiting", "mobile_epifauna"))
text = after(text, "predatory_fish_preys_on_norway_lobster",
             off("whiting", "norway_lobster", comment=(
                 "  # Section 223.26: whiting <= 0.55 % Nephrops (North Sea 1991, Q3-Q4),\n"
                 "  # none in the Skagerrak: unchecked.\n")))
text = replace(text, "benthivorous_fish_preys_on_predatory_fish",
               off("benthivorous_fish", "predatory_fish", comment=(
                   "  # Section 223.26: cod's fish prey in the predator pool is whiting\n"
                   "  # (Skagerrak 3.7 of 3.7 g per 100 g fish prey, box 10.3 of 10.3): the\n"
                   "  # whole 223.10 pair moved to benthivorous_fish_preys_on_whiting.\n"))
               + "  # Section 223.9 flow (Q/B 4.46 x 1.53 x share 0.023 = 0.154 t/km2/yr), all on\n"
                 "  # whiting since 223.26: M2 = 0.154 / 0.446 = 0.345 /yr (user's choice over\n"
                 "  # the SMS 0.05). h = 4800 / 26.18 (the benthivores' bottom-water ceiling).\n"
                 "  # a per WHOLE benthivore tonne: 0.026 per capable tonne (220: large cod,\n"
                 "  # starry ray, gurnard) x the gape share 0.10 = 0.0026 (no prey-side f\n"
                 f"  # carries the gape). Night/day 0.10 at 0 deg. {NOTE}.\n"
               + pair("benthivorous_fish", "whiting", fr=2))
text = replace(text, "predatory_fish_preys_on_benthivorous_fish",
               HDR_PL
               + "  # Water column (saithe 0.77 of a, spurdog, anglerfish) on juvenile\n"
                 "  # haddock, cod and small flatfish: accessible 0.24 (their gape, 19-45\n"
                 "  # cm + 0-group). Night/day 0.48.\n"
               + pair("predatory_fish", "benthivorous_fish", fr=2)
               + "  # Whiting on 0-group and small benthivores: accessible 0.05 (whiting\n"
                 "  # gape ~12 cm + 0-group, J). Night/day 0.64.\n"
               + pair("whiting", "benthivorous_fish", fr=2))
text = after(text, "predatory_fish_preys_on_predatory_fish",
             "  # Section 223.26: saithe 0.066, hake 0.044, spurdog 0.030, anglerfish and ling\n"
             "  # 0.017 /yr on whiting (SMS 2023 saithe on whiting 1+ 0.085 alone): a\n"
             "  # learnable cross-group pair, booked (it was a within-group transfer).\n"
             "  # accessible 0.47 (whiting below the members' gape). Night/day 0.48.\n"
             + pair("predatory_fish", "whiting", fr=2)
             + off("whiting", "predatory_fish", comment=(
                 "  # Traces of hake / horse mackerel in whiting (M2 0.019): unchecked.\n"))
             + off("whiting", "whiting", comment="  # Cannibalism stays out of the budget (section 160).\n"))
for prey in ("phytoplankton", "porpoises", "seabirds", "seals"):
    text = after(text, f"predatory_fish_preys_on_{prey}", off("whiting", prey))
text = replace(text, "predatory_fish_preys_on_zooplankton",
               "  # Section 223.26: krill and copepods; a per whole tonne (layer) on the\n"
               "  # A17-calibrated pool. Water column: mackerel 0.49, saithe 0.28 of a;\n"
               "  # night/day 0.30. Open (209/220): the M2 assumes krill.\n"
               + pair("predatory_fish", "zooplankton")
               + "  # Whiting on krill and mysid-sized zooplankton: encounter 0.0018 x the\n"
                 "  # Staniland 1995 capture factor on shrimp 1.33 (as the pooled pair, J)\n"
                 "  # = 0.0023. Night/day 0.78.\n"
               + pair("whiting", "zooplankton"))
for pred in ("planktivorous_fish", "phytoplankton", "seabirds", "zooplankton"):
    k = f"{pred}_preys_on_predatory_fish"
    text = after(text, k, renamed(entry(text, k), k, f"{pred}_preys_on_whiting"))
text = replace(text, "porpoises_preys_on_predatory_fish",
               "  # Section 209: 0.20 of the porpoise diet; split by member (223.26, J:\n"
               "  # whiting 0.64 of it). a 0.2 and the visibility floors are judgements.\n"
               + pair("porpoises", "predatory_fish", assim=0.84, floor=0.9)
               + pair("porpoises", "whiting", assim=0.84, floor=0.75))

# ---- impacts ----
seg = entry(text, "predatory_fish_impacted_by_bottom_trawling")
text = replace(text, "predatory_fish_impacted_by_bottom_trawling",
               f"  # Section 223.26: water column F {wc['F_bot']:.3f} /yr (saithe 0.30, hake 0.27, ling 0.17,\n"
               "  # anglerfish ~0.08, spurdog 0.008; gillnet, longline and seine on this map).\n"
               + seg.replace("0.00368125719", f"{BF100['predatory_fish']:.11g}")
               + f"  # Section 223.26: whiting F {bed['F_bot']:.3f} /yr.\n"
               + renamed(seg, "predatory_fish_impacted_by_bottom_trawling", "whiting_impacted_by_bottom_trawling")
               .replace("0.00368125719", f"{BF100['whiting']:.11g}"))
seg = entry(text, "predatory_fish_impacted_by_pelagic_trawling")
text = replace(text, "predatory_fish_impacted_by_pelagic_trawling",
               f"  # Section 223.26: water column F {wc['F_pel']:.3f} /yr (mackerel 0.21, horse mackerel\n"
               "  # 0.035 at their shares).\n"
               + seg.replace("0.00548776672", f"{PF10['predatory_fish']:.11g}")
               + f"  # Section 223.26: whiting F {bed['F_pel']:.3f} /yr (industrial bycatch).\n"
               + renamed(seg, "predatory_fish_impacted_by_pelagic_trawling", "whiting_impacted_by_pelagic_trawling")
               .replace("0.00548776672", f"{PF10['whiting']:.11g}"))
for imp in ("rotor", "windfarm_noise"):
    k = f"predatory_fish_impacted_by_{imp}"
    text = after(text, k, renamed(entry(text, k), k, f"whiting_impacted_by_{imp}"))

# ---- observes ----
OBS_WH = {"benthic_infauna": False, "mobile_epifauna": True, "norway_lobster": False,
          "benthivorous_fish": True, "predatory_fish": True, "whiting": True, "planktivorous_fish": True,
          "phytoplankton": False, "porpoises": True, "seabirds": False, "seals": True, "zooplankton": True}
text = replace(text, "predatory_fish_observes_mobile_epifauna",
               "  # Section 223.26: the water-column group has no epifauna pair any more.\n"
               "  predatory_fish_observes_mobile_epifauna:\n    observes: false\n")
for prey, v in OBS_WH.items():
    if prey in ("predatory_fish", "whiting"):
        continue
    k = f"predatory_fish_observes_{prey}"
    text = after(text, k, f"  whiting_observes_{prey}:\n    observes: {str(v).lower()}\n")
text = after(text, "predatory_fish_observes_predatory_fish",
             "  predatory_fish_observes_whiting:\n    observes: true\n"
             "  whiting_observes_predatory_fish:\n    observes: true\n"
             "  whiting_observes_whiting:\n    observes: true\n")
for obs in ("benthivorous_fish", "planktivorous_fish", "porpoises", "seabirds", "seals", "zooplankton"):
    k = f"{obs}_observes_predatory_fish"
    text = after(text, k, renamed(entry(text, k), k, f"{obs}_observes_whiting"))

# ---------------------------------------------------------------------------------------------------
# growth rates (derived) and the epifauna K table
# ---------------------------------------------------------------------------------------------------
tmp = os.path.join(out, "_tmp_lib.yaml")
open(tmp, "w", encoding="utf-8").write(text.replace("GROWTH_whiting", "0.0").replace("GROWTH_predatory_fish", "0.0"))
new = load_config(tmp)
os.remove(tmp)


def active_of(p):
    return [g["group_id"] for key in ("decision_makers", "non_decision_makers")
            for g in (p.get(key) or []) if isinstance(g, dict) and not g.get("muted")]


proj_new = copy.deepcopy(project)
dms = proj_new["decision_makers"]
i = [g["group_id"] for g in dms].index("predatory_fish")
dms[i].update(initial_biomass_min=1360, initial_biomass_max=5430, inference_initial_biomass=2715)
dms.insert(i + 1, {"group_id": "whiting", "initial_biomass_min": 800, "initial_biomass_max": 3210,
                   "inference_initial_biomass": 1605})
proj_new["simulation_settings"]["temperature"]["group_layers"]["whiting"] = "bottom"
act_old, act_new = active_of(project), active_of(proj_new)


def budgets(lib, active):
    res = {}
    for fid in active:
        b = gb.growth_budget(fid, lib["species_definitions"], lib["interaction_definitions"],
                             active_predators=active)
        if b is not None:
            res[fid] = dict(g=b.growth_rate, s_star=b.reference_fill, gross=b.gross_annual,
                            m2=dict(b.predation), best=b.best_prey)
    return res


b_old, b_new = budgets(old, act_old), budgets(new, act_new)
for fid, b in b_new.items():
    new["species_definitions"][fid]["growth_rate"] = b["g"]

# epifauna K: scale the table so the equilibrium start total is kept
depth = habitat.load_depth_map(project, project_path, GRID)
maps = project_baseline_maps(project, project_path, GRID)


def eq_total(lib, active, fid):
    spec = lib["species_definitions"][fid]
    K = habitat.capacity_map(habitat.parse_capacity_table(spec), depth)
    loss = habitat.annual_impact_loss(fid, lib, maps, GRID)
    m2 = habitat.predation_m2(fid, lib, active)
    return float(habitat.equilibrium_map(K, loss, m2, spec["growth_rate"] * TPY).sum())


old_lib_g = copy.deepcopy(old)
tot_old = eq_total(old_lib_g, act_old, "mobile_epifauna")
tot_new_unscaled = eq_total(new, act_new, "mobile_epifauna")
scale = tot_old / tot_new_unscaled
rows = new["species_definitions"]["mobile_epifauna"]["capacity_depth_table"]
caps_new = [round(r["capacity"] * scale, 2) for r in rows]
for r, c in zip(rows, caps_new):
    r["capacity"] = c
tot_new = eq_total(new, act_new, "mobile_epifauna")
K_mean = float(habitat.capacity_map(habitat.parse_capacity_table(new["species_definitions"]["mobile_epifauna"]),
                                    depth).mean())

# write growth rates and the K table into the text
for fid, b in b_new.items():
    g = b["g"]
    if fid in ("whiting", "predatory_fish"):
        text = text.replace(f"growth_rate: GROWTH_{fid}", f"growth_rate: {g:.11g}", 1)
        continue
    g_old = float(old["species_definitions"][fid].get("growth_rate", float("nan")))
    if not math.isclose(g, g_old, rel_tol=1e-6):
        text = set_field(text, fid, "growth_rate", f"{g:.11g}")
seg = entry(text, "mobile_epifauna")
cap_lines = re.findall(r"^      - depth_m: ([\d.]+)\n        capacity: ([\d.]+)$", seg, re.M)
assert len(cap_lines) == len(caps_new), cap_lines
seg2 = seg
for (d, c_old), c in zip(cap_lines, caps_new):
    seg2 = seg2.replace(f"      - depth_m: {d}\n        capacity: {c_old}\n",
                        f"      - depth_m: {d}\n        capacity: {c:.2f}\n", 1)
seg2 = re.sub(r"^(    max_carrying_capacity:)\s*.*$", rf"\g<1> {K_mean:.1f}", seg2, count=1, flags=re.M)
seg2 = seg2.replace("    capacity_depth_table:\n",
                    f"    # Section 223.26: rows x{scale:.4f} so the equilibrium start total is kept\n"
                    "    # (the water-column predators' M2 0.007 is unchecked since the split).\n"
                    "    capacity_depth_table:\n", 1)
text = text.replace(seg, seg2, 1)

lib_out = os.path.join(out, "fg_library_split.yaml")
open(lib_out, "w", encoding="utf-8-sig").write(text)

# ---- manifest copy (text edits) ----
mtext = open(project_path, encoding="utf-8-sig").read()
mtext = mtext.replace("      benthivorous_fish: bottom\n      predatory_fish: at_50m\n",
                      "      benthivorous_fish: bottom\n      predatory_fish: at_50m\n"
                      "      # Section 223.26: whiting forages on the bed (J, as the benthivores).\n"
                      "      whiting: bottom\n", 1)
mtext = mtext.replace(
    "  - group_id: predatory_fish\n    initial_biomass_min: 2150\n    initial_biomass_max: 8600\n"
    "    inference_initial_biomass: 4300\n",
    "  # Section 223.26 (predator split G1): the 1.20 t/km2 predator level\n"
    "  # split by the long-run biomass shares: water column (predatory_fish)\n"
    "  # 0.754, whiting 0.446 t/km2; ranges x0.5-2.\n"
    "  - group_id: predatory_fish\n    initial_biomass_min: 1360\n    initial_biomass_max: 5430\n"
    "    inference_initial_biomass: 2715\n"
    "  - group_id: whiting\n    initial_biomass_min: 800\n    initial_biomass_max: 3210\n"
    "    inference_initial_biomass: 1605\n", 1)
mtext = mtext.replace(
    "        - name: benthivorous_fish\n          weight: 1.0\n          transform: linear\n"
    "        - name: predatory_fish\n          weight: 1.0\n          transform: linear\ninference:",
    "        - name: benthivorous_fish\n          weight: 1.0\n          transform: linear\n"
    "        # Section 223.26: the bed-feeding fish; the water-column predators\n"
    "        # no longer follow the demersal pattern.\n"
    "        - name: whiting\n          weight: 1.0\n          transform: linear\ninference:", 1)
for s in ("whiting: bottom", "group_id: whiting", "        - name: whiting\n"):
    assert s in mtext, s
proj_out = os.path.join(out, "mareld2_split.yaml")
open(proj_out, "w", encoding="utf-8-sig").write(mtext)

# ---------------------------------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------------------------------
L = load_config(lib_out)
P = load_config(proj_out)
assert active_of(P) == act_new, (active_of(P), act_new)
SD, ID = L["species_definitions"], L["interaction_definitions"]
bad = []
for fid in act_new:
    g = gb.derived_growth_rate(fid, SD, ID, active_predators=act_new)
    if g is None:
        continue
    if not math.isclose(g, float(SD[fid]["growth_rate"]), rel_tol=1e-6):
        bad.append((fid, g, SD[fid]["growth_rate"]))
assert not bad, bad


def m2_sums(lib, active):
    return {fid: sum(gb.active_predation(fid, lib["species_definitions"], lib["interaction_definitions"],
                                         active_predators=active).values()) for fid in active}


m_old, m_new = m2_sums(old, act_old), m2_sums(L, act_new)
cfg = load_project_config(project_path, library_path=lib_out, project_config=P)

report = dict(
    T_bottom=T_BOTTOM, T_50=T_50, ceilings=CEIL, RM=RM, feeding_cost=FC, starve=STARVE, M1=M1,
    ben_RM=BEN_RM, h={f"{k[0]}->{k[1]}": v for k, v in H.items()},
    E={k: E[k] for k in ("whiting", "predatory_fish", "planktivorous_fish", "zooplankton",
                         "mobile_epifauna", "benthivorous_fish")},
    growth={fid: dict(old=b_old.get(fid, {}).get("g"), new=b["g"], s_star_old=b_old.get(fid, {}).get("s_star"),
                      s_star_new=b["s_star"], best=b["best"]) for fid, b in b_new.items()},
    m2_sums={fid: dict(old=m_old.get(fid), new=m_new[fid]) for fid in act_new},
    epifauna=dict(scale=scale, caps=caps_new, total_old=tot_old, total_new=tot_new, K_mean=K_mean),
    loader_built=sorted(cfg[1].keys()) if isinstance(cfg, tuple) and isinstance(cfg[1], dict) else str(type(cfg)))
json.dump(report, open(os.path.join(out, "build_split_223.json"), "w"), indent=1, default=float)
print(json.dumps(report, indent=1, default=float))
