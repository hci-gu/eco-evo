"""Mackerel split (G3) -> a library COPY and a manifest COPY (the live files are never touched here).

Usage (from the repository root):
  python3 -I "research_notes/Rovfiskens uppdelning/scripts/build_split_mack.py" \
      fgconfig/fg_library.yaml mareld2.yaml "research_notes/Rovfiskens uppdelning/tables" \
      "research_notes/Rovfiskens uppdelning/data"

Inputs: <tables>/split_fields_mack.json (split_fields_mack.py), split_encounter_mack.json and
split_encounter_mack_rest_on_mackerel.json (split_encounter_mack.py), and the stage access of
research_notes/"Makrillens bytesstorlek" (Swedish letters) summary.md section 6.

User decisions (2026-10-30, resume 228): a new decision maker `mackerel` (mackerel + horse mackerel)
out of the water column `predatory_fish` (which keeps its id: saithe, spurdog, hake, anglerfish, ling,
velvet belly, pollack); stage access on the planktivores at the annual mean [0.079, 0.004]; no
regional K; temperature layer mean_0_50m; reserve-funded growth (no income-funded fat cycle yet).
Taken as in 223.26 (J): the Bjornsson x1.30 anchor on both ceilings (the mackerel GER Cmax rests on
the same cod stomach-fill calibration); seals and porpoises at today's per-tonne M2 on both groups;
the rest water column -> mackerel group pair booked (a learnable cross-group pair, as wc -> whiting);
every DM that observes predatory_fish observes mackerel, and mackerel observes what the water column
observes.

What it does
1. Text-edits the live library (comments kept) into <out>/fg_library_mack.yaml: the predatory_fish
   block rewritten for the rest, a new mackerel block after whiting, every pair / observes / impact
   entry the water column has for both ids, the new cross pair, seal and porpoise pairs.
2. h of every predator pair = prey_energy / the group's ceiling (seals: / 212.5; porpoises 28.6, 209).
3. Derives every active group's growth_rate with lib/world/growth_budget.py and writes the changed ones.
4. Writes <out>/mareld2_mack.yaml (group_layers, decision_makers).
5. Verifies: both copies parse; stored growth_rate == derived for every active group; M2 sums on every
   prey vs the live library (only mackerel is new); load_project_config builds every FG.
"""
import copy
import json
import math
import os
import re
import sys

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
from lib.config.config_loader import load_config, load_project_config  # noqa: E402
from lib.world import growth_budget as gb                               # noqa: E402
from lib.world.energy_balance import prey_energy                        # noqa: E402

lib_path, project_path, tables, out = sys.argv[1:5]
# "_field": the field densities (research_notes/"Makrillens tathet", resume 228); "" the GOV shares
SUFFIX = sys.argv[5] if len(sys.argv) > 5 else ""
STAGE_JSON = os.path.join(ROOT, "research_notes", "Makrillens bytesstorlek", "tables",
                          f"mackerel_stage_access{SUFFIX}.json")
os.makedirs(out, exist_ok=True)
F = json.load(open(os.path.join(tables, f"split_fields_mack{SUFFIX}.json")))
ENC = json.load(open(os.path.join(tables, f"split_encounter_mack{SUFFIX}.json")))
RON = json.load(open(os.path.join(tables, f"split_encounter_mack_rest_on_mackerel{SUFFIX}.json")))
SA = json.load(open(STAGE_JSON))
mk, wc = F["mackerel"], F["predatory_fish"]
TPY = 1460.0
MB = mk["B"]                                    # t/km2 (GOV share, or the field density override)
T0 = int(round(MB * 3600))
SRC = "GOV share" if "B_gov" not in mk else "field level"                      # inference start, tonnes on the 60x60 grid
LEVEL = ("the long-run GOV share" if "B_gov" not in mk else
         'the field density, research_notes/"Makrillens tathet" (Swedish letters: a-umlaut)')
NOTE = 'research_notes/"Rovfiskens uppdelning" (Swedish letters)/params_split_mack.md'
old = load_config(lib_path)
project = load_config(project_path)

# ---- fields ----
CEIL = {"mackerel": round(mk["ceiling_bjornsson"], 2), "predatory_fish": round(wc["ceiling_bjornsson"], 2)}
RM = {"mackerel": round(mk["smr"], 2), "predatory_fish": round(wc["smr"], 2)}
FC = {k: round(1.5 + 0.17 * 0.5 * CEIL[k] / RM[k], 2) for k in CEIL}
LEAN = {"mackerel": round(mk["e_lean"], -1), "predatory_fish": 4180.0}
RES = {"mackerel": round(mk["res_max"], -1), "predatory_fish": 2000.0}
FILL = {"mackerel": round(mk["fill"], 2), "predatory_fish": 0.45}
STARVE = {k: round(1.5 * RM[k] / LEAN[k] / 0.15, 4) for k in CEIL}
M1 = {"mackerel": mk["M1"] / TPY, "predatory_fish": wc["M1"] / TPY}
R = {"mackerel": round(mk["r"], 3), "predatory_fish": round(wc["r"], 3)}
Q10 = {"mackerel": round(mk["q10"], 1), "predatory_fish": round(wc["q10"], 1)}

# ---- pairs: M2 / a / f / night-day ----
enc = lambda g, prey: ENC[g][prey]["layer"]
M2 = {("mackerel", "planktivorous_fish"): round(mk["M2"]["planktivorous_fish"], 4),
      ("mackerel", "zooplankton"): round(mk["M2"]["zooplankton"], 4),
      ("predatory_fish", "planktivorous_fish"): round(wc["M2"]["planktivorous_fish"], 4),
      ("predatory_fish", "zooplankton"): round(wc["M2"]["zooplankton"], 4),
      ("predatory_fish", "benthivorous_fish"): 0.0833,
      ("predatory_fish", "whiting"): 0.157,
      ("predatory_fish", "mackerel"): round(wc["M2"]["mackerel"], 3),
      ("seals", "mackerel"): 0.0098, ("porpoises", "mackerel"): 0.0063}
A = {("mackerel", "planktivorous_fish"): round(enc("G3_mackerel", "planktivorous_fish")["central"], 4),
     ("mackerel", "zooplankton"): round(enc("G3_mackerel", "zooplankton")["central"], 5),
     ("predatory_fish", "planktivorous_fish"): round(enc("G3_rest_wc", "planktivorous_fish")["central"], 5),
     ("predatory_fish", "zooplankton"): round(enc("G3_rest_wc", "zooplankton")["central"], 6),
     ("predatory_fish", "benthivorous_fish"): round(enc("G3_rest_wc", "benthivorous_fish")["central"], 4),
     ("predatory_fish", "whiting"): round(enc("G3_rest_wc", "whiting_prey")["central"], 4),
     ("predatory_fish", "mackerel"): round(RON["a"], 5)}
ND = {("mackerel", "planktivorous_fish"): enc("G3_mackerel", "planktivorous_fish")["night_day"],
      ("mackerel", "zooplankton"): enc("G3_mackerel", "zooplankton")["night_day"],
      ("predatory_fish", "planktivorous_fish"): enc("G3_rest_wc", "planktivorous_fish")["night_day"],
      ("predatory_fish", "zooplankton"): enc("G3_rest_wc", "zooplankton")["night_day"],
      ("predatory_fish", "benthivorous_fish"): enc("G3_rest_wc", "benthivorous_fish")["night_day"],
      ("predatory_fish", "whiting"): enc("G3_rest_wc", "whiting_prey")["night_day"],
      ("predatory_fish", "mackerel"): RON["night_day"]}
ND = {k: round(v, 2) for k, v in ND.items()}
STAGE = {("mackerel", "planktivorous_fish"): [round(x, 4) for x in SA["mackerel_planktivores"]],
         ("predatory_fish", "planktivorous_fish"): [0.69, 0.27],   # same, rest water column
         ("predatory_fish", "benthivorous_fish"): [0.43, 0.027]}   # unchanged (mackerel not a member)
F_ACC = {("predatory_fish", "whiting"): 0.47, ("predatory_fish", "mackerel"): round(RON["f"], 2)}

# ---- new species definitions in memory (prey energies) ----
new = copy.deepcopy(old)
S = new["species_definitions"]
S["mackerel"] = copy.deepcopy(S["predatory_fish"])
for g in ("mackerel", "predatory_fish"):
    S[g].update(max_energy_reserve=RES[g], energy_content=LEAN[g], reserve_reference_fill=FILL[g],
                prey_includes_reserve=True)
E = {fid: prey_energy(S[fid]) for fid in S}
H_SEAL_CEIL = 212.5
H = {k: round(E[k[1]] / CEIL[k[0]], 1) for k in M2 if k[0] in CEIL}
H[("seals", "mackerel")] = round(E["mackerel"] / H_SEAL_CEIL, 1)
H[("seals", "predatory_fish")] = round(E["predatory_fish"] / H_SEAL_CEIL, 1)
H[("porpoises", "mackerel")] = 28.6

# ---------------------------------------------------------------------------------------------------
# text helpers (as build_split_223.py)
# ---------------------------------------------------------------------------------------------------
text = open(lib_path, encoding="utf-8-sig").read()


def block(text, name):
    m = re.compile(rf"^  {re.escape(name)}:\s*$", re.M).search(text)
    if m is None:
        raise KeyError(name)
    nxt = re.compile(r"^  \S|^\S", re.M).search(text, m.end())
    a = m.start()
    lines_before = text[:a].split("\n")[:-1]
    i = len(lines_before)
    while i > 0 and lines_before[i - 1].startswith("  #"):
        i -= 1
    a0 = len("\n".join(lines_before[:i])) + (1 if i > 0 else 0)
    b = nxt.start() if nxt else len(text)
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
    s += f"    max_intake_rate: {A[k]}\n"
    if k in ND:
        s += f"    dark_ratio: {ND[k]}\n    light_threshold_deg: -5.0\n"
    if k in STAGE:
        s += f"    stage_access: [{STAGE[k][0]}, {STAGE[k][1]}]\n"
    return s + extra


def off(pred, prey):
    return f"  {pred}_preys_on_{prey}:\n    preys_on: false\n    assimilation_factor: 1.0\n    handling_time: 0.0\n"


# ---------------------------------------------------------------------------------------------------
# species blocks
# ---------------------------------------------------------------------------------------------------
wcw = wc["weights"]
SPAWN = ("    spawn:\n      mode: colony\n      n_colonies: 25\n      sigma_cells: 1.0\n      anchor: free\n"
         "      amplitude_mode: jitter\n      amplitude_min: 0.0\n      amplitude_max: 1.0\n")
WC_BLOCK = f"""  predatory_fish:
    display_name: "Predatory Fish"
    # Section 223.26 (predator split G1, user 2026-10-27): the WATER-COLUMN
    # predators; whiting, the bed feeder, is its own group `whiting`. Since
    # section 228 (user 2026-10-30) mackerel and horse mackerel are their own
    # group `mackerel` too: this group keeps the id and is the demersal and
    # bathypelagic fish eaters (saithe 0.80 of it). Display only (live viewer
    # hover): the members and their long-run (NS-IBTS 2001-25) biomass shares.
    # {NOTE}.
    member_species:
""" + "".join(f"      - {{name: {k}, share: {v:.3f}}}\n" for k, v in wcw.items()) + f"""    is_decision_maker: true
    # Section 209 / 223.26 / 228: fish that feed on fish and krill: saithe
    # 0.80, spurdog 0.10, hake 0.05, anglerfish, ling, velvet belly, pollack.
    # Saithe (Q3 on the trench slope) is a visitor; the closed pool charges
    # it stock-level r and F, as herring. Every value is the biomass-weighted
    # species value (weighted_fields convention, 209).
    # u 0.5: the engine convention (resume 187.4).
    maintenance_level: 0.5
    # Liver reserve: saithe, hake, spurdog ~2000, anglerfish 1500 -> 2000
    # (1500-2500).
    max_energy_reserve: {RES['predatory_fish']}
    # Measured SMR at the at_50m annual mean 8.37 C: saithe 3.5, spurdog 2.1,
    # hake 3.5, anglerfish 1.2 -> {RM['predatory_fish']} ({wc['smr_lo']:.1f}-{wc['smr_hi']:.1f}).
    resting_metabolism: {RM['predatory_fish']}
    # Cod SMR 2.2-2.6, ICES gastric evacuation exponent 2.25, spurdog 2.5 ->
    # {Q10['predatory_fish']}; annual mean of the at_50m layer.
    metabolism_q10: {Q10['predatory_fish']}
    metabolism_t_ref: annual_mean
    # Every member but anglerfish cruises 6-15 cells/tick (saithe Uopt 1.4
    # BL/s); residence is behaviour, not a speed limit.
    movement_speed: 1.0
    # Routine 1.5 x SMR + SDA 0.17 of half the ceiling {CEIL['predatory_fish']} (GER Cmax
    # {wc['cmax']:.1f} kJ/kg/d at 8.37 C x 1.30 Bjornsson anchoring, 223.26) = {FC['predatory_fish']} x RM;
    # cruising at Uopt 2-2.5 x SMR = 2.0 x RM.
    feeding_cost: {FC['predatory_fish']}
    movement_cost: 2.0
    # Lean tissue as eaten (saithe 4.0, hake 3.7, spurdog 6.0 kJ/g) = 4180;
    # the liver is eaten with the fish: at the reference fill 0.45 (mean
    # reserve 900 / 2000) the static quality is 4180 + 0.45 x 2000 = 5080 MJ/t.
    energy_content: {LEAN['predatory_fish']}
    prey_includes_reserve: true
    reserve_reference_fill: {FILL['predatory_fish']}
    min_split_biomass: 0.5
    # growth_rate is DERIVED by lib/world/growth_budget.py and recomputed
    # by the loader for every run; the number stored here is what the
    # loader computes for mareld2.yaml (seabirds muted) with
    # --mortality on; with --mortality off the loader leaves M1 out. Edit
    # r_max, natural_mortality or the predation matrix, not this value.
    # r_max: today's productivity (Euler-Lotka r_now): saithe 0.44, hake
    # 0.26, spurdog 0.07, anglerfish 0.47, ling 0.55, pollack 0.63 -> {R['predatory_fish']}
    # ({wc['r_lo']:.2f}-{wc['r_hi']:.2f}).
    r_max: {R['predatory_fish']}
    # No regional K since section 223.37: a loss cannot stand for the
    # exchange of a window group that needs a net import (field closure
    # P 0.36 vs L 0.62 /yr), and it counted the box's density dependence
    # twice (research_notes/"Bestandsreglering" (Swedish letters: a-ring)
    # summary.md). Was K = B*/(1 - F/r) = 2.14 t/km2.
    growth_rate: GROWTH_predatory_fish
    # M1 = M0 + predation by groups NOT in the model: saithe 0.29, spurdog
    # 0.10, hake 0.45, anglerfish 0.25 -> {wc['M1']:.3f} /yr ({wc['M1_lo']:.2f}-{wc['M1_hi']:.2f}).
    # Within-group predation is a transfer inside the pool and left out.
    natural_mortality: {M1['predatory_fish']:.6g}
    max_intake_rate: 0.04
{SPAWN}    # 1.5 x SMR / lean energy / 0.15 (the section 209 construction).
    starve_rate: {STARVE['predatory_fish']}
    # Judgement: midwater saithe schools hide badly.
    visibility_floor: 0.7
    # No field evidence of interference for saithe (Rindorf et al. 2006:
    # predator density did not affect intake).
    interference: 0.1
    extinction_threshold_factor: 0.5
"""
mkw = mk["weights"]
MK_BLOCK = f"""  mackerel:
    display_name: "Mackerel"
    # Section 228 (user 2026-10-30): mackerel and horse mackerel out of the
    # water column ({MB:.3f} t/km2, {T0} t: {LEVEL}). A
    # zooplanktivore that takes the small end of the planktivores'
    # juvenile compartment (0-group herring, pout, sprat; SMS Q3 diet other
    # food 0.81, herring 0.066). Summer visitors (IBTS Q3 share 0.95); the
    # closed pool charges stock-level r and F. Display only (live viewer
    # hover): the members and their long-run biomass shares.
    # {NOTE}.
    member_species:
""" + "".join(f"      - {{name: {k}, share: {v:.3f}}}\n" for k, v in mkw.items()) + f"""    is_decision_maker: true
    # u 0.5: the engine convention (resume 187.4).
    maintenance_level: 0.5
    # Fat reserve: mackerel ~6000, horse mackerel ~3800 MJ/t -> {RES['mackerel']:.0f}.
    # Reserve-funded growth (user 228); mackerel's summer fattening has the
    # shape that made herring grow on its fat in winter (184): the income-
    # funded law is the open next step.
    max_energy_reserve: {RES['mackerel']}
    # Measured SMR 3.4 (both members) at 8.37 C re-read at the mean_0_50m
    # annual mean 9.31 C with Q10 {mk['q10']:.2f}: {RM['mackerel']} ({mk['smr_lo']:.1f}-{mk['smr_hi']:.1f}).
    resting_metabolism: {RM['mackerel']}
    # ICES GER temperature term (mackerel 0.080 /C = Q10 2.23), mackerel
    # respirometry 2.0, horse mackerel 2.2 -> {Q10['mackerel']}; annual mean of the
    # mean_0_50m layer (user 228: its prey's layer).
    metabolism_q10: {Q10['mackerel']}
    metabolism_t_ref: annual_mean
    # Schools cruise 0.44-0.72 m/s net; not binding.
    movement_speed: 1.0
    # Routine 1.5 x SMR + SDA 0.17 of half the ceiling {CEIL['mackerel']} (GER Cmax
    # {mk['cmax']:.0f} kJ/kg/d at 9.31 C x 1.30 Bjornsson anchoring: the GER fill is
    # the cod calibration, as for every predator ceiling since 223.26;
    # summer fattening needs >= ~180) = {FC['mackerel']} x RM; cruising 2 x SMR.
    feeding_cost: {FC['mackerel']}
    movement_cost: 2.0
    # Lean tissue as eaten (mackerel 4.7, horse mackerel 4.2 kJ/g) = {LEAN['mackerel']:.0f}; the
    # fat is eaten with the fish: at the reference fill {FILL['mackerel']} (mean reserve
    # {mk['res_mean']:.0f} / {RES['mackerel']:.0f}) the static quality is {E['mackerel']:.0f} MJ/t.
    energy_content: {LEAN['mackerel']}
    prey_includes_reserve: true
    reserve_reference_fill: {FILL['mackerel']}
    min_split_biomass: 0.5
    # growth_rate is DERIVED by lib/world/growth_budget.py (see
    # predatory_fish). r_max: Euler-Lotka r_now mackerel 0.25, horse
    # mackerel 0.33 -> {R['mackerel']} ({mk['r_lo']:.2f}-{mk['r_hi']:.2f}).
    r_max: {R['mackerel']}
    # No regional K (user 228, as the water column since 223.37): the most
    # mobile window group in the box; its density is set outside it. F
    # {mk['F_bot'] + mk['F_pel']:.4f} (the LOCAL 3.a.20 F, 228.1) against r {R['mackerel']} leaves
    # {R['mackerel'] - mk['F_bot'] - mk['F_pel']:.2f} /yr; the other losses are the booked predation (water
    # column {M2[('predatory_fish', 'mackerel')]}, seals, porpoises) and M1.
    growth_rate: GROWTH_mackerel
    # M1 = M0 + predation by groups NOT in the model (tuna, sharks, birds,
    # cetaceans other than porpoises): mackerel 0.15, horse mackerel 0.30
    # -> {mk['M1']:.3f} /yr ({mk['M1_lo']:.2f}-{mk['M1_hi']:.2f}).
    natural_mortality: {M1['mackerel']:.6g}
    max_intake_rate: 0.04
{SPAWN}    # 1.5 x SMR / lean energy / 0.15 (the section 209 construction).
    starve_rate: {STARVE['mackerel']}
    # Judgement: midwater schools hide badly (as the water column).
    visibility_floor: 0.7
    # No field evidence of interference for mackerel (as the water column).
    interference: 0.1
    extinction_threshold_factor: 0.5
"""
text = replace(text, "predatory_fish", WC_BLOCK)
text = after(text, "whiting", MK_BLOCK)

# ---------------------------------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------------------------------
HDR = (f"  # Section 228 (mackerel split): a per tonne of the WHOLE group from the\n"
       f"  # section 220 encounter (layer form) over the group's members, h = prey\n"
       f"  # quality / the group's ceiling, night/day from the same encounter. {NOTE}.\n")
# water column -> planktivores
STALE_PLK = ("  # Water column on planktivores: diet share 0.52 (saithe on Norway pout\n"
             "  # and blue whiting, mackerel, hake on herring); a set by mackerel 0.49,\n"
             "  # saithe 0.33, horse mackerel 0.12. Night/day 0.39 derived.\n")
assert STALE_PLK in text
text = text.replace(STALE_PLK, "", 1)
CEIL_LINE = "  # (Bjornsson-anchored: whiting 31.34, water column 32.71 MJ/t/tick).\n"
assert CEIL_LINE in text
text = text.replace(CEIL_LINE, "  # (Bjornsson-anchored: whiting 31.34, water column 32.71 MJ/t/tick; since\n"
                    f"  # 228 the water column {CEIL['predatory_fish']}, mackerel {CEIL['mackerel']}).\n", 1)
text = replace(text, "predatory_fish_preys_on_planktivorous_fish", keep_comments=True, new_text=
               HDR + "  # Rest water column: M2 = SMS partial M2 scaled to the box (223.37; saithe\n"
               "  # 0.209, hake 0.037); stage access from saithe's, hake's and spurdog's\n"
               "  # windows (Makrillens bytesstorlek (Swedish letters) summary.md section 6).\n"
               + pair("predatory_fish", "planktivorous_fish", floor=0.25, fr=2)
               + "  # Mackerel group: SMS partial M2 horse mackerel 0.008 + mackerel 0.002 at\n"
               "  # the GOV biomass, scaled to the members' field biomass (223.37 box scaling;\n"
               "  # Makrillens tathet (Swedish letters: a-umlaut)). Stage access from the gape\n"
               "  # (p95 prey/predator 0.33 / 0.31, ~10 cm), annual mean over the quarters\n"
               "  # (user 228; Q3 alone ~0.12): the small end of the juveniles only.\n"
               + pair("mackerel", "planktivorous_fish", floor=0.25, fr=2))
# zooplankton
text = replace(text, "predatory_fish_preys_on_zooplankton",
               "  # Section 223.26: krill and copepods; a per whole tonne (layer) on the\n"
               "  # A17-calibrated pool. Open (209/220): the rest's M2 assumes krill.\n"
               "  # Section 228: the rest keeps its share of today's 0.0756 by the members'\n"
               "  # zooplankton consumption (saithe's krill); the mackerel group's M2 is its\n"
               "  # members' consumption at their field biomass, same calibration.\n"
               + pair("predatory_fish", "zooplankton")
               + pair("mackerel", "zooplankton"))
# benthivores and whiting: rest only (mackerel's capability score < 0.1 -> layer a 0)
text = replace(text, "predatory_fish_preys_on_benthivorous_fish",
               "  # Section 228: unchanged M2 and stage access (mackerel and horse mackerel are\n"
               "  # not members of this pair: capability 0.01 / 0.05); a per rest tonne.\n"
               + pair("predatory_fish", "benthivorous_fish", fr=2), keep_comments=True)
text = replace(text, "predatory_fish_preys_on_whiting",
               "  # Section 228: M2 unchanged (all of it saithe, hake, spurdog, anglerfish,\n"
               "  # ling); a per rest tonne.\n"
               + pair("predatory_fish", "whiting", fr=2), keep_comments=True)
# the new cross pair and cannibalism rows
text = after(text, "predatory_fish_preys_on_whiting",
             "  # Section 228: spurdog (0.37 of the a), hake (0.41), velvet belly, ling and\n"
             "  # anglerfish on mackerel and horse mackerel (split_fields_223 diets: spurdog\n"
             "  # W 0.22 x 0.50, hake W 0.21 x 0.10): a learnable cross-group pair, booked\n"
             "  # (it was a within-group transfer). Encounter on pelagic fish prey (the\n"
             "  # planktivore geometry, J) over those members per rest tonne;\n"
             "  # accessible_fraction = the mackerel group's biomass inside the members'\n"
             "  # prey-size windows (spurdog 0.34, hake 0.50, anglerfish 0.57 x length).\n"
             + pair("predatory_fish", "mackerel", fr=2)
             + "  # Cannibalism and mackerel on the other fish stay out of the budget.\n"
             + off("mackerel", "predatory_fish") + off("mackerel", "mackerel") + off("mackerel", "whiting")
             + off("mackerel", "benthivorous_fish") + off("whiting", "mackerel")
             + off("benthivorous_fish", "mackerel"))
# every other pair the water column has as predator / prey: an inert row for mackerel
done = {"planktivorous_fish", "zooplankton", "benthivorous_fish", "whiting", "predatory_fish", "mackerel"}
for k in list(old["interaction_definitions"]):
    m = re.fullmatch(r"predatory_fish_preys_on_(\w+)", k)
    if m and m.group(1) not in done:
        text = after(text, k, off("mackerel", m.group(1)))
    m = re.fullmatch(r"(\w+)_preys_on_predatory_fish", k)
    if m and m.group(1) not in ("predatory_fish", "whiting", "benthivorous_fish", "seals", "porpoises", "mackerel"):
        text = after(text, k, off(m.group(1), "mackerel"))
# seals and porpoises: today's per-tonne M2 on both groups
text = set_field(text, "seals_preys_on_predatory_fish", "handling_time", H[("seals", "predatory_fish")])
text = after(text, "seals_preys_on_predatory_fish",
             "  # Section 228: today's water-column M2 per tonne on the mackerel group (no\n"
             f"  # diet evidence to split it; J); h = {E['mackerel']:.0f} / 212.5.\n"
             f"  seals_preys_on_mackerel:\n    preys_on: true\n    predation_mortality: {M2[('seals', 'mackerel')]}\n"
             f"    assimilation_factor: 0.85\n    handling_time: {H[('seals', 'mackerel')]}\n"
             "    visibility_floor: 0.75\n    max_intake_rate: 0.05\n")
text = after(text, "porpoises_preys_on_predatory_fish",
             "  # Section 228: today's water-column M2 per tonne on the mackerel group (J).\n"
             f"  porpoises_preys_on_mackerel:\n    preys_on: true\n    predation_mortality: {M2[('porpoises', 'mackerel')]}\n"
             "    assimilation_factor: 0.84\n    handling_time: 28.6\n    visibility_floor: 0.9\n"
             "    max_intake_rate: 0.2\n")

# ---- impacts ----
BF_B = {"predatory_fish": wc["bf_bottom_100"], "mackerel": mk["bf_bottom_100"]}
BF_P = {"mackerel": mk["bf_pelagic_10"]}
seg = entry(text, "predatory_fish_impacted_by_bottom_trawling")
text = replace(text, "predatory_fish_impacted_by_bottom_trawling",
               f"  # Section 228: water column F {wc['F_bot']:.3f} /yr (saithe 0.30, hake 0.27, ling 0.17,\n"
               "  # anglerfish ~0.08, spurdog 0.008; gillnet, longline and seine on this map).\n"
               + seg.replace("0.0048361803927", f"{BF_B['predatory_fish']:.11g}")
               + f"  # Section 228.1: mackerel group F {mk['F_bot']:.4f} /yr: the LOCAL 3.a.20 F of mackerel\n"
               "  # (catch 343 t/yr / survey biomass 25 945 t = 0.0132) on its static gears (gillnet,\n"
               "  # hooks) and bottom trawls; horse mackerel 0.003. Not the stock F (0.2).\n"
               + renamed(seg, "predatory_fish_impacted_by_bottom_trawling", "mackerel_impacted_by_bottom_trawling")
               .replace("0.0048361803927", f"{BF_B['mackerel']:.11g}"))
seg = entry(text, "predatory_fish_impacted_by_pelagic_trawling")
text = replace(text, "predatory_fish_impacted_by_pelagic_trawling",
               "  # Section 228: no pelagic F on the water column (all of today's 0.029 was\n"
               "  # mackerel's and horse mackerel's).\n"
               "  predatory_fish_impacted_by_pelagic_trawling:\n    impact_affects: false\n    impact_table: []\n"
               f"  # Section 228.1: mackerel group F {mk['F_pel']:.4f} /yr: mackerel's local midwater-trawl\n"
               "  # share (0.0132 x 0.025), horse mackerel 0.035 (stock = local).\n"
               + renamed(seg, "predatory_fish_impacted_by_pelagic_trawling", "mackerel_impacted_by_pelagic_trawling")
               .replace("0.0079827776176", f"{BF_P['mackerel']:.11g}"))
for imp in ("rotor", "windfarm_noise"):
    k = f"predatory_fish_impacted_by_{imp}"
    text = after(text, k, renamed(entry(text, k), k, f"mackerel_impacted_by_{imp}"))

# ---- observes ----
for k, v in list(old["interaction_definitions"].items()):
    m = re.fullmatch(r"predatory_fish_observes_(\w+)", k)
    if m and m.group(1) != "predatory_fish":
        text = after(text, k, f"  mackerel_observes_{m.group(1)}:\n    observes: {str(bool(v['observes'])).lower()}\n")
    m = re.fullmatch(r"(\w+)_observes_predatory_fish", k)
    if m and m.group(1) != "predatory_fish":
        text = after(text, k, renamed(entry(text, k), k, f"{m.group(1)}_observes_mackerel"))
text = after(text, "predatory_fish_observes_predatory_fish",
             "  predatory_fish_observes_mackerel:\n    observes: true\n"
             "  mackerel_observes_predatory_fish:\n    observes: true\n"
             "  mackerel_observes_mackerel:\n    observes: true\n")

# ---------------------------------------------------------------------------------------------------
# growth rates (derived)
# ---------------------------------------------------------------------------------------------------
tmp = os.path.join(out, "_tmp_lib.yaml")
open(tmp, "w", encoding="utf-8").write(text.replace("GROWTH_mackerel", "0.0").replace("GROWTH_predatory_fish", "0.0"))
new = load_config(tmp)
os.remove(tmp)


def active_of(p):
    return [g["group_id"] for key in ("decision_makers", "non_decision_makers")
            for g in (p.get(key) or []) if isinstance(g, dict) and not g.get("muted")]


proj_new = copy.deepcopy(project)
dms = proj_new["decision_makers"]
i = [g["group_id"] for g in dms].index("predatory_fish")
dms[i].update(initial_biomass_min=1110, initial_biomass_max=4440, inference_initial_biomass=2220)
dms.insert(i + 1, {"group_id": "mackerel", "initial_biomass_min": int(round(T0 / 2)),
                   "initial_biomass_max": 2 * T0, "inference_initial_biomass": T0})
proj_new["simulation_settings"]["temperature"]["group_layers"]["mackerel"] = "mean_0_50m"
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
    g = b["g"]
    if fid in ("mackerel", "predatory_fish"):
        text = text.replace(f"growth_rate: GROWTH_{fid}", f"growth_rate: {g:.11g}", 1)
        continue
    g_old = float(old["species_definitions"][fid].get("growth_rate", float("nan")))
    if not math.isclose(g, g_old, rel_tol=1e-6):
        text = set_field(text, fid, "growth_rate", f"{g:.11g}")
assert "GROWTH_" not in text
lib_out = os.path.join(out, "fg_library_mack.yaml")
open(lib_out, "w", encoding="utf-8-sig").write(text)

# ---- manifest copy (text edits) ----
mtext = open(project_path, encoding="utf-8-sig").read()
s_old = "      predatory_fish: at_50m\n"
assert s_old in mtext
mtext = mtext.replace(s_old, s_old + "      # Section 228: mackerel feeds in its prey's layer (user).\n"
                      "      mackerel: mean_0_50m\n", 1)
s_old = ("  # Section 223.26 (predator split G1): the 1.20 t/km2 predator level\n"
         "  # split by the long-run biomass shares: water column (predatory_fish)\n"
         "  # 0.754, whiting 0.446 t/km2; ranges x0.5-2.\n"
         "  - group_id: predatory_fish\n    initial_biomass_min: 1360\n    initial_biomass_max: 5430\n"
         "    inference_initial_biomass: 2715\n")
assert s_old in mtext
mtext = mtext.replace(s_old,
                      "  # Section 223.26 (predator split G1): the 1.20 t/km2 predator level\n"
                      "  # split by the long-run biomass shares: water column (predatory_fish)\n"
                      "  # 0.754, whiting 0.446 t/km2; since section 228 the water column is\n"
                      f"  # predatory_fish 0.617 + mackerel {MB:.3f} t/km2 ({SRC}); ranges x0.5-2.\n"
                      "  - group_id: predatory_fish\n    initial_biomass_min: 1110\n    initial_biomass_max: 4440\n"
                      "    inference_initial_biomass: 2220\n"
                      f"  - group_id: mackerel\n    initial_biomass_min: {int(round(T0 / 2))}\n    initial_biomass_max: {2 * T0}\n"
                      f"    inference_initial_biomass: {T0}\n", 1)
proj_out = os.path.join(out, "mareld2_mack.yaml")
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
m2_moved = {fid: (m_old.get(fid), m_new[fid]) for fid in act_new
            if m_old.get(fid) is None or not math.isclose(m_old[fid], m_new[fid], rel_tol=1e-9, abs_tol=1e-12)}
cfg = load_project_config(project_path, library_path=lib_out, project_config=P)
# key parity: every predatory_fish key has a mackerel twin
keys_wc = {k.replace("predatory_fish", "@") for k in ID if "predatory_fish" in k}
keys_mk = {k.replace("mackerel", "@") for k in ID if "mackerel" in k and "predatory_fish" not in k}
report = dict(
    ceilings=CEIL, RM=RM, feeding_cost=FC, starve=STARVE, M1={k: v * TPY for k, v in M1.items()}, r=R, q10=Q10,
    E={k: E[k] for k in ("mackerel", "predatory_fish", "planktivorous_fish", "zooplankton", "benthivorous_fish",
                         "whiting")},
    pairs={f"{k[0]}->{k[1]}": dict(M2=M2.get(k), a=A.get(k), h=H.get(k), night_day=ND.get(k), f=F_ACC.get(k),
                                   stage=STAGE.get(k)) for k in set(M2) | set(A)},
    growth={fid: dict(old=b_old.get(fid, {}).get("g"), new=b["g"], s_star_old=b_old.get(fid, {}).get("s_star"),
                      s_star_new=b["s_star"], best=b["best"]) for fid, b in b_new.items()},
    m2_sums_moved=m2_moved,
    key_parity=dict(only_wc=sorted(keys_wc - keys_mk), only_mackerel=sorted(keys_mk - keys_wc)),
    loader_built=sorted(cfg[1].keys()) if isinstance(cfg, tuple) and isinstance(cfg[1], dict) else str(type(cfg)))
json.dump(report, open(os.path.join(out, "build_split_mack.json"), "w"), indent=1, default=float)
print(json.dumps(report, indent=1, default=float))
