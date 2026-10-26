"""Generate ``Strategy_Oct_eng.pdf`` - the English edition of the
October 2026 strategy deck (``Strategi_okt.pdf``).

Layout, diagrams, drawing code and the library tables are shared with
the Swedish generator, ``tools/make_strategi_deck.py``; this module only
supplies English text: the diagram labels (``LABELS_EN``), the footer and
section label, the number format, and the slide content in ``build``.
Keep the two ``build`` functions in step when the mechanics move.

Usage:
    python3 tools/make_strategy_deck_en.py [-o Strategy_Oct_eng.pdf] [--png DIR]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_strategi_deck as sv  # noqa: E402
from make_strategi_deck import (  # noqa: E402
    Deck, PdfPages, dm_value_rows, draw_energy_flow, draw_food_web,
    draw_functional_response, draw_hunger_gate, draw_impact_curves,
    draw_observation, draw_policy_net, draw_spawn_examples,
    draw_tick_pipeline, fmt, observability_rows, pair_value_rows, species)


class EnglishDeck(Deck):
    FOOTER = "Strategy - Mareld / Poseidon Nord - October 2026"
    PART = "Part"


LABELS_EN = {
    "pipeline": [
        ("1  Decision", "observation → policy → masked softmax"),
        ("2  Impact mortality", "impact_table → biomass"),
        ("3  Predation", "Holling II/III + interference"),
        ("4  Energy costs", "rest / eat / move, × Q10(T)"),
        ("5  Movement", "N/E/S/W × speed, energy moves along"),
        ("6  Currents", "optional: current_response"),
        ("7  Growth / starvation", "mass-balanced, + mortality, light for NDMs"),
        ("8  Masks", "accessibility, extinction threshold"),
    ],
    "web_nodes": {
        "phytoplankton": "Phytoplankton",
        "benthic_infauna": "Benthic community",
        "zooplankton": "Zooplankton", "pelagic_fish": "Pelagic fish",
        "gadoids": "Gadoids", "porpoises": "Porpoises", "seals": "Seals",
        "seabirds": "Seabirds",
    },
    "web_legend": "green = non-decision maker (logistic growth)\n"
                  "blue = decision maker (policy network and energy budget)\n"
                  "dashed = muted in mareld2.yaml",
    "web_self": "cannibalism",
    "fr_herr": "pelagic fish  type III, h = {h}",
    "fr_herr_w": "pelagic fish  w = 1, B_pred = 0.5",
    "fr_benth": "benthic community  type II, h = {h}",
    "fr_typeII_plain": "type II",
    "fr_typeIII": "type III (specialist)",
    "fr_ceiling": "ceiling 1/h",
    "fr_ceiling_e": "energy ceiling 4,500/130",
    "fr_title_gad": "Gadoids: a = {a}, h ∝ prey energy",
    "fr_title_zoo": "Zooplankton → phytoplankton: a = {a}  h = {h}",
    "fr_xlabel": "visible prey biomass in the cell  [t]",
    "fr_ylabel": "intake  [t prey / t predator / tick]",
    "fr_ylabel_e": "energy intake  [MJ / t gadoid / tick]",
    "hg_u5": "u = 0.5  (zooplankton, gadoids, porpoises)",
    "hg_u2": "u = 0.2  (pelagic fish)",
    "hg_xlabel1": "satiation  s_X = E_X / ME_X",
    "hg_ylabel1": "hunger factor  h_X",
    "hg_zoo": "zooplankton  g = {g} / SR = {sr}",
    "hg_pel": "pelagic fish  g = {g} / SR = {sr}  (×10)",
    "hg_xlabel2": "satiation  s_X  (q_X = s_X − u_X)",
    "hg_ylabel2": "relative biomass change / tick",
    "ic_title1": "windfarm_noise → porpoises: energy_factor",
    "ic_xlabel1": "map value  (dB)",
    "ic_ylabel1": "extra metabolic cost",
    "ic_title2": "rotor → seabirds: biomass_factor",
    "ic_xlabel2": "map value  (rotor density 0-1)",
    "ic_ylabel2": "biomass loss / tick",
    "obs_centre": "own cell",
    "obs_shape": "plus shape: own cell + four neighbours",
    "obs_caption": "Channels per cell (compact layout)",
    "obs_code": """
own cell     [ B_own, s_own, B_obs_1..k, imp_1..m ]
neighbour d  [ B_own, B_obs_1..k, imp_1..m ]
last         [ light F(t) ]    (with the daylight calendar)

in_dim_X = (2 + k_X + m) + 4 · (1 + k_X + m) + 1
""",
    "obs_bullets": [
        ("-", "k_X = number of OTHER active FGs that X observes according "
              "to the Observability matrix"),
        ("-", "m = number of active impact maps flagged observable"),
        ("-", "B_obs is VISIBLE biomass: the hidden (resting) share is "
              "subtracted"),
        ("-", "Mareld today: k = 1-4, m = 0 (windfarm_noise is muted) and "
              "one light channel, i.e. 12-27 channels"),
    ],
    "net_in": "observation\n(in_dim)",
    "net_h1": "hidden layer 1\n30 nodes, sigmoid",
    "net_h2": "hidden layer 2\n30 nodes, sigmoid",
    "net_out": "logits → mask → softmax",
    "net_actions": ["Move N", "Move E/S/W", "Rest (= hide)",
                    "Eat Y_1 ... Y_n", ""],
    "ef_stages": ["Intake  intake_XY", "Assimilated  · assimilation",
                  "Reserve R_X  (+ intake − cost)", "New biomass  dB t"],
    "ef_spill": "egested",
    "ef_costs": "rest / movement / foraging\n+ impacts",
    "ef_price": "paid from the reserve: dB · energy_content",
    "spawn_panels": ["uniform", "perlin  (phytoplankton)",
                     "colony  (pelagic fish)",
                     "env_driven  (follows phytoplankton)"],
}


def setup():
    """Switch the shared module to English labels and number format."""
    sv.LABELS.update(LABELS_EN)
    sv.NUM.update(dec=".", sep=",")


# ----------------------------------------------------------------------
# The deck
# ----------------------------------------------------------------------
def build(d):
    f = fmt
    phy, ben = species("phytoplankton"), species("benthic_infauna")
    zoo = species("zooplankton")
    # ---------------------------------------------------------------- 1
    d.title_slide(
        "Strategy",
        "For building ecosystem simulators",
        ["Updated edition - October 2026",
         "Mareld / Poseidon Nord",
         "Describes the mechanics the simulator actually runs today"])

    d.bullets(
        "About this edition",
        [("p", "This version replaces the September edition, which in turn "
               "replaced the May edition of the strategy document. The "
               "structure is the same; the content is written against the "
               "code as it stands today, and a new part weighs the model's "
               "strengths against its weaknesses."),
         "",
         "The code is the model. The document describes it - not the other "
         "way round. Where earlier editions and the code differ, the code "
         "applies, and the difference is stated plainly.",
         "Every formula and number is taken from the tick pipeline in "
         "lib/environments/ecosystem_env/, from the ARS trainers in "
         "lib/runners/ and lib/gpu/, from fg_library.yaml or from "
         "mareld2.yaml. The library tables are read straight from the YAML "
         "files when the document is built.",
         "Parameter names are given as in the library (growth_rate, "
         "handling_time, max_intake_rate) so that a slide can be traced to "
         "the YAML key it describes.",
         ("-", "The strategy is still a working document, not a "
               "requirements specification. The code matching the document "
               "is no proof that a term is ecologically right."),
         ],
        kicker="How to read this",
        footnote="Generated by tools/make_strategy_deck_en.py (Swedish "
                 "edition: tools/make_strategi_deck.py). Re-run when the "
                 "mechanics change.")

    d.table(
        "What has changed since May 2026",
        ["Area", "May edition", "Today"],
        [["Growth", "B·(1 + MG·q) - new mass was free",
          "Mass-balanced: growth is paid for from the energy reserve; "
          "growth_rate is derived from a literature budget"],
         ["Predation", "Wanted intake = B·π·I·h, proportional scale-down",
          "Holling type II/III per pair with Beddington-DeAngelis "
          "interference and a light-dependent attack rate"],
         ["Hunger", "All predation masked above s_X = 0.8",
          "No mask - full appetite up to the maintenance level, then "
          "linearly to zero at a full reserve"],
         ["Rest", "Rest = low energy cost",
          "Rest is also a hide action with per-pair visibility floors"],
         ["Fitness", "α·ΔB + β·ΔR over two transitions",
          "Mean over the rollout of log(total energy / start energy)"],
         ["Training", "Alternating: one FG at a time",
          "Co-evolution on CPU (train.py) or GPU (train_gpu.py), one-year "
          "rollouts"],
         ["Tick length", "Fixed 6 h", "--tick-length 1-6 h, the library "
          "is converted in memory at load time"],
         ["New", "-", "Daylight calendar, water temperature, currents, "
          "migration, extinction threshold, per-cell local reward, "
          "diagnostic tools"]],
        [10, 30, 46], kicker="Overview", size=11.5,
        footnote="Part 9 lists the deviations from the May edition that are "
                 "deliberate, and why.")

    d.table(
        "What has changed since September 2026",
        ["Area", "September", "October", "Section"],
        [["growth_rate", "Free number per FG",
          "Derived from r_max, M1 and the predation matrix (M2)", "129-131"],
         ["Hunger", "h = max(0, 1 − s/scale)",
          "h = min(1, (1 − s)/(1 − u)); satiation_scale removed",
          "130, 133"],
         ["Season", "Sine for phytoplankton (switched off)",
          "Daylight calendar: light-dependent attack and light-limited "
          "primary production", "137-138"],
         ["Temperature", "None",
          "Q10 on resting metabolism per FG and water layer", "143"],
         ["Herring", "One energy pool, 7,500 MJ/t",
          "Lean body 4,000 + fat reserve up to 7,000; the reserve is eaten "
          "too", "140"],
         ["Gadoids", "Two prey, type II, ceiling 16 %/day",
          "Four prey, type III on herring, ceiling as an energy limit, "
          "M1 0.415/yr", "141, 147-148, 157"],
         ["Porpoises", "Rest 90, indivisible unit 250 kg",
          "Costs against measured FMR, diet fitted to Ecopath, 50 kg",
          "145, 151-152"],
         ["Plankton", "Free half-saturation, 2,000 MJ/t",
          "Half-saturation 20 t/km², 4,800 MJ/t, feeding_cost 4.0",
          "133-135"],
         ["Movement", "Residues below the threshold were swept",
          "Residues follow the outflow; small emigrations stay", "156"]],
        [11, 25, 50, 12], kicker="Overview", size=10.8,
        footnote="Section numbers refer to mareld_resume.txt, where every "
                 "change is motivated and measured.")

    # ---------------------------------------------------------------- 2
    d.section("1", "Strategy and components",
              "What the model is, what it is for and which parts it is "
              "made of.")

    d.bullets(
        "Strategy",
        ["Ecosystems are modelled as a map with several map layers that "
         "evolve over time.",
         "The layers describe environmental conditions, the distribution of "
         "populations belonging to different functional groups, and "
         "different kinds of environmental impact. A calendar for light "
         "and temperature drives the seasons.",
         "Population behaviour - movement and foraging in particular - is "
         "governed by behaviour models trained with ARS, but can also be "
         "hand-coded (the viability rig runs with no network at all).",
         "Biology drives, not action hacks. Parameters and mechanisms are "
         "adjusted first; shaping terms in the reward are the last resort "
         "and are switched off by default.",
         "The model is Eulerian: we follow values in fixed cells, not "
         "individuals. Biomass can split, move in different directions and "
         "mix with other biomass in the same cell."],
        kicker="Core ideas")

    d.bullets(
        "Objective",
        ["The strategy is used to build simulation models that serve as "
         "scenario and hypothesis tools.",
         "The models do not produce absolute population forecasts.",
         "Results are to be read relatively, comparatively and under stated "
         "assumptions.",
         "",
         ("p", "Before a scenario is compared, the world must be "
               "self-sustaining with the policies actually used - no "
               "functional group may go extinct or drift towards zero over "
               "the horizon. The viability rig, which runs with frozen "
               "behaviour, is a diagnostic for the mechanics, not the "
               "judge."),
         ],
        kicker="What the model answers")

    d.bullets(
        "Components",
        ["An ecosystem model consists of:",
         ("-", "a grid with a length scale and a tick length"),
         ("-", "a set of functional groups (FGs), decision makers and non "
               "decision makers"),
         ("-", "a set of maps - a scenario"),
         ("-", "a calendar: daylight and water temperature for the site"),
         ("-", "a set of properties for every functional group and every "
               "predation pair (fg_library.yaml)"),
         ("-", "a dynamic model with update rules for the maps - the tick"),
         ("-", "a behaviour model for every decision-making functional "
               "group"),
         ("-", "spawn strategies that decide how the initial biomass is "
               "laid out"),
         ("-", "diagnostics: the energy budget gate, the frozen viability "
               "rig and probe tools for annual budgets"),
         "The library is shared by all projects. The project file "
         "(mareld2.yaml) selects the FGs, initial biomasses, impact "
         "variables, grid size and the site's light and temperature "
         "climate."],
        kicker="Parts of the model", size=15.5)

    d.bullets(
        "Grid",
        ["Every ecosystem model has a grid, i.e. a matrix of cells with:",
         ("-", "a dimension - Mareld runs 60 x 60 cells"),
         ("-", "a length scale - one cell side = 1 km"),
         ("-", "a time scale - one tick = 1-6 hours, normally 6"),
         ("-", "a position - 58.15° N, which sets the daylight"),
         "The grid is flat and 4-connected: all movement is to N, E, S or "
         "W, never diagonal.",
         "An optional accessibility map decides which cells are available. "
         "It is shared by all FGs: movement into an inaccessible cell is "
         "masked out, and biomass that still ends up there is zeroed at "
         "the end of the tick.",
         "Without migration the grid edge is closed. With --migration on "
         "biomass leaves the grid and is reintroduced, concentrated along "
         "the edge."],
        kicker="Geography")

    d.table(
        "Tick length is a run parameter",
        ["Parameter type", "Rule at factor k = new/old", "Examples"],
        [["flux - amount transferred per tick", "x · k",
          "max_intake_rate (also per pair), resting_metabolism, "
          "movement_speed, seed_rate"],
         ["growth - share ADDED to a stock", "(1 + x)ᵏ − 1",
          "growth_rate"],
         ["loss - share REMOVED from a stock", "1 − (1 − x)ᵏ",
          "natural_mortality, starve_rate, biomass_factor in impact tables"],
         ["inverse - quantity whose inverse is the rate", "x / k",
          "handling_time (the Holling ceiling 1/h must stay fixed in real "
          "time)"],
         ["tick-independent", "unchanged",
          "max_energy_reserve, energy_content, maintenance_level, "
          "interference, min_split_biomass, the cost multipliers, "
          "dark_ratio, light_threshold_deg, metabolism_q10"]],
        [22, 24, 40], kicker="--tick-length", size=11.5,
        lead="The engine is tick-length agnostic: every library number is "
             "expressed PER TICK. The tick length converts per-tick rates to "
             "real time - and, when it is not 6 h, rescales every "
             "tick-dependent parameter in memory as the FGs are built. The "
             "light and temperature tables are built directly for the run's "
             "tick length.",
        footnote="fg_library.yaml has ONE calibration (6 h) and is never "
                 "rewritten by a run. lib/world/tick_time.py is the single "
                 "definition point.")

    d.table(
        "Functional groups",
        ["Functional group", "Type", "Menu", "Role in the model"],
        [["phytoplankton", "non decision maker", "-",
          "Primary producer, light-limited logistic growth towards "
          "carrying capacity, drifts with the current"],
         ["benthic_infauna", "non decision maker", "-",
          "Bottom food for gadoids, very slow growth, does not move"],
         ["zooplankton", "decision maker", "phytoplankton",
          "Grazer, specialist (type III), moves 0.02 cells/tick"],
         ["pelagic_fish", "decision maker", "zooplankton",
          "Herring/sprat, specialist (type III), lean body + fat reserve"],
         ["gadoids", "decision maker",
          "pelagic_fish, benthic_infauna, zooplankton, gadoids",
          "Generalist; type III on herring, energy ceiling, interference "
          "1.0"],
         ["porpoises", "decision maker", "pelagic_fish, gadoids",
          "Top predator, endotherm, sensitive to underwater noise"],
         ["seals", "decision maker", "pelagic_fish, gadoids",
          "Top predator, colony spawn (muted)"],
         ["seabirds", "decision maker", "pelagic_fish",
          "Top predator, the only FG with rotor mortality (muted)"]],
        [16, 13, 26, 37], kicker="Mareld: eight groups, six with a policy",
        size=11.5,
        footnote="English snake_case IDs are canonical. seabirds and seals "
                 "are muted in mareld2.yaml - defined but not placed in the "
                 "current scenario.")

    d.figure("Food web", draw_food_web, kicker="Menu according to the library",
             footnote="The arrow points from prey to consumer. Every pair has "
                      "its own handling_time, assimilation_factor, "
                      "predation_mortality and - for some pairs - attack "
                      "rate, visibility floor and light dependence.")

    # ---------------------------------------------------------------- 3
    d.section("2", "Maps and calendar",
              "All state in the model is maps over the grid. Light and "
              "temperature come from a calendar for the site.")

    d.bullets(
        "Maps",
        ["A map is a function from the grid to the real numbers.",
         "A map with values in [0, 1] can be seen as a greyscale image. "
         "Other maps - depth, biomass, noise - are shown as heatmaps.",
         "The model keeps two state maps per functional group: biomass "
         "B_X(c) in tonnes and reserve energy R_X(c) in MJ.",
         "The energy map E_X(c) = R_X(c) / B_X(c) is derived, not stored. "
         "That form is what makes mixing on movement trivial: two inflows "
         "add their R and their B, and the ratio is recomputed afterwards.",
         "The satiation s_X(c) = E_X(c) / ME_X is the quantity that "
         "governs hunger, growth and starvation. It lies in [0, 1]."],
        kicker="Definition")

    d.table(
        "Map types and calendar tables",
        ["Type", "Symbol", "Unit", "Interpretation"],
        [["Biomass maps", "B_X(c)", "t/cell",
          "Biomass of functional group X in cell c."],
         ["Reserve energy maps", "R_X(c)", "MJ/cell",
          "Stored energy. Primary state; clipped to B_X · ME_X."],
         ["Energy maps", "E_X(c)", "MJ/t",
          "Derived: R_X / B_X. Zero where biomass is zero."],
         ["Carrying capacity maps", "CC_X(c)", "t/cell",
          "Non decision makers only (max_carrying_capacity)."],
         ["Accessibility", "accessibility(c)", "[0,1]",
          "Shared habitat mask. > 0 = the cell may be occupied and "
          "reached."],
         ["Impact maps", "Impact_p(c)", "arbitrary scale",
          "Noise in dB, trawling intensity, rotor density. Looked up in "
          "the FG's impact_table."],
         ["Current field", "-", "share/tick",
          "Generated procedurally per tick from noise, not stored."],
         ["Calendar tables", "F(t), m(t), P(t), T(t)", "per tick",
          "Light, attack multiplier, primary production factor and water "
          "temperature. Same for every cell; built from the manifest."]],
        [18, 16, 12, 44], kicker="What exists in a run", size=11.5,
        footnote="Geographic maps such as per-cell depth and temperature and "
                 "administrative maps are part of the strategy but are not "
                 "read by the engine today; temperature is a time series per "
                 "water layer, not a map.")

    d.bullets(
        "The daylight calendar",
        ["The calendar gives every tick a date and a time: year tick k "
         "starts k · tick length hours after 1 January 00:00 solar time. "
         "8,760 h divides by every tick length 1-6 h, so the year closes "
         "exactly.",
         "Sun elevation follows from latitude (58.15° N), declination and "
         "solar time. The light level F(t) is the mean of a logistic "
         "function of elevation over the tick's 10-minute steps - the mean "
         "of f, not f of the mean, because f is strongly nonlinear.",
         "The start day is drawn per world (start_day_of_year: random) and "
         "shared by every delta pair and both signs in that world (CRN).",
         "The calendar is used in three places:",
         ("-", "a light-dependent attack rate for visual predators"),
         ("-", "light-limited growth for phytoplankton"),
         ("-", "the monthly temperature that drives metabolism"),
         "Every decision maker sees F(t) as its last observation channel. "
         "Switching the calendar on or off changes in_dim by 1 and "
         "invalidates checkpoints."],
        kicker="simulation_settings.daylight", size=15.5)

    d.table(
        "The calendar month by month",
        ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
         "Oct", "Nov", "Dec"],
        [["light F", "0.37", "0.44", "0.54", "0.66", "0.79", "0.87", "0.83",
          "0.70", "0.58", "0.47", "0.38", "0.34"],
         ["m herring → zoo", "0.70", "0.80", "0.94", "1.10", "1.30", "1.42",
          "1.36", "1.17", "0.99", "0.84", "0.72", "0.66"],
         ["m gadoid → herring", "0.90", "0.93", "0.98", "1.04", "1.10",
          "1.13", "1.11", "1.06", "1.00", "0.95", "0.91", "0.88"],
         ["P/P_ref phyto", "0.16", "0.38", "0.72", "0.99", "1.12", "1.14",
          "1.03", "0.84", "0.58", "0.37", "0.15", "0.10"],
         ["T 0-50 m (°C)", "6.4", "5.5", "5.3", "6.2", "7.9", "9.9", "12.0",
          "13.3", "13.6", "12.6", "10.5", "8.4"],
         ["T 50 m (°C)", "7.8", "7.2", "6.7", "6.6", "6.8", "7.2", "8.0",
          "8.7", "10.2", "11.4", "10.6", "9.3"],
         ["Q10 factor herring", "0.55", "0.51", "0.51", "0.54", "0.62",
          "0.73", "0.85", "0.95", "0.97", "0.90", "0.76", "0.64"],
         ["Q10 factor gadoids", "0.95", "0.90", "0.87", "0.86", "0.88",
          "0.91", "0.96", "1.01", "1.14", "1.26", "1.18", "1.06"],
         ["Q10 factor zoo", "0.80", "0.75", "0.74", "0.79", "0.88", "1.02",
          "1.18", "1.29", "1.32", "1.23", "1.06", "0.92"]],
        [20] + [6] * 12, size=11.5, align=["left"] + ["right"] * 12,
        kicker="58.15° N, 6 h/tick, monthly means",
        footnote="From tools/probes/daylight_table.py. The attack multiplier "
                 "m has an annual mean of exactly 1; the phytoplankton factor "
                 "is 1 on 15 April (day 105) with an annual mean of 0.63; "
                 "herring's Q10 factor is relative to a summer cost at "
                 "14 °C. Temperature: SMHI SHARK A16 + A17, 1991-2020.")

    d.bullets(
        "Water temperature",
        ["Without temperature every ectotherm paid its summer cost all "
         "year. With simulation_settings.temperature a decision maker with "
         "metabolism_q10 pays:",
         ("p", "resting_metabolism · Q10^((T(t) − t_ref) / 10)"),
         "Every action cost is a multiple of resting_metabolism, so rest, "
         "eat and move all follow. T(t) is interpolated circularly from "
         "twelve monthly means for the FG's water layer: 0-50 m for "
         "herring and zooplankton, 50 m for gadoids.",
         ("-", "Herring: Q10 2.2, t_ref 14 °C - the library value is the "
               "Rudstam/Rose cost at summer temperature."),
         ("-", "Gadoids 2.2 and zooplankton 2.0 with t_ref = annual mean: "
               "the factor averages 1 over the year."),
         ("-", "Porpoises: none - endotherm, field metabolic rate is flat "
               "over 0-20 °C."),
         "Attack rates are not temperature-scaled, and no observation "
         "channel is added: checkpoints stay valid.",
         ("p", "Effect in a probe with trained policies: herring winter "
               "starvation disappeared; reserve fill in January-February "
               "came out at 0.31-0.32, inside the literature's 0.28-0.45.")],
        kicker="Q10 on the calendar", size=15)

    # ---------------------------------------------------------------- 4
    d.section("3", "Properties of functional groups",
              "The parameters in fg_library.yaml, their role in the tick "
              "and their calibrated values.")

    d.table(
        "Properties: energy and metabolism",
        ["Property", "Key", "Unit", "Role"],
        [["Decision model", "is_decision_maker", "yes/no",
          "Whether the FG has a policy and an energy budget."],
         ["Maximum reserve energy", "max_energy_reserve (ME_X)", "MJ/t",
          "Ceiling for E_X. The reserve is clipped to B_X · ME_X after "
          "movement."],
         ["Resting metabolism", "resting_metabolism", "MJ/t/tick",
          "Base cost. Multiplied by the action's cost factor."],
         ["Cost factors", "resting_cost, feeding_cost, movement_cost",
          "factor", "Multiplier on resting metabolism per action."],
         ["Temperature", "metabolism_q10, metabolism_t_ref", "-, °C",
          "Q10 factor on resting metabolism; t_ref in °C or annual_mean."],
         ["Energy content", "energy_content", "MJ/t",
          "What a tonne of the FG is worth as food - and the price of a "
          "tonne of its own new biomass."],
         ["Reserve as food", "prey_includes_reserve, "
          "reserve_reference_fill", "yes/no, share",
          "The prey's reserve is eaten with it. Static quality for the "
          "gates at the reference fill."],
         ["Maintenance level", "maintenance_level (u_X)", "share of ME",
          "Zero point: q_X = s_X - u_X, and the knee of the hunger curve. "
          "0.5 for every decision maker except herring (0.2)."],
         ["Energy sensitivity", "impact_table.energy_factor", "factor",
          "Extra metabolic cost from impact p."]],
        [17, 25, 12, 41], size=11.0, kicker="Per functional group")

    d.table(
        "Properties: growth, losses and movement",
        ["Property", "Key", "Unit", "Role"],
        [["Maximum net growth", "r_max", "1/yr",
          "Literature net growth at low density. Basis of the derived "
          "growth_rate."],
         ["Growth rate", "growth_rate (MG_X)", "share/tick",
          "DERIVED by the loader - never edited directly (except the "
          "benthic community)."],
         ["Natural mortality", "natural_mortality (M1)", "share/tick",
          "Only losses that are not predation by an FG. Needs "
          "--mortality on."],
         ["Starvation rate", "starve_rate", "share/tick",
          "Catabolism when q < 0."],
         ["Maximum carrying capacity", "max_carrying_capacity", "t/cell",
          "Non decision makers only."],
         ["Recolonisation", "seed_rate", "share of CC/tick",
          "Floor for non decision makers; 0 for phytoplankton."],
         ["Attack rate", "max_intake_rate (a)", "1/(t·tick)",
          "The slope of the functional response; can be set per pair."],
         ["Interference", "interference (w)", "1/t",
          "Beddington-DeAngelis: own density lowers intake."],
         ["Speed", "movement_speed (v)", "cell sides/tick",
          "Distance per tick for the moving cohort. Clipped to [0, 1]."],
         ["Minimum divisible mass", "min_split_biomass", "kg",
          "Below this the action distribution collapses to argmax."],
         ["Extinction threshold", "extinction_threshold_factor", "factor",
          "Cell zeroed below factor · min_split_biomass."]],
        [18, 24, 14, 39], size=10.8, kicker="Per functional group")

    d.table(
        "Properties: predation pairs and primary producer",
        ["Property", "Key", "Unit", "Role"],
        [["Predation mortality", "predation_mortality (M2)", "1/yr",
          "Literature loss of the prey to the predator. Enters the prey's "
          "derived growth."],
         ["Attack rate per pair", "max_intake_rate", "1/(t·tick)",
          "Replaces the species value. Moves the half-saturation 1/(a·h) "
          "without moving the ceiling."],
         ["Handling time", "handling_time (h)", "tick",
          "Sets the ceiling 1/h tonnes of prey per tonne of predator and "
          "tick."],
         ["Assimilation", "assimilation_factor", "share",
          "Share of the prey's energy the predator keeps."],
         ["Visibility floor", "visibility_floor", "[0,1]",
          "Share that stays visible despite resting; else the prey's own."],
         ["Functional response", "functional_response", "2 | 3",
          "Forces Holling type II or III for the pair."],
         ["Light dependence", "dark_ratio, light_threshold_deg",
          "share, °", "Attack in darkness relative to daylight, and the sun "
          "elevation of half detection."],
         ["Light saturation", "light_saturation (I_k)", "µmol/m²/s",
          "Knee of the phytoplankton P-I curve; one value or twelve."],
         ["Reference day", "light_reference_day", "day",
          "Day on which the light factor is exactly 1 (105 = mid April)."],
         ["Current response", "current_response", "[0,1]",
          "Share of the current field the FG follows."]],
        [19, 25, 12, 40], size=10.8, kicker="Per pair and for producers")

    d.formula(
        "The growth rate is derived from a literature budget",
        [("lib/world/growth_budget.py - the loader, the FG editor and the "
          "tests compute it the same way", [
            r"g_X &= \dfrac{r_{\max,X} + M1_X + \sum_j M2_{Xj}}{(1 - u_X)\cdot 1460} \qquad \mathrm{decision\ makers}",
            r"r_X &= \dfrac{r_{\max,X} + \sum_j M2_{Xj}}{1460} \qquad \mathrm{non\ decision\ makers}",
            ("t", "1,460 = number of 6-hour ticks per year. M2_Xj counts when "
                  "the pair is checked and the predator is a decision maker "
                  "active in the project."),
        ])],
        lead=[("p", "growth_rate is no longer a free number. Growth has to "
                    "pay for the losses the model itself takes, so that net "
                    "growth at low density is the literature r_max.")],
        tail=[("-", "M1 and M2 come from the North Sea Ecopath model for 1991 "
                    "(Mackinson & Daskalov 2007); r_max from FishBase/CMSY, "
                    "observed recovery (seals, porpoises), Huntley & Lopez "
                    "(copepods) and dilution experiments (phytoplankton)."),
              ("-", "Unchecking a predator in the FG editor lowers the prey's "
                    "g by that predator's share. With --mortality off the "
                    "loader leaves M1 out of g."),
              ("-", "The budget is paid in full only at s = 1, where hunger "
                    "closes intake. That is a known weakness (Part 8). The "
                    "benthic community is still hand-set.")],
        kicker="r_max, M1, M2")

    d.table(
        "Library values - decision-making functional groups",
        ["", "zooplankton", "pelagic_fish", "gadoids", "porpoises", "seals *",
         "seabirds *"],
        dm_value_rows(year="yr", annual="annual mean", derived="derived"),
        [22, 13, 13, 13, 13, 13, 13], size=10.2,
        align=["left", "right", "right", "right", "right", "right", "right"],
        kicker="fg_library.yaml, calibrated at 6 h/tick",
        footnote="Read from fg_library.yaml when the document is built. "
                 "Energies in MJ/t, rates per tick. * = muted in "
                 "mareld2.yaml. Herring's energy_content is the lean body; "
                 "the reserve comes on top.")

    d.table(
        "Library values - predation pairs",
        ["Predator → prey", "a", "h", "assim.", "floor", "response",
         "dark / threshold", "M2 (1/yr)"],
        pair_value_rows(prey_word="prey", type_word="type",
                        forced_word="set"),
        [28, 8, 8, 9, 11, 13, 14, 11], size=10.8,
        align=["left", "right", "right", "right", "right", "left", "left",
               "right"],
        lead=f"Non decision makers: phytoplankton grows at growth_rate "
             f"{f(phy['growth_rate'])} (the April rate, light-limited) "
             f"towards max_carrying_capacity "
             f"{f(phy['max_carrying_capacity'])} t/cell, "
             f"{f(phy['energy_content'])} MJ/t; benthic_infauna at "
             f"{f(ben['growth_rate'])} towards "
             f"{f(ben['max_carrying_capacity'])} t/cell, "
             f"{f(ben['energy_content'])} MJ/t, plus seed_rate "
             f"{f(ben['seed_rate'])}.",
        footnote="\"prey 0.2\" = the pair inherits the prey's own "
                 "visibility_floor. \"set\" = functional_response in the "
                 "library; otherwise a specialist gets type III and a "
                 "generalist type II. * = the predator is muted.")

    d.bullets(
        "Herring: lean body and fat reserve",
        ["Herring used to carry a single energy pool (7,500 MJ/t). In "
         "trained runs they starved every winter, yet no source reports "
         "winter starvation of adult herring: real herring drop to "
         "4.4-6.5 kJ/g in the first quarter and live.",
         "The body is now separated from the reserve:",
         ("-", "energy_content 4,000 MJ/t = lean body (≈ 4.1 kJ/g)"),
         ("-", "max_energy_reserve 7,000 MJ/t ≈ 18 % fat"),
         ("-", "maintenance_level 0.2 - starvation only when the fat is "
               "nearly gone"),
         ("-", "prey_includes_reserve: a predator gains "
               "intake · assimilation · (4,000 + R/B in the cell)"),
         "The gates and the FG editor count the reserve at the reference "
         "fill 0.5, i.e. the same quality of 7,500 MJ/t as before, so the "
         "porpoise energy budget is unchanged.",
         ("p", "With the split, resting at low food pays - a herring can "
               "learn to fast. In the ten-year run herring follow a lipid "
               "cycle: s 0.23 in February, 0.87 in June.")],
        kicker="prey_includes_reserve", size=15)

    d.bullets(
        "Gadoids: four prey and an energy ceiling",
        ["Ecopath 1991 gives gadoids 45 % of their food from prey the model "
         "lacks. With only herring and benthos, the model's gadoids put "
         "twice the Ecopath pressure on herring. Zooplankton (krill proxy) "
         "and cannibalism were added; the herring pair got type III, which "
         "gives sparse prey a refuge.",
         "The ceiling was 1/h = 16 % of body mass per day - against 2-3.5 % "
         "for adult cod. Hunger, not the stomach, set the intake. Gastric "
         "evacuation time scales with prey energy density, so the ceiling "
         "is now set as an ENERGY limit per pair:",
         ("p", "h = 130 · E / 4,500   →   benthos 87, zooplankton 130, "
               "gadoids 159, herring 217"),
         "Every pair then saturates at the same MJ per tonne of gadoid and "
         "tick.",
         "M1 was lowered on 12 October from 0.683 to 0.415 per year: "
         "residual M0 0.20 as in ICES SMS plus predation by groups that are "
         "not FGs (0.214). The old benthos part came from duplicated rows "
         "in the Ecopath table."],
        kicker="Sections 141, 147-148, 157", size=15)

    d.bullets(
        "Porpoises: costs, diet and indivisible unit",
        ["Costs against measured metabolism (Rojano-Donate et al. 2018: "
         "wild porpoises 0.35 MJ/kg/day, field metabolic rate including "
         "foraging):",
         ("-", "resting_metabolism 60 MJ/t/tick = 0.24 MJ/kg/day"),
         ("-", "feeding_cost 1.5 → eat = 90 = the measured field metabolic "
               "rate; movement_cost 1.3"),
         "Diet: Ecopath gives ~49 % gadoids by mass. Herring saturated the "
         "porpoise, so the per-pair attack rate was set to 0.015 on herring "
         "and 0.2 on gadoids. The ratio of 13 is a fit to diet data, not a "
         "mechanism.",
         "Indivisible unit: min_split_biomass 250 → 50 kg (one animal), "
         "extinction sweep at 5 kg. At 250 kg every porpoise cell was "
         "argmax, which hid the policy's real distribution.",
         ("p", "Trained at 50 kg: 44 % gadoids in the diet, porpoises spread "
               "over 264 cells, the move probability was trained down from "
               "22 % to 7 %.")],
        kicker="Sections 145, 151-152, 154", size=15)

    # ---------------------------------------------------------------- 5
    d.section("4", "Behaviour models",
              "The policy network, the observation and the masking that "
              "make the output ecologically meaningful.")

    d.figure(
        "Policy network", draw_policy_net, kicker="Architecture",
        lead="A behaviour model is a small fully connected network. The "
             "default is two hidden layers of 30 nodes with sigmoid and a "
             "linear output layer; --policynetwork LAYERS NODES [ACTIVATION] "
             "changes it.",
        items=[("-", "All decision makers are evaluated in ONE batched "
                     "torch.bmm over the whole grid - a policy sees all its "
                     "cells at once."),
               ("-", "The output is logits. Masking happens BEFORE softmax, "
                     "so an impossible action gets zero probability mass."),
               ("-", "The action vector is [N, E, S, W, Rest, Eat Y_1 ... "
                     "Y_n] - a distribution, not a choice: every cell splits "
                     "its biomass across all actions.")])

    d.figure(
        "Observation", draw_observation, kicker="What a population sees",
        footnote="Changing the Observability matrix, the observable impact "
                 "maps, the FG set, the grid layout or the daylight calendar "
                 "invalidates existing checkpoints - the width of the "
                 "observation changes.")

    d.table(
        "The Observability matrix",
        ["Observer", "phyto", "zoo", "pel. fish", "gadoids", "porpoises",
         "seals", "seabirds", "benthos"],
        observability_rows("yes"),
        [20, 10, 10, 11, 11, 12, 9, 11, 10], size=11.5,
        align=["left"] + ["center"] * 8,
        lead="The matrix sets the width of the observation per FG: k_X = "
             "number of yes on the row besides the FG itself, counting "
             "active FGs only (own biomass and satiation are always "
             "included). Today k = 1 for zooplankton, 2 for porpoises, 3 for "
             "pelagic fish and 4 for gadoids.",
        footnote="Read from fg_library.yaml. Gadoids see zooplankton since "
                 "section 141 (in_dim 22 → 27). * = muted. The matrix is "
                 "edited in the FG editor.")

    d.bullets(
        "Action masking",
        ["Predation is masked out when the prey is not in the cell or not "
         "on the menu. Nothing else masks eating.",
         ("-", "The May edition masked all predation above s_X = 0.8. That "
               "rule is not in the code. Hunger is instead a CONTINUOUS "
               "factor h_X = min(1, max(0, (1 − s_X)/(1 − u_X))) that "
               "scales the wanted intake."),
         "Movement in a direction is masked out when the neighbour is "
         "inaccessible, when the direction leaves the grid (without "
         "--migration on) or when the FG's movement_speed is 0.",
         "If every action would be masked, Rest is opened, so the "
         "distribution is always defined.",
         "Below min_split_biomass the distribution in the cell collapses to "
         "argmax - a population that cannot be split picks one action "
         "instead of spreading into arbitrarily small parts.",
         "",
         ("p", "Interpretation: masking keeps the model away from "
               "unreasonable choices, simplifies training and lets the "
               "output follow the model's ecological rules.")],
        kicker="Before softmax")

    d.figure(
        "Hunger, maintenance and growth", draw_hunger_gate,
        kicker="Two curves that govern everything",
        lead="The hunger factor h_X is 1 up to the maintenance level u_X and "
             "then falls linearly to 0 at a full reserve. The surplus energy "
             "q_X = s_X − u_X sets the sign of growth. Both curves have "
             "their knee at u_X.",
        footnote="Below u_X a population eats with full appetite - eating is "
                 "set by food and gut capacity, not by the reserve. The "
                 "energy budget gate therefore works with h(u_X) = 1. "
                 "Herring's growth curve is magnified ten times.")

    d.bullets(
        "Resting is also hiding",
        ["The share of a decision maker that chooses Rest in a tick is "
         "protected from predation that tick - and invisible in the next "
         "tick's observation.",
         ("-", "visible share = 1 - π_rest · (1 - floor)"),
         "The floor is primarily the prey's own visibility_floor, but can "
         "be set per PAIR: detection ability belongs to the pair, not the "
         "prey. Porpoises use biosonar - hence floor 0.95 for porpoises → "
         "pelagic fish against 0.25 for gadoids → pelagic fish.",
         "Light and hiding are two different things: the floor says how "
         "well hiding works, light says how dangerous it is to be visible. "
         "Rest also stops grazing - the real cost of diel vertical "
         "migration.",
         "The cap on removal from a cell is set by the BEST-detecting "
         "predator, while each individual predator is limited by what it "
         "sees itself.",
         ("-", "An exposure-weighted M1 (section 139: a visual part follows "
               "light, a tactile part hits the hidden) exists in the engine "
               "but is not set in the library.")],
        kicker="visibility_floor")

    # ---------------------------------------------------------------- 6
    d.section("5", "Update rules",
              "The tick, step by step, with the formulas the engine runs.")

    d.figure("Pseudocode for one tick", draw_tick_pipeline,
             kicker="Eight steps, in this order",
             footnote="The FG order is shuffled every tick, but no step "
                      "depends on it: in the looped steps each FG touches "
                      "only its own state, and predation and movement are "
                      "vectorised - all predators eat at once from the state "
                      "at t.")

    d.formula(
        "Step 2: impact mortality",
        [("Lookup in the FG's own table", [
            r"(\mathrm{biomass\_factor}_p,\ \mathrm{energy\_factor}_p) = \mathrm{interp}\left(\mathrm{impact\_table},\ \mathrm{Impact}_p(c)\right)",
            "",
            r"m_X^{\mathrm{Impact}}(c) &= B_X(c)\,\sum_p \mathrm{biomass\_factor}_p(c)",
            r"B_X(c) &\leftarrow \max\left(0,\ B_X(c) - m_X^{\mathrm{Impact}}(c)\right)",
            r"R_X(c) &\leftarrow R_X(c)\,\dfrac{B_X^{\mathrm{new}}(c)}{B_X^{\mathrm{old}}(c)}",
        ])],
        lead=[("p", "Impacts are taken BEFORE predation, so the prey biomass "
                    "the predators see is already reduced.")],
        tail=[("-", "The table is a list of (value, biomass_factor, "
                    "energy_factor). Between points it is interpolated "
                    "linearly; outside the support the nearest endpoint is "
                    "used - no extrapolation."),
              ("-", "energy_factor is not used here but in step 4, where it "
                    "raises the metabolic cost by the factor "
                    "(1 + Σ energy_factor)."),
              ("-", "Only decision makers carry impact tables in the current "
                    "model.")],
        kicker="impact_table")

    d.figure(
        "Impact curves in the library", draw_impact_curves,
        kicker="The three active tables",
        items=[("-", "windfarm_noise → porpoises: pure energy cost, 0 at "
                     "60 dB, +80 % metabolism at 120 dB."),
               ("-", "rotor → seabirds: pure mortality, up to 3 % of the "
                     "biomass per tick at full rotor density."),
               ("-", "pelagic_trawling → pelagic_fish: linear from 0 to "
                     "100 % removal at map value 1,000.")],
        footnote="All three impact variables are muted in mareld2.yaml - "
                 "they are switched on when a scenario is to be run.")

    d.formula(
        "Step 3: predation",
        [("Wanted intake, effective attack rate and removal", [
            r"B_Y^{\mathrm{visible}}(c) &= B_Y(c)\,\left(1 - \pi_{Y,\mathrm{rest}}(c)\,(1 - \mathrm{floor}_{XY})\right) \qquad a_{XY}(t) = a_{XY}\,m_{XY}(t)",
            r"f_{XY}(c) &= \dfrac{a_{XY}(t)\,B_Y^{\mathrm{visible}}}{1 + a_{XY}(t)\,h_{XY}\,B_Y^{\mathrm{visible}} + w_X\,B_X(c)} \qquad (\mathrm{type\ II})",
            r"D_{XY}(c) &= B_X(c)\,\pi_{X,\mathrm{eat}\,Y}(c)\,f_{XY}(c)\,h_X(c)",
            r"\mathrm{scale}(c) &= \min\left(1,\ 0.999\,B_Y^{\mathrm{visible}}(c)\ /\ \Sigma_X\,D_{XY}(c)\right) \qquad \mathrm{intake}_{XY} = D_{XY}(c)\,\mathrm{scale}(c)",
        ]),
         ("Bookkeeping", [
            r"R_X(c) &\leftarrow R_X(c) + \sum_Y \mathrm{intake}_{XY}(c)\,\mathrm{assimilation}_{XY}\,\left(\mathrm{energy\_content}_Y + [R_Y/B_Y]\right)",
            r"B_Y(c) &\leftarrow B_Y(c) - \sum_X \mathrm{intake}_{XY}(c) \qquad R_Y\ \mathrm{scaled\ by\ the\ same\ ratio}",
        ])],
        tail=[("-", "Type III replaces B with B² in numerator and "
                    "denominator. It is used by specialists and by pairs with "
                    "functional_response: 3."),
              ("-", "m_XY(t) is the daylight multiplier (1 without the "
                    "calendar); [R_Y/B_Y] applies only to prey with "
                    "prey_includes_reserve, today herring."),
              ("-", "w_X · B_X(c) is Beddington-DeAngelis interference. The "
                    "factor 0.999 keeps an overgrazed cell away from exact "
                    "zero, which is absorbing.")],
        kicker="Holling + interference + light")

    d.figure(
        "Functional response", draw_functional_response,
        kicker="What the formula does",
        items=[("-", "Left: the gadoid ceilings are an energy limit. Herring "
                     "(type III, h 217) and benthos (type II, h 87) saturate "
                     "at the same MJ per tonne of gadoid and tick."),
               ("-", "Right: zooplankton on phytoplankton, ceiling 1/h = "
                     "0.25. The per-pair attack rate sets 1/(a·h) = "
                     "20 t/km²; with type III half the ceiling is reached at "
                     "√20 ≈ 4.5 t/cell."),
               ("-", "Interference w lowers the whole curve when predators "
                     "crowd together - that removed the vertical bands in "
                     "the movement pattern.")],
        footnote="Energy is counted as eaten: herring with the reserve at the "
                 "reference fill (4,000 + 0.5 · 7,000 MJ/t).")

    d.formula(
        "Light-dependent attack rate",
        [("Per pair with dark_ratio ρ < 1", [
            r"f(\mathrm{elev}) &= \mathrm{logistic}\left(\dfrac{\mathrm{elev} - \theta}{2^\circ}\right) \qquad F(t) = \mathrm{mean\ over\ the\ tick's\ 10\ min\ steps}",
            r"m_{XY}(t) &= \dfrac{\rho + (1 - \rho)\,F(t)}{\mathrm{annual\ mean}\left(\rho + (1 - \rho)\,F\right)} \qquad a_{XY}(t) = a_{XY}\,m_{XY}(t)",
        ])],
        lead=[("p", "Visual predators see their prey much better in light. "
                    "The CAUSE is put in the model - the attack rate - while "
                    "the prey's response, diel vertical migration, is left "
                    "to the policies.")],
        tail=[("-", "The normalisation makes the library a the annual mean, "
                    "so the growth budget and the energy budget gate hold on "
                    "average."),
              ("-", "Herring → zooplankton: ρ = 0.1, θ = −8° (biting down to "
                    "0.001 lx, Batty et al. 1990). m: 0.66 in December, 1.42 "
                    "in June - the midnight sun only reaches −8.5°."),
              ("-", "Gadoids → herring: ρ = 0.6, θ = −5° (field stomach "
                    "data: herring in morning and evening stomachs). Other "
                    "pairs lack measurements and are light-independent."),
              ("-", "Elevation, not lux: visual range grows with log(light) "
                    "and saturates at twilight. Clouds are ignored.")],
        kicker="dark_ratio, light_threshold_deg")

    d.formula(
        "Step 4: energy after the action",
        [("Cost per action, before movement", [
            r"\mathrm{cost}_a(c) &= B_X(c)\,\pi_{X,a}(c)\,\mathrm{resting\_metabolism}_X\,Q_{10}^{(T(t) - t_{\mathrm{ref}})/10}\,\mathrm{factor}_a\,\left(1 + \sum_p \mathrm{energy\_factor}_p(c)\right)",
            r"\mathrm{factor}_a &\in \{\mathrm{resting\_cost},\ \mathrm{feeding\_cost},\ \mathrm{movement\_cost}\}",
            "",
            r"R_{\mathrm{rest}}(c) &= \max\left(0,\ R_X(c)\,\pi_{\mathrm{rest}} - \mathrm{cost}_{\mathrm{rest}}\right)",
            r"R_{\mathrm{eat}}(c) &= \max\left(0,\ R_X(c)\,\pi_{\mathrm{eat}} - \mathrm{cost}_{\mathrm{eat}}\right) + \mathrm{energy\ gain}(c)",
            r"R_{\mathrm{move}}(c) &= \max\left(0,\ R_X(c)\,\pi_{\mathrm{move}} - \mathrm{cost}_{\mathrm{move}}\right)",
        ])],
        lead=[("p", "The reserve is split over the three action groups in "
                    "proportion to how the biomass chose, each group pays "
                    "its cost, and the eating group receives the energy "
                    "gain from predation.")],
        tail=[("-", "Every action costs. Rest is the cheapest, not free."),
              ("-", "The Q10 factor is 1 for FGs without metabolism_q10 and "
                    "without a temperature block. Impacts act "
                    "multiplicatively: noise makes every action more "
                    "expensive instead of killing."),
              ("-", "After movement the reserve is clipped to "
                    "[0, B_X · ME_X] in every cell.")],
        kicker="Metabolism")

    d.formula(
        "Step 5: movement",
        [("Outflow, remaining mass and mixing", [
            r"\mathrm{out}_d(c) &= B_X(c)\,\pi_{X,d}(c)\,v_X\,\mathrm{mask}_d(c)",
            r"\mathrm{stay}(c) &= B_X(c) - \sum_d \mathrm{out}_d(c)",
            r"B_X^{+}(c) &= \mathrm{stay}(c) + \sum_d \mathrm{out}_d\left(\mathrm{neighbour}_d^{-1}(c)\right)",
            "",
            ("t", "R follows its biomass: if 20 % of the mass moves, 20 % of "
                  "that action group's reserve moves with it."),
            r"E_X &= \dfrac{R_X}{B_X} \qquad \mathrm{recomputed\ after\ mixing;}\ B = 0\ \Rightarrow\ E = 0",
        ])],
        lead=[("p", "Movement neither creates nor destroys biomass. It only "
                    "splits the cell's mass into outflows and remaining "
                    "mass.")],
        tail=[("-", "v_X is a distance per tick, not a probability: the "
                    "moving cohort's centre of mass advances v cells in one "
                    "tick. That is why it scales linearly with the tick "
                    "length."),
              ("-", "mask_d is accessibility and the grid edge. Blocked flow "
                    "stays in the source cell; the probabilities are never "
                    "renormalised afterwards."),
              ("-", "Movement is pure redistribution - all mortality has "
                    "already been taken in steps 2 and 3.")],
        kicker="Slice-assign, four directions")

    d.bullets(
        "Movement: thresholds, migration and extinction",
        ["An outflow whose DESTINATION cell would still end below the "
         "extinction threshold is cancelled and returned to the source cell.",
         "New in October (section 156): a cell that is then itself below "
         "the threshold but still sends biomass to a neighbour hands ALL its "
         "biomass and reserve to those neighbours, in proportion to the "
         "outflows. And an emigration across the edge that is smaller than "
         "the threshold stays home.",
         ("-", "The reason: residues and emigration dust were swept and "
               "booked as starvation - for porpoises half of the visible "
               "\"starvation loss\"."),
         "With --migration on the edge flow leaves the grid and is "
         "reintroduced, concentrated on the most heavily weighted edge "
         "cells, so that immigration is not spread out below the threshold.",
         "The last step of the tick zeroes cells where "
         "0 < B < extinction_threshold_factor · min_split_biomass and books "
         "the mass as starvation loss. The reason is numerical: below the "
         "threshold the values approach float32 subnormals.",
         "Continuous FGs - plankton, with min_split_biomass 0 - are affected "
         "by neither the argmax collapse nor the extinction threshold."],
        kicker="Discrete populations in a continuous field", size=15.5)

    d.bullets(
        "Currents",
        ["With --currents on an advection is added after movement: a "
         "cloud-like noise field scrolling east and south moves a share of "
         "the biomass every tick.",
         ("-", "The field is sampled procedurally from the tick counter and "
               "a world seed - it is not stored and does not consume the "
               "ecology's random numbers."),
         ("-", "--current-strength sets the maximum share per tick (default "
               "0.1), --current-period how fast the field scrolls, "
               "--current-scale the noise scale in cells."),
         "Which groups are carried is set by current_response in the "
         "library, not by whether they are decision makers: a swimmer that "
         "responds is advected on top of the displacement its policy "
         "already chose.",
         ("-", "Today: phytoplankton 1.0 and zooplankton 1.0, "
               "benthic_infauna 0.0, others 0.0 by default."),
         "The current shares movement's masks - blocked flow stays - and is "
         "conservative: no mass is created or lost."],
        kicker="current_response")

    d.formula(
        "Step 7: growth for decision makers - mass-balanced",
        [("Wish, cap and debit", [
            r"s_X(c) &= \dfrac{E_X(c)}{\mathrm{ME}_X} \qquad q_X(c) = s_X(c) - u_X",
            r"\mathrm{wish} &= B_X(c)\,\mathrm{MG}_X\,q_X(c) \qquad (q \geq 0)",
            r"E_{\mathrm{free}} &= \max\left(0,\ R_X(c) - u_X\,\mathrm{ME}_X\,B_X(c)\right)",
            r"\Delta B &= \min\left(\mathrm{wish},\ \dfrac{E_{\mathrm{free}}}{\mathrm{energy\_content}_X}\right)",
            r"R_X(c) &\leftarrow R_X(c) - \Delta B\,\mathrm{energy\_content}_X \qquad B_X(c) \leftarrow B_X(c) + \Delta B",
        ]),
         ("Starvation (q < 0) is a loss, not a purchase", [
            r"\Delta B &= B_X(c)\,\mathrm{starve\_rate}_X\,q_X(c) \qquad R\ \mathrm{scaled\ with}\ B",
        ])],
        lead=[("p", "Growth is GATED by satiation - B · MG · q is the wish - "
                    "but the wish is capped by the reserve above the "
                    "maintenance level, and that reserve is DEBITED. MG_X is "
                    "the derived growth_rate.")],
        tail=[("-", "Without the debit, new biomass carries energy that was "
                    "never taken from anything: measured at 198-344 % of the "
                    "mass actually eaten, against a thermodynamic ceiling of "
                    "28.9 %."),
              ("-", "Consequence: ME_X / energy_content_X becomes a hard cap "
                    "on the achievable growth_rate. For zooplankton the "
                    f"ratio is {f(zoo['max_energy_reserve'] / zoo['energy_content'])} "
                    f"against growth_rate {f(zoo['growth_rate'])}."),
              ("-", "--no-mass-balance restores the May edition's term. The "
                    "library is NOT calibrated for it.")],
        kicker="--mass-balance, on since September")

    d.figure(
        "Energy bookkeeping in one tick", draw_energy_flow,
        kicker="Where the mass comes from",
        items=[("-", "Egestion is (1 − assimilation_factor) of intake: 0.35 "
                     "for zooplankton on phytoplankton, 0.10 for porpoises "
                     "on pelagic fish."),
               ("-", "The reserve is the only buffer. Every action draws on "
                     "it; temperature and impacts raise the cost "
                     "multiplicatively."),
               ("-", "Only in the growth step does reserve become biomass, "
                     "at the price of energy_content MJ per tonne. A "
                     "population at or below the maintenance level therefore "
                     "cannot grow however much food there is.")],
        footnote="The split between respiration, growth and egestion per FG "
                 "can be read tick by tick from the loss accounting and the "
                 "probe tool tools/probes/fg_year_budget.py.")

    d.bullets(
        "Losses: four channels",
        ["The tick books four loss channels separately per FG: predation, "
         "starvation, impacts and natural mortality. "
         "lib/environments/ecosystem_env/loss_accounting.py is the single "
         "place that turns them into shares; the training log, inference, "
         "the live viewer and plots.html use the same function.",
         "Natural mortality (M1) is density-independent and optional: it "
         "needs --mortality on and is scaled by --mortality_multiplier "
         "(keep = 1 - FACTOR · natural_mortality). It is taken from both "
         "biomass and reserve, before the growth term. Before section 132 "
         "it did not show in the loss accounting at all.",
         "Starvation is catabolism: below the maintenance level biomass "
         "shrinks by starve_rate · |q|, and the reserve is scaled by the "
         "same ratio.",
         "Mass zeroed by the extinction threshold is booked as starvation, "
         "and the number of events is counted separately.",
         "Mortality through impacts is modelled with impact maps: fishing, "
         "hunting, chemicals or mechanical mortality."],
        kicker="Predation, starvation, impacts, natural", size=15.5)

    d.formula(
        "Step 7: growth for non decision makers",
        [("Light-limited logistic growth towards carrying capacity", [
            r"r(t) &= \mathrm{growth\_rate}\,\dfrac{P(t)}{P_{\mathrm{ref}}} \qquad P(t) = \mathrm{mean}_{\mathrm{tick},\,z}\ \tanh\left(\dfrac{I_0(t)\,e^{-K_d z}}{I_k}\right)",
            r"\Delta B(c) &= r(t)\,B(c)\,\left(1 - \dfrac{B(c)}{\mathrm{CC}(c)}\right) + \mathrm{seed\_rate}\,\mathrm{CC}(c)\,u \qquad u = 10^{U(-1,\,1)}",
            r"B(c,t+1) &= \mathrm{clip}\left(B(c) + \Delta B(c),\ 0,\ \mathrm{CC}(c)\right)",
        ])],
        lead=[("p", "Applies to functional groups whose energy is not "
                    "modelled. Grazing has already been taken in step 3. The "
                    "old sine term is removed; the season comes from the "
                    "calendar.")],
        tail=[("-", "I_0: clear-sky irradiance (Haurwitz) × 0.45 PAR × "
                    "4.57 µmol/J × monthly cloud transmission (SMHI "
                    "Nordkoster). K_d 0.29/m; mixed layer 6-17 m (SHARK "
                    "A17); I_k 111-298 µmol/m²/s per month (SMHI 14C, "
                    "Anholt E). P = 0 at night."),
              ("-", f"growth_rate {f(phy['growth_rate'])} per 6 h is the "
                    "highest in situ rate in April dilution experiments minus "
                    "microzooplankton grazing; P_ref is day 105, so the "
                    "factor is 1 in mid April and 0.63 as an annual mean."),
              ("-", "The benthic community is not light-limited: growth_rate "
                    f"{f(ben['growth_rate'])}, plus seed_rate as a "
                    "recolonisation floor.")],
        kicker="light_saturation, light_reference_day")

    d.bullets(
        "Bookkeeping summary",
        ["New biomass = biomass after impacts, predation and movement "
         "+ growth - starvation - natural mortality.",
         "Biomass can only increase when the energy reserve allows it. For "
         "non decision makers the carrying capacity is the ceiling and light "
         "the rate; for decision makers the reserve above the maintenance "
         "level is the ceiling.",
         "Reserve energy always follows its biomass: whenever biomass is "
         "scaled down the reserve is scaled by the same ratio, and whenever "
         "mass moves the corresponding share of the reserve moves with it. "
         "For herring the reserve also follows into the predator.",
         "What is NOT conserved: natural mortality, impacts and the "
         "extinction threshold remove mass from the system, and migration "
         "moves mass across the grid boundary. Predation and movement "
         "conserve it.",
         ("-", "With --migration off and no mortality, the source tracking's "
               "mass balance is an identity, which is what makes the local "
               "reward exact.")],
        kicker="What is conserved and what is not")

    # ---------------------------------------------------------------- 7
    d.section("6", "Training with ARS",
              "How the behaviour models are produced, what they optimise "
              "and what they should not be mistaken for.")

    d.bullets(
        "Training: goal and perspective",
        ["The goal of training is not to find ecologically optimal "
         "policies, but plausible, stable and testable policy networks that "
         "can be used in scenario analysis.",
         "The model is Eulerian. There is no individual or distinct "
         "population to follow over several steps - biomass splits, moves "
         "in different directions and mixes with other biomass. Think of "
         "people on a town square.",
         "So no long-term individual return is optimised, but how well a "
         "FIELD does over a rollout.",
         "What runs is ARS: gradient-free and parallel, on CPU (train.py) "
         "or in tensor form on GPU (train_gpu.py).",
         "The rollout is 15 ticks by default and 100-200 ticks in the "
         "profiles. With the calendar on, training in practice uses "
         "one-year rollouts, 1,460 ticks, so that every rollout holds every "
         "season."],
        kicker="What ARS is meant to achieve")

    d.formula(
        "Fitness",
        [("Default: total energy, integrated over the rollout", [
            r"E_{\mathrm{tot}}(t) &= \sum_c \left(B_X(c,t)\,\mathrm{energy\_content}_X + R_X(c,t)\right)",
            r"\mathrm{Fitness}_X &= \dfrac{1}{H}\sum_{t=1}^{H}\,\log\dfrac{E_{\mathrm{tot}}(t) + \varepsilon}{E_{\mathrm{tot}}(0) + \varepsilon}",
            r"\varepsilon &= \max\left(10^{-6}\,E_{\mathrm{tot}}(0),\ 10^{-9}\right)",
            ("t", "H = --n_eval_ticks"),
        ]),
         ("Kept as --legacyreward: the May edition's form", [
            r"\mathrm{Fitness}_X &= \alpha\,\log\dfrac{B_H}{B_0} + \beta\,\log\dfrac{R_H}{R_0} + \kappa\,\dfrac{t_{\mathrm{survived}}}{H}",
        ])],
        tail=[("-", "Counting biomass and reserve in the SAME currency removes "
                    "the weighting between α and β: a tonne of biomass is "
                    "worth exactly its energy content."),
              ("-", "The integral form (--integral_reward, on by default) "
                    "scores a policy lower if it crashes its prey mid-rollout "
                    "and recovers at the end."),
              ("-", "The logarithm makes the reward scale-invariant and "
                    "symmetric: halving costs exactly what doubling is "
                    "worth.")],
        kicker="What a rollout measures")

    d.formula(
        "Per-cell local reward",
        [("--local_reward: the same question, asked per cell", [
            r"A(c,t) &= B(c,t)\,\mathrm{energy\_content} + R(c,t)",
            r"B(c,t+1) &= \sum_d \mathrm{share}_d(c)\,Q\left(\mathrm{dest}(c,d),\ t+1\right)",
            r"\mathrm{reward}(c) &= \mathrm{clip}\left(\dfrac{B(c,t+1)}{A(c,t)},\ 0.2,\ 5.0\right)",
            r"\mathrm{Fitness}_X &= \mathrm{mean}_t\ \dfrac{1}{N_{\mathrm{cells}}}\sum_c w_c\,\log\,\mathrm{reward}(c)",
        ])],
        lead=[("p", "The global reward is dominated by the large cells: a "
                    "policy that handles a thousand-tonne cell well and a "
                    "kilogram badly gets the same score as the reverse.")],
        tail=[("-", "share_d(c) is the share of the destination cell's moved "
                    "biomass that came from c - an identity, not an "
                    "approximation, because everything after movement is "
                    "cell-wise multiplicative."),
              ("-", "Use --local_reward_norm grid with the log metric: sum "
                    "rewards spreading thin and mean rewards killing the "
                    "worst cells."),
              ("-", "w_c = A_cᶿ. θ = 0 weights every occupied cell equally, "
                    "θ = 1 recovers the global energy reward.")],
        kicker="Scale-invariant per cell")

    d.formula(
        "The ARS update",
        [("Per iteration, per decision maker with weights θ", [
            r"\delta_1, \ldots, \delta_N &\sim \mathcal{N}(0,\ I)",
            r"F_i^{+} &= \mathrm{fitness}(\theta + \sigma\,\delta_i) \qquad F_i^{-} = \mathrm{fitness}(\theta - \sigma\,\delta_i)",
            ("t", "N = --n_deltas. Same world, seed and calendar start for + "
                  "and − (CRN)."),
            r"\mathrm{top}\ b &= \mathrm{the}\ b\ \mathrm{pairs\ with\ the\ highest}\ \max(F_i^{+},\ F_i^{-}) \qquad b = N/2",
            r"\sigma_F &= \mathrm{standard\ deviation\ over\ the\ selected\ pairs'\ fitness}",
            r"\theta &\leftarrow \theta + \dfrac{\eta}{b\,\sigma_F}\sum_{i \in \mathrm{top}\,b}\left(F_i^{+} - F_i^{-}\right)\delta_i",
        ])],
        tail=[("-", "ARS-V2: observations are normalised with a running mean "
                    "and running variance per FG and clipped to ±10."),
              ("-", "CRN - common random numbers - make the difference "
                    "F⁺ − F⁻ meaningful with few rollouts: both halves face "
                    "identical initial fields and the same season."),
              ("-", "Default: η = 0.03, σ = 0.1, N = 10, one world per "
                    "delta. Normalising by σ_F makes the step length "
                    "independent of the reward scale: |step| / probe radius "
                    "≈ (η/σ) · k/√b.")],
        kicker="Augmented Random Search")

    d.bullets(
        "Co-evolution",
        ["The default is co-evolution: ALL decision makers are perturbed "
         "together within the same delta pair and evaluated in ONE shared "
         "rollout. Each FG gets its own fitness from the same simulation, "
         "and the ARS update is then done independently per FG.",
         ("-", "The reason is that a prey policy and a predator policy are "
               "only meaningful against each other. Trained separately, each "
               "optimises against a counterpart that no longer exists when "
               "both are done."),
         "--no_coevolution gives the alternating schedule of the May "
         "edition: one FG at a time, the others frozen.",
         "Policies not trained in the iteration - and non decision makers - "
         "run with their latest weights as a fixed background.",
         "One generation = --iter-per-gen iterations. After every generation "
         "checkpoints are saved and a deterministic probe rollout logs "
         "biomass per FG to results/<run>/biomass.jsonl."],
        kicker="All species at once")

    d.bullets(
        "GPU training: train_gpu.py",
        ["lib/gpu/ is a tensor version of the same tick: every world, delta "
         "pair and sign runs at once on the graphics card, in blocks of "
         "ticks replayed as CUDA graphs. Parity tests compare every step "
         "with the CPU engine.",
         "Current runs (fix53): 16 delta pairs × 2 worlds × 1,460 ticks, "
         "60 × 60, mortality, migration, currents, daylight and temperature "
         "on.",
         "The calendar start is drawn per WORLD and shared by every pair "
         "(section 150). When each pair had its own season, the season set "
         "the pairs' reward level: top-b selection was nearly random and the "
         "step shrank to a third.",
         "Step length: at η = 0.1 and σ = 0.05 the update became "
         "noise-dominated - the weight norm grew like a random walk "
         "(29-35 → 64-75) and the policies sharpened more than the reward "
         "supports. Keep η/σ ≲ 0.6; σ is the knob for more exploration.",
         ("-", "The viewer's probe re-reads fg_library.yaml at every probe, "
               "but the trainer keeps the library it started with.")],
        kicker="lib/gpu/", size=15)

    d.table(
        "Profiles and shaping terms",
        ["Setting", "Default on train.py", "sanity", "info", "deep"],
        [["generations / iter-per-gen", "inf / 20", "10 / 15", "10 / 20",
          "80 / 20"],
         ["n_deltas", "10", "16", "16", "20"],
         ["n_eval_ticks", "15", "100", "150", "200"],
         ["rollouts_per_delta", "1", "3", "3", "3"],
         ["entropy_coef", "0", "0", "0", "0"],
         ["argmax_penalty", "0", "0", "0", "0"],
         ["softmax temperature", "1.0", "1.0", "1.0", "1.0"]],
        [26, 22, 14, 14, 14], size=12,
        align=["left", "right", "right", "right", "right"],
        lead="The entropy bonus, the argmax penalty and temperature annealing "
             "are shaping terms that act on the action distribution rather "
             "than on the biology. They remain as flags but are off by "
             "default, with or without a profile, on both trainers.",
        footnote="--profile sanity | info | deep. Explicit flags on the "
                 "command line always win over the profile's values.")

    d.figure(
        "Training worlds", draw_spawn_examples,
        kicker="Four spawn strategies",
        lead="To avoid overfitting to a single map, the policies are trained "
             "in many randomised worlds. Every FG has its own spawn strategy "
             "in the library and a range for initial biomass in the project "
             "file.",
        items=[("-", "uniform: independent random weight per cell, no "
                     "spatial structure. perlin: natural-looking noise "
                     "(phytoplankton, trawling intensity)."),
               ("-", "colony: n Gaussian colonies with jittered amplitude "
                     "(fish, seals, seabirds, noise sources)."),
               ("-", "env_driven: the weight is built from other fields - "
                     "zooplankton follows phytoplankton, porpoises follow "
                     "pelagic fish.")],
        footnote="Initial biomass is drawn per world from "
                 "[initial_biomass_min, initial_biomass_max] in mareld2.yaml, "
                 "the start day from the calendar. In TRAINING mode the "
                 "energy level is drawn per cell around the maintenance "
                 "level; the inference world starts deterministically at "
                 "0.7 · ME_X.")

    d.bullets(
        "Checkpoints",
        ["A checkpoint is results/<run>/policy_<fg>.pth and contains "
         "{'state_dict': ..., 'obs_stats': {...}} - the weights AND the "
         "running statistics of the observation normalisation.",
         "--resume loads them and continues; inference.py runs them against "
         "a deterministic world.",
         "The following changes invalidate existing checkpoints - the shape "
         "or meaning of the observation changes:",
         ("-", "the Observability matrix (e.g. gadoids seeing zooplankton, "
               "section 141)"),
         ("-", "which impact maps are observable and active"),
         ("-", "the FG set in the project and the grid layout"),
         ("-", "the daylight calendar on or off (the light channel)"),
         "Temperature and a changed calibration do not formally invalidate "
         "them, but a policy trained in a substantially different world "
         "should be retrained before it is cited. Copy a checkpoint before "
         "comparing - a running training overwrites policy_*.pth."],
        kicker="What is saved and what breaks it", size=15.5)

    # ---------------------------------------------------------------- 8
    d.section("7", "Diagnostics and viability",
              "Tools that test the mechanics without ARS, and what trained "
              "policies do over ten years.")

    d.table(
        "The viability rig",
        ["#", "Condition", "Formula", "Default"],
        [["1", "Survival", "B_f(t) > 0 for all t ≤ T", "-"],
         ["2", "Floor", "eq_f ≥ floor · B_f(0)", "floor = 0.10"],
         ["3", "Ceiling", "eq_f ≤ ceiling · B_f(0)", "ceiling = 10"],
         ["4", "Stationarity",
          "max(eq_f/prev_f, prev_f/eq_f) ≤ max_drift", "max_drift = 2.0"]],
        [6, 26, 44, 24], size=12.5, kicker="Frozen behaviour",
        lead="The rig runs the world with FROZEN, hand-coded behaviour. That "
             "makes it fast and deterministic, and therefore good for "
             "measuring a parameter or mechanism change in isolation. eq_f is "
             "the mean biomass over the horizon's last window.",
        footnote="Horizon 5 years (1,460 ticks at 6 h), 3 seeds, last window "
                 "10 % of the horizon; the default corner is --spawn "
                 "colocated --behaviour greedy. The verdict is about THAT "
                 "behaviour: everyone eats as much as possible, which grazes "
                 "down the phytoplankton and drags the chain above it along. "
                 "Trained policies do not do that. python3 "
                 "tools/viability.py, ~4 min.")

    d.bullets(
        "The energy budget gate",
        ["Can every decision maker close its energy budget over its RATION? "
         "The question is answered without a rollout, in seconds: "
         "python3 tools/probes/budget_gate.py",
         ("p", "ration = the ceiling 1/h at full appetite h(u_X) = 1   "
               "[t prey / t predator / tick]"),
         ("p", "cost   = resting_metabolism · feeding_cost   [MJ / t / "
               "tick]"),
         ("p", "need   = cost / ration ≤ the ration's weighted energy "
               "quality"),
         "The gate is evaluated over the whole menu, not per pair: the "
         "engine sums intake over all prey and gives the predator ONE energy "
         "gain. It reports \"min share\" - the minimum share of the best "
         "prey in the diet.",
         ("-", "Today all four active predators close on any diet. "
               "Porpoises: break-even ration on herring 5.3 % of body mass "
               "per day; in the middle of the Kastelein window (6.75 %) at "
               "least 33 % herring is needed - the junk-food hypothesis."),
         ("-", "The gate works with the library's resting_metabolism, i.e. "
               "at t_ref - for herring a summer cost.")],
        kicker="Does the budget close?", size=15)

    d.table(
        "Ten years with trained policies",
        ["Functional group", "Year 1", "Year 10, M1 0.683",
         "Year 10, M1 0.415", "Course with the current library"],
        [["pelagic_fish (herring)", "12,565", "15,938", "6,232",
          "Halves; biomass peak Oct-Nov"],
         ["gadoids", "2,519", "1,430", "3,292",
          "+3 to +9 %/yr, peak in year 8, then decline"],
         ["porpoises", "11.5", "3.99", "5.9",
          "−6 %/yr (was −12 %/yr)"],
         ["benthic_infauna", "18,603", "-", "15,856",
          "Rises to year 3-4, then grazed down"],
         ["zooplankton", "~24,000", "~22,800", "-",
          "Stationary with an annual cycle of ~9×"],
         ["phytoplankton", "~43,400", "~43,400", "-",
          "Stationary, spring bloom in March"]],
        [21, 10, 15, 15, 39], size=11.5,
        align=["left", "right", "right", "right", "left"],
        kicker="The null scenario, 14,600 ticks",
        lead="Frozen fix53 checkpoint (trained at M1 0.683), deterministic "
             "60 × 60 probe world with mortality, migration, currents, "
             "daylight and temperature, all impacts muted. Annual means in "
             "tonnes.",
        footnote="The M1 0.683 column is the end of year 10 (section 156.7), "
                 "M1 0.415 annual means (157.3); plankton from section 156.2 "
                 "(a neighbouring checkpoint). A second global seed stayed "
                 "within 2 %: the cycle is driven by the calendar, not by "
                 "chaos. The policies should be retrained at M1 0.415 before "
                 "the new balance is read.")

    d.table(
        "Realised versus budgeted predation",
        ["Pair", "Library M2 (1/yr)", "Realised, year 1 → 10", "Reading"],
        [["gadoids → gadoids", "0.217", "~0.001",
          "Growth paid for a loss that is never taken"],
         ["gadoids → pelagic_fish", "0.30", "0.19 → 0.14",
          "Herring under-cropped, they grow until food brakes"],
         ["gadoids → benthic", "0.0048", "0.33 → 0.17",
          "Different base (K 10 vs ~100 t/km²); diet ~2× Ecopath"],
         ["zooplankton → phyto", "22.6", "30.5", "Somewhat higher"],
         ["pelagic_fish → zoo", "5.24", "3.2 → 5.2", "Approaching the budget"],
         ["porpoises → gadoids", "0.056", "0.087 → 0.020",
          "Follows the porpoise decline"],
         ["porpoises → pelagic", "0.0117", "0.006 → 0.003",
          "Follows the porpoise decline"]],
        [24, 18, 20, 38], size=11.5,
        align=["left", "right", "right", "left"],
        kicker="Section 156",
        lead="The growth budget assumes every predator takes its M2. In the "
             "ten-year run most pairs roughly do - but not all.",
        footnote="Realised M2 = annual removal / mean prey biomass. The "
                 "figures are for gadoid M1 0.683 (before section 157).")

    d.bullets(
        "What the rig is for",
        ["The rig is a measuring instrument for the mechanics, not a verdict "
         "on the world. With frozen behaviour and a fixed seed it is "
         "deterministic, so a library change can be measured in isolation "
         "without training noise.",
         "It catches faults no policy can rescue: a group that shrinks even "
         "with trained behaviour - today the porpoises - points to a "
         "structural problem. There the rig and the energy budget gate "
         "point the same way.",
         "It does not say that a world is unusable. The premise that a "
         "failing world cannot be rescued by training does not hold: the "
         "trained policies do not graze down the phytoplankton the way the "
         "greedy behaviour does, and the zooplankton learns restraint.",
         "Viability should therefore be judged with the policies actually "
         "used: the same criteria on inference data from several seeds, and "
         "a stationarity requirement that tolerates natural cycles instead "
         "of comparing two windows. That is not built yet."],
        kicker="Diagnosis, not verdict")

    # ---------------------------------------------------------------- 9
    d.section("8", "Strengths, weaknesses and suggested improvements",
              "An overall assessment of the model as it stands in October "
              "2026: what it does well, where it falls short and what "
              "should be done next.")

    d.bullets(
        "Strengths: mechanics and method",
        ["Mass and energy are booked in a closed loop. Predation, movement "
         "and currents conserve mass; growth is paid from the reserve; "
         "every loss is booked under one of four causes. Per-cell source "
         "tracking is an identity.",
         "Parameters are traceable. growth_rate is derived from r_max, M1 "
         "and Ecopath M2; light climate, temperature and P-I curves come "
         "from SMHI data for the site; every library value has a comment "
         "and a section in the resume.",
         "One library, one calibration: the tick length 1-6 h is converted "
         "in memory and never written back.",
         "Behaviour is learned, not hacked. Shaping terms are off; problems "
         "are solved in the biology or in perception.",
         "Reproducible and tested: CRN, deterministic probe worlds, per-step "
         "CPU/GPU parity and a test suite of more than 1,000 tests.",
         "Fast diagnostics in three steps: the energy budget gate in "
         "seconds, the frozen rig in minutes, probe tools for annual and "
         "monthly budgets with trained policies."],
        kicker="What holds up", size=15)

    d.table(
        "Strengths: behaviour that resembles nature",
        ["Phenomenon", "In the model (trained policies, years 2-10)",
         "Comparison"],
        [["Spring bloom", "Phytoplankton 0.75 in January → 1.12 in March",
          "Diatom bloom Feb-Mar in the Skagerrak"],
         ["Zooplankton annual cycle",
          "Minimum in January, maximum May-June (~9×)",
          "Calanus-like, 2-3 months after the bloom"],
         ["Winter rest", "Zooplankton rests 0.78-0.85 in winter, eats most "
          "in Feb-Mar", "Diapause-like (Calanus at depth)"],
         ["Diel vertical migration", "Zooplankton rests day/night "
          "0.93/0.60 (winter)", "DVM where real night exists"],
         ["Visual planktivore", "Herring rest 0.11 day / 0.75 night "
          "(winter)", "Herring feed in light, filter at night"],
         ["Lipid cycle", "Herring s 0.23 in February → 0.87 in June",
          "Lean Q1, fat Q3"],
         ["Seasonal diet switch", "Gadoids: summer herring + zooplankton, "
          "winter benthos", "Direction plausible, share elevated"],
         ["Continuous foraging", "Porpoises eat 97-100 % of the time, Q/B "
          "22/yr", "Near-continuous (Wisniewska et al. 2016)"],
         ["Grazing restraint", "Zooplankton rests ~0.8 and does not graze "
          "the phytoplankton down", "Emergent at the population level"]],
        [21, 43, 32], size=10.8, kicker="Emergent, not programmed",
        footnote="From sections 139 and 156, detrended monthly patterns. None "
                 "of this is written into the rules - it follows from the "
                 "calendar, the energy budget and what the policies have "
                 "learned.")

    d.bullets(
        "Weaknesses: biology and structure",
        ["The growth budget is paid in full only at s = 1, where hunger "
         "closes intake. Real s sits at 0.6-0.75, so gadoids and porpoises "
         "realise 40-50 % of the designed production - for porpoises it "
         "does not even cover M1.",
         "No nutrients: phytoplankton sits on a plateau April-October (1.6× "
         "annual range against 5-10× in chlorophyll in nature). No summer "
         "low, no autumn bloom.",
         "Top predators do not track their prey: porpoises overlap their "
         "prey at 0.15-0.22 and move 0-3 % of the time; gadoids move 1 %. "
         "With filled ceilings a denser cell gives only ~2 % more intake.",
         "Porpoises decline by 6-12 % per year over ten years. The diet is "
         "fitted (attack ratio 13), not emergent. Cannibalism is budgeted "
         "but does not happen.",
         "Depth is collapsed to 2D; no spawning cost, no diapause as a "
         "state. Several values rest on a single source (the phytoplankton "
         "rate, I_k from the Kattegat; the dark ratios are estimates).",
         "Seals, seabirds and every impact are muted: no scenario has yet "
         "been run with the current library."],
        kicker="Where the model falls short", size=14.5)

    d.bullets(
        "Weaknesses: training and evaluation",
        ["Policies are trained on one-year rollouts. Everything after year 1 "
         "of a ten-year run lies outside the training distribution, and the "
         "world is not at equilibrium after ten years.",
         "The ARS step can become noise-dominated: at too high η/σ the "
         "weight norm grows like a random walk and the policies sharpen more "
         "than the reward supports. There is no norm control.",
         "Small groups have noisy rewards: porpoise σ_F was 4-10× the other "
         "DMs with the 250 kg unit and is still ~4× gadoids and zooplankton. "
         "Below min_split_biomass argmax hides the real distribution, and "
         "fragments smaller than one animal die through the extinction "
         "sweep.",
         "Viability is judged with one or a few seeds, and there is no "
         "cycle-tolerant stationarity requirement; the rig that has a "
         "criterion runs frozen behaviour.",
         "Every library change requires retraining before numbers are "
         "cited. The gadoid M1 changed on 12 October - after fix53 was "
         "trained.",
         "The network is small (2 × 30) and has no memory; the policies see "
         "the light but not the date."],
        kicker="Where the method falls short", size=14.5)

    d.table(
        "Suggested improvements: the model",
        ["#", "Suggestion", "Motivation", "Section"],
        [["1", "Reference fill < 1 in the growth budget (e.g. s ≈ 0.7)",
          "Today the budget is paid only at s = 1, which is never reached",
          "156.4"],
         ["2", "Nutrient pool for phytoplankton",
          "Summer low and autumn bloom are missing", "138.7, 156.4"],
         ["3", "Functional response for top predators (a or h per pair) so "
          "that density pays", "Gadoids and porpoises stand still",
          "154.6, 156.4"],
         ["4", "Cannibalism: lower M2 or give it a mechanism",
          "Growth paid for a loss that is not taken", "156.4, 157.2"],
         ["5", "Spawning cost for herring, winter diapause for zooplankton",
          "Lipid minimum in the wrong month; winter survival",
          "143.6, 138.7"],
         ["6", "SDA in assimilation_factor instead of a per-tick fee",
          "The cost scales with intake, not with time", "145.5"],
         ["7", "Temperature-scaled attack rate (E ~ 0.4 eV)",
          "Step 2 after Q10 on metabolism", "143.2"],
         ["8", "Benthic community with K ~100 t/km² as in the literature",
          "Benthos dominance in the gadoid diet", "136, 156.10"]],
        [4, 42, 38, 14], size=11, kicker="Biology and mechanics",
        footnote="The order is a judgement of benefit against cost. Every "
                 "suggestion is first measured with the probe tools and "
                 "trained policies before the library is changed.")

    d.table(
        "Suggested improvements: training and evaluation",
        ["#", "Suggestion", "Motivation", "Section"],
        [["1", "Retrain on the current library and read the balance with "
          "several seeds", "Every checkpoint was trained before the M1 "
          "change", "157"],
         ["2", "Cycle-tolerant viability criterion on inference data",
          "The two-window test fails ordinary predator-prey cycles", "126"],
         ["3", "Norm control in ARS (weight decay or fixed |θ|); σ as the "
          "exploration knob", "Random walk in the weights at high η/σ",
          "154.8-10"],
         ["4", "Sampling instead of argmax below min_split_biomass; handle "
          "fragments smaller than one animal", "Hidden distribution; "
          "porpoises die through the sweep", "152.6, 156.9"],
         ["5", "Evaluation longer than the training episode: multi-year "
          "rollouts in the probe and the reward", "Ten-year behaviour lies "
          "outside the training distribution", "156.2"],
         ["6", "Switch on seals, seabirds and impact scenarios only once the "
          "null scenario is viable", "A comparison needs a self-sustaining "
          "baseline", "-"]],
        [4, 44, 38, 12], size=11, kicker="Method",
        footnote="None of the suggestions needs shaping terms in the reward; "
                 "they concern the biology, the optimiser or the evaluation.")

    # --------------------------------------------------------------- 10
    d.section("9", "Scenarios and runs",
              "What a scenario is, how it is run and what separates this "
              "edition from the May edition.")

    d.bullets(
        "Scenario",
        ["A scenario is a set of maps for a given set of functional groups "
         "and a given grid.",
         "Scenarios are analysed with the update rules and the trained "
         "behaviour models, and are always read as a COMPARISON: null "
         "alternative against project alternative, with the same seeds and "
         "the same policies.",
         "In Mareld, impacts are defined by three impact variables in the "
         "project file - pelagic_trawling (0-1,000), windfarm_noise "
         "(0-140 dB, observable to the policies) and rotor (0-1). All three "
         "stay muted until a scenario switches them on.",
         "Maps can either be spawned procedurally (perlin, colony) or loaded: "
         "the inference block in mareld2.yaml points rotor at "
         "vindparker.npz, built from the wind farms' real geometry.",
         "The difference between two scenarios is to be reported as the "
         "relative change per functional group, under stated assumptions - "
         "never as an absolute forecast."],
        kicker="Definition and practice")

    d.formula(
        "Runs",
        [("Training, inference and gates", """
python3 train.py --project mareld2.yaml --run-name <name> --profile info
python3 train_gpu.py --project mareld2.yaml --run-name <name> ...  # GPU
python3 train.py --project mareld2.yaml --species pelagic_fish \\
        --generations 1 --iter-per-gen 1 --workers 1        # smoke test
python3 inference.py --project mareld2.yaml --run-name <name>
python3 tools/biomass_html.py results/<name>                # plots.html
python3 tools/probes/budget_gate.py        # seconds, no rollout
python3 tools/probes/daylight_table.py     # the calendar month by month
python3 tools/viability.py                 # ~4 min, frozen behaviour
python3 -m pytest tests/ -q
""")],
        tail=[("-", "Important flags: --tick-length 1-6, --mortality on, "
                    "--mortality_multiplier, --migration on, --currents on, "
                    "--local_reward, --no-mass-balance, --policynetwork, "
                    "--resume, --visual, --rnd_baseline."),
              ("-", "The live viewer shows the month, the day length and a "
                    "calendar band under the inference plot; --rnd_baseline "
                    "draws random reference curves.")],
        kicker="What is run")

    d.table(
        "Deliberate deviations from the May edition",
        ["What", "Why"],
        [["Growth is mass-balanced and growth_rate is derived",
          "The documented term B·(1 + MG·q) creates mass without debiting "
          "energy - measured at 198-344 % of what was eaten, against a "
          "thermodynamic ceiling of 28.9 %. The rate is derived so that net "
          "growth is the literature's."],
         ["Hunger is continuous, not a mask",
          "The May edition had both the mask at s = 0.8 and the factor "
          "h = max(0, 1 − s/0.8). Today appetite is full up to the "
          "maintenance level and falls linearly after it: eating is set by "
          "food and gut capacity, not by the reserve."],
         ["Fitness counts biomass and reserve in one currency",
          "α and β become arbitrary weights between two quantities that "
          "already have an exchange rate: energy_content."],
         ["Rollouts are 15 ticks to one year, not 2",
          "The integral reward reads every tick, and the calendar requires "
          "a rollout to hold the seasons."],
         ["The season comes from a calendar",
          "The May edition had no season. Light and temperature for the site "
          "now drive attack, primary production and metabolism."],
         ["The memory extension N(c,t-1) and the size ladder 3×3 → 60×60 "
          "are not included", "The only state between ticks is last tick's "
          "hidden share; variation in training comes from spawning, initial "
          "biomass and the calendar start."]],
        [27, 73], size=11, kicker="The document against the code")

    d.bullets(
        "Open questions",
        ["Porpoises decline by 6-12 % per year even with trained policies. "
         "The largest remaining causes are the growth budget at s = 1, "
         "winter catabolism and fragments smaller than one animal.",
         "With the new gadoid M1 (0.415) the gadoids overshoot and halve "
         "the herring within ten years under the old policies. Is it a slow "
         "cycle or a new level? That needs retraining.",
         "Zooplankton falls to ~13 % of its initial biomass in January and "
         "recovers by May. Diapause as a state is a candidate if that does "
         "not hold with trained policies.",
         "Should the growth budget refer to a realistic satiation (s ≈ 0.7) "
         "instead of s = 1? That changes every DM's growth_rate.",
         "Phytoplankton lacks nutrient limitation; the summer level stays "
         "close to the April level.",
         ("p", "The full history - every calibration, A/B test and open "
               "problem - is in mareld_resume.txt, sections 0-158.")],
        kicker="What remains", size=15.5)

    d.section("", "Thank you",
              "Questions, objections and corrections belong in "
              "mareld_resume.txt as much as in the code.")


# ----------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", default="Strategy_Oct_eng.pdf")
    parser.add_argument("--png", default=None,
                        help="Also write one PNG per slide into this "
                             "directory (for proofreading).")
    args = parser.parse_args(argv)

    setup()
    if args.png:
        os.makedirs(args.png, exist_ok=True)
    with PdfPages(args.out) as pdf:
        deck = EnglishDeck(pdf, args.png)
        build(deck)
        info = pdf.infodict()
        info["Title"] = "Strategy - for building ecosystem simulators"
        info["Author"] = "Mareld / Poseidon Nord"
        info["Subject"] = ("Updated strategy, October 2026 - describes "
                           "the mechanics the simulator runs today")
    print(f"{args.out}: {deck.n} slides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
