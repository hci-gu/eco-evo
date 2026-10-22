"""Generate ``Strategy_Oct2.pdf`` - the English edition of the second
October 2026 strategy deck (``Strategi_okt2.pdf``).

Layout, diagrams, drawing code and the library tables are shared with
the Swedish generator, ``tools/make_strategi_deck2.py``; this module only
supplies English text: the diagram labels (``LABELS_EN``, which must
carry exactly the Swedish keys), the footer and section label, the number
format, and the slide content in ``build``. Keep the two ``build``
functions in step when the mechanics move.

Usage:
    python3 tools/make_strategy_deck2_en.py [-o Strategy_Oct2.pdf] [--png DIR]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_strategi_deck2 as sv  # noqa: E402
from make_strategi_deck2 import (  # noqa: E402
    DM_ORDER, Deck, PdfPages, budget, budget_rows, capacity_rate,
    dm_energy_rows, dm_growth_rows, draw_capacity, draw_energy_flow,
    draw_fishing_maps, draw_food_web, draw_functional_response,
    draw_hunger_gate, draw_ideal_reserve, draw_observation, draw_policy_net,
    draw_spawn_examples, draw_tick_pipeline, draw_water_column, fmt,
    muted_groups, observability_rows, observation_width, pair_value_rows,
    regional_f, species)


class EnglishDeck(Deck):
    FOOTER = "Strategy - Mareld / Poseidon Nord - October 2026, edition 2"
    PART = "Part"


LABELS_EN = {
    "pipeline": [
        ("1  Decision", "observation → policy → masked softmax → whole units"),
        ("2  Impact mortality", "impact_table × map, every FG (fishing)"),
        ("3  Predation", "Holling II/III, interference, light, Q10 on intake"),
        ("4  Energy costs", "rest / eat / move × Q10(T); herring grows here"),
        ("5  Movement", "N/E/S/W × speed, energy moves along"),
        ("6  Currents", "current_response; the nutrient pool moves with water"),
        ("7  Water column", "layer exchange, supply, grazing by non-deciders"),
        ("8  Growth and losses", "M1, regional K, mass-balanced growth"),
        ("9  Masks", "accessibility, extinction threshold"),
    ],
    "web_nodes": {
        "phytoplankton": "Phytoplankton",
        "subsurface_phytoplankton": "Subsurface phyto.",
        "microzooplankton": "Microzooplankton",
        "benthic_community": "Benthic community",
        "zooplankton": "Zooplankton", "pelagic_fish": "Pelagic fish",
        "gadoids": "Gadoids", "porpoises": "Porpoises", "seals": "Seals",
        "seabirds": "Seabirds",
    },
    "web_legend": "green = non-decision maker (nutrient pool, grazing, logistic)\n"
                  "blue = decision maker (policy network and energy budget)\n"
                  "dashed = muted in mareld2.yaml",
    "web_self": "cannibalism (unchecked)",
    "fr_herr": "pelagic fish  type III, h = {h}",
    "fr_herr_w": "pelagic fish  w = 1, B_pred = 0.5",
    "fr_benth": "benthic community  type II, h = {h}",
    "fr_ceiling_e": "energy ceiling 4,500/130",
    "fr_title_gad": "Gadoids: a = {a}, h ∝ prey energy",
    "fr_title_zoo": "Zooplankton: half-saturation in concentration",
    "fr_zoo_m6": "phyto., type II, M = 6 m (summer)",
    "fr_zoo_m17": "phyto., type II, M = 17 m (November)",
    "fr_zoo_micro": "microzoo., type III, M = 10 m",
    "fr_ceiling": "ceiling 1/h",
    "fr_xlabel": "visible prey biomass in the cell  [t]",
    "fr_ylabel": "intake  [t prey / t predator / tick]",
    "fr_ylabel_e": "energy intake  [MJ / t gadoid / tick]",
    "hg_u5": "u = 0.5  (zooplankton, gadoids, seals, porpoises)",
    "hg_u2": "u = 0.2  (pelagic fish)",
    "hg_xlabel1": "satiation  s_X = E_X / ME_X",
    "hg_ylabel1": "hunger factor  h_X",
    "hg_zoo": "zooplankton  g = {g} / SR = {sr}",
    "hg_gad": "gadoids  g = {g} / SR = {sr}  (×10)",
    "hg_sstar": "s*",
    "hg_xlabel2": "satiation  s_X  (q_X = s_X − u_X)",
    "hg_ylabel2": "relative biomass change / tick",
    "ir_title": "{fg}: s* = {s}",
    "ir_income": "intake h(s) · C",
    "ir_need": "cost + P · (ec + s · ME)",
    "ir_xlabel": "satiation s",
    "ir_ylabel": "MJ / t / tick",
    "ir_fg": {"zooplankton": "Zooplankton", "gadoids": "Gadoids"},
    "cap_title": "{fg}: r = {r}, K = {K} t/km², F = {F}",
    "cap_withk": "with K:  r (1 − B/K) − F",
    "cap_nok": "without K:  r − F",
    "cap_today": "today's B",
    "cap_eq": "equilibrium",
    "cap_xlabel": "regional density B  [t/km²]",
    "cap_ylabel": "net growth  [1/yr]",
    "cap_fg": {"gadoids": "Gadoids", "pelagic_fish": "Pelagic fish"},
    "fm_title_b": "Bottom trawl, mean {m} MW-h/km²/yr",
    "fm_title_p": "Pelagic trawl, mean {m} MW-h/km²/yr",
    "wc_light": "light  I0(t) · e^(−K_d z)",
    "wc_mixed": "Mixed layer  0 – M(t)   (6-17 m)",
    "wc_mixed_items": "phytoplankton · nutrient pool N_m · micro- and "
                      "zooplankton",
    "wc_sub": "Subsurface layer  M(t) – 30 m",
    "wc_sub_items": "subsurface phytoplankton (I_k 30) · nutrient pool N_d",
    "wc_res": "Reservoir 50 m  (SHARK A17, nitrate by month)",
    "wc_e": "exchange e(t)\n0.05-0.3 m/d",
    "wc_eb": "exchange e_b(t)\n0.2-1.0 m/d",
    "wc_ent": "entrainment when M(t)\ndeepens, detrainment\nwhen it shoals",
    "wc_regen": "grazing returns\n0.3 (zoo.) / 0.4 (micro)",
    "wc_export": "M1 (sinking)\nleaves the cycle",
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
        ("-", "m = number of active impact maps flagged observable - 0 "
              "today: fishing is not observable, the wind farm is muted"),
        ("-", "B_obs is VISIBLE biomass: the hidden (resting) share is "
              "subtracted"),
        ("-", "Mareld today: k = 3-5 and one light channel, i.e. 22-32 "
              "channels"),
    ],
    "net_in": "observation\n(in_dim)",
    "net_h1": "hidden layer\n16 nodes, tanh",
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
    missing = set(sv.LABELS) ^ set(LABELS_EN)
    assert not missing, f"LABELS_EN keys differ from LABELS: {missing}"
    sv.LABELS.update(LABELS_EN)
    sv.NUM.update(dec=".", sep=",")


# ----------------------------------------------------------------------
# The deck
# ----------------------------------------------------------------------
def build(d):
    f = fmt
    phy, sub = species("phytoplankton"), species("subsurface_phytoplankton")
    ben, mic = species("benthic_community"), species("microzooplankton")
    zoo, pel = species("zooplankton"), species("pelagic_fish")
    gad = species("gadoids")
    b_zoo, b_gad = budget("zooplankton"), budget("gadoids")
    f_gad, f_pel = regional_f("gadoids"), regional_f("pelagic_fish")
    k_gad, k_pel = gad["regional_carrying_capacity"], \
        pel["regional_carrying_capacity"]
    eq_gad = k_gad * (1 - f_gad / gad["r_max"])
    eq_pel = k_pel * (1 - f_pel / pel["r_max"])
    # ---------------------------------------------------------------- 0
    d.title_slide(
        "Strategy",
        "For building ecosystem simulators",
        ["Edition 2 - October 2026",
         "Mareld / Poseidon Nord",
         "Describes the mechanics the simulator runs after section 185"])

    d.bullets(
        "About this edition",
        [("p", "This version replaces the edition of 12 October "
               "(Strategy_Oct_eng.pdf). Since then the model has gained two "
               "functional groups, a two-layer water column with a nutrient "
               "pool, today's fishery, active seals and a regional carrying "
               "capacity for the fish."),
         "",
         "The code is the model. The document describes it - not the other "
         "way round. Every formula and number comes from the tick pipeline, "
         "fg_library.yaml, mareld2.yaml or lib/world/growth_budget.py; the "
         "tables are read from those files when the document is built.",
         "New is Part 4, which explains in detail the biological reasoning "
         "behind every group's growth rate and behind the logistic ceiling: "
         "where r_max comes from, which losses growth has to pay for, why "
         "the budget is paid at s* and why gadoids and pelagic fish have a "
         "regional K.",
         "Part 9 weighs the model's strengths against its weaknesses and "
         "lists suggested improvements in order of priority.",
         ("-", "The strategy is a development document, not a "
               "specification. That the code matches the document is no "
               "evidence that a term is ecologically right."),
         ],
        kicker="How to read this",
        footnote="Generated by tools/make_strategy_deck2_en.py; the Swedish "
                 "edition by tools/make_strategi_deck2.py. Section numbers "
                 "refer to mareld_resume.txt.")

    d.table(
        "What has changed since 12 October",
        ["Area", "12 October", "Today", "Sections"],
        [["Groups", "8 FGs, seals muted",
          "10 FGs: microzooplankton and subsurface phytoplankton new; seals "
          "active", "166, 177, 180"],
         ["Phytoplankton", "Light-limited logistic toward K = 24 t/km²",
          "Nutrient pool, sinking as M1, two layers, grazed by "
          "microzooplankton", "172-182"],
         ["Growth budget", "Paid at a full reserve, s = 1",
          "Paid at s*, the reserve under ideal conditions; cannibalism "
          "unchecked", "160"],
         ["Logistic ceiling", "Non-decision makers only",
          "Regional K for gadoids and pelagic fish; r_max = today's "
          "productivity", "184-185"],
         ["Herring", "Growth out of the fat reserve all year",
          "Growth out of income (κ 0.75, 5.5 months); w 0.05; type II on "
          "zooplankton", "162, 184"],
         ["Zooplankton", "Q10 on costs only",
          "Q10 on intake (2.8) and growth (3.0); grazes on concentration",
          "177-183"],
         ["Fishing", "Every impact muted",
          "Bottom and pelagic trawling from EMODnet maps, F 0.12 / 0.10; "
          "porpoise bycatch", "161, 165"],
         ["Splitting", "Continuous from one unit",
          "Only whole units move (rule D)", "159"]],
        [12, 23, 52, 11], kicker="Overview", size=10.8,
        footnote="Every policy trained before these changes must be "
                 "retrained before numbers are quoted; the FG set changed in "
                 "sections 177 and 180, which invalidates older checkpoints.")

    # ---------------------------------------------------------------- 1
    d.section("1", "Strategy and components",
              "What the model is, what it is for and which parts it is "
              "made of.")

    d.bullets(
        "Strategy",
        ["Ecosystems are modelled as a map with several map layers that "
         "evolve over time.",
         "The layers describe environmental conditions, the distribution of "
         "populations of different functional groups, nutrients and "
         "different kinds of environmental pressure. A calendar for light, "
         "temperature and stratification drives the season.",
         "Population behaviour - above all movement and foraging - is driven "
         "by behaviour models trained with ARS, but can also be hand-coded "
         "(the viability rig runs without any network).",
         "Biology drives, not action hacks. Parameters and mechanisms are "
         "fixed first; shaping terms in the reward are the last resort and "
         "are off by default.",
         "The model is Eulerian: we follow values in fixed cells, not "
         "individuals. Biomass can split, move in different directions and "
         "mix with other biomass in the same cell."],
        kicker="Core ideas")

    d.bullets(
        "Objective",
        ["The strategy is used to build simulation models that serve as "
         "scenario and hypothesis tools.",
         "The models do not give absolute population forecasts.",
         "Results are read relatively, comparatively and under stated "
         "assumptions.",
         "",
         ("p", "Before a scenario is compared, the world must sustain itself "
               "with the policies actually used - no functional group may "
               "die out or drift towards zero over the horizon. The start "
               "biomasses describe the area as it is today, under today's "
               "fishery; a baseline scenario should therefore stay close to "
               "the start."),
         ("p", "The viability rig, which runs frozen behaviour, is a "
               "diagnostic for the mechanics, not the judge."),
         ],
        kicker="What the model answers")

    d.bullets(
        "Components",
        ["An ecosystem model consists of:",
         ("-", "a grid with a length scale and a tick length"),
         ("-", "a set of functional groups (FGs), decision makers and "
               "non-decision makers"),
         ("-", "a set of maps - today's fishery as baseline maps, a scenario "
               "as replacement maps"),
         ("-", "a calendar: daylight, water temperature and mixed-layer "
               "depth for the site"),
         ("-", "a water column: mixed layer, subsurface layer and a nutrient "
               "reservoir"),
         ("-", "properties for every functional group and every predation "
               "pair (fg_library.yaml)"),
         ("-", "a dynamic model with update rules - the tick"),
         ("-", "a behaviour model for every decision-making group"),
         ("-", "spawn strategies that lay out the start biomass"),
         ("-", "diagnostics: energy budget gate, frozen viability rig and "
               "probe tools for annual and monthly budgets"),
         "The library is shared by all projects. The project file "
         "(mareld2.yaml) picks the FGs, start biomasses, impact maps and the "
         "site's light, temperature and nutrient climate."],
        kicker="Model parts", size=15)

    d.bullets(
        "Grid",
        ["Every ecosystem model has a grid, i.e. a matrix of cells with:",
         ("-", "a size - Mareld runs 60 x 60 cells"),
         ("-", "a length scale - one cell side = 1 km, so t/cell = t/km²"),
         ("-", "a time scale - one tick = 1-6 hours, normally 6"),
         ("-", "a position - 58.15° N, which sets the daylight"),
         "The grid is flat with four-neighbour connectivity: all movement "
         "goes N, E, S or W, never diagonally.",
         "An optional accessibility map sets which cells are available. "
         "Movement into an inaccessible cell is masked, and biomass that "
         "ends up there anyway is zeroed at the end of the tick.",
         "Without migration the grid edge is closed. With --migration on "
         "biomass leaves the grid and is reintroduced concentrated along the "
         "edge.",
         ("p", "The grid covers 3,600 km² of the Skagerrak. Its stocks are "
               "shares of much larger stocks - a difference in scale that "
               "returns in Parts 4 and 10.")],
        kicker="Geography")

    d.table(
        "Tick length is a run parameter",
        ["Parameter kind", "Rule for factor k = new/old", "Examples"],
        [["flux - amount moved per tick", "x · k",
          "max_intake_rate (also per pair), resting_metabolism, "
          "movement_speed, seed_rate, capacity_loss_rate"],
         ["growth - share ADDED to a stock", "(1 + x)ᵏ − 1", "growth_rate"],
         ["loss - share REMOVED from a stock", "1 − (1 − x)ᵏ",
          "natural_mortality, starve_rate, biomass_factor in impact tables"],
         ["inverse - quantity whose inverse is the rate", "x / k",
          "handling_time (the Holling ceiling 1/h stays fixed in real time)"],
         ["tick-independent", "unchanged",
          "max_energy_reserve, energy_content, maintenance_level, r_max, "
          "regional_carrying_capacity, Q10 values, nutrient keys"]],
        [22, 24, 40], kicker="--tick-length", size=11.5,
        lead="The engine is tick-length agnostic: every library number is "
             "expressed PER TICK at 6 h. The tick length converts per-tick "
             "rates to real time and, when it is not 6 h, rescales every "
             "tick-dependent parameter in memory as the FGs are built. The "
             "light, temperature, nutrient and layer tables are built "
             "directly for the run's tick length.",
        footnote="fg_library.yaml has ONE calibration (6 h) and is never "
                 "rewritten by a run. lib/world/tick_time.py is the single "
                 "definition point.")

    d.table(
        "Functional groups",
        ["Functional group", "Kind", "Menu", "Role in the model"],
        [["phytoplankton", "non-decision maker", "-",
          "Primary producer in the mixed layer; light- and nutrient-limited, "
          "sinks (M1), drifts with the current"],
         ["subsurface_phytoplankton", "non-decision maker", "-",
          "The same algae in the layer M(t)-30 m: the subsurface chlorophyll "
          "maximum, shade-adapted"],
         ["microzooplankton", "non-decision maker, grazes",
          "phyto, subsurface",
          "Ciliates and heterotrophic dinoflagellates; grows only by "
          "grazing"],
         ["benthic_community", "non-decision maker", "-",
          "Bottom food for gadoids; logistic, depleted by bottom trawling"],
         ["zooplankton", "decision maker", "phyto, subsurface, micro",
          "Copepods; graze on concentration, Q10 on intake and growth"],
         ["pelagic_fish", "decision maker", "zooplankton",
          "Herring, mackerel, sprat; fat reserve, growth out of income, "
          "regional K"],
         ["gadoids", "decision maker", "pelagic, benthic, zooplankton",
          "Cod, haddock, whiting, saithe, hake; energy ceiling, regional K"],
         ["porpoises", "decision maker", "pelagic, gadoids",
          "Harbour porpoise; endotherm, biosonar, bycatch on the trawl map"],
         ["seals", "decision maker", "pelagic, gadoids",
          "Harbour seal; ~95 % gadoids, hunts with its whiskers"],
         ["seabirds", "decision maker", "pelagic_fish",
          "Muted; the only FG with rotor mortality"]],
        [21, 15, 21, 43], kicker="Mareld: ten groups, six with a policy",
        size=10.5,
        footnote="English snake_case ids are canonical. Only seabirds are "
                 "muted in mareld2.yaml.")

    d.figure("Food web", draw_food_web, kicker="Menu from the library",
             footnote="Arrows run from prey to consumer and are drawn from "
                      "the library. Every pair has its own handling_time, "
                      "assimilation and predation_mortality and - for some "
                      "pairs - attack rate, visibility floor, light "
                      "dependence and concentration scale.")

    # ---------------------------------------------------------------- 2
    d.section("2", "Maps, calendar and water column",
              "All state is maps over the grid. Light, temperature and "
              "stratification come from a calendar for the site.")

    d.bullets(
        "Maps",
        ["A map is a function from the grid to the real numbers.",
         "The model holds two state maps per functional group: biomass "
         "B_X(c) in t and a reserve map R_X(c).",
         "For a decision maker R_X is reserve energy in MJ. The energy map "
         "E_X = R_X / B_X is derived, which makes mixing during movement "
         "trivial: two inflows add their R and their B.",
         "The satiation s_X = E_X / ME_X in [0, 1] drives hunger, growth "
         "and starvation.",
         "For a primary producer the same slot holds the nutrient pool N(c), "
         "in the producer's own units (t/km² it could become). Currents and "
         "the open boundary then move it with the water without new code - "
         "and nothing that reads a reserve as energy reads a "
         "non-decision maker's reserve."],
        kicker="Definition")

    d.table(
        "Map types and calendar tables",
        ["Type", "Symbol", "Unit", "Meaning"],
        [["Biomass maps", "B_X(c)", "t/cell",
          "Biomass of functional group X in cell c."],
         ["Reserve energy maps", "R_X(c)", "MJ/cell",
          "Stored energy of decision makers; clipped to B_X · ME_X."],
         ["Nutrient pool", "N(c)", "t/km²",
          "Dissolved nitrogen per producer and layer, in producer units."],
         ["Carrying-capacity maps", "CC_X(c)", "t/cell",
          "Per-cell ceiling for non-decision makers (max_carrying_capacity)."],
         ["Accessibility", "accessibility(c)", "[0,1]",
          "Shared habitat mask."],
         ["Fishing maps", "Impact_p(c)", "MW-h/km²/yr",
          "EMODnet effort for bottom trawls and pelagic trawls."],
         ["Other impact maps", "Impact_p(c)", "arbitrary",
          "Noise in dB, rotor density; muted until a scenario uses them."],
         ["Current field", "-", "share/tick",
          "Generated procedurally per tick from noise, never stored."],
         ["Calendar tables", "F, m, P, T, M, N0", "per tick",
          "Light, attack factor, production factor, temperature, "
          "mixed-layer depth, nitrate below the layer."]],
        [18, 16, 13, 43], kicker="What a run contains", size=11,
        footnote="The calendar tables are the same for every cell and are "
                 "built from the manifest. Temperature and nitrate are time "
                 "series per water layer, not maps.")

    d.bullets(
        "The daylight calendar",
        ["The calendar gives every tick a date and a time: year tick k "
         "starts k · tick length hours after 1 January 00:00 solar time. "
         "8,760 h is divisible by every tick length of 1-6 h.",
         "The sun's elevation follows from latitude (58.15° N), declination "
         "and solar time. The light level F(t) is the mean of a logistic "
         "function of the elevation over the tick's 10-minute steps.",
         "The start day is drawn per world and shared by every delta pair "
         "and both signs in that world (CRN).",
         "The calendar is used in five places:",
         ("-", "light-dependent attack rates of visual predators"),
         ("-", "light-limited growth in both producer layers"),
         ("-", "the monthly temperature that drives metabolism, intake and "
               "growth"),
         ("-", "the mixed-layer depth M(t), which dilutes prey for grazers "
               "and drives the nutrient supply"),
         ("-", "the twelve-month nitrate series below the layer"),
         "Every decision maker sees F(t) as its last observation channel. "
         "Switching the calendar off changes in_dim and invalidates "
         "checkpoints."],
        kicker="simulation_settings.daylight", size=15)

    d.table(
        "The calendar month by month",
        ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
         "Oct", "Nov", "Dec"],
        [["light F", "0.37", "0.44", "0.54", "0.66", "0.79", "0.87", "0.83",
          "0.70", "0.58", "0.47", "0.38", "0.34"],
         ["mixed layer M (m)", "13", "9", "6", "6", "6", "6", "6", "7",
          "10", "10", "17", "15"],
         ["m herring → zoo", "0.70", "0.80", "0.94", "1.10", "1.30", "1.42",
          "1.36", "1.17", "0.99", "0.84", "0.72", "0.66"],
         ["m gadoids → herring", "0.90", "0.93", "0.98", "1.04", "1.10",
          "1.13", "1.11", "1.06", "1.00", "0.95", "0.91", "0.88"],
         ["P/P_ref phyto", "0.16", "0.38", "0.72", "0.99", "1.12", "1.14",
          "1.03", "0.84", "0.58", "0.37", "0.15", "0.10"],
         ["P/P_ref subsurface", "0.27", "0.54", "0.80", "1.00", "1.17",
          "1.25", "1.21", "1.05", "0.82", "0.56", "0.25", "0.18"],
         ["T 0-50 m (°C)", "6.4", "5.5", "5.3", "6.2", "7.9", "9.9", "12.0",
          "13.3", "13.6", "12.6", "10.5", "8.4"],
         ["T 50 m (°C)", "7.8", "7.2", "6.7", "6.6", "6.8", "7.2", "8.0",
          "8.7", "10.2", "11.4", "10.6", "9.3"],
         ["Q10 herring, cost", "0.55", "0.51", "0.51", "0.54", "0.62", "0.73",
          "0.85", "0.95", "0.97", "0.90", "0.76", "0.64"],
         ["Q10 gadoids, cost", "0.95", "0.90", "0.87", "0.86", "0.88",
          "0.91", "0.96", "1.01", "1.14", "1.26", "1.18", "1.06"],
         ["Q10 zoo, cost", "0.80", "0.75", "0.74", "0.79", "0.88",
          "1.02", "1.18", "1.29", "1.32", "1.23", "1.06", "0.92"],
         ["Q10 zoo, intake", "0.71", "0.64", "0.63", "0.69", "0.82",
          "1.01", "1.25", "1.43", "1.48", "1.34", "1.08", "0.87"],
         ["Q10 zoo, growth", "0.69", "0.62", "0.61", "0.67", "0.81",
          "1.01", "1.27", "1.46", "1.52", "1.36", "1.08", "0.86"]],
        [21] + [6] * 12, size=10.2, align=["left"] + ["right"] * 12,
        kicker="58.15° N, 6 h/tick, monthly means",
        footnote="From tools/probes/daylight_table.py and lib/world/"
                 "temperature.py. The attack factors and every Q10 factor "
                 "except the herring's have an annual mean of exactly 1; the "
                 "producer factors are 1 on 15 April. Temperature: SMHI SHARK "
                 "A16 + A17, 1991-2020; mixed layer: A17.")

    d.bullets(
        "Water temperature",
        ["With simulation_settings.temperature an ectotherm follows the "
         "temperature of its water layer (0-50 m for herring, zooplankton "
         "and microzooplankton; 50 m for gadoids). Three separate effects:",
         ("-", "metabolism_q10 on resting metabolism and so on every action "
               "cost: herring 2.2 (t_ref 14 °C, a summer cost), gadoids 2.2 "
               "and zooplankton 2.0 (annual mean 1)"),
         ("-", "ingestion_q10 on the zooplankton's whole functional "
               "response, a and 1/h together: 2.8 (Hansen et al. 1997)"),
         ("-", "growth_q10 on the zooplankton's growth branch: 3.0 (Huntley "
               "& Lopez 1992), starvation unscaled"),
         "Microzooplankton grazing has grazing_q10 2.6 at 10 °C (Nielsen & "
         "Kiørboe 1994).",
         "Porpoises and seals have no Q10: endotherms, and field metabolism "
         "is flat over 0-20 °C water.",
         "Intake and growth are normalised to an annual mean of 1, so the "
         "library's a, h and growth_rate stay annual means and the growth "
         "budget's s* does not change (Part 4).",
         ("p", "Effect: the zooplankton spring starts in March as at A17 "
               "instead of a month early; herring winter starvation "
               "disappeared once its cost followed the temperature.")],
        kicker="Q10 on the calendar", size=14.5)

    d.figure(
        "The water column: two layers and a reservoir", draw_water_column,
        kicker="simulation_settings.nutrients",
        footnote="A linear 2 × 2 map per tick moves water AND algae between "
                 "the layers; each producer has its own pool. New nitrogen "
                 "comes only through the exchange with the reservoir.")

    d.formula(
        "The nutrient pool",
        [("Growth paid out of the pool (Evans & Parslow 1985)", [
            r"\Delta B &= \min\left(r\,L(t)\,B\,\dfrac{N}{N + k_N\,Y\,H(t)},\ N,\ K - B\right) \qquad N \leftarrow N - \Delta B",
            r"\dfrac{dN}{dt} &= Y\left[e\,(N_0(t) - c) + \max(\dot M, 0)\,N_0 - \max(-\dot M, 0)\,c\right] \qquad c = \dfrac{N}{Y\,H}",
            ("t", "r = derived growth_rate, L(t) = the light factor, H = the "
                  "layer's thickness (M(t) in the mixed layer), Y = 0.795 "
                  "t/km² per µmol/l and metre."),
        ])],
        tail=[("-", "k_N 0.5 µmol/l: one pool for nitrate and regenerated "
                    "ammonium. Y = 14 mg N/m² × Redfield 5.68 / C:WW 0.10. "
                    "N0 = A17 nitrate below the layer by month."),
              ("-", "Grazing returns grazing_regeneration 0.3 of the grazed "
                    "biomass (zooplankton) and 0.4 of intake "
                    "(microzooplankton). Sinking (M1) and faecal pellets "
                    "leave the cycle: the budget is open."),
              ("-", "The logistic factor is gone; K = 24 t/km² remains only "
                    "as a ceiling (Part 4)."),
              ("-", "The exchange e(t) 0.05-0.3 m/d and the reservoir "
                    "exchange 0.2-1.0 m/d are fitted to A17 nitrate, not "
                    "measured.")],
        kicker="lib/world/nutrients.py")

    # ---------------------------------------------------------------- 3
    d.section("3", "Functional group properties",
              "The parameters in fg_library.yaml, their role in the tick "
              "and their calibrated values.")

    d.table(
        "Properties: energy and metabolism",
        ["Property", "Key", "Unit", "Role"],
        [["Decision model", "is_decision_maker", "yes/no",
          "Whether the FG has a policy and an energy budget."],
         ["Maximum reserve", "max_energy_reserve (ME_X)", "MJ/t",
          "Ceiling for E_X. The reserve is clipped to B_X · ME_X after "
          "movement."],
         ["Resting metabolism", "resting_metabolism", "MJ/t/tick",
          "Base cost. Multiplied by the action's cost factor."],
         ["Cost factors", "feeding_cost, movement_cost", "factor",
          "Multiplier on resting metabolism per action."],
         ["Temperature", "metabolism_q10, metabolism_t_ref", "-, °C",
          "Q10 on resting metabolism; t_ref in °C or annual_mean."],
         ["Intake and growth", "ingestion_q10, growth_q10", "-",
          "Q10 on the whole functional response or the growth branch "
          "(annual mean 1)."],
         ["Energy content", "energy_content", "MJ/t",
          "Value as food - and the price of one tonne of own new biomass."],
         ["Reserve as food", "prey_includes_reserve, "
          "reserve_reference_fill", "yes/no, share",
          "The prey's reserve is eaten with it (herring)."],
         ["Maintenance level", "maintenance_level (u_X)", "share of ME",
          "Zero point q = s − u and the knee of the hunger curve."]],
        [17, 26, 12, 40], size=10.8, kicker="Per functional group")

    d.table(
        "Properties: growth, losses and movement",
        ["Property", "Key", "Unit", "Role"],
        [["Net growth at low density", "r_max", "1/yr",
          "The literature r; basis of the derived growth_rate (Part 4)."],
         ["Growth rate", "growth_rate (MG_X)", "share/tick",
          "DERIVED by the loader - never edited directly."],
         ["Natural mortality", "natural_mortality (M1)", "share/tick",
          "Losses that are not predation by an FG."],
         ["Income-funded growth", "income_growth_kappa, "
          "feeding_season_months", "share, months",
          "Growth out of the tick's intake; the budget is paid over the "
          "season."],
         ["Regional carrying capacity", "regional_carrying_capacity (K)",
          "t/km²", "Loss r_max/1460 · B_reg/K outside the budget."],
         ["Starvation rate", "starve_rate", "share/tick",
          "Catabolism when q < 0."],
         ["Attack rate", "max_intake_rate (a)", "1/(t·tick)",
          "Slope of the functional response; can be set per pair."],
         ["Interference", "interference (w)", "1/t",
          "Beddington-DeAngelis: own density lowers intake."],
         ["Speed", "movement_speed (v)", "cell sides/tick",
          "Distance per tick of the moving cohort."],
         ["Smallest unit", "min_split_biomass", "kg",
          "One individual: only whole units are split (rule D)."],
         ["Extinction threshold", "extinction_threshold_factor", "factor",
          "A cell is zeroed below factor · min_split_biomass."]],
        [19, 27, 12, 37], size=10.4, kicker="Per functional group")

    d.table(
        "Properties: predation pairs",
        ["Property", "Key", "Unit", "Role"],
        [["Predation mortality", "predation_mortality (M2)", "1/yr",
          "The literature loss of the prey to the predator; part of the "
          "prey's growth budget."],
         ["Attack rate per pair", "max_intake_rate", "1/(t·tick)",
          "Moves the half-saturation 1/(a·h) without moving the ceiling."],
         ["Handling time", "handling_time (h)", "tick",
          "Sets the ceiling 1/h t prey per t predator and tick."],
         ["Assimilation", "assimilation_factor", "share",
          "Share of the prey's energy the predator keeps."],
         ["Visibility floor", "visibility_floor", "[0,1]",
          "Share that stays visible despite resting; per pair or the "
          "prey's."],
         ["Functional response", "functional_response", "2 | 3",
          "Forces Holling type II or III for the pair."],
         ["Light dependence", "dark_ratio, light_threshold_deg", "share, °",
          "Attack in darkness relative to daylight; elevation of half "
          "detection."],
         ["Concentration", "mixed_layer_reference_m", "m",
          "a is scaled by M_ref / M(t): grazers feed on concentration."]],
        [19, 27, 12, 37], size=10.8, kicker="Per pair")

    d.table(
        "Properties: producers and grazers",
        ["Property", "Key", "Unit", "Role"],
        [["Carrying capacity per cell", "max_carrying_capacity", "t/cell",
          "The logistic K; with a nutrient pool only a ceiling."],
         ["Light saturation", "light_saturation (I_k)", "µmol/m²/s",
          "Knee of the P-I curve; twelve monthly values or one."],
         ["Water layer", "water_layer", "mixed | subsurface",
          "Which layer the producer lives in."],
         ["Nutrient pool", "nutrient_half_saturation, nutrient_yield",
          "µmol/l, t/km² per µmol/l·m", "Monod limitation and the "
          "conversion N → biomass."],
         ["Regeneration", "grazing_regeneration, excretion_regeneration",
          "share", "Part of the grazing returned to the pool."],
         ["Grazer efficiency", "growth_efficiency", "share",
          "Gross growth efficiency (GGE) of a grazing NDM."],
         ["Grazer temperature", "grazing_q10, grazing_t_ref", "-, °C",
          "Q10 on grazing in the grazer's layer."],
         ["Food threshold", "food_threshold (F0)", "t/km²",
          "ERSEM's perception term: a refuge at low prey density."],
         ["Edible share", "edibility_floor, edibility_half_saturation_umol_l",
          "share, µmol/l", "The grazer sees e · B; e falls with nitrate."],
         ["Current response", "current_response", "[0,1]",
          "Share of the current field the FG follows."]],
        [17, 32, 15, 31], size=10.4, kicker="Non-decision makers")

    d.table(
        "Library values - energy and movement",
        ["", "zooplankton", "pelagic_fish", "gadoids", "porpoises", "seals",
         "seabirds *"],
        dm_energy_rows(annual="annual mean"),
        [24, 13, 13, 13, 13, 13, 13], size=10.4,
        align=["left", "right", "right", "right", "right", "right", "right"],
        kicker="fg_library.yaml, calibrated at 6 h/tick",
        footnote="Read from fg_library.yaml when the document is built. "
                 "Energies in MJ/t, rates per tick. * = muted in "
                 "mareld2.yaml. The herring energy_content is the lean body; "
                 "the fat reserve comes on top.")

    d.table(
        "Library values - growth and ceilings",
        ["", "zooplankton", "pelagic_fish", "gadoids", "porpoises", "seals",
         "seabirds *"],
        dm_growth_rows(year="yr", derived="derived", months="months"),
        [26, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5], size=11,
        align=["left", "right", "right", "right", "right", "right", "right"],
        kicker="Part 4 explains the numbers",
        lead="growth_rate and capacity_loss_rate are derived by the loader "
             "from r_max, M1, the predation matrix and K. The values below "
             "are what the loader computes for mareld2.yaml with "
             "--mortality on.",
        footnote="Producers: growth_rate "
                 f"{f(phy['growth_rate'])} per tick (gross), M1 "
                 f"{f(phy['natural_mortality'])}, K "
                 f"{f(phy['max_carrying_capacity'])} t/km². Benthos: hand-set "
                 f"{f(ben['growth_rate'])} (0.82/yr), K "
                 f"{f(ben['max_carrying_capacity'])}. Microzooplankton: 0 - "
                 "it grows only by grazing.")

    d.table(
        "Library values - predation pairs",
        ["Predator → prey", "a", "h", "assim.", "floor", "response",
         "other", "M2 (1/yr)"],
        pair_value_rows(prey_word="prey", type_word="type",
                        forced_word="set", graze_word="grazing"),
        [33, 7, 7, 7, 9, 15, 19, 9], size=9.6,
        align=["left", "right", "right", "right", "right", "left", "left",
               "right"],
        kicker="Read from fg_library.yaml",
        footnote="\"prey 0.35\" = the pair inherits the prey's "
                 "visibility_floor; \"set\" = functional_response in the "
                 "library. ρ = dark_ratio / threshold; F0 = food_threshold; "
                 "e = edibility floor / half-saturation (µmol/l). * = the "
                 "predator is muted.")

    d.bullets(
        "Plankton: three groups in two layers",
        ["Microzooplankton (ciliates and heterotrophic dinoflagellates) used "
         "to be a deduction from the phytoplankton rate, \"60 % "
         "micrograzing\". It is now a non-decision group of its own that "
         "grazes both producer layers and is itself eaten by zooplankton:",
         ("-", "Imax 1.7/d at 10 °C (Hansen et al. 1997), half-saturation "
               "70 µg C/l, food threshold 12 mg C/m³ (ERSEM), GGE 0.3 "
               "(Straile 1997), M1 0.05/d, Q10 2.6"),
         "Subsurface phytoplankton is the same algae in the layer between the "
         "mixed-layer depth and 30 m: at A17 ~90 % of the April-August "
         "chlorophyll lies below the 6 m summer layer, on the nitracline.",
         "Zooplankton grazes on CONCENTRATION: a is scaled by 10 m / M(t) "
         "(subsurface: 10 m / H(t)). Half-saturation 50 µg C/l = 5 t/km² in "
         "a 10 m layer.",
         ("-", "Zooplankton → microzooplankton is type III (prey switching, "
               "Kiørboe et al. 1996): with type II it was grazed to 1e-18 "
               "in two weeks."),
         ("-", "Microzooplankton sees only an edible share of the subsurface "
               "layer, e = 0.1 + 0.9 N/(N + 2): the late-summer Ceratium "
               "layer is too large for ciliates (Nielsen & Kiørboe 1994)."),
         ("p", "Effect: gross primary production rose from 13 to ~40 g "
               "C/m²/yr and summer zooplankton from 0.4 to 3-4 t/km² (A17 "
               "4.6-6.9).")],
        kicker="Sections 177-182", size=14.5)

    d.bullets(
        "Herring: fat reserve, winter food and growth out of income",
        ["Body and reserve are separate: energy_content 4,000 MJ/t = lean "
         "body, max_energy_reserve 7,000 ≈ 18 % fat, maintenance_level 0.2. "
         "Predators eat the reserve with the fish (prey_includes_reserve).",
         "Winter food (section 162): no interference has been measured in "
         "herring and every fit of the herring functional response is type "
         "II. Hence w 0.7 → 0.05, type II on zooplankton and a visibility "
         "floor 0.2 → 0.5 (a bank, copepods migrate little).",
         "Growth is built from the tick's intake, not from the fat (section "
         "184): with reserve-funded growth the herring turned fat into "
         "biomass in Oct-Dec, entered the winter lean and lost 25-30 %/yr.",
         "Biomass is held by a regional K = 5.0 t/km² and r_max 0.34 "
         "(Part 4).",
         ("p", "In fix68: no starvation in any month, s 0.35-0.41 in February "
               "and 0.80 in May-June - the fat is kept for the winter. The "
               "remaining winter loss is M1 and predation.")],
        kicker="Sections 140, 162, 184, 185", size=15)

    d.bullets(
        "Gadoids: three prey, an energy ceiling and a regional K",
        ["The menu is herring (type III, light-dependent), zooplankton (a "
         "krill proxy) and benthos. Cannibalism is unchecked (section 160): "
         "a group reward never learns to eat its own group, so the budget "
         "paid 0.22/yr for a loss that was never taken.",
         "The ceiling is an ENERGY limit per pair, h = 130 · E / 4,500: "
         "gastric evacuation scales with prey energy density, so every pair "
         "saturates at the same ~35 MJ per t gadoid and tick (~3 % of body "
         "mass per day).",
         "M1 0.415/yr = residual M0 0.20 (ICES SMS) + predation by groups "
         "that are not FGs (0.214).",
         "Fishing F 0.12/yr on the map, 98 % bottom trawls.",
         "r_max 0.37 (today's productivity) and K 1.02 t/km² - explained in "
         "Part 4.",
         ("p", "Without K the gadoids grew at r − F ≈ 0.35/yr, ate the "
               "herring down and then starved in winter; with K they sit at "
               "0.63-0.66 t/km² and the herring pool is stable.")],
        kicker="Sections 141, 147-148, 157, 160, 185", size=15)

    d.bullets(
        "Porpoises and seals",
        ["Porpoises: costs against measured metabolism (Rojano-Donate et al. "
         "2018), rest 60 MJ/t/tick, eat = field metabolic rate 90. The diet "
         "is fitted to Ecopath (attack rate 0.015 on herring, 0.2 on "
         "gadoids); biosonar gives herring a floor of 0.95. One unit = "
         "50 kg.",
         ("-", "Gillnet bycatch 0.017/yr rides on the bottom-trawl map: "
               "gillnet effort in the grid is ~0."),
         "Seals (sections 166-168, 185): harbour seals only, ~55 kg per unit. "
         "Rest 50, eat 1.5×, M1 0.10 and r_max 0.11 (Carroll et al. 2025). "
         "Spawned on gadoids.",
         ("-", "Gadoids carry ~95 % of the diet (flatfish, haddock, saithe "
               "lumped in): M2 0.12/yr, a 0.2, floor 0.85 - seals find "
               "resting bottom fish with their whiskers."),
         ("-", "Pelagic fish is type III with a 3e-4 per (t/cell)²: a low "
               "attack rate stands in for the vertical separation a "
               "one-layer model lacks. Target ~10 % of the diet."),
         ("p", "Open: the trained seal policy eats ~0 % pelagic fish. The eat "
               "channel has saturated (logit ~−8) and the ARS gradient "
               "through a softmax share is proportional to the share "
               "(Part 9).")],
        kicker="Top predators", size=14.5)

    # ---------------------------------------------------------------- 4
    d.section("4", "Growth rate and logistic ceiling",
              "The biological reasoning: where r_max comes from, which "
              "losses growth pays for, why the budget is paid at s* and "
              "why the fish have a regional K.")

    d.bullets(
        "What the growth rate has to mean",
        ["In the May edition growth_rate (MG_X) was a free number per group. "
         "A free number cannot be tested against data: the same value gives "
         "very different net growth depending on which predators, which "
         "mortality and which energy the model has.",
         "What can be measured in nature is r_max: how fast a sparse stock "
         "grows when food does not limit it - recruitment and individual "
         "growth minus all natural mortality. It is the parameter of the "
         "logistic model dB/dt = r B (1 − B/K).",
         "The model takes losses itself: M1 (natural mortality that is not "
         "an FG) and M2 (predation by the model's FGs). The growth term must "
         "therefore be GROSS, r_max + M1 + ΣM2, for the net growth at low "
         "density to come out as r_max.",
         "Checking or unchecking a predator changes the prey's rate by "
         "exactly the predator's M2 - the net is unchanged. The model "
         "survives a changed FG set without recalibration.",
         "Fishing, bycatch and the regional ceiling lie OUTSIDE the budget: "
         "r_max is unfished growth at low density, and fishing must be able "
         "to lower the net.",
         ("p", "The budget is a premise, not an outcome: what is realised "
               "depends on food and on what the policies learn.")],
        kicker="Principle (section 129)", size=15)

    d.formula(
        "From literature budget to growth rate",
        [("lib/world/growth_budget.py - the loader, the FG editor and the "
          "tests compute the same", [
            r"g_X &= \dfrac{r_{\max,X} + M1_X + \sum_j M2_{Xj}}{(s^{*}_X - u_X)\cdot 1460}\cdot\dfrac{12}{n_X} \qquad \mathrm{decision\ maker}",
            r"r_X &= \dfrac{r_{\max,X} + M1_X + \sum_j M2_{Xj}}{1460} \qquad \mathrm{non\ decision\ maker}",
            "",
            r"\Delta B_X &= B_X\,g_X\,\theta_X(t)\,(s_X - u_X) \qquad s_X \geq u_X,\ \mathrm{debited\ from\ the\ reserve}",
            ("t", "1,460 = number of 6-hour ticks per year; θ_X(t) = the "
                  "growth_q10 factor (annual mean 1); n_X = "
                  "feeding_season_months (12 when not set)."),
        ])],
        tail=[("-", "M2_Xj counts while the pair is checked and the predator "
                    "is active - a grazing non-decision predator only when "
                    "its pair has an attack rate."),
              ("-", "With --mortality off the loader leaves M1 out of g: "
                    "removing a loss removes its compensation."),
              ("-", "Groups without r_max (benthos, microzooplankton) keep "
                    "their hand-set growth_rate.")],
        kicker="r_max, M1, M2, s*")

    d.table(
        "The budget per functional group",
        ["FG", "r_max", "M1", "ΣM2 per predator", "gross", "s*",
         "g / tick", "Pmax", "K"],
        budget_rows(),
        [22, 7, 7, 17, 7, 6, 15, 7, 9], size=10.4,
        align=["left", "right", "right", "left", "right", "right", "right",
               "right", "right"],
        kicker="Read from growth_budget when the document is built",
        lead="All rates per year except g. The gross budget is what growth "
             "has to produce per year at s*; Pmax is the most the energy at "
             "the Holling ceiling allows. K is regional (t/km²) for the fish "
             "and per cell for the producers.",
        footnote="Producers' \"–\": non-decision makers have no reserve, so "
                 "the denominator is 1. The herring g is ×12/5.5 because it "
                 "grows only while it eats. Porpoises and seals have no "
                 "predators in the model.")

    d.table(
        "r_max: the biology behind the numbers",
        ["FG", "r_max/yr", "What the number is", "Source"],
        [["phytoplankton", f(phy["r_max"]),
          "Dilution-experiment growth in April, 0.67/d gross (net of "
          "respiration and lysis), minus what the model takes explicitly: "
          "micrograzing 147, zooplankton 22.6, sinking 7.3",
          "Stelfox-Widdicombe et al. 2004; Calbet & Landry 2004"],
         ["subsurface_phyto.", f(sub["r_max"]),
          "The same algae; only light and nutrients differ", "as above"],
         ["zooplankton", f(zoo["r_max"]),
          "Food-unlimited copepod production, 0.16/d at 11.5 °C = 58.4/yr, "
          "minus total mortality 25/yr", "Huntley & Lopez 1992"],
         ["pelagic_fish", f(pel["r_max"]),
          "Today's low-density growth for the site's mix (weights: herring "
          "0.73, mackerel 0.19, sprat 0.06, horse mackerel 0.02)",
          "FishBase CMSY; ICES SAG; Euler-Lotka"],
         ["gadoids", f(gad["r_max"]),
          "The same method for cod, haddock, whiting, saithe and hake",
          "Froese et al. 2017; ICES SAG"],
         ["porpoises", f(species("porpoises")["r_max"]),
          "Observed growth after bycatch ended, 9.6 %/yr",
          "Lockyer 2003; Forney et al. 2021"],
         ["seals", f(species("seals")["r_max"]),
          "Harbour seals in the Kattegat-Skagerrak after 1988, 10.2-13.6 "
          "%/yr", "Carroll et al. 2025"],
         ["benthic_comm.", "(0.82)",
          "Benthic community biomass recovery after trawling; hand-set g",
          "Hiddink et al. 2017; Pitcher et al. 2022"]],
        [15, 9, 50, 26], size=10.2, kicker="One number per group, with a "
        "source",
        footnote="Microzooplankton has no r_max: it grows with GGE 0.3 of "
                 "what it grazes (Straile 1997). Seabirds (muted) 0.10/yr "
                 "(Niel & Lebreton).")

    d.formula(
        "From historical to present-day productivity",
        [("Euler-Lotka: recruits per spawner fallen by a factor d", [
            r"r_{\mathrm{now}} &= r_{\mathrm{prior}} - \dfrac{\ln d}{T} \qquad T = \dfrac{3}{r_{\mathrm{prior}}}\ \ \mathrm{(generation\ time)}",
            ("t", "Gadoids: cod 0.35, haddock 0.24, whiting 0.49, saithe "
                  "0.44, hake 0.26 → pool 0.34-0.40, chosen 0.37."),
            ("t", "Pelagic fish: NSAS herring d 1.7, WBSS 2.25, mackerel 3.6, "
                  "horse mackerel 1.6, sprat 1 → 0.34 (0.27-0.40)."),
        ])],
        lead=[("p", "The CMSY priors (r 0.50-0.51 for the gadoids, 0.44 for "
                    "the site's pelagic mix) are 1960s-90s growth at low "
                    "density. Recruits per spawner have fallen since: cod "
                    "2.6×, haddock 4.7×, hake 3.9× (ICES SAG), linked to "
                    "warming and hysteresis (Sguotti et al. 2019).")],
        tail=[("-", "A fall by a factor d lowers r by ln(d)/T - not by the "
                    "factor d. The trial with 0.51/2.5 = 0.20 was therefore "
                    "too low (sections 184.17-18)."),
              ("-", "The old pelagic 0.648 weighted in 36 % sandeel, which "
                    "has no grounds in the grid."),
              ("-", "Today's r is thus low productivity at low density - "
                    "not crowding near a ceiling.")],
        kicker="Sections 184.18, 185.18")

    d.bullets(
        "M1 and M2: the losses growth pays for",
        ["M2 per pair comes from the North Sea Ecopath model for 1991 "
         "(Mackinson & Daskalov 2007), with every species under the "
         "umbrella term. Herring: gadoids 0.30, porpoises 0.0117, seals "
         "0.0015 per year.",
         "M1 = residual natural mortality M0 + predation by groups the "
         "model lacks:",
         ("-", "gadoids 0.20 (ICES SMS) + 0.214 (squid, halibut, gurnards, "
               "rays, monkfish) = 0.415"),
         ("-", "pelagic fish 0.391 + 0.189 (sharks, rays, other fish) = "
               "0.580"),
         ("-", "zooplankton: total mortality 25/yr minus the 21 % taken by "
               "fish = 19.7; porpoises and seals 0.10 from Leslie models"),
         ("-", "phytoplankton: sinking only, 0.02/d - respiration and lysis "
               "are already in the dilution rate and must not be counted "
               "twice"),
         "The microzooplankton M2 of 147/yr is a RATIO, 60 % of the gross "
         "0.67/d: it makes the phytoplankton g the gross rate.",
         "Not in the budget: fishing (F 0.12 / 0.10), porpoise bycatch "
         "0.017, the regional ceiling and cannibalism (unchecked)."],
        kicker="Ecopath 1991, ICES SMS", size=14.5)

    d.formula(
        "Why the budget is paid at s*",
        [("The reserve under ideal conditions (section 160)", [
            r"h(s)\,C &= \mathrm{cost} + P\,(ec + s\,\mathrm{ME}) \qquad h(s) = \dfrac{1 - s}{1 - u}",
            r"s^{*} &= \dfrac{C/(1-u) - \mathrm{cost} - P \cdot ec}{C/(1-u) + P \cdot \mathrm{ME}} \qquad P = \dfrac{\mathrm{gross}}{1460}",
            ("t", "C = ceiling 1/h × assimilation × energy of the best checked "
                  "prey; cost = resting_metabolism × feeding_cost."),
        ])],
        lead=[("p", "Growth per tick is g (s − u). Before section 160 g was "
                    "sized at a full reserve, s = 1. But the hunger factor "
                    "closes intake exactly there, so no group ever produced "
                    "its budget: production/budget = (mean s − u)/(1 − u), "
                    "i.e. herring 60 %, gadoids 48 %, porpoises 39 %, "
                    "zooplankton 38 %.")],
        tail=[("-", "s* is where the reserve settles when the group eats its "
                    "best prey at the ceiling every tick: income meets cost, "
                    "the price of growth and the dilution. Closed form - no "
                    "policy, no rollout (calibrating on a trained policy's s "
                    "would be circular)."),
              ("-", "The budget is rejected if P ≥ Pmax = (C − cost)/(ec + "
                    "u ME), or if g > ME/ec (the mass-balance ceiling).")],
        kicker="Hunger closes at a full reserve")

    d.figure(
        "s* for zooplankton and gadoids", draw_ideal_reserve,
        kicker="Where income meets need",
        items=[("-", f"Zooplankton: best prey microzooplankton, C = "
                     f"{f(b_zoo.ideal_intake)} MJ/t/tick against cost "
                     f"{f(b_zoo.ideal_cost)}; s* {f(b_zoo.reference_fill)}, "
                     f"g {f(b_zoo.growth_rate)} - close to the mass-balance "
                     f"ceiling ME/ec = "
                     f"{f(zoo['max_energy_reserve'] / zoo['energy_content'])}."),
               ("-", f"Gadoids: C = {f(b_gad.ideal_intake)} against cost "
                     f"{f(b_gad.ideal_cost)}; s* {f(b_gad.reference_fill)}. "
                     "A low s* gives a high g for the same budget.")],
        footnote="s* is computed at the library cost (the reference "
                 "temperature); the Q10 factors have an annual mean of 1 and "
                 "are left out, so the budget stays a property of the "
                 "library.")

    d.formula(
        "Herring: growth out of income",
        [("Income-funded growth (section 184)", [
            r"G &= \min\left(\kappa\,I - C,\ \ B\,g\,(s - u)\,ec\right) \qquad \Delta B = G / ec",
            r"g &= \dfrac{r_{\max} + M1 + \sum M2}{(s^{*} - u)\cdot 1460}\cdot\dfrac{12}{5.5}",
            ("t", "I = the tick's assimilated intake, C = the tick's action "
                  "costs (MJ); the rest of the income goes to the reserve. "
                  "No intake - no growth."),
        ])],
        lead=[("p", "With reserve-funded growth the herring kept turning fat "
                    "into biomass in October-December (0.17-0.27 %/d) after "
                    "the food had gone. It entered the winter lean and lost "
                    "25-30 % a year.")],
        tail=[("-", "Biology: structure is built from food - fat holds no "
                    "nitrogen, and fasting fish lose the GH → IGF-1 signal. "
                    "Herring somatic growth runs April - mid-September "
                    "(Bachiller et al. 2018); 65-93 % of the year's length "
                    "gain at the site happens before August (DATRAS)."),
              ("-", "The same form as Wisconsin bioenergetics and DEBlipid "
                    "(Martin et al. 2017). κ 0.75 is where the budget closes "
                    "at today's s*."),
              ("-", "The months only size the rate: WHEN the herring grows is "
                    "decided by its food. A calendar switch was tried and "
                    "removed.")],
        kicker="income_growth_kappa, feeding_season_months")

    d.bullets(
        "Temperature and growth",
        ["Zooplankton growth has its own Q10 (growth_q10 3.0): Huntley & "
         "Lopez (1992) give the food-saturated copepod growth 0.0445 "
         "e^(0.111 T), ~0.08/d at 5 °C. The flat rate let the model's "
         "zooplankton grow 13 %/d in February.",
         ("-", "Effect: the spring starts at A17's resident stock (March 1.7 "
               "vs 1.6 t/km²) and the peak falls in May."),
         "Intake has ingestion_q10 2.8 on the whole functional response, so "
         "the half-saturation stays put while the ceiling follows the "
         "temperature.",
         "Both are normalised to an annual mean of 1: the library "
         "growth_rate is the annual mean, and s* and the budget do not "
         "change.",
         "A Q10 on herring growth was tried and rejected: the 0-50 m water is "
         "warmest in August-October, so growth moved INTO the autumn - the "
         "wrong phase. Temperature is not the herring's growth clock; food "
         "is.",
         ("p", "Gadoids, seals and porpoises have no growth Q10. The gadoid "
               "cost follows the 50 m temperature.")],
        kicker="Sections 181, 183", size=15)

    d.formula(
        "Primary producers: gross, light and nutrients",
        [("Phytoplankton and subsurface phytoplankton", [
            r"g &= \dfrac{r_{\max} + M1 + M2_{\mathrm{micro}} + M2_{\mathrm{zoo}}}{1460} = \dfrac{0.67\,\mathrm{d}^{-1}\cdot 365}{1460}",
            r"r(t) &= g\,L(t) \qquad L(t) = \dfrac{\mathrm{mean}_{z}\,\tanh\left(I_0(t)\,e^{-K_d z}/I_k\right)}{\mathrm{the\ same\ on\ 15\ April}}",
            r"\Delta B &= \min\left(r(t)\,B\,\dfrac{N}{N + k_N Y H},\ N,\ K - B\right)",
        ])],
        lead=[("p", "The dilution technique measures the algae's own growth "
                    "in closed bottles over 24 h: net of respiration and "
                    "lysis (they act in every bottle), gross of grazing and "
                    "sinking. Hence g is gross, and grazing and sinking are "
                    "taken out explicitly.")],
        tail=[("-", "0.67/d is the highest in situ rate in April at 7-9.5 °C "
                    "in the southern North Sea; L(t) = 1 on 15 April and 0 "
                    "at night."),
              ("-", "I_k 111-298 µmol/m²/s by month (SMHI 14C, Anholt E); the "
                    "subsurface layer 30 (shade-adapted, below the regional "
                    "measurements of 63-117)."),
              ("-", "The logistic factor is gone: what ends a bloom is the "
                    "nitrate running out, not crowding.")],
        kicker="Sections 131, 138, 172-173")

    d.bullets(
        "The logistic ceiling: why a K is needed",
        ["The r_max priors are the r of a Schaefer model, dB/dt = r B (1 − "
         "B/K) − catch. The model kept r and dropped K: with good food the "
         "pool grew at r − F at EVERY density.",
         "Measured (section 184.13): the gadoids grew +0.35/yr, 2.7× in four "
         "years, ate the herring down and then fell by starvation in the "
         "winter of year 3 - a predator-prey cycle regulated by food only. "
         "No such gadoid-herring cycle is documented; SMS gives the gadoid "
         "M2 on herring as 0.15/yr with no trend.",
         "Interference cannot be the brake: w = 1 already gives 2-3 % less "
         "intake per doubling (empirically 3-6 %), and zero growth would "
         "need w ≈ 11 (section 161.1).",
         "Without K the pelagic pool grew 15-25 %/yr and grazed the "
         "zooplankton 10-14 times its standing stock per year (Ecopath "
         "1.9×). The literature finds no top-down control of North Sea "
         "zooplankton by herring.",
         "Density dependence in gadoids is mainly stock-recruit "
         "compensation (68-78 % of NE Atlantic stocks) plus cannibalism."],
        kicker="Sections 184.13-185", size=14.5)

    d.formula(
        "Regional carrying capacity",
        [("The loss after M1, in every cell (section 185)", [
            r"c &= \dfrac{r_{\max}}{1460\,K} \qquad B(c) \leftarrow B(c)\,\left(1 - c\,\bar B\right) \qquad R\ \mathrm{scaled\ alike}",
            r"\dfrac{1}{B}\dfrac{dB}{dt} &= r_{\max}\left(1 - \dfrac{\bar B}{K}\right) - F \qquad \mathrm{when\ M1\ and\ M2\ are\ taken\ as\ budgeted}",
            r"K &= \dfrac{B_{\mathrm{today}}}{1 - F/r_{\max}} \qquad \mathrm{gadoids:}\ \dfrac{0.69}{1 - 0.12/0.37} = 1.02",
            r"K &= \dfrac{B_{\mathrm{today}}}{1 - P/r_{\max}} \qquad \mathrm{pelagic:}\ \dfrac{3.33}{1 - 0.115/0.34} = 5.0",
            ("t", "B_reg = the grid-mean density (t/km²); P = the stocks' "
                  "surplus production at the site's weights."),
        ])],
        tail=[("-", "The loss lies outside the derived g, so the net at good "
                    "food is the logistic model the r_max prior came from."),
              ("-", "It acts with or without --mortality (it closes r_max, "
                    "which g always pays) and is not scaled by "
                    "--mortality_multiplier."),
              ("-", f"capacity_loss_rate: gadoids "
                    f"{f(capacity_rate('gadoids'))}, pelagic fish "
                    f"{f(capacity_rate('pelagic_fish'))} per tick and t/km².")],
        kicker="regional_carrying_capacity")

    d.figure(
        "Net growth with and without the ceiling", draw_capacity,
        kicker="F from today's fishing maps",
        items=[("-", f"The equilibrium K (1 − F/r): gadoids {f(eq_gad)} t/km² "
                     f"(= today's 0.69), pelagic fish {f(eq_pel)} against the "
                     "start's 3.33 - K was set with the surplus production "
                     "0.115, not with F 0.10."),
               ("-", "Without K the net is a positive constant: nothing stops "
                     "growth until the food runs out."),
               ("-", "Realised (frozen policies): gadoids 0.63-0.66 - the "
                     "seals take 0.17-0.19 against the budget's 0.12; "
                     "pelagic fish 2.75 - ~0.05/yr more loss than the "
                     "budget.")],
        footnote="The curves assume M1 and M2 are taken exactly as budgeted. "
                 "Every deviation in realised predation moves the "
                 "equilibrium.")

    d.bullets(
        "Effective capacity, not the historical ceiling",
        ["Gadoids: today's 0.69 t/km² is a depleted stock - decades of "
         "overfishing, trawled nurseries and warming. 2010-19 was depleted "
         "too, and the historical stock was much larger. K is therefore "
         "today's EFFECTIVE capacity, separate from the historical ceiling, "
         "and r is today's low productivity.",
         ("-", "Tried: K 0.9 and 1.4 with r 0.511; r 0.20 with K 3.0; r 0.37 "
               "with K 1.02 (sections 184.15-19). The last keeps the stock "
               "stationary and the herring pool stable."),
         "Pelagic fish: K stands for compensation at the stock scale "
         "(density-dependent recruitment) plus net emigration. The grid "
         "holds ~0.4 % of the autumn-spawning herring, the adults winter in "
         "the Øresund/Køge Bay and the winter stock is ~90 % 1-2-year-olds.",
         "Why regional and not per cell: compensation acts on the stock in "
         "the nurseries, not in a 1 km cell. A per-cell term would give the "
         "policy a gradient to spread out that nature does not have. B_reg "
         "gives the same loss rate in every cell.",
         "K as a function of bottom trawling (habitat) was checked in the "
         "literature: no published function, floating turbines, an effect of "
         "0.03-0.3 %. Left out of the baseline (section 185.9)."],
        kicker="Sections 185.1, 185.9, 185.18", size=14.5)

    d.table(
        "Ceilings for non-decision makers",
        ["FG", "K per cell", "How the ceiling acts", "Biological reading"],
        [["phytoplankton", f(phy["max_carrying_capacity"]),
          "Only a ceiling when the nutrient pool is on; growth is Monod",
          "The real ceiling is the nitrate supply; K is not calibrated. "
          "Ungrazed, the stock sits at K."],
         ["subsurface_phyto.", f(sub["max_carrying_capacity"]),
          "As above; the layer has its own pool and reservoir",
          "Trained zooplankton grazes it off K in April-July"],
         ["microzooplankton", f(mic["max_carrying_capacity"]),
          "Inert: the group grows only by grazing",
          "Field biomass 0.3-5 t/km², peaks ~10"],
         ["benthic_community", f(ben["max_carrying_capacity"]),
          "Logistic, r 0.82/yr hand-set",
          "The user's value; literature ~45-150 t/km². The model's stock is "
          "10-35× smaller, so the gadoid M2 of 0.0048 is inert."]],
        [17, 9, 34, 40], size=10.8, kicker="max_carrying_capacity",
        lead="For a non-decision maker without a nutrient pool the classic "
             "logistic ΔB = r B (1 − B/K) applies per cell, plus seed_rate as "
             "a recolonisation floor (benthos 1e-6, phytoplankton 0).",
        footnote="Bottom trawling depletes the benthos by d × SAR (d 0.10 per "
                 "pass, Rijnsdorp et al. 2020), ~0.2/yr at today's effort.")

    d.table(
        "What rate and ceiling give today",
        ["Trial", "Gadoids", "Pelagic pool", "Zooplankton", "Section"],
        [["Gadoids without K, r 0.511",
          "0.92 → 1.94 → 1.24, starvation in winter of year 3",
          "Jan 4.36 / 3.87 / 2.83, falling", "-", "184.13"],
         ["Gadoids K 1.02, r 0.37", "0.63-0.66, stationary",
          "Jan 5.35 → 7.06, stable from year 3", "-", "184.19"],
         ["Pelagic without K, r 0.648", "0.66 / 0.66 / 0.68",
          "4.07 / 5.54 / 6.77, +15-25 %/yr",
          "5.88 / 3.59 / 2.94; grazed 14×/yr", "185.17"],
         ["Pelagic K 5.0, r 0.34", "0.66 / 0.62 / 0.58",
          "3.09 / 2.75 / 2.73, stationary",
          "6.44 / 5.66 / 5.50; grazed 3.6×/yr", "185.19"]],
        [22, 22, 23, 22, 9], size=11,
        kicker="Annual means t/km², year 1 / 2 / 3",
        lead="Frozen policies in the sandbox, deterministic probe world with "
             "today's fishing. With K the stock stays near today's level "
             "instead of overshooting; there the ceiling takes about the "
             "surplus r − F (gadoids 0.23-0.24/yr).",
        footnote="The policies were trained WITHOUT the ceiling they were "
                 "tested with; the equilibrium (2.75 vs 3.33 for pelagic "
                 "fish, the gadoid decline of 9 %/yr in the last row) should "
                 "be read only after retraining.")

    d.bullets(
        "Open questions on rate and ceiling",
        ["s* assumes the best prey at the ceiling every tick. Realised "
         "satiation is lower, so the budget is an upper bound on the "
         "production the group actually reaches.",
         "K is phenomenological: it stands for recruitment and migration and "
         "is derived from today's biomass, so the equilibrium is today's "
         "state by construction. The model cannot predict a recovery beyond "
         "K - a recovery needs K itself to change.",
         "Scale: K and F apply to the grid, not the stock. Closing trawling "
         "in the parks drops the gadoid F in the grid from 0.12 to ~0.088 "
         "and the equilibrium rises ~13 % - but the stock's F barely "
         "changes.",
         "Realised predation departs from the budget: seals take 0.17-0.19 "
         "of the gadoids (0.12), porpoises 0.003-0.009 (0.056); the gadoid "
         "diet is 62-76 % pelagic against Ecopath's 26-39 %.",
         "Seals and porpoises have no K. Harbour seals are near their "
         "carrying capacity (Carroll et al. 2025) and should grow 0-0.03/yr, "
         "not r_max.",
         "The benthos rate and K are hand-set; the microzooplankton M2 is a "
         "ratio, not a measurement."],
        kicker="What is not solved", size=14.5)

    # ---------------------------------------------------------------- 5
    d.section("5", "Behaviour models",
              "The policy network, the observation and the masking that "
              "make the output ecologically meaningful.")

    d.figure(
        "Policy network", draw_policy_net, kicker="Architecture",
        lead="A behaviour model is a small fully connected network. The "
             "default is two hidden layers of 30 sigmoid nodes; the current "
             "runs use --policynetwork 1 16 tanh, one layer of 16 nodes.",
        items=[("-", "Every decision maker is evaluated in ONE batched "
                     "torch.bmm over the whole grid - a policy sees all its "
                     "cells at once."),
               ("-", "The output is 5 + N logits, N = every FG in the "
                     "project. Masking happens BEFORE the softmax, so an "
                     "impossible action gets zero probability mass."),
               ("-", "The action vector [N, E, S, W, Rest, Eat Y_1 ... Y_n] "
                     "is a distribution: every cell splits its biomass over "
                     "all actions.")])

    d.figure(
        "Observation", draw_observation, kicker="What a population sees",
        footnote="Changing the Observability matrix, the observable impact "
                 "maps, the FG set, the grid layout or the daylight calendar "
                 "invalidates existing checkpoints.")

    widths = {fg: observation_width(fg) for fg in DM_ORDER}
    d.table(
        "The Observability matrix",
        ["Observer", "phyto", "subs.", "micro", "zoo", "pel.", "gad.",
         "porp.", "seal", "bird", "benth."],
        observability_rows("yes"),
        [17] + [8.3] * 10, size=11,
        align=["left"] + ["center"] * 10,
        lead="The matrix sets the observation width: k_X = number of yes on "
             "the row besides the FG itself, counting active FGs only. "
             "Today " + ", ".join(
                 f"{fg} k = {widths[fg][0]} ({widths[fg][1]} channels)"
                 for fg in DM_ORDER if fg not in muted_groups()) + ".",
        footnote="Read from fg_library.yaml. Zooplankton sees its three prey "
                 "but not the herring - no predator-dependent behaviour is "
                 "possible for it. * = muted.")

    d.bullets(
        "Masking and whole units",
        ["Predation is masked when the prey is absent from the cell or not "
         "on the menu. Hunger is not a mask but a continuous factor "
         "h = min(1, max(0, (1 − s)/(1 − u))).",
         "Movement is masked when the neighbour is inaccessible, when the "
         "direction leaves the grid (without --migration on) or when "
         "movement_speed is 0. If every action is masked, Rest opens.",
         "min_split_biomass is an indivisible unit - one individual "
         "(porpoise 50 kg, seal 55 kg, fish 0.5 kg):",
         ("-", "below one unit the distribution collapses to argmax and the "
               "cell moves whole"),
         ("-", "from one unit (rule D, section 159) the cell splits over five "
               "spatial fractions - stay and N/E/S/W - and a fraction below "
               "one unit is handed to the fractions holding one, "
               "proportionally; the eat/rest mix inside stay is kept"),
         "Before rule D 88 % of porpoise cells held less than one animal, 95 % "
         "of that mass created by splits. No mass leaked, but \"21 kg "
         "porpoises\" were meaningless.",
         ("p", "Masking keeps the model from impossible choices and lets the "
               "output follow the model's ecological rules.")],
        kicker="Before and after the softmax", size=15)

    d.figure(
        "Hunger, maintenance and growth", draw_hunger_gate,
        kicker="Two curves that drive everything",
        lead="The hunger factor h_X is 1 up to the maintenance level u_X and "
             "then falls linearly to 0 at a full reserve. The surplus "
             "q_X = s_X − u_X sets the sign of growth; s* is the point where "
             "the growth budget is paid.",
        footnote="Below u_X a population eats with full appetite. Herring "
                 "follows the same hunger curve but grows out of income "
                 "(Part 4). The gadoid growth curve is magnified ten times.")

    d.bullets(
        "Resting is also hiding",
        ["The share of a decision maker that picks Rest in a tick is "
         "protected from predation that tick - and invisible in the next "
         "tick's observation.",
         ("-", "visible share = 1 − π_rest · (1 − floor)"),
         "The floor is the prey's own visibility_floor or is set per PAIR: "
         "detection belongs to the pair. Porpoise biosonar gives herring a "
         "floor of 0.95, seal whiskers 0.85 on gadoids, herring vision 0.5 "
         "on zooplankton.",
         "Light and hiding are different things: the floor says how well "
         "hiding works, the light how dangerous it is to be seen. Resting "
         "also stops grazing - the real cost of diel migration.",
         "The cap on what leaves a cell is set by the BEST detecting "
         "predator, while each predator is limited by what it sees itself.",
         ("-", "An exposure-weighted M1 (section 139) exists in the engine but "
               "is not set in the library.")],
        kicker="visibility_floor")

    # ---------------------------------------------------------------- 6
    d.section("6", "Update rules",
              "The tick, step by step, with the formulas the engine runs.")

    d.figure("Pseudocode for one tick", draw_tick_pipeline,
             kicker="Nine steps, in this order",
             footnote="The FG order is shuffled every tick, but no step "
                      "depends on it: predation and movement are vectorised, "
                      "and in steps 7-8 the water column and the grazers go "
                      "first in a fixed order. The GPU engine mirrors every "
                      "step.")

    d.formula(
        "Step 2: impact mortality",
        [("Lookup in the FG's own table", [
            r"(\mathrm{biomass\_factor}_p,\ \mathrm{energy\_factor}_p) = \mathrm{interp}\left(\mathrm{impact\_table},\ \mathrm{Impact}_p(c)\right)",
            "",
            r"B_X(c) &\leftarrow B_X(c)\,\left(1 - \sum_p \mathrm{biomass\_factor}_p(c)\right) \qquad R_X\ \mathrm{scaled\ alike}",
        ])],
        lead=[("p", "Impacts are taken BEFORE predation, so the prey biomass "
                    "the predators see is already reduced. They act on every "
                    "group with a table - non-decision makers through the "
                    "biomass factor only.")],
        tail=[("-", "The table is interpolated linearly and clipped at the "
                    "endpoints; no extrapolation."),
              ("-", "In step 4 energy_factor raises every action cost by "
                    "(1 + Σ energy_factor)."),
              ("-", "Training: an impact with a baseline map gets that map in "
                    "every world; an observable impact without a map is "
                    "spawned per world; the rest are absent. Inference reads "
                    "only inference.impact_maps - one map per impact, never "
                    "stacked."),
              ("-", "Impacts lie outside the growth budget.")],
        kicker="impact_table")

    d.figure(
        "Today's fishery", draw_fishing_maps,
        kicker="EMODnet, vessels > 12 m, mean 2015-2024",
        items=[("-", f"Gadoids F {f(f_gad, sig=2)}/yr (98 % bottom trawls), "
                     f"pelagic fish {f(f_pel, sig=2)} (98 % pelagic trawls): "
                     "biomass_factor = q × effort, q = F / 1460 / the map's "
                     "mean."),
               ("-", f"Benthos d × SAR, "
                     f"~{f(regional_f('benthic_community'), sig=2)}/yr; "
                     f"porpoise bycatch {f(regional_f('porpoises'), sig=2)}/yr "
                     "on the bottom-trawl map (gillnets do not show in VMS)."),
               ("-", "Wind-farm noise and rotor stay muted until a scenario "
                     "uses them.")],
        footnote="tools/make_fishing_maps.py builds the maps. The same maps "
                 "are baseline maps (training) and inference maps, so "
                 "training, probe and inference all fish today's effort.")

    d.formula(
        "Step 3: predation",
        [("Wanted intake, effective attack rate and removal", [
            r"B_Y^{\mathrm{visible}} &= B_Y\,\left(1 - \pi_{Y,\mathrm{rest}}\,(1 - \mathrm{floor}_{XY})\right) \qquad a_{XY}(t) = a_{XY}\,m_{XY}(t)\,\dfrac{M_{\mathrm{ref}}}{M(t)}",
            r"f_{XY} &= \theta_X(t)\,\dfrac{a_{XY}(t)\,B_Y^{\mathrm{visible}}}{1 + a_{XY}(t)\,h_{XY}\,B_Y^{\mathrm{visible}} + w_X\,B_X} \qquad (\mathrm{type\ II})",
            r"D_{XY} &= B_X\,\pi_{X,\mathrm{eat}\,Y}\,f_{XY}\,h_X \qquad \mathrm{intake}_{XY} = D_{XY}\,\min\left(1,\ \dfrac{0.999\,B_Y^{\mathrm{visible}}}{\Sigma_X D_{XY}}\right)",
        ]),
         ("Bookkeeping", [
            r"R_X &\leftarrow R_X + \sum_Y \mathrm{intake}_{XY}\,\mathrm{assimilation}_{XY}\,\left(\mathrm{energy\_content}_Y + [R_Y/B_Y]\right) \qquad B_Y \leftarrow B_Y - \sum_X \mathrm{intake}_{XY}",
        ])],
        tail=[("-", "Type III replaces B with B² in numerator and denominator "
                    "(specialists and pairs with functional_response 3)."),
              ("-", "m_XY(t) = the daylight factor; M_ref/M(t) only for pairs "
                    "with mixed_layer_reference_m; θ_X(t) = the "
                    "ingestion_q10 factor (a and 1/h together); [R_Y/B_Y] "
                    "for herring only."),
              ("-", "w_X B_X is Beddington-DeAngelis interference. 0.999 keeps "
                    "an overgrazed cell away from exactly zero.")],
        kicker="Holling + interference + light + concentration")

    d.figure(
        "Functional response", draw_functional_response,
        kicker="What the formula does",
        items=[("-", "Left: the gadoid ceilings are an energy limit - herring "
                     "(type III, h 217) and benthos (type II, h 87) saturate "
                     "at the same MJ per t and tick."),
               ("-", "Right: the zooplankton half-saturation is 5 t/km² in a "
                     "10 m layer. In the 6 m summer layer 3 t/km² suffices, "
                     "in November's 17 m it takes 8.5 - the same "
                     "concentration."),
               ("-", "Type III on microzooplankton gives the prey a refuge at "
                     "low density: zooplankton switches away.")],
        footnote="ingestion_q10 scales the whole curve with the season (0.63 "
                 "in March, 1.48 in September) without moving the "
                 "half-saturation.")

    d.formula(
        "Light-dependent attack rate",
        [("Per pair with dark_ratio ρ < 1", [
            r"f(\mathrm{elev}) &= \mathrm{logistic}\left(\dfrac{\mathrm{elev} - \theta}{2^\circ}\right) \qquad F(t) = \mathrm{mean\ over\ the\ tick's\ 10\ min\ steps}",
            r"m_{XY}(t) &= \dfrac{\rho + (1 - \rho)\,F(t)}{\mathrm{annual\ mean}\left(\rho + (1 - \rho)\,F\right)} \qquad a_{XY}(t) = a_{XY}\,m_{XY}(t)",
        ])],
        lead=[("p", "Visual predators see their prey much better in light. "
                    "The CAUSE goes into the model - the attack rate - while "
                    "the prey's answer, diel migration, is left to the "
                    "policies.")],
        tail=[("-", "The normalisation makes the library a the annual mean, "
                    "so the growth budget and the energy budget gate hold on "
                    "average."),
              ("-", "Herring → zooplankton: ρ = 0.1, θ = −8° (biting down to "
                    "0.001 lx, Batty et al. 1990)."),
              ("-", "Gadoids → herring: ρ = 0.6, θ = −5° (stomach data)."),
              ("-", "Seals: no light dependence - they hunt with their "
                    "whiskers, and the diel rhythm is left to Rest and the "
                    "light channel (section 167).")],
        kicker="dark_ratio, light_threshold_deg")

    d.formula(
        "Grazing non-decision makers",
        [("Microzooplankton, at the start of the population step", [
            r"F &= e\,B_p \quad \mathrm{or} \quad \dfrac{(e\,B_p)^2}{e\,B_p + F_0} \qquad e = e_0 + (1 - e_0)\,\dfrac{N}{N + K_e}",
            r"G &= B_g\,\theta_g(t)\,\dfrac{a'\,F}{1 + a'\,h\,F} \qquad a' = a\,\dfrac{M_{\mathrm{ref}}}{M(t)}",
            r"\Delta B_g &= \mathrm{gge}\cdot G \qquad \Delta N_p = \mathrm{excretion\_regeneration}\,(1 - \mathrm{gge})\,G",
        ])],
        lead=[("p", "A non-decision maker with a checked pair that carries an "
                    "attack rate grazes without a policy, with a fixed "
                    "Holling II and a food threshold F0 (ERSEM's perception "
                    "term).")],
        tail=[("-", "θ_g(t) = grazing_q10^((T − 10)/10) in the grazer's "
                    "layer; the take is capped at MAX_HARVEST_FRAC."),
              ("-", "Booked as predation, and the grazer counts as a predator "
                    "in the prey's growth budget."),
              ("-", "The edibility e exists only on the subsurface pair: e0 "
                    "0.1, K_e 2 µmol/l.")],
        kicker="Sections 177, 182")

    d.formula(
        "Step 4: energy after the action",
        [("Cost per action, before movement", [
            r"\mathrm{cost}_a &= B_X\,\pi_{X,a}\,\mathrm{resting\_metabolism}_X\,Q_{10}^{(T(t) - t_{\mathrm{ref}})/10}\,\mathrm{cost}_a\,\left(1 + \sum_p \mathrm{energy\_factor}_p\right)",
            r"R_{a} &= \max\left(0,\ R_X\,\pi_{a} - \mathrm{cost}_{a}\right) \quad (+\ \mathrm{energy\ gain\ for\ eat})",
            "",
            r"G &= \min\left(\kappa\,I - C,\ B\,g\,(s - u)\,ec\right) \qquad \mathrm{income\ funded\ FG\ only\ (herring)}",
        ])],
        lead=[("p", "The reserve is split over the action groups in "
                    "proportion to how the biomass chose, each group pays its "
                    "cost, and the eat group receives the predation energy "
                    "gain.")],
        tail=[("-", "Every action costs. Rest is the cheapest, not free."),
              ("-", "Impacts act multiplicatively: noise makes every action "
                    "dearer instead of killing."),
              ("-", "The herring growth G is taken here, out of the stationary "
                    "reserve, before movement; the growth branch of step 8 "
                    "is then zero for herring."),
              ("-", "After movement the reserve is clipped to [0, B · ME].")],
        kicker="Metabolism")

    d.formula(
        "Step 5: movement",
        [("Outflow, remaining mass and mixing", [
            r"\mathrm{out}_d(c) &= B_X(c)\,\pi_{X,d}(c)\,v_X\,\mathrm{mask}_d(c)",
            r"B_X^{+}(c) &= B_X(c) - \sum_d \mathrm{out}_d(c) + \sum_d \mathrm{out}_d\left(\mathrm{neighbour}_d^{-1}(c)\right)",
            "",
            ("t", "R follows its biomass: when 20 % of the mass moves, 20 % of "
                  "that action group's reserve moves with it."),
        ])],
        lead=[("p", "Movement creates and destroys no biomass. It only splits "
                    "the cell's mass into outflows and remaining mass.")],
        tail=[("-", "v_X is a distance per tick, not a probability; it scales "
                    "linearly with the tick length."),
              ("-", "Blocked flow stays in the source cell. An outflow whose "
                    "target would end below the extinction threshold is "
                    "returned; a cell below the threshold that still sends "
                    "gives ALL its mass to its neighbours (section 156)."),
              ("-", "With --migration on the edge flow leaves the grid and is "
                    "reintroduced concentrated along the edge.")],
        kicker="Slice-assign, four directions")

    d.bullets(
        "Currents",
        ["With --currents on an advection is added after movement: a "
         "cloud-like noise field scrolling east and south moves a share of "
         "the biomass every tick.",
         ("-", "The field is sampled procedurally from the tick counter and a "
               "world seed - it is not stored and does not consume the "
               "ecology's random numbers."),
         ("-", "--current-strength sets the largest share per tick (default "
               "0.1)."),
         "Which groups drift is set by current_response: phytoplankton, "
         "subsurface phytoplankton, microzooplankton and zooplankton 1.0, "
         "benthos and fish 0.",
         "The nutrient pool sits in the producer's reserve slot and therefore "
         "moves with the water, conservatively.",
         "The current shares movement's masks - blocked flow stays - and "
         "creates or destroys no mass."],
        kicker="current_response")

    d.formula(
        "Step 7: the water column",
        [("Layer exchange per tick, before any growth", [
            r"(X_m,\ X_d)' &= A(t)\,(X_m,\ X_d) \qquad X = \mathrm{water,\ nutrients,\ algae}",
            r"f_{\mathrm{up}} &= \dfrac{\max(\Delta M, 0)}{H} \qquad f_{\mathrm{down}} = \dfrac{\max(-\Delta M, 0)}{M} \qquad + \ e(t),\ e_b(t)",
            ("t", "M = the mixed-layer depth, H = 30 m − M; e between the "
                  "layers, e_b between the subsurface layer and the reservoir "
                  "at 50 m."),
        ])],
        lead=[("p", "When the mixed layer deepens in autumn the subsurface "
                    "layer's water, nutrients and algae are entrained into "
                    "the mixed layer; when it shoals in spring they are left "
                    "behind. The exchange happens before any group grows.")],
        tail=[("-", "New nitrogen comes only through e_b (N_res − c_d): the "
                    "reservoir is A17's nitrate at 50 m by month."),
              ("-", "Subsurface light: surface light × e^(−0.29 M) × "
                    "e^(−0.17 z) over the layer (Kd 0.17 = 1.7 / Secchi "
                    "depth)."),
              ("-", "The subsurface I_k of 30 and the summer exchange of "
                    "0.2 m/d are generous against the literature (63-117; "
                    "0.02-0.05) - fitted to A17 nitrate and column "
                    "chlorophyll.")],
        kicker="Section 180")

    d.formula(
        "Step 8: growth for decision makers",
        [("M1, the regional ceiling, then growth or starvation", [
            r"B, R &\leftarrow (B, R)\,(1 - M1) \qquad B, R \leftarrow (B, R)\,(1 - c\,\bar B)",
            r"\mathrm{wish} &= B\,\mathrm{MG}_X\,\theta_X(t)\,q \qquad q = s - u \geq 0",
            r"\Delta B &= \min\left(\mathrm{wish},\ \dfrac{\max(0,\ R - u\,\mathrm{ME}\,B)}{ec}\right) \qquad R \leftarrow R - \Delta B\,ec",
        ]),
         ("Starvation (q < 0) is a loss, not a purchase", [
            r"\Delta B &= B\,\mathrm{starve\_rate}\,q \qquad R\ \mathrm{scaled\ with}\ B",
        ])],
        lead=[("p", "Growth is GATED by satiation but limited by the reserve "
                    "above the maintenance level - and that reserve is "
                    "DEBITED. MG_X is the derived growth_rate.")],
        tail=[("-", "Without the debit new biomass carries an energy content "
                    "that was never taken from anything: measured at "
                    "198-344 % of the mass eaten, against a thermodynamic "
                    "ceiling of 28.9 %."),
              ("-", "ME/ec becomes a hard ceiling on growth_rate: zooplankton "
                    f"{f(zoo['max_energy_reserve'] / zoo['energy_content'])} "
                    f"against g {f(zoo['growth_rate'])}."),
              ("-", "The herring growth branch is zero here (it grew in step "
                    "4); starvation applies to everyone.")],
        kicker="Mass-balanced since September")

    d.figure(
        "Energy bookkeeping in one tick", draw_energy_flow,
        kicker="Where the mass comes from",
        items=[("-", "The egested share is (1 − assimilation) of intake: 0.35 "
                     "for zooplankton on phytoplankton, 0.10 for porpoises on "
                     "herring."),
               ("-", "The reserve is the only buffer. Every action draws on "
                     "it; temperature and impacts raise the cost."),
               ("-", "Reserve becomes biomass at the price energy_content per "
                     "tonne. A population at the maintenance level cannot "
                     "grow however much food there is.")],
        footnote="tools/probes/fg_year_budget.py and the hloss probes read "
                 "the split between respiration, growth and egestion per FG.")

    d.bullets(
        "Losses: four channels",
        ["The tick books four loss channels per FG: predation, starvation, "
         "impact and natural mortality. lib/environments/ecosystem_env/"
         "loss_accounting.py is the only place that turns them into shares.",
         "Natural mortality (M1) needs --mortality on and is scaled by "
         "--mortality_multiplier. It is taken from biomass and reserve "
         "before growth - for non-decision makers from biomass only.",
         "The regional ceiling is booked as natural loss too, but always acts "
         "and is not scaled.",
         "Starvation is catabolism: below the maintenance level biomass "
         "shrinks by starve_rate · |q|, and the reserve is scaled by the same "
         "ratio.",
         "Mass zeroed by the extinction threshold is booked as starvation.",
         "Impact mortality - fishing, bycatch, noise, rotors - comes from the "
         "impact maps."],
        kicker="Predation, starvation, impact, natural")

    d.formula(
        "Step 8: growth for non-decision makers",
        [("Producers with a nutrient pool; the rest logistic", [
            r"B &\leftarrow B\,(1 - M1) \qquad N \leftarrow N\,\mathrm{keep}(t) + \mathrm{supply}(t)",
            r"\Delta B &= \min\left(r\,L(t)\,B\,\dfrac{N}{N + k_N Y H},\ N,\ \mathrm{CC} - B\right) \qquad \mathrm{producer}",
            r"\Delta B &= r\,B\,\left(1 - \dfrac{B}{\mathrm{CC}}\right) + \mathrm{seed\_rate}\cdot\mathrm{CC}\cdot 10^{U(-1,1)} \qquad \mathrm{others}",
        ])],
        lead=[("p", "Grazing was already taken in step 3 (zooplankton) and at "
                    "the start of step 7 (microzooplankton). The old sine "
                    "term is gone; the season comes from the calendar.")],
        tail=[("-", "I_0: clear-sky irradiance × 0.45 PAR × 4.57 µmol/J × "
                    "cloud transmission (SMHI). K_d 0.29/m; mixed layer "
                    "6-17 m (A17)."),
              ("-", "A nutrient-stress loss (stress_loss_max, ERSEM form) "
                    "exists in the engine but is off: the summer minimum "
                    "comes from grazing."),
              ("-", "The benthos is logistic with a hand-set r of 0.82/yr.")],
        kicker="Sections 138, 172-177")

    d.bullets(
        "Bookkeeping summary",
        ["New biomass = biomass after impacts, predation, movement and layer "
         "exchange + growth − starvation − natural mortality − regional "
         "ceiling.",
         "Biomass can only grow when the energy allows it: for decision "
         "makers the reserve above the maintenance level (herring: the "
         "tick's income), for producers the nutrient pool, for "
         "microzooplankton its grazing.",
         "The reserve always follows its biomass; for herring it follows into "
         "the predator too.",
         "Nitrogen has an open budget: supply from the reservoir, "
         "regeneration from grazing, export by sinking and faecal pellets.",
         "What is NOT conserved: mortality, impacts, the ceiling and the "
         "extinction threshold take mass out of the system; migration moves "
         "mass across the edge. Predation, movement, currents and layer "
         "exchange conserve it."],
        kicker="What is conserved and what is not")

    # ---------------------------------------------------------------- 7
    d.section("7", "Training with ARS",
              "How the behaviour models are made, what they optimise and "
              "what they must not be confused with.")

    d.bullets(
        "Training: aim and perspective",
        ["The aim is not ecologically optimal policies but plausible, stable "
         "and testable policy networks for scenario analysis.",
         "The model is Eulerian. There is no individual to follow over "
         "several steps - biomass splits, moves and mixes. What is "
         "optimised is therefore how well a FIELD does over a rollout.",
         "What runs is ARS: gradient-free and parallel, on the CPU "
         "(train.py) or in tensor form on the GPU (train_gpu.py).",
         "With the calendar, training uses one-year rollouts, 1,460 ticks, so "
         "that every rollout holds every season. Multi-year effects are read "
         "in separate inference runs.",
         "A group reward rewards restraint: the zooplankton rests 62-83 % "
         "although eating pays individually (section 170) - a commons "
         "discipline real copepods do not show."],
        kicker="What ARS is meant to achieve")

    d.formula(
        "Fitness",
        [("Default: total energy, integrated over the rollout", [
            r"E_{\mathrm{tot}}(t) &= \sum_c \left(B_X(c,t)\,\mathrm{energy\_content}_X + R_X(c,t)\right)",
            r"\mathrm{Fitness}_X &= \dfrac{1}{H}\sum_{t=1}^{H}\,\log\dfrac{E_{\mathrm{tot}}(t) + \varepsilon}{E_{\mathrm{tot}}(0) + \varepsilon}",
            ("t", "H = --n_eval_ticks, ε = max(1e-6 E_tot(0), 1e-9)"),
        ])],
        tail=[("-", "Biomass and reserve in the SAME currency: one tonne of "
                    "biomass is worth exactly its energy content."),
              ("-", "The integral form scores lower a policy that crashes its "
                    "prey mid-rollout and recovers at the end."),
              ("-", "The logarithm makes the reward scale invariant: halving "
                    "costs exactly what doubling is worth."),
              ("-", "--legacyreward gives the May edition's α log(B) + "
                    "β log(R) + κ t/H.")],
        kicker="What a rollout measures")

    d.formula(
        "Local per-cell reward",
        [("--local_reward: the same question, asked per cell", [
            r"A(c,t) &= B(c,t)\,\mathrm{energy\_content} + R(c,t)",
            r"\mathrm{reward}(c) &= \mathrm{clip}\left(\dfrac{B(c,t+1)}{A(c,t)},\ 0.2,\ 5.0\right) \qquad B(c,t+1) = \sum_d \mathrm{share}_d(c)\,Q(\mathrm{target}(c,d),\ t+1)",
            r"\mathrm{Fitness}_X &= \mathrm{mean}_t\ \dfrac{1}{N_{\mathrm{cells}}}\sum_c w_c\,\log\,\mathrm{reward}(c)",
        ])],
        lead=[("p", "The global reward is dominated by the big cells: a "
                    "policy that handles a thousand-tonne cell well and a "
                    "one-kilogram cell badly scores the same as the "
                    "reverse.")],
        tail=[("-", "share_d(c) is an identity, not an approximation: "
                    "everything after movement is cell-wise multiplicative."),
              ("-", "Use --local_reward_norm grid with the log metric: sum "
                    "rewards spreading thin and mean rewards killing the "
                    "worst cells."),
              ("-", "w_c = A_cᶿ; θ = 1 gives back the global reward.")],
        kicker="Scale invariant per cell")

    d.formula(
        "The ARS update",
        [("Per iteration, per decision maker with weights θ", [
            r"\delta_1, \ldots, \delta_N &\sim \mathcal{N}(0,\ I) \qquad F_i^{\pm} = \mathrm{fitness}(\theta \pm \sigma\,\delta_i)",
            ("t", "The same world, seed and calendar start for + and − (CRN)."),
            r"\theta &\leftarrow \theta + \dfrac{\eta}{b\,\sigma_F}\sum_{i \in \mathrm{top}\,b}\left(F_i^{+} - F_i^{-}\right)\delta_i \qquad b = N/2",
        ])],
        tail=[("-", "ARS-V2: observations are normalised by a running mean "
                    "and variance per FG and clipped to ±10."),
              ("-", "Top-b picks the pairs with the highest max(F⁺, F⁻); σ_F "
                    "is their standard deviation, so the step is independent "
                    "of the reward's scale."),
              ("-", "The gradient through a softmax share is proportional to "
                    "the share: an action pushed close to zero cannot recover "
                    "(the seals' pelagic eat channel, section 185.14).")],
        kicker="Augmented Random Search")

    d.bullets(
        "Co-evolution",
        ["The default is co-evolution: ALL decision makers are perturbed "
         "together within the same delta pair and evaluated in ONE shared "
         "rollout. Each FG gets its own fitness from the same simulation.",
         ("-", "A prey policy and a predator policy only make sense against "
               "each other."),
         "--no_coevolution gives the alternating schedule: one FG at a time, "
         "the rest frozen.",
         "One generation = --iter-per-gen iterations. After each generation "
         "checkpoints are saved and a deterministic probe logs the biomass "
         "per FG.",
         "The evaluation probe follows the rollout length and runs after "
         "every update by default (--eval-every 1; the viewer's slider paces "
         "the biomass probe too)."],
        kicker="Every species at once")

    d.bullets(
        "GPU training: train_gpu.py",
        ["lib/gpu/ is a tensor version of the same tick: every world, delta "
         "pair and sign runs at once, in blocks of ticks replayed as CUDA "
         "graphs. Parity tests compare every step - nutrient pool, layer "
         "exchange, grazing, Q10, income growth and regional K included - "
         "with the CPU engine.",
         "Current runs (fix64-fix69): 32 delta pairs × 1 world × 1,460 ticks, "
         "1 × 16 tanh, η 0.06, σ 0.1; mortality, migration, currents, "
         "daylight, temperature, nutrient pool and fishing on; ~85 s per "
         "update.",
         "The calendar start is drawn per WORLD and shared by every pair "
         "(section 150).",
         "Keep η/σ ≲ 0.6: at η 0.1 and σ 0.05 the weight norm grew like a "
         "random walk.",
         "--init-from starts from another run's policies when the "
         "observation and the output are unchanged.",
         ("-", "The viewer's probe re-reads fg_library.yaml at every probe, "
               "but the trainer keeps the library it started with.")],
        kicker="lib/gpu/", size=14.5)

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
        lead="Entropy bonus, argmax penalty and temperature annealing act on "
             "the action distribution instead of on the biology. They remain "
             "as flags but are off by default, with or without a profile, on "
             "both trainers.",
        footnote="--profile sanity | info | deep. Explicit flags always win "
                 "over the profile's values.")

    d.figure(
        "Training worlds", draw_spawn_examples,
        kicker="Four spawn strategies",
        lead="To avoid overfitting to a single map the policies train in many "
             "randomised worlds. Every FG has its own spawn strategy and a "
             "range for the start biomass.",
        items=[("-", "uniform: independent random weight per cell. perlin: "
                     "natural-looking noise (phytoplankton, benthos)."),
               ("-", "colony: Gaussian colonies with jittered amplitude "
                     "(herring, gadoids)."),
               ("-", "env_driven: the weight is built from other fields - "
                     "zooplankton and microzooplankton follow phytoplankton, "
                     "porpoises herring, seals gadoids.")],
        footnote="The start biomass is drawn per world from the range in "
                 "mareld2.yaml, the start day from the calendar, the "
                 "nutrient pool from the surface nitrate on the start day.")

    d.bullets(
        "Checkpoints",
        ["A checkpoint is results/<run>/policy_<fg>.pth with "
         "{'state_dict': ..., 'obs_stats': {...}} - the weights AND the "
         "observation normalisation's statistics.",
         "--resume continues a run; --init-from starts a new one from "
         "another run's policies; inference.py runs them against a "
         "deterministic world.",
         "The following invalidate existing checkpoints:",
         ("-", "the FG set: the output has 5 + N columns in alphabetical "
               "order, so a new FG moves every eat column (sections 177, "
               "180)"),
         ("-", "the Observability matrix, observable impacts, the grid "
               "layout and the daylight calendar"),
         "Library changes (K, r_max, attack rates) do not formally invalidate "
         "them, but a policy trained in a different world must be retrained "
         "before it is quoted.",
         "Copy a checkpoint before comparing - a running job overwrites "
         "policy_*.pth."],
        kicker="What is saved and what breaks it")

    # ---------------------------------------------------------------- 8
    d.section("8", "Diagnostics and evaluation",
              "Tools that test the mechanics without ARS, and what the model "
              "does against data.")

    d.table(
        "The viability rig",
        ["#", "Condition", "Formula", "Default"],
        [["1", "Survival", "B_f(t) > 0 for every t ≤ T", "-"],
         ["2", "Floor", "eq_f ≥ floor · B_f(0)", "floor = 0.10"],
         ["3", "Ceiling", "eq_f ≤ ceiling · B_f(0)", "ceiling = 10"],
         ["4", "Stationarity",
          "max(eq_f/prev_f, prev_f/eq_f) ≤ max_drift", "max_drift = 2.0"]],
        [6, 26, 44, 24], size=12.5, kicker="Frozen behaviour",
        lead="The rig runs the world with FROZEN, hand-coded behaviour: fast "
             "and deterministic, and so good for measuring a parameter or "
             "mechanics change in isolation. eq_f is the mean biomass in the "
             "last window of the horizon.",
        footnote="Arms: greedy, greedy_hide, greedy_flee, eat, eat_pays, _stay "
                 "variants and mixed arms (policy for every group but one). "
                 "The verdict is about THAT behaviour, not the world.")

    d.table(
        "Hand-coded against trained",
        ["FG", "greedy", "greedy_stay", "eat", "trained (fix58)"],
        [["phytoplankton", "0.12", "0.12", "0.14", "0.59-0.62"],
         ["zooplankton", "0.16", "0.16", "0.16", "0.72"],
         ["benthic_community", "0.13-0.27", "0.45-0.62", "1.24-1.32",
          "0.55-0.62"],
         ["pelagic_fish", "0.001", "0.01", "0.25", "0.55-0.59"],
         ["gadoids", "0.92-1.03", "0.53-0.64", "0.20-0.24", "3.2"],
         ["porpoises", "extinct t 43-49", "0.34-0.40", "extinct t 1,755+",
          "1.00-1.18"]],
        [22, 18, 18, 18, 18], size=11.5,
        align=["left", "right", "right", "right", "right"],
        kicker="Final-year mean / spawn, 5 years, 3 seeds (section 169)",
        lead="Is the behaviour just \"go to food, flee predators, eat when "
             "hungry\"? The trained policies beat every hand-coded arm for "
             "every group.",
        footnote="It is the THRESHOLDS that decide: when to stop moving and "
                 "how hard to graze. Only the porpoise can be replaced by a "
                 "plain rule; zooplankton must rest ~80 %, herring stop "
                 "travelling sooner and gadoids keep searching for herring.")

    d.bullets(
        "The energy budget gate",
        ["Can every decision maker close its energy budget over its RATION? "
         "The question is answered without a rollout, in seconds: "
         "python3 tools/probes/budget_gate.py",
         ("p", "ration      = ceiling 1/h at full appetite   [t prey / t "
               "predator / tick]"),
         ("p", "requirement = resting_metabolism · feeding_cost / ration ≤ "
               "the ration's energy quality"),
         "The gate is evaluated over the whole menu: the engine sums intake "
         "over every prey and gives the predator ONE energy gain.",
         ("-", "Today all five active predators close on any diet. "
               "Zooplankton breaks even at 0.65 t/km² of visible "
               "phytoplankton or 0.28 of microzooplankton; seals at 0.14 t of "
               "visible gadoids per cell."),
         ("-", "The seals' pelagic pair does not close on its own "
               "(62.9 t/cell): low-quality food, on purpose."),
         ("-", "The gate computes at t_ref - for herring a summer cost.")],
        kicker="Does the budget close?", size=15)

    d.table(
        "Plankton against A17",
        ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
         "Oct", "Nov", "Dec"],
        [["zooplankton, model", "0.27", "0.72", "2.04", "6.74", "18.3",
          "10.2", "6.25", "3.33", "2.08", "1.02", "0.30", "0.39"],
         ["zooplankton, A17 target", "0.21", "0.18", "1.56", "19.3", "26.0",
          "11.4", "6.9", "4.6", "3.7", "2.9", "3.5", "0.92"],
         ["nitrate 0-10 m, A17", "5.1", "4.8", "2.3", "0.3", "0.1", "0.1",
          "0.1", "0.1", "0.1", "0.3", "1.3", "3.6"],
         ["chlorophyll 0-10 m, A17", "1.21", "3.88", "3.28", "1.48", "0.94",
          "0.80", "0.46", "0.65", "0.79", "1.81", "1.54", "0.77"]],
        [24] + [6.3] * 12, size=10.5, align=["left"] + ["right"] * 12,
        kicker="t/km², µmol/l, µg/l",
        lead="fix68 generation 3, three probe years (section 184.11). The "
             "target is the resident stock in Dec-Mar (A17 minus the Calanus "
             "and Metridia rising from the Norwegian Trench) and A17's total "
             "in Apr-Nov.",
        footnote="Nitrate log-RMSE 0.54, r(P, chlorophyll) +0.81, "
                 "microzooplankton 0.85 t/km². May-September match; February "
                 "is 4× high, April 3× low (A17's April is Calanus) and "
                 "Oct-Dec 2-12× low. "
                 "Gross primary production ~40 g C/m²/yr against 190-290 in "
                 "the Kattegat.")

    d.table(
        "Realised against budgeted mortality",
        ["Quantity", "Budget / target", "Realised", "Reading"],
        [["Gadoids: seals", "0.12 /yr", "0.17-0.19",
          "Seals take more gadoids than budgeted"],
         ["Gadoids: porpoises", "0.056", "0.003-0.009",
          "Porpoises take almost no gadoids"],
         ["Gadoids: fishing", "F 0.12", "0.10-0.115", "As intended"],
         ["Herring: gadoids", "0.30", "0.26-0.31", "As Ecopath (0.30)"],
         ["Gadoid diet, pelagic share", "26-39 % (Ecopath)", "62-76 %",
          "1.6-2.5× the pelagic ration"],
         ["Seal diet, pelagic share", "~10 %", "0.2-0.3 %",
          "The policy's eat channel is saturated (Part 9)"],
         ["Zooplankton grazed by herring", "1.9× stock/yr (Ecopath)",
          "3.6× (with K), 14× (without)", "K removed the overgrazing"]],
        [28, 20, 20, 32], size=11,
        kicker="Sections 184.15-19, 185.10-19",
        lead="The growth budget assumes every predator takes its M2. The "
             "sandbox with frozen policies and the regional ceiling shows "
             "where that holds and where it does not.",
        footnote="Realised M2 = annual take / mean prey biomass. The policies "
                 "were trained before 185.20; read again after retraining.")

    d.bullets(
        "What the rig and the probes are for",
        ["The rig is a measuring instrument for the mechanics, not a verdict "
         "on the world. With frozen behaviour and a fixed seed it is "
         "deterministic, so a library change can be measured in isolation "
         "without training noise.",
         "It catches faults no policy can rescue. It does not say a world is "
         "unusable: the trained policies do not graze the phytoplankton "
         "down the way greedy behaviour does.",
         "The sandbox probes (phyto_bloom, pair_intake, hloss, gloss, "
         "sealscan) run trained policies with hooks into the engine, so a "
         "mechanism can be tried before it is built in - every built-in "
         "mechanism has reproduced its probe within ~1 %.",
         "Viability should be judged with the policies actually used, on "
         "several seeds and with a stationarity test that tolerates natural "
         "cycles. That is not built yet."],
        kicker="Diagnosis, not verdict")

    # ---------------------------------------------------------------- 9
    d.section("9", "Strengths, weaknesses and suggested improvements",
              "An overall assessment of the model after section 185: what it "
              "does well, where it falls short and what should come next.")

    d.bullets(
        "Strengths: mechanics and method",
        ["Mass, energy and nitrogen are booked closed. Predation, movement, "
         "currents and layer exchange conserve mass; growth is paid from the "
         "reserve, the income or the nutrient pool; every loss has a cause.",
         "Growth is traceable: every g is derived from r_max, M1 and Ecopath "
         "M2 at s*, and every r_max has a source. A ceiling is added only "
         "where a measured overshoot shows it is needed, and it is derived "
         "from today's biomass and fishing.",
         "Site data drive the season: light climate, temperature, mixed "
         "layer, nitrate and P-I curves from SMHI and SHARK A17; the fishery "
         "from EMODnet.",
         "Behaviour is learned, not hacked. Trained policies beat every "
         "hand-coded rule for every group.",
         "Reproducible and tested: CRN, deterministic probe worlds, CPU/GPU "
         "parity per step and over 1,300 tests.",
         "Fast diagnostics in three steps: the energy budget gate in seconds, "
         "the frozen rig and the sandbox probes in minutes."],
        kicker="What holds", size=15)

    d.table(
        "Strengths: behaviour that resembles nature",
        ["Phenomenon", "In the model", "Comparison"],
        [["Spring bloom and summer minimum",
          "Phytoplankton blooms in February-March, minimum in summer; "
          "nitrate taken up in spring", "A17 chlorophyll and nitrate, "
          "r(P, chlorophyll) +0.8"],
         ["The zooplankton spring", "Starts in March, peaks in May",
          "A17 resident stock: March 1.6, peak May"],
         ["Herring fat cycle", "s 0.35-0.41 in February, 0.80 in May-June, "
          "no starvation", "Lean Q1, fat Q3; no winter starvation of adult "
          "herring"],
         ["Growth when food is there", "Herring grows only while it eats",
          "Somatic growth April - September"],
         ["Diel vertical migration", "Zooplankton rests more in light",
          "DVM where there is a real night"],
         ["Visual planktivore", "Herring eats in light, rests at night",
          "Herring feed at dawn and dusk"],
         ["Seasonal diet shift", "Gadoids: herring and zooplankton in summer, "
          "benthos in winter", "Direction plausible, share elevated"],
         ["Continuous foraging", "Porpoises eat 97-100 % of the time",
          "Wisniewska et al. 2016"],
         ["Stable stocks with a ceiling", "Gadoids 0.63-0.66, herring stable",
          "Stocks roughly stationary 2010-2020"]],
        [22, 42, 32], size=10.6, kicker="Emergent, not programmed",
        footnote="From sections 139, 156, 169, 177, 181-185. None of it is "
                 "written into the rules - it follows from the calendar, the "
                 "energy budget and what the policies have learned.")

    d.bullets(
        "Weaknesses: biology and structure",
        ["Primary production is ~40 g C/m²/yr against 190-290 in the "
         "Kattegat; the subsurface layer that carries it is generously "
         "calibrated (I_k 30 vs 63-117, summer exchange 0.2 vs 0.02-0.05 "
         "m/d) and blooms two months early.",
         "Zooplankton is 2-12× too low in October-December and April is 3× "
         "low: the model has no import of Calanus from the Norwegian Trench.",
         "The herring decline followed the gadoid build-up; with K it has "
         "moved to an equilibrium below today's level (2.75 vs 3.33).",
         "The top predators' diets are fitted, not emergent: porpoises take "
         "almost no gadoids, seals too many, and the gadoid diet is 2× too "
         "pelagic.",
         "Seals and porpoises lack density dependence. The model's gadoids "
         "(0.69 t/km²) are ~4× below the IBTS density.",
         "Depth is two layers for plankton but one for fish and mammals: "
         "seal and herring share the same cell, so a low attack rate has to "
         "carry the vertical separation.",
         "Several values rest on one source or are fitted [D]: dark ratios, "
         "edibility, exchanges, κ."],
        kicker="Where the model falls short", size=14)

    d.bullets(
        "Weaknesses: growth budget and ceiling",
        ["K is phenomenological and derived from today's biomass: the "
         "equilibrium is today's state by construction, and a recovery "
         "beyond K cannot emerge in the model.",
         "K and F apply to the grid but stand for processes at the stock "
         "scale (recruitment, migration). A local scenario that closes "
         "fishing can overstate the effect.",
         "s* is an upper bound: realised satiation is lower, so the groups "
         "produce below budget when food limits.",
         "The budget assumes every M2 is taken. When realised predation "
         "departs (seals, porpoises) the prey's equilibrium moves, and no "
         "mechanism corrects it.",
         "The benthos is hand-set (r, K 10 vs ~100), and the "
         "microzooplankton M2 of 147 is a ratio - both lie outside the "
         "derived budget.",
         "r_max has a wide range: gadoids 0.29-0.42, pelagic 0.27-0.40. The "
         "equilibrium is sensitive to r close to F."],
        kicker="Part 4 read critically", size=14.5)

    d.bullets(
        "Weaknesses: training and evaluation",
        ["Policies train on one-year rollouts. Multi-year trends lie outside "
         "the training distribution.",
         "A group reward teaches commons discipline: the zooplankton holds "
         "the phytoplankton near K/2, which no individual copepod does.",
         "ARS cannot revive an action that has been pushed down: the seals' "
         "pelagic eat channel sits at a logit of ~−8 and does not train "
         "back, even after the attack rate was raised.",
         "The generation-to-generation spread with one evaluation seed and "
         "one world is as large as many of the effects being measured.",
         "Every library change needs retraining before numbers are quoted. "
         "The pelagic K (185.20) has not been trained in yet.",
         "The network is small (1 × 16) and has no memory; the policies see "
         "the light but not the date."],
        kicker="Where the method falls short", size=15)

    d.table(
        "Suggested improvements: growth and ceiling",
        ["#", "Suggestion", "Reason", "Sections"],
        [["1", "Retrain with both K and read the equilibrium on several "
          "seeds", "No K is trained into the policies yet", "185.19-20"],
         ["2", "Seasonal immigration and emigration for pelagic fish instead "
          "of K", "Mechanistic version of the net emigration", "183.10, 185.18"],
         ["3", "Density-dependent growth for seals (pups, fecundity)",
          "Harbour seals are near their carrying capacity", "168.5"],
         ["4", "K as a function of habitat - as a sensitivity case only",
          "No published function; double-counts the benthos pathway", "185.9"],
         ["5", "Benthos with r_max and the literature K",
          "Hand-set; the gadoid M2 is inert", "136, 168.1"],
         ["6", "Realised satiation as the budget's reference",
          "s* overstates production when food limits", "160.3"]],
        [4, 44, 38, 14], size=11, kicker="Part 4",
        footnote="Every suggestion is measured in the sandbox with trained "
                 "policies before the library changes.")

    d.table(
        "Suggested improvements: the model",
        ["#", "Suggestion", "Reason", "Sections"],
        [["1", "A Calanus import as a seasonal source for zooplankton",
          "April and autumn too low; A17 receives deep-water animals",
          "183.5-6"],
         ["2", "Calibrate the subsurface layer to the literature with "
          "recycling within the layer", "I_k and exchange generous; spring "
          "too early", "181.8, 182.10"],
         ["3", "A remineralisation pool (detritus/DOM)",
          "Plankton M1 vanishes from the nitrogen cycle", "179"],
         ["4", "Seal and porpoise diets as targets when retraining",
          "Realised M2 departs from the budget", "185.10-17"],
         ["5", "Spawning cost for herring, diapause for zooplankton",
          "Fat minimum and winter survival", "143.6, 183"],
         ["6", "Temperature-scaled attack rates for fish",
          "The next step after Q10 on metabolism", "143.2"]],
        [4, 44, 38, 14], size=11, kicker="Biology and mechanics",
        footnote="The order is a judgement of benefit against cost.")

    d.table(
        "Suggested improvements: training and evaluation",
        ["#", "Suggestion", "Reason", "Sections"],
        [["1", "Reset saturated eat channels when retraining (logit copy)",
          "The ARS gradient cannot lift a share near zero", "185.15"],
         ["2", "Several evaluation seeds and worlds per update",
          "The spread between generations is large", "184.11"],
         ["3", "A cycle-tolerant viability criterion on inference data",
          "The two-window test fails natural cycles", "126"],
         ["4", "Norm control in ARS; σ as the exploration knob",
          "Random walk in the weights at high η/σ", "154.8-10"],
         ["5", "Multi-year evaluation next to the one-year training",
          "Multi-year trends are never optimised", "156.2, 163"],
         ["6", "Activate wind-farm scenarios only once the baseline is "
          "viable after retraining", "A comparison needs a self-sustaining "
          "base", "-"]],
        [4, 44, 38, 14], size=11, kicker="Method",
        footnote="None of the suggestions needs shaping terms in the reward.")

    # --------------------------------------------------------------- 10
    d.section("10", "Scenarios and runs",
              "What a scenario is, how it is run and what separates this "
              "edition from the May edition.")

    d.bullets(
        "Scenario",
        ["A scenario is a set of maps for a given set of functional groups "
         "and a given grid.",
         "Scenarios are analysed with the update rules and the trained "
         "behaviour models, and are always read as a COMPARISON: baseline "
         "against project, with the same seeds and the same policies.",
         "Today's fishery is in both: the bottom-trawl and pelagic maps are "
         "baseline maps in training and inference maps. A project "
         "alternative replaces a map - a closed park, a changed effort - and "
         "never stacks a second impact.",
         "Wind-farm noise (observable) and rotor are muted. vindparker.npz "
         "builds the rotor map from the parks' real geometry.",
         "The project map must state whether the parks' trawl effort is "
         "removed or displaced within the grid: the gadoid F in the grid "
         "falls from 0.12 to ~0.088 if it is removed.",
         "The difference is reported as a relative change per functional "
         "group, under stated assumptions."],
        kicker="Definition and practice", size=15)

    d.formula(
        "Runs",
        [("Training, inference and gates", """
python3 train.py --project mareld2.yaml --run-name <name> < /dev/null
python3 train_gpu.py --project mareld2.yaml --run-name <name> \\
        --policynetwork 1 16 tanh --mortality on --migration on ...
python3 inference.py --project mareld2.yaml --run-name <name>
python3 tools/biomass_html.py results/<name>                # plots.html
python3 tools/probes/budget_gate.py        # seconds, no rollout
python3 tools/probes/daylight_table.py     # the calendar month by month
python3 tools/make_fishing_maps.py         # EMODnet -> maps/*.npz
python3 tools/viability.py                 # ~4 min, frozen behaviour
python3 -m pytest tests/ -q
""")],
        tail=[("-", "Key flags: --tick-length 1-6, --mortality on, "
                    "--migration on, --currents on, --local_reward, "
                    "--init-from, --resume, --visual, --rnd_baseline, "
                    "--eval-every."),
              ("-", "The document: python3 tools/make_strategy_deck2_en.py and "
                    "tools/make_strategi_deck2.py.")],
        kicker="What runs")

    d.table(
        "Deliberate departures from the May edition",
        ["What", "Why"],
        [["Growth is mass-balanced and growth_rate is derived at s*",
          "The term B·(1 + MG·q) created mass without debiting energy; the "
          "budget at a full reserve was never reached. The net at low "
          "density becomes the literature r_max."],
         ["Gadoids and pelagic fish have a regional K",
          "r_max is a Schaefer r; without K the pools grew at r − F until "
          "the food ran out, and crashed."],
         ["Herring grows out of income",
          "Fat became biomass after the food had gone; the herring starved "
          "every winter."],
         ["Phytoplankton has a nutrient pool and two layers",
          "Logistic growth toward a constant K gave no spring bloom and a "
          "summer maximum instead of a minimum."],
         ["Hunger is continuous, not a mask",
          "Full appetite up to the maintenance level; eating is set by food "
          "and stomach capacity."],
         ["Fitness counts biomass and reserve in one currency",
          "α and β were arbitrary weights between quantities that already "
          "have an exchange rate."],
         ["Rollouts are a year and the season comes from a calendar",
          "The calendar needs a rollout to hold every season."]],
        [32, 68], size=10.8, kicker="The document against the code")

    d.bullets(
        "Open questions",
        ["What does the equilibrium look like once the policies have been "
         "trained with both K? Does pelagic fish settle near 3.33 and gadoids "
         "near 0.69?",
         "Should the zooplankton get a Calanus source, and how large is it on "
         "a 100 m bank without diapause habitat?",
         "Should the subsurface layer be calibrated to the literature, and "
         "what then keeps the autumn zooplankton and the herring up?",
         "How does the seal policy get its pelagic eat channel back - a logit "
         "reset, or a different parametrisation of eating?",
         "Do seals and porpoises need density dependence, and should it act "
         "on growth (pups) rather than adult mortality?",
         "At what scale should a fishing scenario be read - grid or stock?",
         ("p", "The whole history is in mareld_resume.txt, sections 0-186.")],
        kicker="What remains", size=15)

    d.section("", "Thank you",
              "Questions, objections and corrections belong in "
              "mareld_resume.txt as much as in the code.")


# ----------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", default="Strategy_Oct2.pdf")
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
        info["Subject"] = ("Strategy, October 2026, edition 2 - the "
                           "mechanics after section 185 and the biological "
                           "reasoning behind growth rates and carrying "
                           "capacities")
    print(f"{args.out}: {deck.n} slides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
