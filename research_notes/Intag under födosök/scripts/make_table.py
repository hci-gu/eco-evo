"""Write tables/mechanisms.csv (columns as in the brief). Run: python3 -I make_table.py ../tables/mechanisms.csv"""
import csv, sys

COLS = ["mechanism", "species", "quantity", "low", "central", "high", "unit", "source", "tag"]
R = [
    # 0 yardstick
    ("0 yardstick", "whiting", "uncapped potential while foraging (T x saturation; frozen fix105; yardstick only)", 0.75, 0.765, 0.78, "fraction of model ceiling", "resume 289; scripts/intake_arithmetic.py s1", "[D]"),
    ("0 yardstick", "water column", "uncapped potential while foraging (yardstick only)", 0.75, 0.757, 0.76, "fraction of model ceiling", "resume 289", "[D]"),
    ("0 yardstick", "whiting", "krill share of uncapped handling time", 0.80, 0.82, 0.84, "fraction", "resume 289", "[D]"),
    ("0 yardstick", "water column", "krill share of uncapped handling time", 0.46, 0.46, 0.46, "fraction", "resume 289", "[D]"),
    ("0 yardstick", "whiting", "potential with krill only (every non-krill pair at 0, epifauna included)", 0.717, 0.740, 0.762, "fraction of model ceiling", "scripts/intake_arithmetic.py s1", "[D]"),
    ("0 yardstick", "water column", "potential with krill only", 0.601, 0.601, 0.601, "fraction of model ceiling", "scripts/intake_arithmetic.py s1", "[D]"),
    ("0 yardstick", "water column", "potential with fish and other prey only (krill at 0)", 0.637, 0.637, 0.637, "fraction of model ceiling", "scripts/intake_arithmetic.py s1", "[D]"),
    ("0 yardstick", "whiting", "generic factor on every a for potential 0.40 / 0.45 / 0.50", 0.145, 0.181, 0.226, "multiplier on a h B_vis", "scripts/intake_arithmetic.py s1", "[D]"),
    ("0 yardstick", "water column", "generic factor on every a for potential 0.40 / 0.45 / 0.50", 0.198, 0.244, 0.300, "multiplier on a h B_vis", "scripts/intake_arithmetic.py s1", "[D]"),
    ("0 yardstick", "whiting; water column", "equivalent factor on the foraging time share for 0.40-0.50", 0.52, 0.59, 0.66, "multiplier on T", "scripts/intake_arithmetic.py s1", "[D]"),
    # 1 pursuit and handling
    ("1 pursuit/handling", "whiting 23-30 cm on shrimp 6-8 cm", "hunting (orientation to capture) time, median", 1.94, 1.94, 2.54, "s per prey", "Staniland 1995 PhD Leicester Table 4.20", "[F]"),
    ("1 pursuit/handling", "whiting 23-30 cm on shrimp 6-8 cm", "handling time in the mouth, median", 0, 0.99, 4.26, "s per prey", "Staniland 1995 Table 4.21", "[F]"),
    ("1 pursuit/handling", "0-group cod on 0-group flatfish (PPR 0.20-0.49)", "pursuit time, least-square means prey 1/2/3", 1.02, 1.40, 1.99, "s per prey", "Ellis & Gibson 1997 MEPS 149:83 doi:10.3354/meps149083", "[F]"),
    ("1 pursuit/handling", "0-group cod on 0-group flatfish", "handling time (pursuit + ingestion), prey 1/2/3", 15.1, 20.3, 30.3, "s per prey", "Ellis & Gibson 1997", "[F]"),
    ("1 pursuit/handling", "adult cod (model)", "external handling time bounds assumed", 3, 30, 600, "s per prey", "Fall & Fiksen 2020 Fish Fish 21:63 doi:10.1111/faf.12415 eq 3", "[F]"),
    ("1 pursuit/handling", "whiting 25-30 cm on fish prey (3-9 g)", "intake lost to handling at 4/30/600 s per capture vs digestion time per item (additive disc)", 0.00001, 0.0003, 0.011, "fraction of intake", "scripts/intake_arithmetic.py s2", "[D]"),
    ("1 pursuit/handling", "whiting 25-30 cm on Meganyctiphanes (0.12 g)", "same, krill", 0.0013, 0.012, 0.23, "fraction of intake", "scripts/intake_arithmetic.py s2", "[D]"),
    ("1 pursuit/handling", "saithe 45 cm on Meganyctiphanes", "same; breakeven for a 10 % loss 0.9 min per capture", 0.008, 0.055, 0.54, "fraction of intake", "scripts/intake_arithmetic.py s2", "[D]"),
    ("1 pursuit/handling", "any (theory)", "asymptote set by max(handling, digestion), not their sum", None, None, None, "-", "Jeschke, Kopp & Tollrian 2002 Ecol Monogr 72:95", "[F]"),
    # 2 capture success
    ("2 capture success", "gadoids on fish prey", "field capture success (open water, tags, video, acoustics)", None, None, None, "-", "none found (also Fangstsannolikhet for fiskbyten gaps)", "[J]"),
    ("2 capture success", "cod on capelin", "acoustic attack rates and success (CODFUN survey aim)", None, None, None, "-", "Skaret et al. 2016 IMR report 41-2016 (not obtained; host unreachable)", "[S]"),
    ("2 capture success", "whiting on shrimp", "strike success falls with stomach content (x0.6 at field fullness)", 0.47, 0.6, 0.77, "multiplier on p", "Staniland 1995 via Fangstsannolikhet D4", "[S]"),
    ("2 capture success", "any digestion-limited predator", "hunger-dependent search a(x) = alpha h(x) yields the disc equation with digestion time: the fullness effect is inside Holling h", None, None, None, "-", "Jeschke et al. 2002 eqs 8-13", "[F]"),
    ("2 capture success", "fish on krill", "per-encounter capture p used vs literature span (no fish-euphausiid measurement)", 0.3, 0.5, 0.7, "probability", "Krillens flykt och energibudget", "[S]"),
    # 3 search efficiency
    ("3 search efficiency", "cod (encounter model, Barents Sea)", "encounter theory predicts gut-limited feeding at field capelin density; modelled stomachs fuller than observed", None, None, None, "-", "Fall & Fiksen 2020 sections 3.3-3.4", "[F]"),
    ("3 search efficiency", "planktivores, whiting on krill", "library a / field-implied a in JANUARY on 100-150 m beds (field ration at bracketing densities); annual and water-column use is extrapolation", 50, 100, 260, "ratio", "Krillparen mot vintermagar (Onsrud et al. 2004 stomachs)", "[S]"),
    ("3 search efficiency", "whiting", "uncapped potential if whiting's krill a were the January field-implied a all year (extrapolation; other pairs unchanged)", 0.452, 0.458, 0.468, "fraction of model ceiling", "scripts/intake_arithmetic.py s5", "[D]"),
    ("3 search efficiency", "water column", "same", 0.638, 0.639, 0.641, "fraction of model ceiling", "scripts/intake_arithmetic.py s5", "[D]"),
    ("3 search efficiency", "largemouth bass", "field consumption at very high prey density", 0.5, 0.5, 0.5, "fraction of Cmax", "Essington, Hodgson & Kitchell 2000 CJFAS 57:548 doi:10.1139/f99-289 (via Rindorf & Gislason 2005)", "[S]"),
    ("3 search efficiency", "whiting 20-35 cm", "per fish-prey-group plateau (density independent)", 0.07, 0.16, 0.29, "fraction of model ceiling", "Rindorf & Gislason 2005 JEMBE 324:1 doi:10.1016/j.jembe.2005.04.013", "[F/D]"),
    ("3 search efficiency", "whiting 20-35 cm", "'other food' intake, constant over 5 locations (x3 at most)", 0.08, 0.15, 0.22, "fraction of Wmax", "Rindorf & Gislason 2005 Results", "[F]"),
    ("3 search efficiency", "whiting 20-35 cm", "fish intake variation between locations", 17, 29, 36, "fold", "Rindorf & Gislason 2005 Results", "[F]"),
    ("3 search efficiency", "whiting (21-28 cm per Krillparen note), Oslofjord, krill 82 % of diet", "krill per stomach (mean, day/night, Nov-Jun)", 1.44, 3.1, 5.92, "krill per stomach", "Onsrud et al. 2004 ICES JMS 61:1278 doi:10.1016/j.icesjms.2004.09.005 Table 3", "[F]"),
    ("3 search efficiency", "whiting, Oslofjord", "ration on krill as fraction of field Cmax", 0.1, 0.3, 0.7, "fraction of Cmax", "Krillparen mot vintermagar (from Onsrud 2004)", "[S]"),
    ("3 search efficiency", "cod shoal on spawning capelin (Barents Sea, Feb 2014)", "capelin eaten per cod in a few hours (10.5 million by ~2.5 million cod); reading as gut-limited is judgement (masses not checked)", 4, 4, 4, "capelin per cod", "Pednekar, Jain, Godo & Makris 2024 Commun Biol 7 doi:10.1038/s42003-024-06952-6 (news and abstract)", "[S/J]"),
    ("3 search efficiency", "whiting", "in-situ search speed / chain speed (0.6 vs 0.8 BL/s; night x0.7)", 0.525, 0.75, 1.0, "multiplier on a", "Onsrud et al. 2005 CJFAS 62:1822 doi:10.1139/f05-090 via Attackrater ur motesteori addendum; chain value scripts/pred_encounter_a.py v_whiting 0.8", "[S/D]"),
    ("3 search efficiency", "saithe", "in-situ speed / chain speed (chain 1.0 BL/s; in-situ gadoids mostly < 1)", 0.5, 1.0, 1.0, "multiplier on a", "Attackrater ur motesteori predators_encounter.md 2.3", "[S/J]"),
    # 4 windows
    ("4 prey availability window", "whiting on clupeids and gadoids", "fish prey ingested almost exclusively at dawn and dusk; midday low", None, None, None, "-", "Rindorf & Gislason 2005", "[F]"),
    ("4 prey availability window", "whiting on krill", "krill eaten at dawn and afternoon-evening; midday intake limited; krill patchy (periods with no krill in stomachs)", None, None, None, "-", "Rindorf 2003 MEPS 249:265 doi:10.3354/meps249265", "[F]"),
    ("4 prey availability window", "whiting on krill", "no systematic day/night difference in krill digestion state or occurrence", None, None, None, "-", "Onsrud et al. 2004", "[F]"),
    ("4 prey availability window", "whiting on benthic prey", "benthic prey eaten at night at 4 of 5 locations", None, None, None, "-", "Rindorf 2003", "[F]"),
    ("4 prey availability window", "any visual predator, 58.5 N", "moderate-light window (sun -12 to +6 deg; definition is judgement, R&G civil twilight ~37 min each) as share of the day", 0.19, 0.25, 0.34, "fraction of day", "scripts/intake_arithmetic.py s3", "[D/J]"),
    ("4 prey availability window", "fish prey pairs", "residual window factor on a after F (day schooling) and the vision rule (night)", 0.25, 0.5, 1.0, "multiplier on a", "judgement on Rindorf & Gislason 2005, Rindorf 2003", "[J]"),
    ("4 prey availability window", "krill pairs (whiting)", "residual window factor on a", 0.5, 0.75, 1.0, "multiplier on a", "Rindorf 2003 vs Onsrud 2004", "[J]"),
    # 5 satiation-independent
    ("5 meals and appetite", "whiting 20-36 cm, North Sea", "probability of a fresh meal depressed when partly digested food present; not tied to absolute gut limitation", None, None, None, "-", "Rindorf 2002 J Fish Biol 61:579 doi:10.1111/j.1095-8649.2002.tb00897.x (abstract via search; reading by Fall & Fiksen 2020)", "[S]"),
    ("5 meals and appetite", "largemouth bass", "share of field diets above the maximum sustainable stomach fullness", 0.13, 0.13, 0.13, "fraction", "Essington et al. 2000 (abstract via search)", "[S]"),
    ("5 meals and appetite", "whiting", "maximum daily intake in the field study", 3.1, 4.3, 6.2, "% body weight wet per day", "Rindorf & Gislason 2005", "[F]"),
    ("5 meals and appetite", "whiting", "resume feeding at stomach fill, interval", 0.40, 0.40, 0.40, "fraction of capacity (~21 h)", "Seyhan et al. 1998 via Mattnad hos rovfisk", "[S]"),
    ("5 meals and appetite", "saithe 34-54 cm coastal", "wild / farm-associated intake (stomach 8.6 vs 20.2 g)", 0.43, 0.54, 0.65, "ratio", "Dempster et al. 2011 via Mattnad hos rovfisk M4", "[S]"),
    # 6 other
    ("6 other", "whiting", "aggregative response to prey: none; predator and prey catches slightly negatively correlated", None, None, None, "-", "Rindorf & Gislason 2005", "[F]"),
    ("6 other", "whiting", "predator interference w in the engine (no field evidence)", 0.1, 0.1, 0.1, "per t/km2", "AGENTS.md section 5", "[S]"),
    ("6 other", "any", "km-scale prey avoidance / interference factor on a", 0.9, 1.0, 1.0, "multiplier on a", "judgement", "[J]"),
    # combination
    ("combination", "whiting", "forward product of sized factors: uncapped potential low / central / high", 0.379, 0.672, 0.795, "fraction of model ceiling", "scripts/intake_arithmetic.py s5", "[D]"),
    ("combination", "water column", "forward product of sized factors: uncapped potential low / central / high", 0.268, 0.665, 0.783, "fraction of model ceiling", "scripts/intake_arithmetic.py s5", "[D]"),
]
w = csv.writer(open(sys.argv[1], "w", newline=""))
w.writerow(COLS)
for r in R:
    w.writerow(["NA" if v is None else v for v in r])
print(len(R), "rows")
