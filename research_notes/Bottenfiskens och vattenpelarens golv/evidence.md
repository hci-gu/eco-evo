# Evidence - hide floors of the benthivores and the water column (2026-11-11)

Grades: [F] field, [L] lab, [A] abstract or catalogue record only, [S] secondary (another note, review,
encyclopedia), [D] derived here, [J] judgement. Model facts M, literature E, derived D.
PDFs: `papers/` (Unpaywall helper only; see papers/README.md). Nothing in the library, code, tests,
manifest, AGENTS.md or resume was changed; no training; no value set from model output.

## Model facts (verified 2026-11-11)

M1. Rest = hide. Visible prey = B (1 - rest (1 - floor)), floor = the pair's `visibility_floor` on a CHECKED
    `<pred>_preys_on_<prey>` pair, else the prey's species floor
    (`lib/environments/ecosystem_env/interactions.py` visible_prey_biomass, ~764-809). The same floor matrix
    sets each observer's view of the hider next tick (`observations.py:39-78`, `state.py` ~249-336). One
    rest share per DM, cell and tick, applied to J and A alike (the stage split is after the floor).
M2. Shared availability cap: the total removal from a prey is bounded by the visible share of its best
    detecting checked predator (max over the pair floors; interactions.py ~800).
M3. Pairs and observers (`scripts/floor_mechanics_stdout.txt`):
    - benthivores, species floor 0.55 ("Judgement: gadoids 0.6, flatfish and rays bury and are cryptic
      (0.4)"). Checked predators: water column (WC) a 0.0418, stage_access [0.392, 0.0388], night/day 0.165,
      M2 0.0833, no pair floor; whiting a 0.103, [0.038, 0.0064], night/day 0.565, M2 0.1067, no pair floor;
      porpoises pair 0.75; seals pair 0.85 (decided 0.9). Cap today 0.85 (seals).
      Observers: WC and whiting (predators, species floor), planktivores (the benthivores eat them; species
      floor), mackerel (no food link; species floor), seals and porpoises (pair keys).
    - water column, species floor 0.7 ("Judgement: midwater saithe schools hide badly"). Checked predators:
      porpoises pair 0.9, seals pair 0.75 (decided 0.7); no fish eats it. Cap 0.9 (porpoises).
      Observers: planktivores, benthivores, whiting, mackerel - for all four the WC is a predator; species
      floor. Seals, porpoises by pair key.
M4. Costs: rest pays 1.0 x resting_metabolism. Benthivores RM 3.33 MJ/t/tick, feeding_cost 1.02 (rest saves
    0.07 MJ/t/tick against eating, before SDA); WC RM 3.29, feeding_cost 2.14 (rest saves 3.75 MJ/t/tick).
    Both fish groups `attack_choice: fixed`.
M5. Already in a (combined derivation, not yet applied; research_notes/"Fangst och stim i fiskparen",
    "Fangstsannolikhet for fiskbyten", "Overlapp i fiskparen", "Bottenfiskens hojdprofil", "Rekryternas
    bottendjup"): benthivores as prey in a 2 m bed slab (uniform, day and night); rule A vision with one
    contrast |C0| 0.7 for every fish prey and the measured light at depth; night/day from the rule; F = 1
    for benthivores (juvenile sigma 0 by day); capture p with pre-strike escape q 0.8 of a DETECTED prey;
    the proposed recruit_depth_table (J over < 100 m beds). No term lowers the detection of a buried or
    cryptic flatfish below that of a gadoid of the same size.
M6. WC members by biomass (research_notes/"Rovfiskens uppdelning" params_split_mack.md): saithe 0.797,
    spurdog 0.097, hake 0.048, anglerfish 0.027, ling 0.014, velvet belly 0.009, pollack 0.008. Members of the
    WC -> benthivores band-A a (269): saithe 0.892, hake 0.051, ling 0.025, anglerfish 0.015, pollack 0.007.
M7. Rest shares (training-world means of `results/pzbpgp248nsrpfix97/training.jsonl`, descriptive only):
    brief / "Gom och sinnen" benthivores 0.026, WC 0.032; this file at 624 updates: all-update mean 0.026 /
    0.018, last 20 updates 0.0059 / 0.0028, max 0.110 / 0.111 (early training). No monthly action series
    exists (the long-run probe stores biomass, diets, viability only): seasonal swings are not documented.

## Literature

E1. Ellis T, Gibson RN (1995). Size-selective predation of 0-group flatfishes on a Scottish coastal nursery
    ground. MEPS 127:27-37. doi:10.3354/meps127027. [F, full text, papers/]
    - Tralee Beach (0-1.5 m), night beach seines, May-Sep 1992-93, 2338 predator stomachs. 0-group plaice,
      dab, flounder eaten by I-group grey gurnard, poor-cod, whiting and 0-group cod (Table 1: 1993 whiting
      116-139 mm, 25 % with plaice, 25 % flounder, n 4; 1992 cod 5.2 % with dab, n 344). Flatfish > 45 mm
      (largest eaten 42 mm) had a size refuge from the fish sampled.
    - Burial is not measured; the samples are night samples, when plaice on this beach are on the sediment.
      Reading [J]: gadoids including whiting find and eat small flatfish on sand; protection is by size.
E2. Ellis T, Gibson RN (1997). Predation of 0-group flatfishes by 0-group cod: handling times and
    size-selection. MEPS 149:83-90. doi:10.3354/meps149083. [L, full text p.1-2 read, papers/]
    - Tank, sand-free Perspex arenas; capture and ingestion of 0-group dab and plaice; selection explained by
      handling and capture probability. Burial not tested. Used only to place E1's predation in lab context.
E3. Ansell AD, Gibson RN (1993). The effect of sand and light on predation of juvenile plaice by fishes and
    crustaceans. J Fish Biol 43:837-845. doi:10.1111/j.1095-8649.1993.tb01159.x. [A, Crossref abstract]
    - Aquaria, sand x light; predators Crangon, Liocarcinus, cod, pollack. Predation higher in the dark for
      all but pollack; removing sand raised predation ONLY for pollack; "Burying in sand appears to provide
      only a partial refuge from predation". Magnitudes not in the abstract (no open PDF).
    - The only burial-vs-no-burial design found. Direction: burial protects against a visual pollack-type
      gadoid, not against cod.
E4. Ryer CH, Stoner AW, Titgen RH (2004). Behavioral mechanisms underlying the refuge value of benthic habitat
    structure for two flatfishes with differing anti-predator strategies. MEPS 268:231-243.
    doi:10.3354/meps268231. [L, full text, papers/]
    - Age-0 Pacific halibut (flee) vs northern rock sole (immobility, burial, crypsis) against Age-2 halibut
      predators (26-34 cm) on sand: 24-h consumption 14 rock sole vs 19 halibut (of 20); per unit predator
      effort 0.6 vs 1.4 (Fig. 3, 5); 82 % of encounters began with the prey flushing at 20.7 cm; rock sole
      were less likely to flush but, once pursued, more often caught (Fig. 8).
    - Age-2 halibut (26-34 cm at the trials; collected at 13-17 cm a year earlier) buried 50-60 % on the day after a meal and 20-35 % three days
      later, and raised the head as hunger grew (Fig. 2b-c): "a hunger-mediated trade-off between refuging
      and foraging". The protective value of that burial was not measured.
    - Reading [J]: burial/crypsis is a resting state that a flatfish trades against foraging (criterion a),
      and a species contrast of ~x0.4-0.75 in consumption exists between a cryptic burier and a flusher;
      that contrast is not a buried-vs-exposed protection value.
E5. Gibson RN, Robb L (2000). Sediment selection in juvenile plaice and its behavioural basis. J Fish Biol
    56:1258-1275. doi:10.1111/j.1095-8649.2000.tb02138.x. [A, abstract via search index]
    - Burial reduces activity; activity, dependent on light and endogenous rhythm, sets the sediment
      selectivity. Buried = inactive [A].
E6. Gibson RN, Robb L (1992). The relationship between body size, sediment grain size and the burying ability
    of juvenile plaice. J Fish Biol 40:771-778. doi:10.1111/j.1095-8649.1992.tb02623.x. [A, record only]
    - Cited (E4) for: burial ability rises with body size, and fine sediment allows burial. The box is mud
      and clay (E10), so burial is physically possible for its flatfish [J].
E7. Gotceitas V, Fraser S, Brown JA (1995). Habitat use by juvenile Atlantic cod in the presence of an
    actively foraging and non-foraging predator. Mar Biol 123:421-430. doi:10.1007/BF00349220. [A, abstract
    copy via search index]
    - Age-0 cod with an age-3 cod predator, tank. Non-foraging predator: habitat use barely changed, the
      juveniles mostly avoided the predator's position and preferred finer substrates. Foraging predator: they
      hid in cobble, or in kelp without cobble; both refuges lowered predation.
    - Two readings: (i) the shelter needs cobble or kelp; on fine substrate the response is spatial
      avoidance (the engine's move action); (ii) prey read a predator's ACTIVITY STATE at tank (metre) scale.
E8. Gotceitas V, Brown JA (1993). Substrate selection by juvenile Atlantic cod: effects of predation risk.
    Oecologia 93:31-37. doi:10.1007/BF00321187. [S, via research_notes/"Trålning och uppväxtområden" and
    "Torskfiskars ljusberoende vid jakt"] Juvenile cod hide in cobble interstices when a larger cod is present.
E9. Shelter on structure for juvenile gadoids [S, research_notes/"Trålning och uppväxtområden"
    mechanism_evidence.md, not re-read at source]: Lindholm et al. 1999 (MEPS 180:247, tank: 0-year cod eaten
    93 % on sand, 67 % on cobble, 32-47 % on sponge); Tupper & Boutilier 1995; Lough et al. 1989 (settled
    0-group cod and haddock "abundant only on the gravel bed", 70-100 m, Georges Bank); Elliott et al. 2017
    (Clyde: cod on shallow gravel-maerl, "M. aeglefinus and M. merlangus predominated over deeper sand and
    mud"). That note's conclusion: "No study links mud epifauna (sea pens, burrow mounds) to juvenile gadoid
    survival"; for the park area "the shelter pathway does not apply to any documented gadoid life stage".
E10. Box substrate [S, research_notes/"Trålning och uppväxtområden" nursery_areas.md, EIA 9.1]: mud and clay,
    accumulation bottom, with sea pens (Funiculina, Virgularia, Pennatula; 0.005-0.03 colonies/m2 on
    trawled-to-protected mud), Nephrops and Pandalus; Bratten rocky ravines lie W-NW, outside the park cells.
E11. Grant SM, Brown JA (1998). Diel foraging cycles and interactions among juvenile Atlantic cod at a
    nearshore site in Newfoundland. CJFAS 55:1307-1316. doi:10.1139/f97-291. [A, search summary; thesis
    snippet] Age-0 cod keep a strict diurnal foraging cycle; the nocturnal drop in feeding coincides with
    more older cod in the catch (avoidance of cannibals suggested). Nearshore, NW Atlantic.
E12. Strand et al. 2020, JEMBE 524:151356; Anderson, Laurel & Brown 2007, JEMBE. [S, via
    research_notes/"Torskfiskars ljusberoende vid jakt" prey_side_and_benthic.md] Large cod and sculpin were
    most effective on 0-group cod and saithe at dusk, night and dawn; 0-group fish fled into cobble or
    vegetation; age-0 cod kept less distance from a larger gadoid in the dark. Night inactivity of juveniles
    is not a refuge from gadoids; darkness enters the pairs through rule A.
E13. Settled 0-group cod and haddock hold within a few cm of the bed by day, most stay on the bed at night
    (Lough 1989; Aglen et al. 1997, small haddock rise at night) [S, research_notes/"Sejens hojd over botten"
    section 3, "Bottenfiskens hojdprofil"]. In a as the 2 m prey slab (M5).
E14. Cod vertical time budget, North Sea DSTs (Hobson et al. 2007, MEPS 347:101): stationary on the bed
    (SB) 70 %, SW 20.5 %, midwater 9.5 % [S, "Bottenfiskens hojdprofil"]. The SB class is the engine's
    resting fish at the bed; it is the occupancy behind the band profile in a.
E15. Saithe vertical position [S/F, research_notes/"Sejens hojd over botten" summary 1]: in-area June-July
    adult saithe exp from the bed with Hs 15 m (9-31); non-feeding pre-spawning and winter aggregations
    (Scotian Shelf, Neilson 2003; DFO 2003/075) Hs 2.5-9 m, nearer the bed at night; activity much reduced
    at night and in winter (Armannsson & Jonsson 2012); saithe feed mainly by day.
E16. Pitcher TJ, Misund OA, Ferno A, Totland B, Melle V (1996). Adaptive behaviour of herring schools in the
    Norwegian Sea as revealed by high-resolution sonar. ICES J Mar Sci 53 (1996). doi:10.1006/jmsc.1996.0063.
    [A, abstract via search index] Cod and haddock accompanied feeding herring schools, attacks modified but
    did not disperse them; the rapid approach of schooling saithe made herring dive steeply and stop feeding.
    Prey distinguish a fast attacking saithe school at sonar range (tens of m), not at km.
E17. Hake diel behaviour [S, encyclopedic and Mediterranean juvenile studies via search; not read at source]:
    juveniles near muddy bottoms by day, rise at night to feed on euphausiids and return to the bottom after
    feeding. Adult North Sea / Skagerrak hake: no source read; HERAS takes hake in 32/43 day bed hauls
    ("Sejens hojd over botten"). Graded [J] for the box.
E18. Smell and senses at km scale [S, research_notes/"Gom och sinnen" section 3, "Lukt och
    rovdjursundvikande", "Sal och tumlare golv" D3]: predator odour persists through a hide (minutes to hours,
    downstream filaments); sight and lateral line reach metres; a cue spreads 1-21 m vertically in a 6 h tick.
E19. Atkinson CJL, Bergmann M, Kaiser MJ (2004). Habitat selection in whiting. J Fish Biol 64:788-793.
    [A, via research_notes/"Lukt och rovdjursundvikande" E4] Context only (the whiting floor is decided).

## Derived

D1. `scripts/take_composition.py` -> `take_composition_stdout.txt`, `take_composition.json`. Species and stage
    composition of the accessible tonnes of the two fish pairs on the benthivores, with the library window
    machinery (reproduces [0.392, 0.0388] and [0.038, 0.0064]); JB 0.15.
    - WC -> benthivores: J 0.641 (haddock 0.555, cod 0.086), gadoid A 0.140, flatfish 0.210 (long rough dab
      0.127, dab 0.042, plaice 0.026, witch 0.008, lemon sole 0.007), rays + gurnard 0.010. Saithe's take is
      J gadoids only (identity haddock + cod); the flatfish come through hake 0.093, ling (on anglerfish's
      window) 0.073 and anglerfish 0.044.
    - Sensitivity, hake's identity haddock + cod (its in-area diet holds 0.4 % benthivores): flatfish 0.129
      (long rough dab 0.074).
    - Whiting -> benthivores: J 0.513 (haddock 0.441, cod 0.070), flatfish 0.486 (long rough dab 0.399, dab
      0.083): the 12-15 cm long rough dab and dab are "all adult" by the engine's J/B bookkeeping
      (research_notes/"Fonsterviktning" open point 3).
    - Long rough dab is 0.60 of the WC's flatfish share (0.57 in the hake sensitivity) and 0.82 of whiting's.
    - Pair-equivalent floor if flatfish had floor x and the rest 1 (one rest share per cell, M1):
      1 - w_flat (1 - x). WC: x 0.6 -> 0.916, x 0.3 -> 0.853, x 0 -> 0.790 (hake sensitivity 0.948 /
      0.910 / 0.871). Whiting: x 0.6 -> 0.806, x 0.3 -> 0.660.
D2. `scripts/floor_mechanics.py` -> `floor_mechanics_stdout.txt`: pairs, observers, cap (M2, M3) and the
    visible share 1 - rest (1 - floor). Benthivores at rest 0.026: 0.988 (floor 0.55), 0.997 (0.9), 1.000 (1);
    at the latest rest 0.006: 0.997 -> 1.000; fully resting cell 0.55 -> 1. WC at rest 0.032: 0.990 (0.7) ->
    1.000; latest 0.003: 0.999 -> 1.000.
D3. WC rest-state factor (cue of a resting member reaching an observer / cue of the same member foraging),
    weighted by member biomass (M6) [D on S/J]:
    - saithe + pollack 0.805: a resting saithe sits ~6-12 m nearer the bed than a foraging one (E15: Hs 2.5-9
      vs 15 m), inside the 1-21 m per-tick vertical spread of a cue (E18) -> 1.0 for bed and pelagic observers.
    - spurdog 0.097: obligate swimmer, schools; rest changes no position -> 1.0 [J].
    - anglerfish 0.027, ling 0.014: bed-bound ambushers whether resting or foraging -> 1.0 [J].
    - hake 0.048: on the bed by day, rises to feed at night (E17, [S/J]): for pelagic observers the resting
      hake is farther away, 0.5 (0-1) [J]; for bed observers nearer, 1.0.
    - velvet belly 0.009: 1.0 [J].
    Weighted: pelagic observers (planktivores, mackerel) 0.976 (0.952-1.0); bed observers (benthivores,
    whiting) 1.0.
