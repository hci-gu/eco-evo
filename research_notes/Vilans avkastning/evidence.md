# Vilans avkastning (the payoff of rest) - evidence trail

Research only (main research agent plus four literature subagents). Nothing in the library, manifest,
code, tests, AGENTS.md or the resume was changed; no model run; no commit. No value is chosen from model
output: the frozen fix105 decomposition (research_notes/"Mattnad hos rovfisk"/model, a-umlaut) is used only
as a yardstick for how much rest a full fish could take.

Tags: [F] full text read, [A] abstract only, [S] secondary (cited by another source or a prior project
note), [D] derived here, [J] judgement. PROXY = not a model species, or not marine.

Files:
- `lit_activity_risk.md` (Q1 fish and generic predators), `lit_mammals.md` (Q1 seals and porpoises),
  `lit_activity_budgets.md` (Q2), `lit_vertical_refuge.md` (vertical / layer refuge): subagent trails.
- `tables/activity_risk_raw.csv`, `tables/mammals_raw.csv`, `tables/activity_budgets_raw.csv`,
  `tables/vertical_refuge_raw.csv`: one row per sourced number (source, doi, species, predator, method,
  quantity, value, unit, tag); subagent helper scripts in `scripts/` (make_activity_csv.py, vert_make_csv.py,
  mammal_activity_risk.py -> `tables/mammal_activity_risk_stdout.txt`, fetch/xml/openalex/crossref readers).
- `tables/derived.csv`: the derived numbers of this note.
- `scripts/rest_payoff.py` -> `tables/rest_payoff_stdout.txt` (Q3 arithmetic, library only);
  `scripts/candidate_forms.py` -> `tables/candidate_forms_stdout.txt` (implied floors per form);
  `scripts/vertical_refuge.py` -> `tables/vertical_refuge_stdout.txt` (form R);
  `scripts/fill_tradeoff.py` -> `tables/fill_tradeoff_stdout.txt` (why a rest payoff lowers T, not fill).
- `data/`: downloads, untrusted, not to be committed.

```
cd scripts
python3 -I rest_payoff.py /home/simon/eco-evo > ../tables/rest_payoff_stdout.txt
python3 -I candidate_forms.py /home/simon/eco-evo > ../tables/candidate_forms_stdout.txt
python3 -I vertical_refuge.py > ../tables/vertical_refuge_stdout.txt
python3 -I fill_tradeoff.py > ../tables/fill_tradeoff_stdout.txt
```

## M. Model facts (verified 2026-11-20 against fgconfig/fg_library.yaml and prior notes)

M1. Rest = hide: the rested share neither eats nor moves and pays 1.0 x resting_metabolism; eating pays
    feeding_cost x RM (planktivores 1.65, benthivores 1.02, whiting 1.50, water column 2.14, mackerel 1.02).
    Visible prey = B (1 - rest (1 - floor)), floor = the pair key on a checked pair, else the prey's
    species floor; the same floor sets the observer's view next tick (option A, research_notes/"Gom och
    sinnen"). One rest share per cell for J and A.
M2. Floors today (section 270, user decisions 2026-11-11): every fish -> fish floor 1.0 (fish species floors
    1.0; the four fish -> planktivore pair keys 1.0); seals -> benthivores 0.9, -> whiting 0.85, -> water
    column 0.7, -> planktivores 0.35 (species); porpoises -> planktivores 1.0, -> whiting 0.9, ->
    benthivores 0.75, -> water column 0.9.
M3. Why 270 set the fish floors to 1.0: once schooling F, per-encounter capture p with pre-strike escape q
    (q = P(strike | detected), central 0.8 [J], research_notes/"Fangstsannolikhet for fiskbyten"), the band
    profiles and vision rule A are in a, no documented feeding-exclusive refuge STATE remained for any fish
    group in the box (open mud, no cobble or kelp; herring keep feeding and schooling in winter; whiting's
    Cyanea shelter is 0-group only; flatfish burial unmeasured against the box's predators). The "resting
    prey is motionless" convention (v_rel(v, 0)) was offered for whiting (benthivores -> whiting 0.9) and
    NOT adopted (research_notes/"Vitlingens golv", User decisions 2026-11-11); the note said it must be
    derived library-wide or not at all.
M4. Every fish-pair a is an encounter for a prey CRUISING at 0.10 m/s (small fish, juvenile gadoids) or
    0.20 m/s (herring) in a Gerritsen-Strickler v_rel, predator search speeds saithe 1.0 BL/s (night x0.6),
    whiting 0.8 BL/s, cod 0.4 BL/s, mackerel 1.8 BL/s, hake 0.5 BL/s, spurdog 0.5 BL/s; anglerfish an
    ambusher (v replaced by the prey's speed) (research_notes/"Attackrater ur motesteori" pred_encounter_a.py,
    benth_encounter.py) [S]. So the fish-pair a's are ACTIVE-prey encounters. The mammal a's are not
    encounter-derived (seals a section 166 starting value; porpoises -> planktivores 1.12 from encounter
    theory since 273, the other porpoise pairs x1.5 fits), i.e. field averages.
M5. Booked M2 by predator class (tables/rest_payoff_stdout.txt part A) [D on library]:
    planktivores fish 0.349 (water column 0.174, benthivores 0.102, whiting 0.055, mackerel 0.019),
    mammals 0.071 (porpoises 0.066, seals 0.005); benthivores fish 0.165 (whiting 0.107, water column
    0.058), mammals 0.085 (porpoises 0.048, seals 0.037); whiting fish 0.337 (benthivores 0.180, water
    column 0.157), mammals 0.155 (porpoises 0.141, seals 0.014); water column fish 0, mammals 0.018;
    mackerel fish 0.021 (water column, pair OFF on overlap), mammals 0. Booked, not realised (whiting's
    predators realise ~0.24 of its booking; research_notes/"Bytessidans predation"): upper bounds.
    Fish-pair stage_access puts most fish predation on juveniles (e.g. benthivores -> whiting [0.68, 0.24],
    whiting -> benthivores [0.038, 0.0064]).
M6. Rest payoff already in the model at today's floors, per unit rest share, sum_p M2_p (1 - floor_p):
    whiting 0.0162 /yr, benthivores 0.0156, planktivores 0.0033 (seals' 0.35 species floor), water
    column 0.0025, mackerel 0 [D]. At the yardstick rest share rho this is 0.86 %/yr (whiting), 0.55
    (benthivores) of biomass on BOOKED M2; fix105 still forages 89-97 % of the time (fix97 whiting rested
    0.066). Whether that payoff is too small to learn is [J]: the realised mammal M2 is smaller than booked
    and no run without those floors exists to compare.
M7. Yardstick rest share a full fish could take, rho = 1 - realised / S (S = uncapped/T, fix105 seed 0):
    planktivores 0.55, benthivores 0.35, whiting 0.53, water column 0.49, mackerel 0.51 [D on model output,
    sizing only].
M8. At need, one foraging tick is worth 17-40 MJ/t (S x ceiling x (1 - sda) - (feeding_cost - 1) RM) while
    complete safety at rest would save only 0.08-2.6 MJ/t/tick of body value (booked M2 x (E + ME)/1460):
    risk never makes rest pay at need [D]. At exact surplus the marginal foraging tick is worth ~0 (the
    reserve-room cap refills any cost), so in theory any r > 1 makes rest pay there; the size of that payoff
    is part B of rest_payoff_stdout.txt.

## L1. Activity-dependent risk, fish and generic predators (lit_activity_risk.md, tables/activity_risk_raw.csv)

No ratio of risk for active vs resting prey was found for any model species (marine adult or sub-adult
gadoid or clupeid) [J; three searches, not exhaustive]. All numbers are PROXY:
- Martel & Dill 1995, doi:10.1111/j.1439-0310.1995.tb01095.x [F]: merganser on coho parr, white background:
  attack per head-dip 15.2 % on moving vs 1.4 % on still fish -> r 10.6 per look [D] (pooled; indicative).
- Pohlmann, Grasso & Breithaupt 2001, doi:10.1073/pnas.121026298 [F]: catfish in infrared darkness, 80 % of
  94 attacks on moving guppies that swam 43 % of the time -> r 5.3 per time [D]; wakes followed up to 10.3 s
  old, 121 cm. Bundles wake detection with the prey's own speed.
- Biro, Post & Parkinson 2003, doi:10.1890/02-0416 [F]: age-0 trout in whole lakes, mortality 0.58 (food
  added, less active) vs 0.75 -> instantaneous ratio 1.60 [D], upper bound for activity (habitat and
  clarity confounded).
- Ioannou & Krause 2009, doi:10.1098/rsbl.2008.0758 [F]: motion matters only on a background-matching
  substrate (stickleback on chironomids; no effect on a conspicuous background, p 0.80).
- Wright & O'Brien 1984 via Wetterer & Bishop 1985 [S]: crappie locate moving copepods at 3x the distance
  (~9x area [D]).
- Krause & Godin 1996, doi:10.1093/beheco/7.3.264 [A]: foraging guppies attacked preferentially, react later
  (sign only).
- Opposite sign: Yokogawa et al. 2007, doi:10.1371/journal.pbio.0050277 [F] (zebrafish still >= 6 s need
  stronger stimuli); Menidia fed fish slower, full stomachs more predation, fast growers more vulnerable
  (Lankford et al. 2001, doi:10.1111/j.0014-3820.2001.tb00836.x; Billerbeck et al. 2001; Munch & Conover
  2003) [A]; no consistent risky-behaviour/survival link in a meta-analysis (Moiron et al. 2020,
  doi:10.1111/ele.13438) [A]; McCormick & Meekan 2010, doi:10.3354/meps08583 [F]: reef-fish bite rate and
  distance moved did not predict survival.
- Encounter by own speed: Gerritsen-Strickler ratio <= 1.33 for u <= v [D] (rest_payoff_stdout.txt D).

## L2. Seals and porpoises (lit_mammals.md, tables/mammals_raw.csv, tables/mammal_activity_risk_stdout.txt)

- Seal trail following: direction judged above chance to 35 s; following success continuous vs gliding
  (non-propelling) source 62.3 vs 43.1 % at 15 s, 33.3 vs 21.1 % at 25 s (Wieskotten et al. 2010,
  doi:10.1242/jeb.041699 and jeb.047134) [F, 1-2 seals, artificial trails]. Breathing-current detection of a
  still flatfish: threshold ~4 cm/s, current 2-2.5 cm/s at 36 cm, the seal must slow to 0.2-0.7 m/s
  (Niesterok et al. 2017, doi:10.1242/jeb.148676, jeb.158055) [F]. Trail vs gill-current detection width
  2.8 / 9.4 / 59 on the bed [D, upper bounds].
- Field (Bowen et al. 2002, doi:10.3354/meps244235) [F]: capture success cryptic sand lance 63.7 %, buried
  flounder 79.9 %, schooling sand lance 37.5 %; captures per 10 min about equal (3.50 vs 3.45 [D]).
- Porpoises: blindfolded animals caught dead drifting fish in 67/71 trials [F]; detection 15-27 m for cod by
  aspect [A]; no Doppler or target-motion use found; prey fled only at ~50 cm (after detection) [F]; hearing
  too poor at 50-400 Hz for gadoid sounds [D]; porpoise foraging peaks at night (herring dispersed,
  non-feeding) [A/S].
- Subagent's synthesis [J]: seals r 2 (1-5) on bed gadoids, 1.3 (0.6-4) flatfish, 1.5 whiting, 1.3
  saithe/hake, 1.1 planktivores; porpoises 1.1 bed gadoids, 2 flatfish, 1.05 whiting/WC, 1.0 (0.6-1.2)
  planktivores. Seals -> benthivores as one pair 0.61 vs today's 0.9 [D/J].

## L3. Activity budgets and state (lit_activity_budgets.md, tables/activity_budgets_raw.csv)

- No adult foraging-time share for any model species (tags record vertical activity, not feeding) [J].
- Cod: Hobson et al. 2007, doi:10.3354/meps07047 [F]: ~0.70 of days on the seabed without vertical movement
  (0.64-0.81 by area); Hobson et al. 2009, doi:10.3354/ab00144 [F]: horizontal travel even on quiet days, so
  'seabed' is not 'resting'. Righton et al. 2000 (grey literature, pointer only): vertically active 0.13
  of time on summer feeding grounds.
- Plaice in midwater < 5 h/day (Hunter et al. 2009, doi:10.3354/meps08213) [F]. Haddock fed ~75 % by day
  with dusk/dawn peaks; wild haddock at 0.15-0.22 of maximum ration (Hall 1987) [A].
- Satiation: cod 29-42 cm swam 0.6 BL/s at ~half maximum ration and 0.2 BL/s when food was unlimited or
  absent (Bjornsson 1993, doi:10.1139/f93-277) [A]: a fed cod swims at 1/3 the speed. Same peaked pattern in
  juvenile walleye pollock (Sogard & Olla 1996, doi:10.3354/meps133043) [F]; juvenile whiting activity falls
  with ration (Hammer 1997) [S].
- State-dependent risk: fed juvenile pollock stay tight after a threat up to an hour, hungry ones recover
  quickly (Sogard & Olla 1997, doi:10.1023/A:1007393307007) [A]; trout in food-rich lakes move less, survive
  68 % better (Biro 2003) [A/F]; salmonid direction-only results (Damsgard & Dill 1998; Gotceitas & Godin
  1991). No adult gadoid or clupeid hunger x risk experiment.

## L4. Vertical refuge by state (lit_vertical_refuge.md, tables/vertical_refuge_raw.csv)

- No within-period split into a foraging and a refuge profile is identified for any box prey group, and no
  mix share; every documented switch follows the period (light), the form the engine has [J on F/A].
- Norway pout, Norwegian Deep incl. the Skagerrak (Albert 1994, doi:10.1006/jmsc.1994.1005) [F]: by day in
  the ~1.5 m near-bottom zone, at night homogeneous in the lowest 5-10 m, feeding more intense at night
  (krill); antipredator reading offered as a hypothesis, time-of-day catch effect not significant. Oslofjord
  (Onsrud et al. 2004) [F]: pout in midwater at night, no day/night fullness difference.
- Herring: the deep day layer is a FEEDING layer (Oslofjord, night/day fullness ~0.79) [F]; sprat's hypoxic
  refuge (Kaartvedt et al. 2009, doi:10.3354/meps08196) [F] needs fjord water the box lacks.
- Whiting: krill day and night, no diel stomach trend [F]; GOV diel catch effects within x2 [F].
- 0-group cod/haddock (Lough et al. 1989, doi:10.3354/meps056001) [F, PROXY stage/area]: on the bed by day,
  'part of the population' rises 3-5 m at night to feed; share not counted.
- Hunger-dependent DVM: theory (Clark & Levy 1988, doi:10.1086/284789; Pearre 2003) [A]; mesopelagic proxies
  by interpretation (Staby et al. 2011 [F]; Dypvik et al. 2012 [A]).
- Cross-predator: Hixon & Carr 1997, doi:10.1126/science.277.5328.946 [A, reef PROXY]; sprat leaving the
  hypoxic refuge at night become whiting and cod prey (authors' text) [F].

## D. Derived here (tables/derived.csv)

D1. Today's rest payoff and the yardstick rho: evidence M6, M7.
D2. Uniform-r payoffs at rho (rest_payoff_stdout.txt B): fish predators r 1.2 -> planktivores 3.2, whiting
    3.0, benthivores 1.0 %/yr; mammals at r 2 -> whiting 4.7, benthivores 1.7, planktivores 2.2 %/yr; water
    column at most 0.9 %/yr for any r (mammal M2 0.018), mackerel at most 0.7 %/yr (pair OFF).
D3. Energy at need (rest_payoff_stdout.txt C): complete safety at rest is worth 0.005-0.095 of one foraging
    tick: risk never makes rest pay at need, only at surplus.
D4. Form V (motionless rest, the a's own v_rel) per pair (candidate_forms_stdout.txt): benthivores ->
    whiting 0.885, WC -> whiting 0.951 (1.0 if whiting hover), whiting -> benthivores 0.938, WC ->
    benthivores 0.950, benthivores -> planktivores 0.931, WC -> planktivores 0.971, whiting -> planktivores
    0.963, mackerel pairs 1.0; mammals 0.99-1.0. Payoff at rho: whiting 2.4, planktivores 1.0, benthivores
    0.9 %/yr.
D5. Form VM central (V x motion detection, m 0.2, d 0.5 [J]; mammals per L2): whiting 3.7, planktivores
    1.9, benthivores 1.8, WC 0.1 %/yr; the low-floor corner 10.3 / 5.7 / 4.1 %/yr.
D6. Form R (vertical refuge, vertical_refuge_stdout.txt): pout's near-bed refuge (0-1.5 m vs 0-10 m)
    against the box's predators gives exposure ratios 1.17 (saithe), 1.45 (hake), 4.1 (cod), 0.93
    (whiting), 1.97 (seal), 1.82 (porpoise) by day at a 140 m bed (seal and porpoise height profiles [J]):
    the refuge layer is MORE exposed to every box predator but whiting (benthivores -> planktivores is a
    checked pair, so cod x4.1 is real). 0-group gadoids on the bed vs 3-5 m up, against the CHECKED
    predators of juvenile benthivores: saithe x1.14, hake x1.42, whiting 0.93, seals and porpoises 1.0 (near
    neutral; cod x19 is printed but benthivores -> benthivores is unchecked, so it does not apply).
D7. Fill vs risk (fill_tradeoff_stdout.txt): holding fill 1 instead of 0.1 costs whiting 0.137 of time
    in extra foraging (feeding level 0.388 vs 0.275, Mattnad hos rovfisk 1.3) and buys 0.59 /yr of
    production; resting that time would save 0.6 %/yr (V), 1.0 (VM central), 2.7 (VM low floor) - 0.01-0.05
    of the production at stake. Water column 0.03-0.06 %/yr vs 0.41 /yr. A rest payoff in the evidence range
    lowers T but leaves the reserve full.
    Check: a mix of the two Albert layers cannot reproduce the library's pout profile (bottom-10-m share
    1.0 for every mix vs 0.7 / 0.4).
