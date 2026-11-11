# Planktivore hide floors - evidence (2026-11-11)

Grades: [F] field (full text or data read), [L] lab, [A] abstract or catalogue record only,
[S] secondary (search summary or another note's reading), [D] derived here, [J] judgement.
"Read there" = read in full by an earlier note in this repository; the number is quoted from it.

## Model facts (code and repository, verified 2026-11-11)

- M1. Hide: visible prey = B (1 - rest (1 - floor)); pair floor if the CHECKED pair carries one, else
  the prey's species floor (`lib/environments/ecosystem_env/interactions.py:764-809`, `visible_prey_biomass`). The same floor
  sets what that observer sees of the prey next tick (`observations.py:39-78`; user decision option A,
  research_notes/"Gom och sinnen").
- M2. The total removal from a prey is capped by the visible biomass of its BEST detector
  (`interactions.py:801-809`, `shared_visible = max over eating rows`). For the planktivores that is
  porpoises (pair floor 0.95), so the cap is >= 0.95 B whatever the fish floors are; the fish floors act
  on each fish pair's own Holling term only.
- M3. Library floors on the planktivores (`fgconfig/fg_library.yaml`): species 0.35 (line 796); pairs
  0.25 for benthivores, water column, mackerel, whiting (lines 1909, 1969, 2006, 2040); 0.95 porpoises
  (2083); seals none (the seal pair, 3226-3237, has no override, so 0.35). Among the observers of the
  planktivores only seals use the species floor ("Gom och sinnen" section 2).
- M4. History (mareld_resume.txt 72.1-72.5, 73): 0.35 was the herring default before pair floors
  existed; porpoises 1.0 (biosonar, Au 1993, Verfuss 2009) then "tuned to 0.95 ... a small residual
  advantage left to schooling"; gadoids -> pelagic 0.25 "a real, deliberate-looking calibration and
  should be motivated", never motivated.
- M5. Band profiles of the planktivores (research_notes/"Krill i Skagerrak"/wc_depth/scripts/
  krill_pairs_wc.py:171-195, used by "Overlapp i fiskparen"/scripts/band_pairs.py): herring and sprat by
  day = the HERAS JULY share in the bottom 10 m by bed depth (0.66 at 70 m, 0.04-0.06 over >= 120 m beds)
  + the rest uniform 10-100 m; by night uniform 0-50 m. Pout 0.7 in the bottom 10 m by day, 0.4 at night.
  Blue whiting below 150 m by day. NONE is seasonal: no winter position is in phi_j.
- M6. Combined derivation (research_notes/"Fangst och stim i fiskparen", not applied): schooling factor
  F per prey species and PERIOD (herring/sprat sigma 0.9 day / 0 night; pout 0.4 / 0; blue whiting
  0.2 / 0), capture p per encounter of a hungry predator with pre-strike escape q 0.8 (0.6-1.0). F and q
  are properties of the prey species and period, not of the prey's activity.
- M7. Mammal pairs carry no F or q: seals -> planktivores a 0.002 Type III is a starting value
  (resume 166.3: "Both attack rates are starting values"); porpoises -> planktivores a 0.0225 is the
  section 151/206 diet fit (research_notes/"Tumlarens byten", flagged).
- M8. Planktivore winter presence: outside_season Jan 0.80 vs Aug 1.33 (of the annual mean), i.e. the
  box keeps ~0.6 of its August planktivores in winter (fg_library.yaml planktivorous_fish).
- M9. budget_gate applies no hide floor ("Stimskydd i motes-a" M4).
- M10. Box deep water: 100-300 m oxygen 5.2-6.3 ml/l in every month (SHARK A16/A17 1991-2020;
  research_notes/"Djurplanktonets diapaus" evidence E2) [F/D there].
- M11. Trained winter hiding (descriptive only): herring visible 0.28-0.50 of B in Nov-Feb in fix78 and
  fix80 (resume 223.27, 223.31, 239.3); fix97 annual mean rest share 0.043 ("Gom och sinnen" section 4).
  Both runs predate the krill group (245).

## Literature

- E1. ICES 2026, stock annex her.27.20-24, section 1.1 [F, read there: research_notes/"Ekolodsdata
  planktivorer"/evidence/seasonality_herring_sprat_3a.txt]: WBSS 2+ ringers feed in the Skagerrak and
  eastern North Sea in summer, aggregate in the eastern Skagerrak/Kattegat at the end of summer, then
  "migrate to the main wintering areas in the southern part of the Kattegat, the Sound and the western
  Baltic"; "The school formation is retained during the overwintering, which mainly occurs in the
  Southern Kattegat and the Sound."
- E2. ICES 2022 NSAS stock annex; HERAS 2023 p. 8 [F, read there]: NSAS juveniles (0-2 ringers) live in
  3.a all year; in July the eastern Skagerrak holds mainly juvenile NSAS plus WBSS.
- E3. Onsrud, Kaartvedt, Rostad & Klevjer 2004, ICES JMS 61:1278, inner Oslofjord, 120 m [F, read
  there: research_notes/"Sillens vinterfoda"/winter_food_and_feeding.md Q2]: herring > 25 cm feed on krill
  by day in the upper part of the krill layer (> 70 m) and shallowest (0-50 m) at night; Nov 2000
  day/night fullness 2.91/2.29, 0 % / 0 % empty; Mar 2001 3.43/2.70, 6.25 % / 0 % empty.
- E4. Segers, Dickey-Collas & Rijnsdorp 2007, ICES JMS 64:60, southern North Sea, February [F, read
  there]: herring 82-288 mm feeding on copepods (59 % by number), amphipods, euphausiids, fish eggs.
- E5. Simmonds & Rivoirard 2000, ICES CM 2000/K:32, IBTS Q1 1983-97 [F, pp. 1-6 read here]: "Some kind
  of time of day effect in catch rate of herring is well supported for the younger herring (1 and 2
  ring)"; "evidence of predictable lower catches in the dark period of the early morning". (A daytime
  bottom trawl catching more 1-2 ring herring by day is consistent with a near-bed daytime position in
  February; the paper does not state depth.)
- E6. Cardinale, Casini, Arrhenius & Hakansson 2003, Aquat Living Resour 16:283, Baltic, October [A]:
  "pelagic fish were dispersed during the night at the surface and aggregated during the day at the
  bottom"; larger herring deeper.
- E7. "Diurnal variations in acoustic density measurements of wintering Norwegian spring spawning
  herring", Huse et al. (IMR Brage record 11250/105417; author list, year and journal from a search
  summary, record not reachable 2026-11-11) [A/S]: downward
  migration at dawn, upward at dusk, part descending again in the dark; "attributed to schooling and
  predator avoidance during the day, and energy conservation and swimbladder refilling during the
  night". Schools 100-400 m by day, 50-400 m at night [S, search summary].
- E8. Huse & Ona 1996, ICES JMS 53:863 [A]: whole adult NSS stock wintered in Ofotfjord and Tysfjord
  from 1987; the fish hold depth by steady swimming at 0.25-0.42 m/s; tilt near horizontal by day, up to
  40 deg at night.
- E9. Dommasnes, Rey & Rottingen 1994, ICES JMS 51:63 (doi 10.1006/jmsc.1994.1006) [A]: wintering
  herring in Ofotfjord in water down to 1-2 ml/l O2; oxygen fell steadily after 1987, plausibly by the
  herring's own respiration.
- E10. Nottestad & Axelsen 1999, Can J Zool 77:1540 (doi 10.1139/z99-124), Tysfjord [A]: 54 observed
  predator-prey events between killer whales and overwintering herring; attacked schools more circular
  and denser than unattacked ones.
- E11. Nottestad 1998, ICES JMS 55:1133, Vestfjorden [A]: overwintering herring released gas during
  attacks "from killer whales and saithe".
- E12. "Seasonal distribution and abundance of killer whales around Lofoten and Vesteralen Islands,
  northern Norway" (IMR Brage record 11250/104994; record, not a paper citation) [A/S]: ~500 killer whales in the
  herring overwintering area (capture-recapture); most leave in January with the herring.
- E13. Kaartvedt, Rostad & Klevjer 2009, MEPS 390:237, Norwegian fjord, 4 winters [A]: overwintering
  sprat in severely hypoxic water (lower limit ~7 % O2 saturation); fed on dormant Calanus by day or not at
  all; gadoid predators avoided water below ~15-20 % saturation but "foraged on migrating sprat at night";
  hypothesis: sprat use hypoxia because they tolerate it better than their predators.
- E14. Solberg, Rostad & Kaartvedt 2015, Prog Oceanogr 138:116 (and thesis) [A]: overwintering sprat
  migrate vertically every winter; near-bed solitary swimming under moderate hypoxia, midwater schools
  when deep water is anoxic; large sprat often with empty stomachs; two tactics "hiding at depth" and
  "hiding in schools".
- E15. Pitcher, Misund, Ferno, Totland & Melle 1996, ICES JMS 53:449 [A]: 14 FEEDING herring schools at
  15-40 m; cod and haddock stayed close and their attacks "caused frequent modifications to, but not
  dispersal of, school structure", so most herring kept feeding with the schooling protection; fast
  schooling predators (saithe) and the vessel made herring dive steeply; loss of feeding and regrouping
  was the cost of evading those.
- E16. Krause & Godin 1996, Behav Ecol 7:264 (doi 10.1093/beheco/7.3.264) [A; L]: guppies foraging nose
  down responded latest and least to a predator model; cichlids preferred to attack foraging guppies;
  risk nose-down > horizontal forager > non-forager. No magnitudes in the abstract.
- E17. Godin & Smith 1988, Nature 333:69 (doi 10.1038/333069a0) [S, citation only]: "a fitness cost of
  foraging" (higher predation risk while foraging) in the guppy.
- E18. Andersen et al. 2017, Bornholm Basin [F, read there: research_notes/"Stimskydd i motes-a"
  school_factor_stdout.txt]: cod on sprat schools sitting at the cod's depth by day: per hour day /
  twilight 0.0011, day / night 0.042 (schooling by day; dissolving schools at twilight).
- E19. Wisniewska et al. 2016, Curr Biol 26:1441 (doi 10.1016/j.cub.2016.03.069) [A]: tagged harbour
  porpoises forage nearly round the clock, up to 550 capture attempts per hour on 3-10 cm fish, success
  > 90 %, from echoes of their own clicks.
- E20. Harbour-seal refuge note (research_notes/"Salars jakt i ljus och morker"/prey_refuge.md) [S/J
  there]: captive harbour seals pursue herring schools actively; Baltic grey seals take mostly herring;
  suggested seals -> pelagic floor 0.4 (0.3-0.5), "the floor should reflect only the refuge value of the
  school"; North Sea and Baltic EwE give seals no prey-specific refuge.
- E21. SMS 2023 North Sea key run, partial M2 by quarter 2013-22 [D here: scripts/sms_quarter_m2.py ->
  sms_quarter_m2_stdout.txt; data in research_notes/"Fiskgrupper i rutan"/planktivores/data/sms2023]:
  fish-predator M2 on herring (annualised): age 1 Q1 0.38, Q2 0.35, Q3 0.18, Q4 0.21 /yr (Q1 1.58 x the
  biomass-weighted annual 0.24); ages 2+ Q1 0.13, Q3 0.10, Q4 0.15 (annual 0.15); all ages Q1 0.75 x
  annual (saithe's Q2 peak). Norway pout all ages Q1 0.61 x annual; sprat 0.55 (mackerel absent in Q1).
  Seals 0.001 and porpoises 0.012-0.015 /yr on herring in every quarter. Confounds: SMS suitability is
  size-based and not by quarter; quarterly M2 moves with predator presence, ration (temperature) and other
  food, so it cannot isolate a refuge. It does say the stomach-based model puts no winter low on
  juvenile herring.

## Derived here

- D1. Winter feeding payoff (scripts/winter_feeding_payoff.py -> _stdout.txt): per tonne of
  planktivores at eat share 1, library pairs on zooplankton (1.5 t/km2 in January) and krill (the open pool's
  per-cell TARGET 5 t/km2 x season x depth shape, before any depletion), shared handling, light and ingestion Q10 as in the library, prey fully visible,
  no depletion (an upper bound): January net gain 24-40 MJ/t/tick by bed class against an eat premium of
  3.8 MJ/t/tick (6.5-10.5 x); July 66-69 against 4.8. Zooplankton alone in January: 16.5 MJ/t/tick
  (4.3 x the premium) - the relevant comparison for the pre-krill runs of M11. Eating pays in every cell
  in January on today's library, so a floor of 1 does not strand the planktivores energetically. It does
  NOT say why fix78/fix80 hid in winter (no krill then; unknown).
- D2. Activity-schooling floor (scripts/floor_readings.py part 1): if a resting fish were fully schooled
  by day (sigma 1 instead of the population sigma) on the combined derivation's terms, the pair floor
  would be 0.93-0.98 (herring and sprat only: WC 0.948, whiting 0.969, mackerel 0.925, benthivores
  0.983), or 0.39-0.63 if resting pout were fully schooled too (no evidence: pout sigma by day 0.4 is a
  judgement, "Vitlinglojans stim"). An upper bound: the population sigma already contains resting fish,
  and E15 shows feeding herring stay schooled.
- D3. Visible share at the implied winter rest (part 2): visible 0.28 / 0.40 / 0.50 at floor 0.25 means
  rest 0.96 / 0.80 / 0.67; at floor 1 the same rest leaves 1.0 visible (x3.6 / x2.5 / x2.0 per-capita fish
  predation in that cell below saturation); at the annual mean rest 0.043 the floor moves visible only
  0.968 -> 1.000.

## Downloads

- Unpaywall helper (2026-11-11): 10.1093/beheco/7.3.264, 10.1038/333069a0, 10.1139/z99-124,
  10.1006/jmsc.1994.1006, 10.1080/00364827.1998.10413679, 10.1006/jmsc.1998.0415 - no open PDF obtained.
- Simmonds & Rivoirard 2000 (E5) was read from the copy WebFetch saved in the session's tool-results
  directory; it is not in papers/.

## Not obtained

- Any per-capita predation rate in a herring wintering layer against the same fish while feeding.
- Winter (Q1) acoustics or depth of the box's herring; no winter vertical distribution of NSAS juveniles in
  3.a was found.
- Full texts of E6-E17 and E19 (abstracts only); Ferno et al. 1998 Sarsia 83:149 (wintering choices).
- Norway pout and blue whiting winter behaviour beyond the profiles (pout spawns Jan-Apr; no
  non-feeding refuge state documented).
