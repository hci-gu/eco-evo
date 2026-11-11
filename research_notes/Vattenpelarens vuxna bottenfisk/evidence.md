# Evidence - the adult part of water column -> benthivores (2026-11-11)

Grades: [F] field data parsed here or full text read, [L] lab, [A] abstract only, [S] secondary / via another note,
[D] derived here, [J] judgement. Model facts M1-M6 are read from the library and the exec'd derivations.

## Model facts

- M1 Library pair (fgconfig/fg_library.yaml `predatory_fish_preys_on_benthivorous_fish`): predation_mortality
  0.0833, h 151.1, a 0.0418 (section 269), stage_access [0.392, 0.0388], depth_access_table 0.0693 / 0.057 /
  0.0234 / 0.0128 (mean 1: 1.63 / 1.34 / 0.55 / 0.30). Combined capture-p x schooling factor x2.96 -> a 0.1238
  (research_notes/"Fangst och stim i fiskparen" combined.json).
- M2 Members (library weights, research_notes/"Rovfiskens uppdelning" params_split_mack.md): saithe 0.797, spurdog
  0.097, hake 0.048, anglerfish 0.027, ling 0.014, velvet belly 0.009, pollack 0.008 of 0.617 t/km2.
- M3 Encounter form (section 220, research_notes/"Attackrater ur motesteori" pred_encounter_a.py; band + rule A,
  research_notes/"Overlapp i fiskparen" band_pairs.py): hake and ling TYPE "midambush" with speed v_hake 0.5 BL/s
  ("J: ambusher in midwater, no measurement"), no night speed reduction; anglerfish "bedambush" (strike R 0.4 m [J],
  prey speed 0.1 m/s [J], scan 1); spurdog / velvet belly non-visual R 0.3 m. Member profiles: saithe exp 15 m,
  hake exp 8 m by day / bottom 80 m (not above 30 m) at night, ling exp 5 m day and night ("[J] demersal; no field
  profile found"), anglerfish bottom 1 m. ALL benthivore stages are met as prey in a uniform 2 m slab at the bed
  (H_juv) with u_smallfish 0.1 m/s; the stage window then splits the encounter by stage.
- M4 Membership: a member stage enters the pair when its capability score >= 0.1 (pred_scores.csv; a 0/1 switch,
  not a weight). Hake adult benth_fish 0.55 with the basis "near bottom by day; gadoids 16-48 % of the
  Skagerrak/Kattegat 2024 hake diet (whiting the main one)"; ling 0.60 "judgement: demersal piscivore".
- M5 Windows (research_notes/"Arsungarnas andel" stage_access.py, "Fonsterviktning" aw method): prey length <= p95
  PPR x predator length over the member's box length mix, a(L)-weighted (kexp 1.73), normalised over all
  benthivore tonnes. p95 ratios: saithe 0.30, spurdog 0.34, hake 0.50, anglerfish 0.57; ling has no ratio and
  takes anglerfish's window [J]; diet identity sets: saithe haddock + cod, spurdog flatfish + haddock, hake,
  anglerfish (ling) all benthivores.
- M6 Capture p (research_notes/"Fangstsannolikhet for fiskbyten"): p = q p_s / (1 - r (1 - p_s)), p_s = 0.973 -
  1.230 PPR (bluefish line), q 0.8, r 0.3; the combined derivation applied p(PPR 0.25) = 0.592 to the whole pair.

## Field and literature

- E1 ICES stomach database, public download on disk (research_notes/"Torskfiskarnas bytesandelar"/data/diet/
  ices_stomach, CC BY 4.0), parsed here (`scripts/ices_member_diets.py`), North Sea + Skagerrak 51-62.5 N, 4 W-13 E,
  every year on disk (1981-2025; predators of interest only 1991, 2022-25). Prey mass in g (mg rows converted),
  stage by prey length or, when missing, by length back-calculated from prey weight (lower bound). [F data, D]
  - Hake: 284 non-empty stomachs (2024 mainly; Skagerrak/Trench 176). Benthivores by mass 0.017 in the
    Skagerrak/Trench (one 13 cm plaice, one 6.6 cm flatfish), 0.37 in the rest of the North Sea from ONE 72 cm
    hake with two 24-25 cm haddock (August, 54 m) plus a 12 cm cod; stomach-equal 0.031. Every benthivore item
    is a juvenile (< 27 cm haddock/cod, < 16 cm plaice). No adult benthivore in 284 stomachs. Main prey herring
    0.58, whiting 0.21 (Skagerrak).
  - Anglerfish: 34 non-empty (2022, 2024, 1991). Benthivores 0.28 by mass (A 0.24: one 32 cm haddock by weight in
    a 62 cm anglerfish, a 22 cm plaice, 17-20 cm flatfish n.i.); stomach-equal 0.34 (A 0.18). Other: hake 0.27-0.38,
    herring, horse mackerel, whiting, cephalopods.
  - Ling: 50 non-empty, only 8 with prey mass; benthivores 0.39 by mass = one 16.6 cm flatfish n.i. in a 64 cm
    ling (31 m depth) and a 6 cm cod; Skagerrak 19 ling: herring 0.81, Trisopterus 0.19. Anecdotal.
  - Spurdog: 160 non-empty; benthivores 0.098 by mass (two dab 10-15 cm, 1991); Skagerrak 2024 91 stomachs with
    1.8 g of prey.
  - Pollack 30, saithe (1981 counts only, no masses): no benthivore mass.
- E2 Ecopath North Sea 1991 (Mackinson & Daskalov 2007, Cefas Tech. Rep. 142; EcoBase model 457 XML on disk),
  parsed here (`scripts/ecopath_bf_items.py`). Diet fractions on the model's benthivore members [F]:
  hake 0 (its "Bf 0.069" in evidence_C is "other gadoids (small)"); monkfish 0.186 (haddock adult 0.066, cod adult
  0.0115, cod juvenile 0-40 cm 0.0115, haddock juvenile 0.003, dab 0.030, long rough dab 0.045, lemon sole 0.011,
  plaice and witch 0.004 each: adult part ~0.17); spurdog 0.068 (plaice 0.048, dab, witch, haddock adult 0.006);
  saithe adult 0.004 (haddock adult 0.002). No ling group exists.
- E3 research_notes/"Fiskgrupper i rutan"/predators/evidence_C_diet_consumption.md [S]: Q/B saithe 3.0 (2.87-3.6),
  hake 3.5 (2.2-5.8), spurdog 2.0 (1.5-2.5), anglerfish 1.8 (1.5-2.2); hake diet central P 0.77, W 0.21, Bf 0.01
  (ICES Skagerrak Aug 2024 n 50; model box n 33: Bf 0); Bergstad 1991 (Norwegian Deep 1984-87) hake: blue whiting
  95-98 % [S/A].
- E4 SMS North Sea key run 2023 (ICES WGSAM 2024; files on disk), parsed here (`scripts/sms_adult_share.py`)
  [F data, D]: hake's SMS prey are herring and Norway pout only (no cod or haddock M2); anglerfish, ling and
  spurdog are not SMS predators. Saithe's haddock flow 2001-22 is 0.17 adult (>= 27 cm, Lsea per age and quarter,
  CV 0.10 [J]): age 2 haddock are 25-30 cm. SMS overshot the saithe stomach haddock share 1980-91 (9.1 vs 4.1 %),
  research_notes/"Sejens hojd over botten" E13.
- E5 research_notes/"Sejens diet pa unga torskfiskar" (a-ring) summary and evidence_saithe.md [S]: saithe 5 % (2-9)
  of food, "almost all 0-1 group haddock", cod a trace, flatfish <= 2 % (only saithe > 70 cm).
- E6 same note, evidence_spurdog_anglerfish.md [S]: spurdog benthivores ~6 % (2-15), mostly plaice; anglerfish
  ~15-20 % (8-25): Ecopath 19 %, Iceland MFRI 2024 ~21 % (haddock 4.7, cod ~16 % by weight, read from bars),
  Laurenson & Priede 2005 Shetland (FishBase rows) flatfish 7.9 %, haddock/cod 0, Crozier 1985 Irish Sea > 60 cm
  8 %. Anglerfish detection c. 5 m (Laurenson 2004, via pred_encounter_a.py).
- E7 research_notes/"Bottenfiskens hojdprofil" summary (Hobson et al. 2007, Aglen et al. 1997, Engas et al. 1988,
  Boudreau 1992) [S]: share in the bottom 2 m: cod and haddock 20-34 cm 0.70, haddock >= 35 cm 0.30 day / 0.50
  night, flatfish and long rough dab 1. Weighted by the adult biomass here: 0.71 day, 0.76 night [D].
- E8 research_notes/"Fangstsannolikhet for fiskbyten" [L/D]: success per strike declines with PPR (bluefish
  0.73 / 0.48 at PPR 0.20 / 0.40; every size-resolved set declines). Applied here per member length and prey
  length inside each window (D): mean p of the encountered A tonnes hake 0.50, anglerfish (and ling as
  anglerfish) 0.48, saithe 0.56, spurdog 0.59, against 0.59 for J.
- E9 research_notes/"Rekryternas bottendjup" [F data / D]: J and A bed-class shapes (GOV), the proposed
  recruit_depth_table 2.71 / 1.00 / 0.21 / 0.05, class-level F_eff of the pair J 0.655, A 0.882.
- E10 research_notes/"Sejens hojd over botten" [F data / D]: saithe Hs 15 m stands; haul-level co-occurrence of
  saithe with in-gape haddock 0.09-0.30; booking 0.0833 at the top of the field range.
- E11 research_notes/"Vattenpelarens bottenfiskbyte" (a-umlaut)/tables/rest_diet_ices.csv (ICES stomach extract by
  area and size, %W of identified prey) [F data, via a literature subagent]: ling NNS 2000+ 25 fed (< 40 cm):
  Norway pout 92.3, cod 5.7 (~1.8 g, a ~6 cm fish); Skagerrak 16 fed: herring 80.8, Trisopterus 19.2, benthivores
  0; CSNS 3 fed: "flatfish/ray/gurnard" 100 (89.5 g). Hake CSNS 24 fed (mostly < 40 cm): haddock 79.5, cod 4.3 of
  344 g (the same 2024 juveniles as E1); NNS 66 fed: benthivores 0; Kattegat 83 fed: flatfish/ray/gurnard 4.9.
  Anglerfish Skagerrak 8 fed: flatfish/ray/gurnard 58.8, cod 3.9 (97 g); NNS 6 fed: haddock 24.9 of 1235 g;
  CSNS 16 fed: flatfish/ray/gurnard 35.6. No prey lengths in that extract.
- E12 MFRI 2024 (HV 2024-01, Faeda 36 tegunda botnfiska a Islandsmidum 1996-2023, pp. 138-139; excerpt in
  research_notes/"Sejens diet pa unga torskfiskar" (a-ring)/data/excerpts) [F figure pages]: haddock "about 8 %"
  of the prey weight of halibut and ling; cod ~6-7 % of ling's (Fig. 160c, read from bars): ling haddock + cod
  ~15 %, all sizes and years pooled. Flatfish share, ling n and sizes not on these pages.
- E13 Bergstad 1991, Sarsia 75:269-313, doi:10.1080/00364827.1991.10413455 [S via FishBase food items]: ling in
  the Norwegian Deep eat cod (juveniles and adults), saithe, blue whiting, Norway pout; no shares reachable (no
  open copy).
- E14 Lokkeborg, Skajaa & Ferno 2000, J. Exp. Mar. Biol. Ecol. 247:195-208, doi:10.1016/S0022-0981(00)00148-9 [A]:
  5 tracked ling: activity peaks at dawn and dusk, small core area, restricted-area search near bait, reaction to
  bait by day not at night; more visual than cod, taking more mobile prey; stomachs mostly herring.
- E15 Brooks, Beylich & Hawley 2026, Front. Mar. Sci. 13:1863630, doi:10.3389/fmars.2026.1863630 [F HTML, weak]:
  tags at ~300 m: three stayed just above the sea floor at all hours, two rose 250-280 m at night; the paper
  names the tagged species ling in the abstract and tusk in the discussion.
- E16 Mahe et al. 2007, ICES J. Mar. Sci. 64:1210-1219, doi:10.1093/icesjms/fsm100 [F HTML table]: Biscay and
  Celtic Sea hake 7-53 cm, 405 non-empty of 1995 (79.6 % empty, daylight GOV hauls): no haddock, cod or
  pleuronectids; fish prey 3-24 cm, mostly 0-group.
- E17 de Pontual et al. 2012, J. Fish Biol. 81:728-734, doi:10.1111/j.1095-8649.2012.03345.x [A]: hake DSTs show
  diel vertical migration. FishBase [S]: hake "live close to the bottom during day-time, but move off-bottom at
  night"; ling "mainly on rocky bottoms", feeds on "cod, herring, flatfish ... lobsters, cephalopods".
- E18 Bozzano, Sarda & Rios 2005, Fish. Res. 73:29-36, doi:10.1016/j.fishres.2005.01.006 and related 24-h
  Mediterranean series [A, attribution of quotes unverified]: juvenile hake feeding starts in the early night,
  fullness peaks at sunrise as they return to the bottom (euphausiid prey).
- E19 Mauchline & Gordon 1984, JMBA 64:657-665, doi:10.1017/S0025315400030320 [A]: Rockall Trough ling
  "primarily fish eaters". Not found anywhere: a ling Q/B, a ling prey/predator length ratio, a post-2005
  size-resolved in-area anglerfish diet.
