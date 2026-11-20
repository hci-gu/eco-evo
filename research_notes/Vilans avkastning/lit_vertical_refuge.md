# Vertical / layer refuge as a state-dependent choice: literature trail (2026-11-20)

Literature subagent. No model run, no library/code edit, no commit, nothing chosen from model output.
Tags: [F] full text read (pdftotext, or page images of a scanned PDF read directly), [A] abstract only,
[S] secondary (incl. prior project notes and the sibling `lit_activity_budgets.md`), [D] derived here,
[J] judgement. PROXY = not a model species, not marine, or another life stage.
Numbers: `tables/vertical_refuge_raw.csv` (47 rows; built by `scripts/vert_make_csv.py`).

## 0. Test applied to every source

A day-deep / night-shallow pattern alone does not answer the question: the engine already has day and
night profiles f(z) and the light rule. A source counts as evidence of a STATE-dependent refuge only if,
within the same period, it shows (a) fullness or condition differing by depth, (b) only part of the
population migrating, or (c) a position change with predator presence lasting hours. Minute-scale dives
under attack belong to the pre-strike escape q ("Planktivorernas golv", E15) and are not recounted.
For each group the feeding layer is identified from stomach evidence, not assumed from depth.

## 1. Sources (new reads)

### Albert 1994, ICES J Mar Sci 51:45-61, doi:10.1006/jmsc.1994.1005 [F] (scanned PDF, page images read; data/vert_albert94)
Norway pout, Norwegian Deep (incl. Skagerrak subarea 4), 16 cruises 1984-87, bottom trawl + echo sounder.
- Abstract p. 45: "generally 5-10 m above the sea bed, but somewhat closer by day ... The diet was highly
  dominated by euphausiids and the feeding was more intense at night than during the day."
- p. 48: "At night the recordings were typically more or less homogeneous in the lowest 5-10 m of the
  water column ... In daytime, however, this type of recording was observed infrequently and then only
  over short distances."
- p. 49: in the bottom-trawl GLM "The factor for time of day was not significant in any of the
  age-groups" (bottom trawl catches the pout in both periods).
- p. 55 Discussion: recordings "seldom made higher than 20 m above the bottom. Within this range ... a
  distinct diurnal vertical migration was observed, the fish rising at dusk and sinking at dawn to about
  the 1.5-m near-bottom zone." Juveniles (Bailey 1975; Nakken & Rorvik 1979) "perform more extensive
  diurnal vertical migrations, staying close to the bottom during the day but being mesopelagic at
  night" [S]. Dickson (1960) caught pout "primarily 2-4 m off the bottom by day compared with 0-2 m by
  night", Bailey et al. (1983) no diel difference [S] (both p. 55): "adult Norway pout do not always
  perform diurnal vertical migrations ... triggered not only by light, but also by a more variable
  stimulus such as prey availability. Neilson and Perry (1990) point out that teleosts are opportunists"
  (p. 55-56).
- Table 8 p. 55: empty stomachs winter 33.6 % (< 14 cm), 24.5 % (>= 14 cm); summer 27.8 % (< 16 cm),
  18.6 % (>= 16 cm).
- p. 54 text: "The stomach content data from seven hauls from the same locality made over a period of
  22 h ... From midnight until the early afternoon the proportion of empty stomachs steadily increased
  together with the proportion of the contents which were heavily digested (Fig. 11). At the same time
  the mean weight of the stomach contents (excluding empty stomachs) relative to the fish weight was
  halved. At the last haul at 2020 h the stomach contents were only slightly or moderately digested, the
  relative mean weight had increased and the proportion of empty stomachs decreased ... the foraging
  activity of Norway pout preying on euphausiids and copepods was low in the middle of the day and that
  the main feeding period was at night." Month and sunset time of the series are NOT given (text or
  caption), so the hours cannot be labelled day/night beyond "midnight" and "middle of the day".
- Fig. 11 p. 58 (values read by eye; N 13-29 per haul; relative weight excludes empty stomachs): ~2.2
  mg/g near 01 h falling to ~1.0 at ~14-17 h and back to ~2.0 at 20:20; empty stomachs ~0 % near 01 h,
  ~20 % at 06-09 h, ~45 % at ~12 h, ~70 % at ~14 h, ~30 % at ~17 h and at 20:20.
- p. 59: "the feeding was more intense at night than during the day ... the fish are feeding more
  actively when distributed above the bottom at night. Feeding in the dark may be possible because the
  main prey items are bioluminescent euphausiids. It is further possible that by staying close to the
  bottom in daytime, the Norway pout reduces the predation pressure on itself from visual predators.
  Evidence for diurnal feeding behaviour of Norway pout is weak, however, and if present it is probably
  liable to change in response to changes in abundance of prey and predators."
- [J] The clearest refuge/forage split for any box prey group: refuge = within ~1.5 m of the bed by day
  (lower feeding), foraging = lowest 5-10 m at night (krill). It is a DAY/NIGHT switch reported as a
  population pattern, not a within-period mix; the authors flag it as variable (prey- and
  predator-dependent) and the antipredator reading as a hypothesis.

### Albert 1995, ICES J Mar Sci 52:873-885, doi:10.1006/jmsc.1995.0084 [F] (copy in research_notes/"Krillintag och jaktsatt"/data/q2_albert1995)
Coastal bank off SW Norway, 75 and 140 m, 14 trawls over 48 h, July (haddock, pout) and March (poor-cod).
- Small pout (11-15 cm) fed on copepods "mostly in the middle of the day"; "Mean total dry weight by day
  was twice that at night (p<0.01)" (p. 876). Large pout (15-20 cm): "no regular diel variation ...
  at nearly all times most fish had not been eating much, whereas a few fish had eaten a lot".
- Haddock 15-26 cm: two feeding periods (early morning; noon to a few hours after sunset); "The
  proportion of individual fish that had newly eaten ... was less than 10% at midnight and dawn, while
  more than 60% in the evening"; dry weight in feeding periods double that in non-feeding periods; "no
  clear diel variation in catch rate of larger haddock" (p. 875-876).
- Discussion p. 883: contrasts with Albert (1994) night feeding of pout in the Norwegian Deep: prey
  availability (Calanus, krill DVM) is "a reasonable explanation", but "foraging at night might be
  associated with lower predation pressure ... it is possible that Norway pout choose to forage at night
  whenever they are able to capture their prey. When they are not, they are forced to forage in daytime
  at the expense of increased predation risk." [F; the risk reading is the author's hypothesis]
- [J] Pout feeding time follows the prey (copepods by day on the bank, krill by night in the Deep); the
  within-trawl spread of large pout (few heavy feeders, many near-empty) shows that at any hour only part
  of the population is feeding, but depth was not resolved per state.

### Onsrud, Kaartvedt, Rostad & Klevjer 2004, ICES J Mar Sci 61:1278-1290, doi:10.1016/j.icesjms.2004.09.005 [F] (copy in research_notes/"Vitlingens fiskbyten"/data/step2_lit/vertical/onsrud2004.txt)
Inner Oslofjord, 120 m, Nov, Mar, Jun.
- Abstract: herring, whiting and pout "foraging on krill ... during both day and night. During daytime,
  herring and whiting were foraging in the upper and middle part of the krill assemblage, while the
  deep-living, and often benthopelagic Norway pout was approaching the krill from below ... At night,
  herring and whiting were feeding near the surface ... Norway pout foraged in midwater."
- p. 1284: pout "No significant differences in stomach fullness, or normalized stomach wet weight could
  be found between day/night (chi-square and Kruskal-Wallis tests, p > 0.05)"; whiting: "No systematic
  trends ... during the day and night"; herring "contained krill prey during both day and night", deep
  day prey (polychaetes, isopods) "indicating feeding at depth during the day". Pout "had the deepest
  nocturnal distribution, with maximum abundance in midwater and no catches in the upper 25 m".
- Herring visual biting threshold 3.6e-2 to 7e-3 lx (Blaxter 1964) [S, cited p. 1286].
- Cross-predator (krill as prey, PROXY prey): krill DVM gives "relative safety in the darkness of deep
  water, but their pelagic predators were still active. Further descent might have been unrewarding,
  bringing them into the realm of large-eyed, deep-living predators (Norway pout)" (p. 1288) [F, the
  authors' reading].
- [J] For herring, whiting and pout in this fjord the deep day layer IS a feeding layer, not a refuge;
  no day/night fullness difference for pout or whiting.

### Onsrud, Kaartvedt & Breien 2005, CJFAS 62:1822-1832, doi:10.1139/f05-090 [F] (copy in research_notes/"Krillens flackighet"/data/onsrud2005)
- Deep scattering layer (pout, whiting) below ~90 m by day (bed 120 m), into midwater at night; krill
  70-90 m by day. Swimming speed modes 14-16 cm/s day, 10-11 cm/s night (~1 vs 0.7 BL/s); "We ascribe
  the reduction of swimming speed in deep-living fish at night to a switch from visual feeding during
  daytime to nonvisual feeding by night"; "Norway pout appeared to forage on krill in the lower part of
  the water column, even by night". After midnight "the lower edge of the fish scattering layer was
  sinking continuously at ... about 0.3 cm/s" (midnight sinking; Onsrud 2004 offers "cessation of
  feeding" as one explanation).
- [J] Both periods are foraging in the deep layer; no refuge state resolved.

### Kaartvedt, Rostad & Klevjer 2009, MEPS 390:237-249, doi:10.3354/meps08196 [F] (data/vert_upw1)
Overwintering sprat in hypoxic Bunnefjorden (inner Oslofjord), 4 winters. PROXY habitat (box water is
5.2-6.3 ml/l at 100-300 m, "Planktivorernas golv" M10).
- Sprat to ~7 % O2 saturation; gadoids (whiting, cod) confined to > ~15-20 %; "few predators occurred in
  the deep, low-oxygen sprat habitat. However, gadoids were foraging on the vertically migrating sprat at
  night." Sprat fed by day at depth on dormant Calanus "or did not feed at all".
- Table 1: sprat with prey 93 % by day (10 Feb 2005, 55-85 m) vs 47-53 % at night; 0 % in the dense day
  catch of 28 Jan 2004 (70-80 m).
- Table 2 (whiting, share with prey / sprat per stomach): day 14 Jan 2004 60-77 % / 0.73-1.46; night
  14 Jan 2004 50-75 % / 0.5-1; night 28 Jan 2004 64-89 % / 1-1.44; day 8 Feb 2005 42 % / 0.33. No clear
  day/night contrast in the whiting stomachs; the night-predation claim rests on the text: sprat "were
  primarily captured during their migrations to upper waters at night, and was then the primary prey
  for whiting and cod ... fish captured by lure from the ship ... had freshly eaten sprat" (p. 246).
- p. 246: "Asynchronous vertical migrations are normally explained in terms of the hunger satiation
  hypothesis ... However, this may not apply in our case, since foraging of sprat ... was basically
  restricted to deep-living copepods during the day. These migrations possibly rather relate to refilling
  of the swimbladder or for repaying oxygen debts." Part of the population did normal nocturnal DVM; "We
  do not have sufficient data to distinguish these fish from those in deep water". "Vertically migrating
  sprat to some extent became vulnerable to predators, even though they ascended in the shelter of
  darkness."
- [J] A real refuge layer (predator-free by physiology) where feeding is by day and the exposure comes
  from leaving it at night; partial migration present, share unmeasured, driver not hunger by the authors'
  reading. Not a box state.

### Lough et al. 1989, MEPS 56:1-12, doi:10.3354/meps056001 [F] (data/vert_upw1)
Settled 0-group cod and haddock 3-9 cm, Georges Bank gravel, 70-100 m, submersible + trawl. PROXY area
and stage (smaller than the model's juvenile window).
- Abstract: "During the day, demersal juveniles remain within a few centimeters of the bottom ... At
  night, regardless of the current, they rise off the bottom to feed on invertebrates in the near-bottom
  water column." 1986: "After sunset, part of the population rose off the bottom ca 3 to 5 m"; 1987: "At
  night, most juveniles were on the bottom, but some fish were several meters up in the water column";
  stomachs: euphausiids (1986), hyperiids (1987) that swarm off the bed after sunset. Gravel coloration
  read as the predator refuge; predators named silver hake and yearling cod.
- [J] Partial nocturnal ascent of a FEEDING fraction (share not counted); the day refuge is a substrate
  (crypsis on gravel), which the box's mud does not offer.

### Adlerstein & Ehrich 2000, ICES CM 2000/K:01 [F] (OCR text in research_notes/"Vitlingens fiskbyten"/data/step2_lit/vertical)
GOV, 30 hauls, northern North Sea, Nov 1997: "Rates of Norway pout, and of small and large haddock tend
to be higher during the day while catch rates of dab, grey gurnard and small whiting [< 20 cm, 0-group]
are higher at night"; "Catch rates except those of adult whiting varied with time of day ... within
twofold between day and night"; adult whiting no time-of-day effect. [J] Day/night availability to a
bottom trawl (position plus gear vision), within x2; consistent with 0-group whiting closer to the bed at
night and pout/haddock off it at night, but not a state split.

### Bromley, Watson & Hislop 1997, ICES J Mar Sci 54:846-853, doi:10.1006/jmsc.1996.0211 [F] (same folder)
Pelagic 0-group gadoids (cod, haddock, whiting, saithe, pout), May-June, Shetland: "congregating at the
surface in the evening and dispersing downwards in the day ... Few fish were caught near the bottom";
peak feeding "probably related to the vertical migration patterns of their invertebrate prey ... but
might also be modified by predation pressures". PROXY stage (pre-settlement; outside the box's model
window). [F]

### Staby, Rostad & Kaartvedt 2011, MEPS 441:241-255, doi:10.3354/meps09363 [F] (data/vert_upw1)
Maurolicus muelleri, 15 months bottom-mounted 38 kHz, Masfjorden. PROXY (mesopelagic).
- Varied patterns: evening ascent suspended in autumn 2007 (non-migrators, ascent after midnight), no
  dusk ascent Nov-Mar, reverse DVM by part of the layer (20-30 m upward by day; light x7-20), interrupted
  ascents. Abstract: "capability of changing its behaviour in response to ontogeny and internal state
  (satiation and hunger)". The state attribution is interpretive: "A possible explanation ... might
  therefore be that non-migrators were in better condition"; "hunger would motivate ascent in the
  morning, as long as there is sufficient food in the surface layer for taking the risk" (p. 252-253).
  No stomach-by-depth test. Predators named: blue whiting, saithe, haddock.

### Abstract-only sources [A]
- Clark & Levy 1988, Am Nat 131:271-290, doi:10.1086/284789: theory of brief "antipredation windows"
  at dawn and dusk for visual planktivores under visual predators; "Our model does not predict any
  specific nocturnal migration pattern."
- Scheuerell & Schindler 2003, Ecology 84:1713-1720, doi:10.1890/0012-9658(2003)084[1713:DVMBJS]2.0.CO;2:
  juvenile sockeye (PROXY freshwater) migrate to keep a constant light level; predators showed no diel
  depth change; zooplankton did not migrate; window continuous in early summer, split into two
  crepuscular periods in late summer. No hunger state measured.
- Pearre 2003, Biol Rev 78:1-79, doi:10.1017/S146479310200595X: review of gut-content "tracer" evidence
  that individual vertical movements are driven by hunger and satiation (mostly zooplankton); population
  profiles "can not distinguish migratory activity of individual organisms". No fish numbers obtained.
- Dypvik, Klevjer & Kaartvedt 2012, Mar Biol 159:443-453, doi:10.1007/s00227-011-1822-4: Benthosema
  glaciale (PROXY) split into normal, inverse and no DVM within one population; the inverse group
  (200-270 m by day) fed more by day on overwintering Calanus.
- Hixon & Carr 1997, Science 277:946-949, doi:10.1126/science.277.5328.946: juvenile Chromis (PROXY reef)
  mortality density-dependent "only in the presence of two suites of predators: transient piscivores
  attacking from above, and reef-resident piscivores attacking from below".
- Neilson & Perry 1990, Adv Mar Biol 26:115-168, doi:10.1016/S0065-2881(08)60200-X: no abstract; cited by
  Albert 1994 for fish DVM being facultative (opportunism overriding light) [S].
- Linehan, Gregory & Schneider 2001, JEMBE 263:25-44 [S, search summary; doi not checked]: tethered age-0
  cod, predation ~3x higher in deep (to 20 m) than shallow coastal water by day and dusk, lower at night.
  PROXY habitat (0.7-20 m).
- Rosland & Giske 1994, Prog Oceanogr 34:1-43 [S, search summary only]: stochastic dynamic model of
  Maurolicus (PROXY) in January; unequal growth motivation of juveniles and adults reproduces the two
  observed scattering layers. State = stage / growth motivation, not hunger measured in the field.
- Blue whiting, Irminger Sea, 624 stomachs, 22-39 cm (DTU, 2021) [S, search summary; doi not checked]:
  highest intake noon to late evening, minimum in the morning; no depth-by-state data.

### Reused (not redone) [S]
- "Planktivorernas golv": no non-feeding refuge state of the box's planktivores; winter herring layers are
  adult NSS/WBSS outside the box and attacked there (killer whales, saithe); E15 Pitcher et al. 1996:
  feeding herring schools at 15-40 m stay schooled and keep feeding under cod/haddock attack; saithe make
  them dive (minutes, = q); E21 SMS Q1 fish M2 on 1-ring herring 0.38 /yr (1.6x annual).
- "Sillens vinterprofil": box 1-2 ringers nearer the bed by day in winter (bottom-10-m share x2, k
  1.5-3); night 0-50 m in both seasons; annual a moves -5 % to +3 %; July profile kept.
- "Sillens vinterfoda": Oslofjord herring Nov/Mar fullness night/day 0.79/0.79, 0-6 % empty.
- `lit_activity_budgets.md` (sibling): Bernreuther et al. 2013 Baltic sprat fed at night, herring little
  or not at night [A]; Darbyson et al. 2003 herring fullness peaks morning and evening [A]; Hall 1987
  haddock 75 % of food by day [A]; Sogard & Olla 1996 starved juvenile pollock moved 3-4 C colder, field
  pollock stayed shallow when food was high and migrated deeper with low prey [F there, PROXY]; Sogard &
  Olla 1997 hungry juvenile pollock less responsive to predator threat [A]; Vollset et al. 2013 unfed cod
  larvae 4.3 cm deeper in a 115 cm column [A]; Biro et al. 2003 food-rich lakes: less deep risky habitat
  use, +68 % survival [A, PROXY].
- "Overlapp i fiskparen" / "Bottenfiskens hojdprofil" / research_notes/"Krill i Skagerrak"/wc_depth/scripts/
  krill_pairs_wc.py `pout_prof` (checked): bottom-10-m share 0.7 day / 0.4 night [J there]; the rest by
  day uniform 10-30 m above the bed, by night uniform from 25 m depth to bed - 10 m ("Onsrud 2004: no pout
  in the upper 25 m at night, maximum in midwater" [F there]). Albert-1994 day layer (within 1.5 m) was a
  sensitivity only (roughly doubles the saithe -> pout day term), not adopted. Cod ~70 % in a
  near-inactive on-bed class (Hobson et al. 2007); saithe exp. 15 m scale height by day, higher at night.
- "Sejens hojd over botten" section 3 [S/F there]: settled 0-group cod/haddock within a few cm of the bed
  by day, most stay at night, some rise 3-5 m (Lough 1989); Barents small haddock rise 30-40 m at night;
  the band phi_j (uniform bottom 2 m day and night) "right in kind", night-rise variants move the pair
  x0.96-0.99. Decisive is horizontal: in-gape haddock (< 19 cm) over < 100 m beds (box Q3 0.82), adult
  saithe over 150-250 m (0.80).
- Aglen et al. 1997, ICES CM 1997/Y:16 (via "Overlapp i fiskparen") [S]: Barents haddock, large and medium
  from the bed up to 100 m above it by day, down at night; small haddock the reverse (near the bed by day,
  up at night).
- "Torskfiskars ljusberoende vid jakt": gadoid half-detection ~0.005-0.5 lx, no measured adult threshold;
  rule A uses Ke 0.02 lx.

## 2. Per prey group [J unless tagged]

| group | feeding layer (stomachs) | refuge layer | state split within a period? | mix share measured? |
|---|---|---|---|---|
| Norway pout | Norwegian Deep: lowest 5-10 m at night, krill (Albert 1994 [F]); Oslofjord: krill layer from below by day, midwater at night, no fullness difference (Onsrud 2004 [F]); coastal bank: copepods by day (Albert 1995 [F]) | within ~1.5 m of the bed by day (Albert 1994 [F], authors' antipredator hypothesis) | no depth-by-state data; at any hour few fish heavy, most near-empty (Albert 1995 [F]) | no; Fig. 11 empty-stomach shares (~0 % near 01 h, rising to ~70 % at ~14 h; month unknown) are an indicator only [F/D] |
| herring / sprat | herring: day in or above the deep krill layer and at dusk/dawn (Onsrud 2004 [F]; Darbyson [A]); night shallow with digested food; sprat: day at depth on Calanus (fjord, [F]) or night (Baltic [A]) | herring: no non-feeding refuge layer in the box ("Planktivorernas golv" [S]); sprat: hypoxic deep layer (fjord only, [F]) | sprat: synchronous plus asynchronous night migrations, share unmeasured, not hunger by the authors ([F]) | no |
| blue whiting | Irminger Sea intake noon to late evening [S]; depth not resolved | not found (one search) | not found | no |
| whiting juvenile / adult | krill day and night, no diel stomach trend (Onsrud 2004 [F]); 0-group GOV catch higher at night, adults no diel effect (Adlerstein & Ehrich [F]) | Cyanea shelter for 0-group only ("Vitlingens golv" [S]) | not found | no |
| juvenile haddock / cod | haddock 15-26 cm feed by day, <10 % fresh food at midnight/dawn vs >60 % in evening (Albert 1995 [F]); settled 0-group rise 3-5 m at night to feed (Lough 1989 [F], PROXY stage); Barents small haddock up at night (Aglen 1997 [S]) | settled 0-group: within cm of the bed by day on gravel (crypsis) | partial night ascent ("part of the population", "most on the bottom, some several m up") | no |

Can a field profile be split into forage and refuge profiles?
- Norway pout is the only box prey group with a documented refuge-vs-forage geometry: refuge = within
  ~1.5 m of the bed (day), forage = lowest 5-10 m (night, krill) in the Norwegian Deep [F]; but the same
  species forages by day from below the krill layer in the Oslofjord with no diel fullness difference
  [F]. The mix weight per period is NOT identified; any split is [J]. [D] Under Albert's geometry both
  profiles lie inside the bottom 10 m (bottom-10-m share ~1 day and night), so the library's 0.7 / 0.4
  (shares [J]; the night remainder spread from 25 m to bed - 10 m after Onsrud 2004's midwater night
  layer, checked in `pout_prof`) is not reproduced by a mix of Albert's two profiles; the two sources
  describe different sites, not two states of one site.
- Juvenile gadoids: refuge = on the bed (day), forage = 3-5 m above it (night) [F, PROXY stage]; share up
  "part of the population" (1986) to "some" (1987); not counted.
- Herring, sprat, whiting: the deep day layer is a feeding layer; no refuge layer distinct from it in the
  box; no split.
- In every case the observed switch is keyed to the PERIOD (light), not to hunger or predators. The
  engine has that form (day/night f(z), rule A); whether its day/night shares match the field is a
  profile question (pout's Albert day layer is a sensitivity only, not in the library), not a refuge
  state.

## 3. Hunger and predator state [J on F/A/S]
- No measurement in a box species links stomach fullness or condition to depth within a period.
- Direction from PROXY: hungry or food-limited fish search more and accept more risk (pollock, salmon,
  trout; sibling notes), fed or better-conditioned migrators skip the risky ascent (Maurolicus,
  interpretive [F]); magnitudes small (cod larvae 4.3 cm in 115 cm [S]) or not given. The sprat case
  argues against hunger as the driver of its asynchronous migrations [F].
- Predator presence: hours-scale position changes found only with a physical refuge (hypoxia, gravel).
  In open water under attack herring dive for minutes and keep schooling and feeding (E15 [S]) = q.

## 4. Cross-predator trade-offs [J on F/A/S]
- Measured: Hixon & Carr 1997 [A, PROXY reef]: from-above and from-below predators together make
  mortality density-dependent; either alone does not.
- Sprat [F]: the deep refuge from gadoids is left at night; the authors state migrating sprat are then
  the primary prey of whiting and cod (freshly eaten sprat in lure-caught fish); the whiting stomach
  table shows no clear day/night contrast (60-77 % with prey by day, 50-89 % at night).
- Krill vs pout (PROXY prey) [F]: deeper refuge from pelagic visual predators runs into near-bed pout.
- Box geometry [D/J, not measured]: pout and juvenile gadoids on the bed by day leave the saithe band
  (exp. 15 m, higher at night) but sit in cod's on-bed band (~70 % on-bed); rising at night re-enters the
  saithe and whiting bands in the dark. Winter herring nearer the bed meet the bed-band gadoids, matching
  SMS Q1 M2 on 1-ring herring 1.6x the annual mean ([S]). None of these is a per-capita measurement.

## 5. Light (Q4, brief) [S]
Herring biting threshold 7e-3 to 3.6e-2 lx (Blaxter 1964 via Onsrud 2004); gadoid half-detection
~0.005-0.5 lx with no measured adult threshold ("Torskfiskars ljusberoende vid jakt"); Maurolicus layer
follows a light level, x7-20 per 20-30 m (Staby 2011 [F]). Nothing here beyond rule A.

## 6. Not found / not searched
Searched, not found:
- Stomach fullness or condition by depth within one period for pout, herring, sprat, blue whiting,
  whiting or juvenile haddock/cod in open shelf water (herring: one search; pout/gadoids: within the
  sources read).
- The share of a population migrating vs staying (partial migration) for any box species.
- A herring, pout or gadoid vertical shift lasting hours in response to predator presence outside
  hypoxic fjords (one search on herring/predators; one on cross-predator facilitation).
- Blue whiting feeding by depth (one search; only diel timing in the Irminger Sea).
Not obtained: Bailey 1975 (JMBA 55:133) and Neilson & Perry 1990 full texts; Pearre 2003 beyond the
abstract (not OA); Huse & Korneliussen 2000; Rosland & Giske 1994 full text (an open PDF exists at
bio.uib.no, not read).
Not searched (prior notes only or nothing): capelin; Raitt 1968 (pout); Fiksen & Giske models; juvenile
whiting vertical position beyond the GOV day/night ratio.

## 7. Files
- `tables/vertical_refuge_raw.csv`, `scripts/vert_make_csv.py`.
- Downloads (untrusted, not to be committed): `data/vert_albert94/paper.pdf` (scanned; read as page
  images), `data/vert_upw1/` (Unpaywall records; PDFs and pdftotext output for meps08196, meps09363,
  meps056001), `data/vert_cr1`, `vert_cr2`, `vert_cr4`, `data/vert_oa1` (Crossref / OpenAlex records).
- Note on paths: prior-note directory names are written here partly without Swedish letters; on disk
  they are e.g. "Krillintag och jaktsätt", "Krillens fläckighet", "Sillens vinterföda" (the others as
  written).
