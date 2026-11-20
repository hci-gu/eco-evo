# Activity budgets, satiation-dependent activity and state-dependent risk taking in the model's fish species (2026-11-20)

Literature subagent. Research only: nothing outside this folder was edited, no model runs.
Tags: [F] full text read, [A] abstract only (verbatim from publisher/OpenAlex), [S] secondary (search
summary or another paper's citation), [D] derived here (arithmetic shown), [J] judgement.
PROXY = not one of the model species, or not marine, or not the life stage.
Rows: `tables/activity_budgets_raw.csv` (written by `scripts/make_activity_csv.py`).
Downloads: `data/bud_*/` (each its own directory); readers `scripts/fetch.py`, `scripts/xml2txt.py`,
`scripts/oa_search.py` (OpenAlex), `scripts/unpaywall.py`.

Prior work not redone (research_notes/"Mättnad hos rovfisk"/lit_foraging_limits.md section A): Armannsson &
Jónsson 2012 (saithe DST, less activity at night and in winter), Skilbrei & Otterå 2016 (farm saithe),
de Pontual et al. 2012 (hake DVM), Módica et al. 2011 (juvenile hake rest phase), Dempster et al. 2011
(farm saithe stomach contents 2.35x), Persson et al. 2012 (juvenile cod intake 0.22-0.44 of control
under a predator cue).

**What the methods measure (read before using any number).** Depth tags (DST) and acoustic depth
telemetry measure vertical position and its short-term change, not feeding. "On the seabed with tidal
signal" means no vertical movement; it can include horizontal swimming and bottom feeding. Acoustic
positioning measures displacement (m/h). Accelerometers measure body movement (tail beat, "active"
state). None of them measures time spent feeding; a step from any of them to a foraging-time share
is marked [J] or [D].

## 1. Cod (benthivores member; also the best-studied gadoid)

### Hobson, Righton, Metcalfe & Hays 2007, MEPS 347:101-110, doi:10.3354/meps07047 [F]
- Method: DSTs on 92 cod (11 007 fish-days) plus acoustic tags (n = 23), southern and central North Sea
  (Lowestoft, Jutland Bank, Flamborough, eastern Channel). Each day classified by its depth trace into
  4 types: midwater (MW), continuous search (CS), seabed with ascents (SW), seabed (SB: "the continuous
  rise and fall of a tidal wave above the cod").
- Abstract: "cod were closely associated with the seabed for 90% of the time".
- Visual classification of the DST days (p. 105): MW-like large descents 4.3 % of days, CS 5.7 %,
  SW 21.1 %, SB 68.9 %.
- Objective classification (p. 105-106): "The predominant form of behaviour across all areas was SB
  (Fig. 6), accounting for 70% of the behaviour each month (SD = 0.05%), with 20.5% ... attributed to
  behaviour also associated with the seabed (SW)."
- By area (p. 106): Jutland Bank 63.9 % SB, 35.9 % SW, 0.2 % MW. Flamborough 78 % SB + 17 % SW, MW 5 %;
  eastern Channel 80.7 % SB + 11.8 % SW, MW 7.5 %. Southern North Sea (Lowestoft) "from May to
  September they spent more than 95% of their time very close to the seabed"; MW + CS > 25 % in
  November-March (migration). "Across all areas, over 55% of their time was spent within 5 m of the
  seabed."
- Interpretation by the authors (p. 107): "One of the behavioural states (SB) seems to indicate general
  inactivity by the fish, with almost no vertical excursions, although we cannot exclude the
  possibility that cod were moving horizontally across a flat seabed." The other three states "show
  greater vertical movement, implying greater foraging activity". A 10-s-resolution DST record over
  2 weeks in July "did not show any vertical activity".
- p. 109: "at other times cod showed very little activity, spending nearly all their time near or on
  the seabed ... This relatively inactive behaviour might reflect the fact that meal sizes for cod can
  be very large (Adlerstein & Welleman 2000) and hence animals may feed only occasionally, being
  quiescent the rest of the time."
- [J] Share of days with vertical activity (SW + CS + MW): ~30 % across areas. In the southern North Sea
  summer < 5 % of time away from the seabed (the > 95 % "very close to the seabed" can include SW days). SB days are vertically quiet, not proven resting: cod eat benthos on the bed. This is
  NOT a foraging-time share; it is consistent with long quiescent periods but cannot measure them.

### Hobson, Righton, Metcalfe & Hays 2009, Aquat Biol 5:133-142, doi:10.3354/ab00144 [F]
- Method: tidal geolocation of 14 cod with time-depth recorders (southern North Sea 9, Jutland Bank 1,
  English Channel 4), tracks 40-468 d; days classed MW/CS/SW/SB as in Hobson et al. 2007.
- Abstract: "Cod spent time both in midwater (43% of total time) and near the seabed (57% of total
  time)." p. 135: time in extended directed movement / extended localised movement / residence "were 8,
  25 and 67% respectively".
- p. 139: "Our a priori expectation was that when there is little vertical movement, cod might always be
  expected to be sedentary. This certainly is the case in some cases ... However, at other times cod
  showed extended horizontal travel even when there was little vertical movement. Similarly, resident
  cod sometimes showed quite extensive vertical movements ... it is perhaps most likely that these
  vertical movements reflect prey searching behaviour."
- [J] This undercuts reading "seabed" days as rest: a vertically quiet cod can be travelling. The 43 %
  midwater (14 geolocatable fish, mostly southern North Sea migrants) differs strongly from the 10 % of
  the 92-fish 2007 set; subset selection, not a contradiction.

### Freitas, Olsen, Moland, Ciannelli & Knutsen 2015, Ecol Evol 5:2070, doi:10.1002/ece3.1496 [F]
- Method: acoustic positioning and depth telemetry, 181 cod 30-80 cm, Sømskilen (Norwegian Skagerrak
  coast), 2008-2012, 2.8 million depth records.
- Activity metrics (Methods): "(4) activity level, that is, average short-term changes in depth; and
  (5) average distance moved per hour". No time share is reported.
- Abstract: "Diel vertical migration, which consists in a nighttime rise to shallow feeding habitats,
  was stronger among smaller cod ... smaller cod were more active and travelled larger distances ...
  Cold temperatures during winter tended, however, to reduce the magnitude of diel vertical
  migrations, as well as the activity level and distance moved by those smaller individuals."
- Gives direction (night feeding rise; winter lower activity) but no share of time active.

### Espeland et al. 2010, MEPS 405:29-37, doi:10.3354/meps08524 [F]
- Acoustic depth telemetry, 26-45 cm coastal cod, Skagerrak, year round. Discussion (p. 36): "Night is
  the primary feeding period in ... shallow depths, whereas daytime is the resting period in deeper
  waters. Due to the lack of accurate observations of fish behaviour (resting or feeding) and because
  we only have data on depth, we are prevented from drawing strong conclusions". Qualitative only;
  sub-adult (PROXY for the adult stage).

### Righton, Turner & Metcalfe 2000, ICES CM 2000/Q:09 "Behavioural switching in North Sea cod: implications for foraging strategy?" [F] (grey literature; vliz.be imisdocs 276036)
- CAUTION: the document is headed "Not to be cited without permission of the authors". Use only as a
  pointer; the published successors are Righton, Metcalfe & Connolly 2001, Nature 411:156,
  doi:10.1038/35075667, and Righton & Metcalfe 2002, "Multi-torsking: simultaneous measurements of cod
  behaviour show differences between North Sea and Irish Sea stocks", Hydrobiologia 483:193-200,
  doi:10.1023/A:1021387831445 [both title only; abstracts not obtained].
- Method: DSTs (depth every 10 min) on 58 adult cod tagged off Lowestoft in March 1999, 22 returned
  (up to 225 d). "An active fish was therefore defined as an individual that adjusted its depth at a
  rate greater than 7cm min-1, or that changed its direction of movement more than 50% of the time in
  the twenty-minute period either side of the data point" (p. 3). So "active" = vertically active.
- Result (p. 4): "Phase I was characterised by high activity (mean proportion of time active = 0.51 ±
  0.01), followed by a rapid decline into Phase II (mean 0.13 ± 0.01) and then rising through Phase III
  (0.23 ± 0.01) and Phase IV (0.44 ± 0.03)." Phase I = April-May (northward post-spawning migration,
  midwater); Phase II = June to late August (on the seabed "for most or all of the time"); Phase III =
  late August to mid-October (leave the seabed at nightfall, return before dawn); Phase IV = October
  (pre-spawning).
- Diel: no light/dark difference in Phase I; night/day differences in II and III (nocturnal activity in
  III).
- Authors (p. 5-6): Phase II "very low levels of vertical movement, which challenges any notion of
  active foraging ... a significant behavioural switch from high activity to inactivity that has not
  previously been recorded for adult cod. Such a rest period, or 'sit and wait' strategy, may be
  favoured over active feeding in order to conserve energy if prey are not abundant." Abstract:
  "Although we did not measure feeding directly, the lack of activity during summer months indicates
  that feeding may have taken place only rarely."
- [J] This is the only number found for adult cod that is close to a time-active share: 0.13 on the
  summer feeding grounds, 0.23 in late summer/autumn (night active), 0.44-0.51 during migrations. It is
  vertical activity, it undercounts horizontal search on the bed, and migration activity is not
  foraging. Hobson et al. 2007 above is the later, larger analysis.

### Løkkeborg 1998, Anim Behav 56:371-378, doi:10.1006/anbe.1998.0772; Løkkeborg & Fernö 1999, Environ Biol Fish 54:345-353, doi:10.1023/A:1007504712163 [S]
- Abstracts not obtainable (closed; Crossref/Semantic Scholar elide them; Unpaywall: not OA).
- Secondary, Righton et al. 2000 p. 2: "Lokkeborg (1998) found that cod in a fjord in northern Norway
  were more active by day than at night, corroborating evidence from a previous study (Lokkeborg et
  al., 1989) in which cod were found to increase activity levels at dawn. These results were
  interpreted as diurnal changes in foraging activity associated with visual feeding." Search-index
  records: a stationary positioning system tracked cod in late spring; the feeding rhythm was linked
  to a shift to more active prey by day with vision more important; cod made area-restricted search
  on encountering a string of baits by day; higher swimming speeds and a larger range by day than at
  night (Løkkeborg & Fernö 1999). No time-active share obtained.

### Reubens, De Rijcke, Degraer & Vincx 2014, J Sea Res 85:214-221, doi:10.1016/j.seares.2013.05.005 [A] (juvenile/sub-adult cod at offshore wind-farm reefs, Belgian North Sea)
- Abstract (marineinfo.org record): "we integrated acoustic telemetry with stomach content analysis to
  quantify diel activity and evaluate diel feeding patterns at a windmill artificial reef (WAR) ...
  Atlantic cod exhibited crepuscular movements related to feeding activity; a 12 h cycle was found
  and the highest catch rates and stomach fullness were recorded close to sunset and sunrise ...
  Foraging at low ambient light levels (i.e. at dusk and dawn) probably causes a trade-off between
  foraging success and reducing predation pressure. Fish did not leave the area in-between feeding
  periods."
- [J] Feeding concentrated in two crepuscular windows, inactive between them; no share given in the
  abstract. Also relevant to Q4 (a rich, structured food site with residence).

## 2. Flatfish (benthivores members: plaice, dab, long rough dab, witch)

### Hunter, Cotton, Metcalfe & Reynolds 2009, MEPS 392:167-178, doi:10.3354/meps08213 [F]
- Method: DSTs (depth every 10 min) on 139 mature female plaice, 19 182 fish-days, North Sea. Depth
  record converted to "a binary file of activity, or 'swimming' (= 1) versus inactivity (resting on
  seabed, = 0)" (p. 169).
- Abstract: "Plaice infrequently spent > 5 h in mid-water, and rarely left the sea-bed during summer."
  Results (p. 170-171): "On average the fish rarely spent > 5 h swimming per day. Average swimming
  times in excess of 5 h per day were observed only for DST1 West fish during the months of December
  and January. Very little time was spent swimming in mid-water during the summer months, when few
  fish left the sea-bed at all."
- [D] Midwater swimming < 5/24 = 0.21 of the day in winter (migration/spawning), ~0 in summer.
- [J] "Resting on seabed" here is "not in midwater": plaice feed ON the bed, so the tag cannot split
  bottom foraging from rest. The midwater share is transport (selective tidal stream transport), not
  feeding. Hobson et al. 2007 note the same flat depth signal in plaice tags (Hunter et al. 2004).

## 3. Satiation- and ration-dependent activity (Q2)

### Björnsson 1993, Can J Fish Aquat Sci 50:2542-2551, doi:10.1139/f93-277 [A] (Atlantic cod 29-42 cm, i.e. sub-adult; lab, 15 m circular tank, live fish prey)
- Abstract (Crossref, verbatim): "The mean swimming speed was highest (0.6 bl/s) when food intake was
  about half the maximum intake and lowest (0.2 bl/s) when food was unlimited or not available.
  Swimming metabolism was related to food intake in a similar way, but with a larger difference
  between the most active (50 mg O2·kg-1·h-1) and the least active group (10 mg O2·kg-1·h-1) ... The
  estimated energetic cost of swimming as a percentage of energy obtained from the food decreased with
  food intake from 24% at maintenance ration to 2% at maximum food intake. On a log-log plot swimming
  speed decreased linearly with time after encounter of prey."
- [D] Satiated cod swim at 0.2/0.6 = 1/3 of the speed of half-fed cod; swimming metabolism 10/50 = 0.2.
- [J] The load-bearing gadoid result for Q2: activity is HUMP-shaped in food supply. Fed to
  satiation -> low activity; moderately food-limited -> 3x the swimming; no food -> low again (energy
  saving). Speed is a proxy for search effort, not a time share.

### Sogard & Olla 1996, MEPS 133:43-55, doi:10.3354/meps133043 [F] (juvenile walleye pollock Gadus chalcogrammus, 112-213 mm; PROXY gadoid, juvenile)
- Six ration treatments from 10 % BW/d (ad lib) to 0 % for 54 d; motor activity = block crossings per
  5 min, ambit = blocks used per 5 min (Fig. 7).
- p. 50: "there was a general pattern of intermediate motor activity levels for fish on the highest
  ration, and then increased activity as ration initially declined, followed by a marked decrease in
  activity as ration declined to zero levels ... Across all rations, fish typically swam continuously
  but at different rates; periods of passive sinking were not observed." Ambit: "fish in the highest
  ration treatment dropping to the level of fish in the zero ration treatments ... fish on the
  highest ration and the 2 zero rations searched a low proportion of the available area".
- Abstract: "The initial response to food limitation was increased activity, indicative of greater
  searching behavior. With extended food deprivation, a switch to energy-conserving behavior was
  evident"; starved fish occupied water 3-4 C colder, "energy savings of up to 34 %".
- p. 44 cites the cod case: "In juvenile cod, swimming speeds initially increased as food levels
  declined, then decreased with further food deprivation (Bjornsson 1993); O2 consumption rates were 4
  to 5 times higher for cod on intermediate rations compared to higher or lower rations."
- Field corollary quoted in the abstract (Bering Sea): juvenile pollock "remained in surface waters if
  food availability was high, but initiated vertical migration into deeper, colder water with reduced
  prey densities".

### Hammer 1997, Arch Fish Mar Res 45:1-16, "The spontaneous swimming activity of juvenile whiting (Merlangius merlangus) and cod (Gadus morhua) under tank conditions, with regard to feeding levels" [S] (search-index abstract paraphrases; full text not obtained)
- Satiation ration (chopped smelt) 5.9 % BW/d juvenile whiting, 9.1 % BW/d juvenile cod.
- Whiting: mostly inactive apart from steady swimming against a slow current; activity rose during and
  just after feeding or when lights dimmed; spontaneous activity DECLINED with declining food
  consumption.
- Cod: spontaneous swimming increased conspicuously with decreasing ration; at the lowest ration cod
  raised activity up to 6 h before feeding (appetitive search); activity after a dissatisfying meal
  declined only gradually.
- Author's reading: cod search more when food is scarce, whiting save energy. The only whiting number
  found; juvenile, tank.

### Metcalfe & Thorpe 1992, J Anim Ecol 61:175 (end page not checked), doi:10.2307/5520 [A] (juvenile Atlantic salmon, freshwater; PROXY)
- Abstract (OpenAlex, verbatim): "an index of appetite dropped by an average of 96% between early
  September and late December"; after 3 weeks' deprivation "The fish responded by increasing their
  appetite; appetite was negatively correlated with the estimated energy reserves of individual fish
  ... made up their lost fat within 4 weeks of resuming feeding; their appetite then decreased to the
  level of control fish ... the salmon effectively having a 'defended energy level' below which
  appetite is increased until lost energy reserves have been restored."
- [J] Lipostatic appetite in a fish, but a seasonal (winter anorexia) PROXY, not a marine fish.

### Bull, Metcalfe & Mangel 1996, Proc R Soc B 263:13-18, doi:10.1098/rspb.1996.0003 [S]
- Abstract not obtained (OpenAlex has none; not OA). Title: "Seasonal matching of foraging to
  anticipated energy requirements in anorexic juvenile salmon". PROXY. Not used for numbers.

### Robinson & Pitcher 1989a, J Fish Biol 34:631-633, doi:10.1111/j.1095-8649.1989.tb03341.x; 1989b, J Fish Biol 35:459-460, doi:10.1111/j.1095-8649.1989.tb02997.x [S]
- Short communications; no abstract obtainable (Wiley 403). Titles: "The influence of hunger and ration
  level on shoal density, polarization and swimming speed of herring" and "Hunger motivation as a
  promoter of different behaviours within a shoal of herring". Cited in later work (search summary)
  for "nutritional deprivation can lead to increased inter-individual distances". Direction for
  herring: hungry shoals looser and (per the citing study's own experiments on another species)
  faster. No herring numbers obtained. Related: Robinson 1995, Mar Behav Physiol 24:237-242,
  doi:10.1080/10236249509378898 "Food competition in a shoal of herring: the role of hunger" [title
  only].

## 4. State-dependent risk taking (Q3)

### Sogard & Olla 1997, Environ Biol Fish 50:405-413, doi:10.1023/A:1007393307007 [A] (juvenile walleye pollock; PROXY gadoid, juvenile)
- Abstract (NOAA AFSC archived abstract page, verbatim): "Food availability, simulated by a gradient of
  six ration treatments, had a major influence on group cohesion, with increasing dispersion as food
  level decreased. Group cohesion also decreased at night relative to daytime levels. Small juveniles
  (x = 53 mm TL) maintained on high rations were highly responsive to the potential threat of a
  predator, with groups becoming more cohesive and remaining so for up to an hour after the initial
  threat ... Small juveniles maintained on low rations were less responsive to predation threats and
  recovered quickly, supporting the hypothesis that hunger induces risk-taking behavior. Large
  juveniles (x = 149 mm TL) did not change their degree of aggregation in response to either type of
  predation threat."
- [J] Fed small juveniles: anti-predator response lasting "up to an hour"; hungry: weak and brief.
  Large juveniles: no response either way. No foraging-time share.

### Damsgård & Dill 1998, Behav Ecol 9:26-32, doi:10.1093/beheco/9.1.26 [S] (juvenile coho salmon; PROXY)
- Abstract via the OUP page (fetch-tool extraction, one quoted phrase): after food restriction "the
  proportion of fish in the risky patch was significantly higher than before deprivation"; food intake
  and specific growth rate rose in the weeks after deprivation (compensatory growth); deprived fish
  habituated faster to the predator and took larger shares of group meals. No numbers in the
  abstract; full text paywalled.

### Gotceitas & Godin 1991, Behav Ecol Sociobiol 29:255-261, doi:10.1007/BF00163982 [S] (juvenile Atlantic salmon; PROXY)
- Search summary of the abstract: pairs of known dominance; at equal hunger the dominant resumed
  feeding first after a simulated predator; when hunger differed, the hungrier fish resumed first
  regardless of rank. "Hunger level, not social status per se" sets the order. No latency numbers
  obtained.

### Biro, Post & Parkinson 2003, Ecology 84:2419-2431, doi:10.1890/02-0416 [A] (age-0 rainbow trout, whole-lake experiment, cannibal predators; PROXY freshwater, but a FIELD food manipulation - also Q4)
- Abstract (OpenAlex, verbatim): "In high-food lakes, age-0 trout spent less time moving, fewer
  individuals swam continuously, and those swimming continuously swam at slower speeds relative to
  those in low-food lakes. Age-0 trout also used deep, risky habitats less when food was abundant.
  This lower activity, combined with avoidance of risky habitats, coincided with 68% higher survival
  in high-food lakes."
- [J] The clearest field demonstration that more food -> less time moving -> lower predation mortality.
  Time shares are in the full text (not obtained).

### Vollset, Catalán, Fiksen & Folkvord 2013, MEPS 475:191-201, doi:10.3354/meps10129 [A, from the PDF in data/bud_vollset13] (Atlantic cod larvae and early juveniles; PROXY stage)
- Abstract: 16 h unfed vs ad lib: "Unfed larvae distributed deeper in the column than fed larvae, but
  this effect was small (on average, 4.3 cm difference in a 115 cm column) ... at 44 dph ... there was
  an increased number of fish swimming up and down in the column, suggesting an increased motivation to
  sample new environments." Hunger -> more search, small effect.

### Persson et al. 2012 (prior, cited): a predator cue cut juvenile cod consumption to 0.22-0.44 of control (research_notes/"Mättnad hos rovfisk").

No adult gadoid or adult clupeid experiment crossing hunger state with predator risk was found.

## 5. Other model species: activity budgets and diel feeding windows

- **Haddock**: no activity-budget telemetry (DST, accelerometer) found. Diel feeding from stomachs on
  a coastal bank off SW Norway: Albert 1995, ICES J Mar Sci 52:873-885 [S, search summary of abstract:
  14 trawls over 48 h; small Norway pout fed on copepods mostly around midday; stomach dry weight
  varied diurnally in all groups except large Norway pout]. No time share.
- **Saithe**: beyond the prior DST/farm notes: "Diurnal movement patterns have been observed, with the
  highest activity and feeding levels occurring during daytime (Bjordal & Johnstone 1993, Sarno et al.
  1994)" [S, cited in Uglem et al. 2009 p. 252]. No time share; no saithe accelerometer study found.
- **Whiting**: no adult whiting activity budget found (no DST, acoustic or accelerometer study). Only
  the juvenile tank result of Hammer 1997 [S] (section 3): mostly inactive, activity tied to meals and
  dimming light, activity falls with ration.
  Hammer & Schwarz 1994, ICES CM 1994/G:41 [F, vliz copy, data/bud_vliz275264] (0-group whiting, tank):
  undirected spontaneous swimming at 0.2-0.3 BL/s cost 2561 J/kg/h gross, 1022 J/kg/h net, "14% higher" net
  per swimming hour than forced swimming at 1 BL/s. A cost, not an activity share.
- **Hake**: nothing new beyond the prior notes (de Pontual et al. 2012 DVM; Módica et al. 2011 juvenile
  rest phase). No adult activity share found.
- **Norway pout**: no telemetry (too small to tag at sea). Diel feeding of pelagic 0-group in the
  northern North Sea: Bromley, Watson & Hislop 1997, ICES J Mar Sci 54:846-853, doi:10.1006/jmsc.1996.0211
  [F, vliz copy]: diel feeding "size-dependent and species-specific"; Norway pout of 2-4 cm "appeared to feed
  throughout the day but feeding was most intense from 0400-1600 h. Larger Norway pout tended to start
  feeding earlier in the day and appeared less prone to eat over the period 1600-2400 h"; larger
  0-group haddock "appeared to eat throughout the 24 h" (0-group; PROXY stage). Adults: Albert 1995 above (midday copepod feeding,
  small pout).
- **Herring and mackerel**: Darbyson, Swain, Chabot & Castonguay 2003, J Fish Biol 63:1235-1257,
  doi:10.1046/j.1095-8649.2003.00247.x [A] (Gulf of St Lawrence, June): "Two peaks in stomach fullness
  occurred for herring, one in the morning and the other in the evening. Mackerel showed an increase in
  feeding intensity throughout the day with a peak in mid-afternoon ... The estimate of the
  instantaneous evacuation rate for herring was twice that for mackerel." Feeding is daytime/crepuscular
  for both; no feeding-time share.
- **Mackerel (Iceland, summer)**: Kvaavik, Óskarsson, Daníelsdóttir & Marteinsdóttir 2019, PLoS ONE
  14:e0225552, doi:10.1371/journal.pone.0225552 [S, full text read through the fetch tool]: time of day
  had "only small impacts on the variation in the stomach weight"; only 18:00-23:00 significant (lower);
  under near-24 h daylight "a more continuous feeding time frame". Mackerel has no swim bladder and
  swims continuously (no rest state exists for it; the model already carries this as an obligate
  swimming floor) [J].
- **Herring light threshold**: Batty, Blaxter & Richard 1990, Mar Biol 107:383-388,
  doi:10.1007/BF01313419 - title only (abstract not obtainable); content not verified, not used.
- **Sprat (and herring), Baltic**: Bernreuther, Schmidt, Stepputtis & Temming 2013, MEPS 489:177-195,
  doi:10.3354/meps10405 [A] (Bornholm Basin, June; 165 sprat 10-15 cm, 214 herring 12-30 cm): "A
  comparison of the field stomach contents with the estimated gastric evacuation predicted by parameters
  based on laboratory experiments indicated that sprat fed during the night, while herring did not or
  only to a minor extent." Cardinale et al. 2003 (Aquat Living Resour, doi 10.1016/S0990-7440(03)00007-X
  per URL) [S, search summary]: in a 1997 Baltic survey sprat "seemed to continue feeding during the whole
  day time"; herring fullness peaks tracked their depth distribution. Sprat diel feeding is variable
  (night in one study, all day in another); no time share.
- **Haddock and dab, continuous food (demand/ad lib feeding)**: Hall 1987, J Fish Biol 31:479-491,
  doi:10.1111/j.1095-8649.1987.tb05253.x [A] (marineinfo record): "Haddock consumed approximately 75% of
  all food during the day, with a peak of feeding at dusk and an inferred peak at dawn; peaks of feeding
  were associated with larger meals. By contrast, dabs were mainly night-time feeders with daily
  consumption rates that were approximately 40% lower than those shown by haddock. Comparison of haddock
  maximum daily ration with published estimates of consumption in the wild indicate that wild haddock
  feed at between 15 and 22% of their maximum capacity." (Already used in earlier project notes.)
- **Dab, satiation and appetite**: Gwyther & Grove 1981, J Fish Biol 18:245-259,
  doi:10.1111/j.1095-8649.1981.tb03767.x [A] (dab 35-225 g, demand feeders): "fed during daytime ... under
  natural photoperiods"; "changes in daily intake are primarily caused by changes in feeding frequency;
  the amount eaten at a meal is relatively constant"; "interfeeding periods closely followed GET (P =
  1.09 GET - 2.6) supporting the proposal that stomach fullness is a major factor controlling appetite in
  the dab" (except at 6.5 C, "when appetite is suppressed"). Note Hall 1987 found dab mainly night
  feeders: diel timing is not fixed in this species. [J] A direct satiation rule for a model member: a dab
  does not feed again until the stomach has (nearly) emptied.

## 6. Food supply and activity in the field (Q4)

- **Biro et al. 2003** (section 4): whole-lake food manipulation; more food -> less time moving, slower
  swimming, less use of risky habitat, 68 % higher survival [A, PROXY].
- **Uglem, Dempster, Bjørn, Sanchez-Jerez & Økland 2009, MEPS 384:251-260, doi:10.3354/meps08001 [F]**
  (wild saithe at salmon farms, acoustic tags): "When resident at a farm, saithe spent 8 to 10 h d-1
  close to the sea-cages"; Steinviknes-released saithe: daily residence at ANY of the 3 farms fell from 12.7 h/d in week 1
  to 4.3 h/d in week 7, then 4.5-7.7 h/d (weeks 7-12); means 8.4 (Steinviknes) and 10.0 h/d (Lille
  Skognes) (p. 256); Table 2: share
  of tagged fish at the release farm 45 % (01-04 h) to 96 % (05-08 h, start of salmon feeding) at
  Steinviknes; "the diurnal pattern in the residence ... broadly correlated with farm feeding times".
  Residence and timing only; activity not measured.
- **Skjæraasen et al. 2022, Aquacult Environ Interact 14:229-242, doi:10.3354/aei00440 [A]** (cod at
  salmon farms, n = 535 tagged): 54 % detected at farms, 70 % of those for < 1 week; 48 cod > 1 month,
  one 720 d under a farm. Residence only.
- **Dempster et al. 2011** (prior): farm saithe stomach contents 2.35x, empty-stomach share halved.
- **Reubens et al. 2014** (section 1): cod at a wind-farm reef feed in two crepuscular windows and stay
  between them [A].
- No field study found that measured an activity or foraging-time share of a model species with and
  without a food subsidy.

## 7. Synthesis [J]

1. **No source measures a foraging-time share of adult fish of any model species.** Tags measure
   vertical activity, displacement or residence. The closest numbers are adult cod:
   - share of time vertically active (DST, > 7 cm/min or frequent direction changes): 0.13 on the summer
     feeding grounds (June-August), 0.23 in late summer/autumn (nocturnal), 0.44-0.51 in the migration
     phases (Righton et al. 2000, grey literature headed "Not to be cited without permission of the
     authors" - a pointer only);
   - share of fish-days classed "seabed" (no vertical movement): ~0.70 across the North Sea, 0.64-0.81 by
     area, > 0.95 of time in the southern North Sea summer (Hobson et al. 2007); the authors read SB as
     "general inactivity" but cannot exclude horizontal movement.
   But Hobson et al. 2009 show vertically quiet cod can be travelling horizontally, and a geolocated
   14-fish subset spent 43 % of time in midwater. "Seabed" is not "resting".
   Plaice: midwater swimming < 5 h/d (< 0.21), ~0 in summer; bottom feeding is invisible to the tag.
2. **Diel feeding windows are the rule where measured**: cod crepuscular at a wind-farm reef (two windows,
   staying between them), cod nocturnal rise to shallow feeding grounds (Skagerrak coast), herring morning
   and evening stomach-fullness peaks, mackerel a daytime peak, small Norway pout around midday, saithe
   most active by day; with food always available, haddock took ~75 % of their food by day (dusk and
   dawn peaks) and dab fed mainly at night (Hall 1987), or by day in another set-up (Gwyther & Grove
   1981); Baltic sprat fed at night in one study and all day in another. None gives the window length; the windows are consistent with substantial
   non-feeding time but do not size it.
3. **Satiation lowers activity in gadoids, but activity is hump-shaped in food supply.** Sub-adult cod
   swam 0.2 BL/s when fed without limit or not fed, 0.6 BL/s at about half the maximum intake
   (swimming metabolism 10 vs 50 mg O2/kg/h; Björnsson 1993). Juvenile pollock: the same hump, and
   ad-lib fish searched as little area as starved fish (Sogard & Olla 1996). Juvenile whiting
   differ: activity fell with ration (Hammer 1997, [S]). Juvenile cod raise activity before an expected
   meal at low ration (Hammer). Dab: the next meal waits until the stomach has emptied (interfeeding
   period P = 1.09 GET - 2.6 h; Gwyther & Grove 1981), and daily intake is set by meal FREQUENCY, not
   meal size. Field-scale: trout in high-food lakes spent less time moving (Biro et
   al. 2003). Salmon: appetite falls as reserves are restored ("defended energy level"; Metcalfe &
   Thorpe 1992).
4. **Hunger raises risk taking** in every experiment found, all juveniles and mostly PROXY species:
   hungry pollock schools looser and "recovered quickly" (fed: cohesive up to an hour) after a threat
   (large juveniles unresponsive either way); hungry coho use the risky patch more; the hungrier salmon
   resumes feeding first after a predator; well-fed trout avoid risky habitat (68 % higher survival).
   No adult gadoid or clupeid case.
5. **Effect size for the model question [J].** The literature supports the direction - field fish
   spend a large part of the day not searching, and a fed or fat fish searches less and takes fewer
   risks - but it gives no adult foraging-time share to calibrate against. The adult cod numbers
   (vertically active 0.13-0.5 of the time; ~0.7 of days vertically quiet) are an upper/lower bracket on
   a different quantity, not a foraging share. Do not set a parameter from them. A related field
   number: wild haddock eat 15-22 % of their maximum daily ration (Hall 1987, [A]; already in earlier
   notes); [J] read as: gut capacity is far from binding in the field.

## 8. Sought, not found

- Adult activity budgets (time active vs resting) from tags for haddock, saithe (accelerometer), whiting,
  hake, Norway pout, herring, sprat. No whiting telemetry of any kind found.
- Løkkeborg 1998 and Løkkeborg & Fernö 1999 abstracts and numbers (closed; only secondary summaries).
- Broell et al. 2013 (J Exp Biol 216:1255): a method paper on accelerometer sampling frequency; its
  species was not confirmed as cod and it gives no wild activity budget - not used.
- Wild-cod accelerometer time shares. Checked: Skjæraasen et al. 2026, Ecol Evol, doi:10.1002/ece3.73316
  [A] (154 mature cod, accelerometer transmitters, spawning grounds): activity levels by sex and depth
  during spawning, no foraging or time-share content. A 2021 seismic-exposure cod study (van der Knaap
  et al.) seen only in a search summary. Method limit [A, Movement Ecology review 2023,
  doi:10.1186/s40462-023-00403-3]: acoustic accelerometer transmitters "send data summaries ... root-mean
  square error or overall dynamic body action ... (often at 5-12.5 Hz)", i.e. an activity LEVEL, not a
  time-in-state share, unless thresholded by the user.
- Robinson & Pitcher 1989a,b herring hunger numbers (short notes; no abstract obtainable).
- Numbers (latencies, risky-patch shares) for Gotceitas & Godin 1991 and Damsgård & Dill 1998 (paywalled;
  only directions seen). Godin & Crossman 1994 and Bull et al. 1996: abstracts not seen at all.
- Hammer 1997 full text (whiting and cod activity vs ration).
- Any field study measuring a model species' activity or foraging time with and without a food subsidy.

## 9. Downloads not used

- `data/bud_vliz253799` (Van Ginderdeuren et al. 2013, pelagic fish diets, Belgian North Sea): no diel
  content.
- `data/bud_sogard97` (archive wrapper page; the abstract came from `data/bud_sogard97b`).
- `data/bud_codspawn26` (Skjæraasen et al. 2026): spawning-ground activity only.
- `data/bud_oa_*`: OpenAlex records; most returned "no abstract".
