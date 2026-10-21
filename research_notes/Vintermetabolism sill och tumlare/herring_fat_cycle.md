# Seasonal fat and energy-density cycle of Atlantic herring (Skagerrak / Kattegat / North Sea / western Baltic) and adult winter starvation

Model context being checked: herring = lean body ~4.0-4.1 kJ/g + fat reserve (max ~18 % fat, ~0.396 kJ/g per fat %). The model currently drives herring to ~2.7-3.4 % fat in Jan-Feb and loses ~13 % of the stock to starvation in one winter.

Quick model mapping used below (my arithmetic, not a source): ED_model = 4.05 + 0.396 x fat%. So 2.7-3.4 % fat -> 5.1-5.4 kJ/g; 6 % -> 6.4 kJ/g; 18 % -> 11.2 kJ/g.

Search budget note: ~25 tool calls. Pedersen & Hislop 2001, Slotte 1999, Nielsen et al. 2001 and the HAWG reports could not be read in full (paywall / PDF too large); values from them come via secondary tables or search abstracts and are flagged.

---

## 1. Peak and minimum fat % and energy density by season/stock; timing of the minimum and the effect of spawning

### Takeaway
Seasonal minimum fat in herring occurs AFTER spawning, not in mid-winter: late May for North Sea autumn spawners (muscle fat ~4.5 %), March-May (spent fish) for Baltic spring spawners (whole-body ~2-5 %). In Q1 (Jan-Mar) North Sea adult herring of 20-25 cm still have ~6.5 kJ/g (~6 % fat in the model's terms), and in Q3 10.1-11.9 kJ/g. A Jan-Feb floor of 2.7-3.4 % fat is at the very low edge of anything observed and is mistimed by 1-4 months.

### Cited Findings
**North Sea autumn spawners (NSAS) - muscle fat, industry data 2006-2020 (Kenyon et al. 2022, ICES JMS 79(1):88):**
- Abstract (verbatim): "Fat content increased from 4.5% 95% CI [0.03, 0.06] to 16.1% 95% CI [0.15, 0.17] during May and June before plateauing and decreasing to 9.1% 95% CI [0.08, 0.10] in September, which coincided with the onset of spawning." — [Kenyon et al. 2022](https://academic.oup.com/icesjms/article/79/1/88/6459961)
- Seasonal values from the fitted curve: late May 4.5 % (minimum), late June/early July 16.1 % (peak), late July ~15.8 %, mid-Sept 9.1 %, late Oct ~9.1-9.7 %, late Nov 5.3 %. Decline during spawning ~0.11 % fat per day (mid-July to mid-Sept). — [Kenyon et al. 2022](https://academic.oup.com/icesjms/article/79/1/88/6459961)
- Inter-annual maximum fat ranged 16-22 % (mean 18.6 %) over 2006-2020. — [Kenyon et al. 2022](https://academic.oup.com/icesjms/article/79/1/88/6459961)
- Tissue: MUSCLE fat (Fish Fatmeter, Dutch processors; evaporation method, Scottish processors), n = 4,975. Not whole-body; not directly comparable with whole-fish lipid. — [Kenyon et al. 2022](https://academic.oup.com/icesjms/article/79/1/88/6459961)
- Comparison quoted in the same paper: Norwegian spring-spawning herring (NSS, other stock) fat declines only ~0.03 % per day (vs 0.11 %/day for NS herring during spawning). — [Kenyon et al. 2022](https://academic.oup.com/icesjms/article/79/1/88/6459961)
- Data gap noted: no Jan-Apr industry data (no fishery), so the curve is unconstrained in winter. (Inferred from the fitted months reported, late May to late Nov.) — [Kenyon et al. 2022](https://academic.oup.com/icesjms/article/79/1/88/6459961)

**North Sea energy density (kJ/g wet weight) by quarter and length - Pedersen & Hislop 2001 via JNCC Report 633, Table 2 (p. 5-6):**
- JNCC Report 633 = Ransijn, Booth & Smout 2019, "A calorific map of harbour porpoise prey in the North Sea". Herring rows (ref 2 = Pedersen & Hislop 2001; ref 3 = Wanless et al. 2005): — [JNCC Report 633](https://data.jncc.gov.uk/data/c12c1b45-73ba-4402-a8f5-ec0275a72cf1/JNCC-Report-633-FINAL-WEB.pdf)

  | Quarter | Length (cm) | ED (kJ/g) | Ref |
  |---|---|---|---|
  | Q1 | 1.0-9.9 | 4.6 | Wanless et al. 2005 |
  | Q1 | 10.0-11.9 | 4.7 | Pedersen & Hislop 2001 |
  | Q1 | 12.0-14.9 | 4.4 | Pedersen & Hislop 2001 |
  | Q1 | 15.0-19.9 | 4.4 | Pedersen & Hislop 2001 |
  | Q1 | 20.0-24.9 | 6.5 | Pedersen & Hislop 2001 |
  | Q1 | 25.0-39.9 | 8.5 | Wanless et al. 2005 (not P&H) |
  | Q3 | 1.0-4.9 | 4.1 | Pedersen & Hislop 2001 |
  | Q3 | 5.0-5.9 | 4.2 | Pedersen & Hislop 2001 |
  | Q3 | 6.0-7.9 | 3.9 | Pedersen & Hislop 2001 |
  | Q3 | 8.0-9.9 | 4.5 | Pedersen & Hislop 2001 |
  | Q3 | 10.0-11.9 | 4.4 | Pedersen & Hislop 2001 |
  | Q3 | 12.0-14.9 | 5.2 | Pedersen & Hislop 2001 |
  | Q3 | 15.0-19.9 | 10.1 | Pedersen & Hislop 2001 |
  | Q3 | 20.0-24.9 | 11.0 | Pedersen & Hislop 2001 |
  | Q3 | 25.0-39.9 | 11.9 | Pedersen & Hislop 2001 |

- This confirms the claim "Pedersen & Hislop 2001: 10.1-11.9 kJ/g in Q3 North Sea" (for herring >= 15 cm). The "3.9-5.2 kJ/g" figure that appears in some secondary summaries is the small (6-15 cm) Q3 rows of the same table, not a contradiction. — [JNCC Report 633](https://data.jncc.gov.uk/data/c12c1b45-73ba-4402-a8f5-ec0275a72cf1/JNCC-Report-633-FINAL-WEB.pdf); [Pedersen & Hislop 2001 (ResearchGate record)](https://www.researchgate.net/publication/229537916_Seasonal_variations_in_energy_density_of_fishes_in_the_North_Sea)
- Pedersen & Hislop (J. Fish Biol. 59:380-389): seasonal differences in ED were obvious in mature fish, geographical differences insignificant; herring ED = -4.6395 + 0.4170 x (% dry mass). — [Pedersen & Hislop 2001 (ResearchGate record / search abstract)](https://www.researchgate.net/publication/229537916_Seasonal_variations_in_energy_density_of_fishes_in_the_North_Sea)
- JNCC 633 (p. 5): seasonal/length-specific ED could only be applied for herring and whiting because published data are limited; herring "added a substantial amount of energy" in summer. — [JNCC Report 633](https://data.jncc.gov.uk/data/c12c1b45-73ba-4402-a8f5-ec0275a72cf1/JNCC-Report-633-FINAL-WEB.pdf)

**Baltic herring - whole-body lipid (Rojbek et al. 2014, ICES JMS 71(1):56-71; Bornholm Basin, ICES SD 25, i.e. central Baltic, NOT WBSS):**
- Whole-fish lipid ranged 2.1-13.6 % of wet weight over the study (Table 5 / Fig. 4e); sampling months Nov 2008, Mar, May, Aug 2009. — [Rojbek et al. 2014](https://academic.oup.com/icesjms/article/71/1/56/791963)
- 205-214 mm class: ~10 % in November, 3-5 % in March (spring-spawning, maturing), 8-14 % in August for autumn-spawning herring of the same size. Quote: "Lipid content varied between 8 and 14% in the four samples of autumn spawning herring in August 2009 compared with 3 and 5% in the spring-spawning fish of same size classes and maturity stages in March." — [Rojbek et al. 2014](https://academic.oup.com/icesjms/article/71/1/56/791963)
- Quote: "Low lipid content in spent herring in the present study is most likely due to energy spent on reproduction in spring." — [Rojbek et al. 2014](https://academic.oup.com/icesjms/article/71/1/56/791963)
- Pairwise tests: all months except March and May differed significantly (p <= 0.03) (Table 4), i.e. March and May are the joint low. — [Rojbek et al. 2014](https://academic.oup.com/icesjms/article/71/1/56/791963)
- Method: whole fish, Bligh & Dyer extraction of a 10 g blended subsample. No energy density reported. — [Rojbek et al. 2014](https://academic.oup.com/icesjms/article/71/1/56/791963)

**Baltic herring (northern Baltic, Archipelago Sea, C. h. membras) - muscle fat:**
- Muscle fat was somewhat higher in overwintering fish (mean 2-3 %) than at spawning (mean 1.5 %); gonad weight related to fat content. Muscle tissue, northern Baltic - not WBSS. — [Relationship between Food, Fat, Sexual Maturation, and Spawning Time of Baltic Herring in the Archipelago Sea (ResearchGate; authors not confirmed)](https://www.researchgate.net/publication/237183879_Relationship_between_Food_Fat_Sexual_Maturation_and_Spawning_Time_of_Baltic_Herring_Clupea_harengus_membras_in_the_Archipelago_Sea) (search-snippet level; full text not read)
- Baltic herring fillet oil content highest in autumn (~10 %), dropping to 2-3 % in May-June. — [search summary citing ACS J. Agric. Food Chem. paper](https://pubs.acs.org/doi/10.1021/jf000389%2B) (fillet, Finnish Baltic herring; snippet only)

**NW Atlantic (Gulf of Maine / Georges Bank; other stocks, for shape only):**
- Spring spawners: somatic ED peak ~7.5 kJ/g (late May), secondary peak ~7.2 (late Oct), winter minimum ~5.5 kJ/g; fall spawners: peak ~7.8 kJ/g (late June), winter minimum ~5.8 kJ/g; overall range ~5.0-8.5 kJ/g. Spring spawners build gonads over winter. — [Intra-annual ED cycles of spring- and fall-spawning herring (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13397281/)

### Inferences
- Timing: in every series found, the minimum is post-spawning (late May for NSAS muscle fat; March-May spent fish for Baltic spring spawners). WBSS spawn March-May, so their minimum should be ~April-May, not Jan-Feb. In Jan-Feb a spring spawner is still carrying developing gonads (energy moved from soma to gonad, not lost).
- Level in Jan-Feb: linear interpolation of Rojbek's ~10 % (Nov) -> 3-5 % (March) gives ~5-7 % whole-body fat in Jan-Feb for (central-Baltic) spring spawners. NS Q1 adult ED 6.5 kJ/g (20-25 cm) corresponds to ~6 % fat in the model's ED formula; 8.5 kJ/g (25-40 cm, Wanless) to ~11 %. So a realistic Jan-Feb target is ~5-8 % fat (ED ~6-7 kJ/g), with the floor of ~2-4 % reached only in spent fish after spawning.
- The model's 18 % maximum fat is consistent with NSAS peak muscle fat (mean annual max 18.6 %, range 16-22 %) and with Q3 ED 11.0-11.9 kJ/g (model ED at 18 % = 11.2 kJ/g). The upper end looks well calibrated.
- 2.7-3.4 % fat (5.1-5.4 kJ/g) is not impossible in absolute terms (Baltic spent fish reach 2.1 %), but those values belong to spent fish in spring and to the lean, declining-condition Bornholm Basin population, not to North Sea/Skagerrak herring in mid-winter.
- Spawning loss: Kenyon's 0.11 %/day decline during NS spawning vs 0.03 %/day in NSS herring suggests the spawning (gonad + spawning behaviour) phase drives fat loss ~3-4x faster than non-spawning overwintering.

### Gaps
- No Skagerrak/Kattegat-specific fat or ED time series found.
- No WBSS-specific (SD 20-24) whole-body fat or ED series found; WBSS values must be bridged from NSAS (same summer feeding area in Skagerrak/Kattegat/eastern North Sea) and Baltic data.
- Pedersen & Hislop 2001 Q2 and Q4 values not obtained (JNCC only reproduces Q1 and Q3).
- Wood (1958) North Sea fat cycles and Iles (1984) not accessed.
- Arrhenius & Hansson 1996, "Growth and seasonal changes in energy content of young Baltic Sea herring", ICES JMS 53:792 (https://academic.oup.com/icesjms/article/53/5/792/704307) found but not read (young fish, Baltic proper).
- No conversion between muscle fat (Kenyon) and whole-body fat for herring was found; they are not directly comparable.

---

## 2. Verification: Rojbek et al. 2014 "healthy herring have a spring low of ~2 % fat"; Pedersen & Hislop 2001; JNCC Report 633

### Takeaway
Rojbek et al. 2014 does NOT say "healthy herring have a spring low of ~2 % fat". It reports a whole-fish lipid range of 2.1-13.6 % in Bornholm Basin (SD 25) herring, with 3-5 % in maturing spring spawners in March and the lowest values in spent fish; it gives no energy density and no health/survival threshold, and it explicitly says winter lipid was lower than previously reported. The Pedersen & Hislop Q3 claim (10.1-11.9 kJ/g) is confirmed via JNCC Report 633 Table 2.

### Cited Findings
- Area: Bornholm Basin, ICES Subdivision 25 (central Baltic) - not western Baltic. — [Rojbek et al. 2014](https://academic.oup.com/icesjms/article/71/1/56/791963)
- Overall range 2.1-13.6 % ww; March 3-5 % (205-214 mm, spring spawning maturity stages); November ~10 %; low lipid in spent herring attributed to reproduction in spring. — [Rojbek et al. 2014](https://academic.oup.com/icesjms/article/71/1/56/791963)
- Quote: "Lipid content in whole herring in the present study was in general lower from November to March than previously reported" (compared with older fillet data) - i.e. the authors frame the low values as reflecting reduced condition, not a healthy baseline. — [Rojbek et al. 2014](https://academic.oup.com/icesjms/article/71/1/56/791963)
- The paper reports no energy density (kJ/g) and no conversion, and makes no mention of minimum lipid thresholds for survival or starvation. — [Rojbek et al. 2014](https://academic.oup.com/icesjms/article/71/1/56/791963)
- Pedersen & Hislop 2001 Q3: 10.1 (15-19.9 cm), 11.0 (20-24.9 cm), 11.9 (25-39.9 cm) kJ/g wet weight - confirmed in JNCC Report 633 Table 2, p. 5-6. — [JNCC Report 633](https://data.jncc.gov.uk/data/c12c1b45-73ba-4402-a8f5-ec0275a72cf1/JNCC-Report-633-FINAL-WEB.pdf)
- JNCC Report 633 is about harbour porpoise prey (not a seabird report); its herring Q1 25-39.9 cm value (8.5 kJ/g) is from Wanless et al. 2005, not Pedersen & Hislop. — [JNCC Report 633](https://data.jncc.gov.uk/data/c12c1b45-73ba-4402-a8f5-ec0275a72cf1/JNCC-Report-633-FINAL-WEB.pdf)

### Inferences
- Correct citation form for the model notes: "Rojbek et al. 2014: whole-body lipid of central Baltic (SD 25) herring 2.1-13.6 % ww; spring spawners 3-5 % in March; lowest in spent fish after spawning; condition below historical levels." Any wording implying "~2 % is a healthy spring low" or "herring survive at 2 %" goes beyond the paper (though the fish sampled were evidently alive at 2.1 %).
- Precisely which sample gave 2.1 % (month/size/maturity) could not be pinned from the full text via the fetch tool; the two fetches indicated it is a spring (March/May) sample - treat the "spent fish in May" attribution as likely but unconfirmed.

### Gaps
- The exact month/length class of the 2.1 % minimum in Rojbek et al. Table 5 / Fig. 4e is unconfirmed (PDF fetch redirected to a tokenised URL).

---

## 3. Do adult herring feed in winter or fast? Overwintering behaviour (Skagerrak/Kattegat/Oresund)

### Takeaway
Overwintering is a low-intake, reserve-driven phase but not strict fasting: in the Oslofjord (Skagerrak) 31 % of herring stomachs held krill in November and 50 % in March. WBSS herring migrate from Skagerrak/Kattegat/eastern North Sea feeding grounds to overwinter in the Oresund; quantitative overwintering feeding data for the Sound were not found. Strict fasting from Oct-Nov is documented for Norwegian spring spawners (different stock).

### Cited Findings
- WBSS migrate from summer feeding grounds in the North Sea / Kattegat / Skagerrak / eastern North Sea to "constrained overwintering areas in the western Baltic Sea (i.e., Oresund, DK, SWE)". — [Polte et al. 2021, Front. Mar. Sci.](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2021.589242/full)
- Nielsen et al. 2001 (Fish. Res. 50:235-258), "Distribution, density and abundance of the western Baltic herring in the Sound (ICES SD 23) in relation to hydrographical features": the Sound is possibly the major overwintering area for the Rugen spring-spawning stock, and the study indicated prolonged overwintering in the Sound. — [Nielsen et al. 2001 (ResearchGate record)](https://www.researchgate.net/publication/223583219_Distribution_density_and_abundance_of_the_western_Baltic_herring_Clupea_harengus_in_the_Sound_ICES_Subdivision_23_in_relation_to_hydrographical_features) (abstract/snippet level only)
- Oslofjord (Skagerrak region; 120 m site; samples March 1997/98/2000/01, November 2000, June 2001; Onsrud, Kaartvedt, Rostad & Klevjer 2004, ICES JMS 61:1278): krill found in herring stomachs in November in 31.0 % (day) / 17.2 % (night) of fish, in March 50 % (day) / 20 % (night); March daytime mean 5.13 (+-4.64) krill per stomach, night 1.50 (+-0.70). "Herring stomachs contained krill prey during both day and night"; herring fed in the upper/middle krill layer by day and near the surface at night. — [Onsrud et al. 2004](https://academic.oup.com/icesjms/article/61/8/1278/629825)
- Norwegian spring spawners (other stock): peak condition and start of the fasting period at early overwintering (Oct-Nov); energy declines from overwintering to pre-spawning, with losses up to ~50 % of remaining energy during the spawning migration, and no substantial feeding during the spawning migration despite available prey (Slotte 1999, J. Fish Biol. 54:338-355). — [Slotte 1999 (DOI)](https://doi.org/10.1111/j.1095-8649.1999.tb00834.x) (content from search summaries only; full text paywalled - treat as secondhand)

### Inferences
- For the model: Skagerrak/Kattegat/Oresund herring should have reduced but non-zero winter intake (opportunistic krill/zooplankton feeding), not zero. A near-total winter intake shutdown is more appropriate for NSS herring in Norwegian fjords than for WBSS/NSAS.
- Overwintering herring in dense schools with low activity spend less than the active-season maintenance rate; a model that keeps full summer metabolic cost plus near-zero intake through Dec-Feb will overshoot fat loss.

### Gaps
- No quantitative stomach/feeding data for herring overwintering in the Oresund were obtained; Nielsen et al. 2001 full text not read.
- Overwintering depth/activity for the Oresund not sourced. (A search result mentioning 160-370 m and "densest aggregations" refers to Norwegian fjord-wintering NSS herring, NOT the Sound, which is <60 m deep; do not transfer.)
- No measured winter metabolic rate / lipid depletion rate (% per day) for WBSS or NSAS overwintering herring found. Only Kenyon's NSS comparator (0.03 %/day) is available.

---

## 4. Winter starvation mortality in adults vs predation-dominated M; typical adult M

### Takeaway
No source found reports winter starvation mortality of adult herring in these stocks; starvation-related winter mortality in the literature concerns larvae and age-0 fish. Adult natural mortality is treated as predation-dominated (gadoids, seabirds, marine mammals) and stable. Assessment M values found are low (historical constant 0.1 for NS herring; 0.2 constant in an older WBSS assessment, unverified); current assessments take M from the SMS multispecies model (cod-predation-dependent for WBSS).

### Cited Findings
- North Sea herring: older herring (age 2+) "are mainly eaten by gadoids and birds"; predation mortality M2 on 2+ herring "has been very stable and only immediately after the collapse and in recent years has predation removed more 2+ herring than the fishery did"; M2 on 0- and 1-group herring has always been substantially higher than F. — [Dickey-Collas et al. 2010, ICES JMS 67:1875](https://academic.oup.com/icesjms/article/67/9/1875/618208)
- Historical NS herring assessments used constant M = 0.1 for all ages; current NS herring M is time-varying from the SMS multispecies model. — [search summary, source not confirmed](https://academic.oup.com/icesjms/article/67/9/1875/618208) (UNVERIFIED; the Dickey-Collas fetch itself gave no M numbers)
- WBSS (her.27.20-24): predation mortalities are based on the latest key run of the stochastic multispecies model (SMS; Lewy & Vinther 2004) for the western Baltic; herring natural survival depends on cod predation, which increases with cod stock size. — [HAWG report (ICES library)](https://ices-library.figshare.com/articles/report/Herring_Assessment_Working_Group_for_the_Area_South_of_62_N_HAWG_/19249010) (search summary; PDF exceeded fetch limit)
- An earlier (1990) WBSS/Rugen herring assessment reportedly used constant M = 0.2. — search result summary pointing to [Dana vol. 10 (1994), DTU Aqua](https://www.aqua.dtu.dk/-/media/institutter/aqua/publikationer/dana/dana_vol_10_pp_29_59.pdf) (UNVERIFIED; Payne et al. 2009 was checked and does not state M values: [Payne, Clausen & Mosegaard 2009](https://academic.oup.com/icesjms/article/66/8/1673/677024))
- Herring larvae: experimental work shows larvae are little affected by sporadic restricted feeding, indicating predation rather than starvation is the main mortality regulator in nature even at that stage. — [search summary, Illing thesis 2016](https://www.bjornilling.com/publication/journal-article/2016-illing-thesis/)
- First-winter mortality of juvenile (age-0) herring can be very high (estimates up to 95 %), with survival promoted by late-autumn growth - this is a juvenile, not adult, phenomenon. — [NOAA MEPS 623:195-208 (2019)](https://repository.library.noaa.gov/view/noaa/31519/noaa_31519_DS1.pdf) (search summary; NW Atlantic)
- In NW Atlantic herring, consumption by predators other than cetaceans was almost equal to fishery landings in 2013, i.e. predation is a major component of adult M. — [Principal predators of herring, southern Gulf of St Lawrence](https://www.researchgate.net/publication/308520126_Principal_predators_and_consumption_of_juvenile_and_adult_Atlantic_Herring_Clupea_harengus_in_the_southern_Gulf_of_St_Lawrence) (other stock)
- NSS herring (other stock): size-dependent starvation risk is discussed in the literature (smaller fish have higher mass-specific metabolism and lower lipid stores) - found only in a mixed search summary and not attributable to a specific paper; treat as unverified.

### Inferences
- Negative finding (not merely a gap): across the sources checked, adult herring winter starvation mortality is not reported as a component of M for NSAS, WBSS or Baltic stocks. Baltic herring live at whole-body fat as low as ~2 % (spent fish) and Baltic muscle fat of 1.5-3 % without reported die-offs.
- Scale check: if annual adult M ~ 0.2 (= ~18 % loss/yr from all natural causes, predation-dominated), a 13 % starvation loss in a single winter would be most of the total natural mortality and would be on top of predation - inconsistent with the predation-dominated, stable M2 picture for 2+ herring. Even with M = 0.1 (~10 %/yr), 13 % exceeds total M. The model's winter starvation is therefore very likely an artefact (too-low winter intake, too-high winter metabolic cost, and/or a starvation threshold set too high relative to observed fat floors).
- Implication for the model's starvation onset ("starvation from 0.2 of reserve" = ~3.6 % fat if max is 18 %): observed live herring reach ~2-4 % fat after spawning, so a starvation threshold at ~3.6 % would kill healthy spent fish; a threshold nearer ~1-2 % fat (or lean-tissue catabolism rather than direct mortality) is more consistent with the data.

### Gaps
- Age-specific M values from the current HAWG SMS key run for NSAS and WBSS were not obtained (report PDF >10 MB). Expected location: HAWG report, WBSS and NSAS sections, "natural mortality" input tables; and WGSAM SMS key-run reports.
- No direct study of adult herring winter condition-linked mortality in the North Sea / Skagerrak / Kattegat / western Baltic found (searches returned only larval/juvenile or Pacific herring).

---

## 5. Fat % vs energy density (kJ/g per % fat) and lean-tissue energy

### Takeaway
The model's lean body of ~4.0-4.1 kJ/g matches measured ED of small, near-fat-free North Sea herring (3.9-4.5 kJ/g, P&H Q3 6-12 cm). The slope of ~0.396 kJ/g per % fat corresponds to ~39.6 kJ/g of lipid, a standard combustion value, but no herring-specific fat%->ED regression was retrieved; published herring regressions use % dry mass.

### Cited Findings
- Herring ED = -4.6395 + 0.4170 x (% dry mass) (North Sea; P&H 2001); highly significant correlation between % dry mass and ED. — [Pedersen & Hislop 2001 (ResearchGate record)](https://www.researchgate.net/publication/229537916_Seasonal_variations_in_energy_density_of_fishes_in_the_North_Sea)
- NW Atlantic herring: % dry weight of somatic tissue, ovary, testes and whole fish are excellent predictors of ED (r^2 = 0.990-0.994); changes in somatic lipid were the biggest driver of ED; somatic protein stayed at 15.4-19.4 % as % DW increased. — [Intra-annual ED cycles (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13397281/) (other stock)
- Small North Sea herring (6-12 cm, Q3) 3.9-4.5 kJ/g; Q1 juveniles 10-20 cm 4.4-4.7 kJ/g. — [JNCC Report 633, Table 2](https://data.jncc.gov.uk/data/c12c1b45-73ba-4402-a8f5-ec0275a72cf1/JNCC-Report-633-FINAL-WEB.pdf)

### Inferences
- Lean-tissue check: ~4.0-4.4 kJ/g for near-zero-fat herring tissue is supported (juveniles have little lipid; their ED is mostly protein ~16-19 % x ~23.6 kJ/g protein ~ 3.8-4.5 kJ/g - protein calorific value is a standard figure, not verified in this session).
- Slope check: 0.396 kJ/g per % fat = 39.6 kJ/g lipid, consistent with the standard lipid calorific value (~39.5 kJ/g; textbook value, not retrieved here). Since fat largely replaces water (protein fraction ~constant), the additive "lean + fat" model is a reasonable approximation.
- Cross-check of the model's ED formula against data: Q3 adults 10.1-11.9 kJ/g -> 15-20 % fat equivalent; Q1 20-25 cm 6.5 kJ/g -> ~6 %; Q1 25-40 cm 8.5 kJ/g (Wanless) -> ~11 %. The model's Jan-Feb 2.7-3.4 % (5.1-5.4 kJ/g) is ~1-3 kJ/g below the observed Q1 adult values.

### Gaps
- No herring-specific regression of ED on whole-body fat % was retrieved; the 0.396 kJ/g per % value is not sourced to a herring paper here.
- No conversion from muscle fat % (industry Fatmeter data) to whole-body fat % for herring was found.
