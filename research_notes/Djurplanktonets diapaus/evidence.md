# Evidence: zooplankton dormancy, its mortality and metabolism, and DVM (Mareld box)

Quotes and data used in the summary of this folder. Tags: [F] field or lab data read in the source (full text, or abstract where marked ABS); [S] search snippet or secondary citation only, not verified in the source; [D] derived here (script named); [J] judgement.
Earlier notes that settled points are cited, not redone: `../Zooplankton vinterdödlighet/` (winter_stock.md, zoo_winter_mortality.md, zoo_autumn_closure.md), `../Zooplankton energibudget/copepod_energetics.md`, `../Sillens vinterföda/zooplankton_refuge.md`.

## E1. A17 0-25 m by taxon and Calanus stage (SHARK)

Source: SHARK zooplankton, Å17, WP2 90 um, 25-0 m, 2007-2023 (abundance) and 2019-2023 (carbon). Reused download, see data/README.md. Script `scripts/a17_stage_season.py`, output `tables/a17_stage_season_stdout.txt` (visit means and medians).
Note: `sample_id` is empty for the 2007-2018 visits, so visits are keyed on date + id + depth. A key on `sample_id` alone pools all of them into one false "visit" (this inflated June 2007 into a 1.6 million ind/m3 outlier in a first run).

Carbon, ug C/m3, mean of visits per month (2019-23; n = 2-5) [D]:

| | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Total | 3923 | 2477 | 14490 | 64867 | 87326 | 38426 | 23059 | 14651 | 12287 | 9733 | 11642 | 4008 |
| Calanus, all stages | 2750 | 1755 | 9228 | 56983 | 66327 | 13556 | 10943 | 9200 | 9548 | 2866 | 2686 | 475 |
| Calanus C5 | 1472 | 275 | 2359 | 18615 | 37704 | 6092 | 3953 | 4438 | 1870 | 1463 | 1432 | 266 |
| Calanus C5 / total | 0.38 | 0.11 | 0.16 | 0.29 | 0.43 | 0.16 | 0.17 | 0.30 | 0.15 | 0.15 | 0.12 | 0.07 |
| Resident (total minus Calanus and Metridia) | 710 | 615 | 5252 | 7135 | 20496 | 24867 | 12065 | 5237 | 2652 | 6812 | 8934 | 3075 |

- May to June, Calanus falls by 52,800 ug C/m3 (C5 alone by 31,600), while the resident carbon rises from 20,500 to 24,900 [D]. In t WW/km2 over 0-25 m at C/WW 0.084, Calanus is 19.7 in May and 4.0 in June: a loss of 15.7 t/km2 in one month [D].
- In 2019-23 Calanus copepodites are reported to genus only, so the carbon cannot be split by species. In 2007-18 abundance, C. finmarchicus C5 averages Apr 329, May 141, Jun 83, Jul 131, Aug 71, Sep 25 ind/m3. C. helgolandicus copepodites appear only in August (C5 9.9 ind/m3) [D].
- Small taxa in abundance (mean of 2007-23 visits, ind/m3) [D]:
  - Centropages hamatus: the median is 0 in every month except June (41). The means are 104, 91 and 27 ind/m3 in May-Jul, with single-visit traces in Sep-Oct and Jan. This fits a resting-egg phase.
  - Pseudocalanus drops 2,103 to 321 from July to August. The median drops 1,280 to 128.
  - Temora drops 738 to 72 from July to August.
  - Paracalanus, Acartia, Oithona and C. typicus stay high or peak in Jul-Oct.
  - Microcalanus, a deep-living form, is present only Nov-May.

## E2. Temperature and oxygen at depth, A16/A17 (SHARK, 1991-2020)

Script `scripts/a17_depth_temperature.py`, output `tables/a17_depth_temperature_stdout.txt` [D]. A17 (median water depth 340 m), monthly means:
- 0-10 m: 3.8 C (Feb) to 17.4 C (Aug).
- 50 m: 6.7-12.0 C.
- 100 m: 6.9-9.1 C.
- 150-300 m: 6.6-8.6 C (annual 7.0-7.4 C).
- Oxygen at 100-300 m: 5.2-6.3 ml/l in every month. The deep box water is never hypoxic, so it offers no hypoxic predator refuge like the inner Oslofjord (Kaartvedt et al. 2021, winter_stock.md row 10).
- A16 (202 m) is within 0.3 C of A17 at 100-200 m.

## E3. Calanus in a 120 m Skagerrak-coast fjord: Hirche (1983) [F, full text]

Hirche H-J (1983) Overwintering of Calanus finmarchicus and Calanus helgolandicus. Mar Ecol Prog Ser 11:281-290. doi:10.3354/meps011281. PDF in data/hirche1983/.
- p. 281 (abstract): "in a shallow (120 m) Swedish fjord and in a deep (650 m) Norwegian fjord ... The deep population, consisting of Copepodid stage V (CV) and a few females, was torpid, had large oil sacs and empty guts. Their respiratory and digestive enzyme activities were very low. In the Swedish fjord CV in deep layers weighed much less than those in the Norwegian fjord ... Metabolic rates allowed successful overwintering only in the Norwegian fjord."
- p. 281-282 (methods): the Gullmarfjord is at "58°20'N 11°30'E, maximum depth 120 m". "In the Gullmarfjord during autumn and winter virtually no CV of Calanus finmarchicus nor C. helgolandicus occurred in the surface layers although many CV's occurred in deeper waters". The deep-water temperature "below 50 m throughout most of the year ... is 6 °C".
- p. 283: "In the Gullmarfjord, from August to December 99 % of the population were CV. They were found almost exclusively below 50 m". Antennules stretched backwards, "copepods did not show any escape reaction when touched" (continued on p. 284).
- p. 283, Table 1, specific respiration of CV (ul O2 mg-1 DW h-1):
  - Gullmarfjord: 17.11.77 0.30; 6.10.78 0.44; 14.12.78 0.35; **13.6.79 1.49**; 21.3.80 0.48.
  - Korsfjord, same dates, deep (d) vs surface (s): 23.8-4.9.78 CVd 0.11 vs CVs 0.72; 21/23.11.78 CVd 0.18 vs CVs 0.79.
  - The in situ ETS assays were all incubated at 6 °C (p. 282), so the Korsfjord pairs compare equal temperature [D]: deep/surface = 0.15 and 0.23. The Gullmarfjord winter/June ratio is 0.20-0.30 [D].
- p. 284-285: "Although respiration of overwintering CV was only 1/3 to 1/5 of that in summer these results cannot be used to investigate overwintering trends since they were obtained from measurements in different years."
- p. 287 (discussion): "For the lowest respiration rate measured ... the CV from Gullmarfjord could survive for only 120 d." "In places like the Gullmarfjord the species considered here cannot overwinter." "the formation of the overwintering stock commences at the end of May by accumulation of CV ... In the Gullmarfjord, however, in July both Calanus species have virtually disappeared, although by the end of August and during September and October large numbers of CV are found again, their origin is unclear".

## E4. Skagerrak Calanus seasonality: Falkenhaug et al. (2022) [F, full text, first 100k characters read]

Falkenhaug T, Broms C, Bagøien E, Nikolioudakis N (2022) Front Mar Sci 9:779335. doi:10.3389/fmars.2022.779335. Arendal station 2: 75 m deep, 1 nmi offshore, on the northern slope of the Norwegian Trench, WP2 50-0 m.
- Discussion, "Ecology": "Abundances of C. finmarchicus decreased to low numbers in July when surface temperatures reached >12°C." "During the summer months it is likely that the species stayed below the thermocline". "Predation and descent to overwintering depths further off-shore may also contribute to the abrupt decline in July." They cite Jónasdóttir et al. 2005 for descent starting in July and complete absence from surface waters by September.
- "C. finmarchicus is believed not to overwinter in the North Sea - Skagerrak in substantial numbers"; "diapausing individuals enter the area during winter via the deep Norwegian Trench Atlantic Inflow"; "overwintering aggregations are known to exist in the Norwegian Trench".
- C. helgolandicus has its "main occurrence during July-November with a long-lasting peak from about August to October".
- "no information on vertical distributions are available from this site."

## E5. Dormant vs active metabolism

- Maps F, Record NR, Pershing AJ (2014) A metabolic approach to dormancy in pelagic copepods ... J Plankton Res 36:18-30. doi:10.1093/plankt/fbt100 [F, ABS]: "the metabolic rate of dormant copepods is about one-fourth of the values for actively growing copepods, a level that remains consistent across a large range of body size or environmental conditions" (meta-analysis, 15 species).
  - Note: the comparison is with *actively growing* copepods, so it includes the feeding and growth increment (SDA).
- Grigor JJ et al. (2022) Front Mar Sci 9:909528 [F, full text; PDF in data/grigor2022/]:
  - Introduction: "respiration rate is reduced to 10-20% of that recorded in shallower conspecifics (e.g., Hirche, 1983; Ingvarsdóttir, 1998; Ingvarsdóttir et al., 1999; Auel et al., 2003) ... diapause, which may last 6 to 8 months".
  - Abstract: mesopelagic CV "show highly reduced activity (88.5 ± 3.4% reduction)".
  - Results: in late-summer Fram Strait, "no significant difference in respiration rate between the meso- and epipelagic copepods" (4 °C and 2.8 °C).
- Wilson RJ, Banas NS, Heath MR, Speirs DC (2016) Glob Change Biol 22:3332-3340, doi:10.1111/gcb.13282 (Strathprints postprint p. 22, lines 372-373) [F]: "a consistent pattern across Calanus species, with overwintering metabolism typically 25% of that in surface animals (Maps et al., 2014)". They also caution that respiration of animals brought up from depth may be raised by the disturbance.
- Saumweber WJ, Durbin EG (2006) Deep-Sea Res II 53:2597 [S, abstract via search summary]: potential C5 diapause from about 280 d at 0 °C to about 90 d at 11 °C. Interpolated at 7.0-7.4 °C (the box's 150-300 m): 130-136 d [D, `scripts/rest_payoffs.py`].
- Johnson CL et al. (2008) ICES J Mar Sci 65:339-350 [F, ABS]: "individuals can only enter dormancy if their food and temperature history allows them to accumulate sufficient lipid to endure overwintering, moult, and undergo early stages of gonad maturation."
- Jónasdóttir SH (1999) Lipid content of Calanus finmarchicus during overwintering in the Faroe-Shetland Channel. Fish Oceanogr 8:61-72. doi:10.1046/j.1365-2419.1999.00003.x [F, ABS]: "maintenance of diapause and molting is not expensive in terms of lipid utilization". In the deep water "the wax ester (WE) content did not change between the autumn and spring".
- Small residents: no measured respiration drop beyond temperature for Temora, Paracalanus or Oithona (winter_stock.md Q3: not found). Fed vs starved Acartia: respiration more than 4x higher when fed (copepod_energetics.md, Kiørboe et al. 1985).

## E6. Mortality while dormant vs active

- Bagøien E, Kaartvedt S, Aksnes DL, Eiane K (2001) L&O 46:1494-1510. doi:10.4319/lo.2001.46.6.1494 [F, ABS via OpenAlex]. Four Norwegian fjords, 380-1300 m deep:
  - "The highest daily per capita mortality rate for Calanus was estimated at 0.024-0.027 d-1 (95% CI) in a fjord with high fish abundance and with the clearest water. Predation by M. muelleri and B. glaciale alone could explain the estimated winter mortality."
  - "The fjord devoid of mesopelagic fish but particularly rich in invertebrate predators gave the lowest estimated mortality rate; 0.008-0.009 d-1".
  - "mesopelagic fish pose a stronger predatory threat than invertebrates to overwintering Calanus."
- Gislason A, Eiane K, Reynisson P (2007) Mar Biol 150:1253-1263, doi:10.1007/s00227-006-0400-7 (citation checked via Crossref; the rate is from a search summary only) [S]: overwintering mortality in the Irminger and Iceland basins, Nov-Apr, "0.004 ± 0.0028" per day.
- Melle W et al. (2014) Prog Oceanogr 129:244-284 (Strathprints postprint) [F]:
  - pdf p. 50: "data from the two major overwintering basins ... indicate that mortality during overwintering is low and less variable than during the active phase ... Still, losses of almost the entire population from the Norwegian shelf during winter indicate that predatory or advective losses during winter may be substantial in shallower regions".
  - pdf p. 52, North Sea: "The primary factor controlling abundance in the North Sea appears to be predation mortality during the early life stages ..., which probably also holds for the older stages, which cannot migrate vertically far enough to avoid predation, because of the relatively shallow bathymetry."
  - pdf p. 37: median dormancy duration "250 d in the northeast Atlantic".
- Kaartvedt S (2000) ICES J Mar Sci 57:1819 [F, ABS]: "Late-stage copepodites are heavily exploited by seasonally migrating fish during summer, so representing a driving force for the seasonal descent of C. finmarchicus. Overwintering takes place below the main domain of mesopelagic fish."
- Tönnesson K, Tiselius P (2005) MEPS 289:177-190 [F, ABS], Gullmar fjord (same fjord as E3): "S. elegans consumed 5 to 18% of the deeper dwelling Calanus community daily"; S. setosa removed up to 48 % of single small-copepod species daily at the end of October.
  - This is tactile predation on the deep Calanus in a 120 m fjord, of the same order as or above the active-phase rates [J].
- Active-phase reference (zoo_winter_mortality.md rows 4, 8, 9):
  - Hirst & Kiørboe 2002: broadcast spawners 0.048/d at 5.3 °C and 0.089/d at 13.8 °C; predation-free floor 0.019/d at 5 °C.
  - Hirst et al. 2007, C. helgolandicus CV-adult female at L4: 0.099/d at 13.8 °C.
  - Eiane & Ohman 2004, Fladen Ground Pseudocalanus C5: 0.05/d.

## E7. Dormancy in the small residents

- Wesche, Wiltshire & Hirche 2007 (Helgoland; winter_stock.md rows 13-15) [F]:
  - Acartia clausi: spawning zero in Dec-Jan (reproductive dormancy of the adults).
  - Centropages hamatus: absent Oct-Mar, returns in May "probably from resting eggs".
  - Temora: pelagic all winter, and "known to produce resting eggs in the North Sea".
- Norrbin MF (1996) MEPS 142:99-109. doi:10.3354/meps142099 [F, ABS]: Pseudocalanus acuspes and Acartia longiremis in North Norwegian fjords "switch from developing populations ... to resting populations, with immature gonads, within a few weeks in early fall".
- Renz J dissertation (Bremen 2006) [S, search summary]: "a vertical distribution below the summer thermocline is compellent for P. acuspes" in the Baltic. Peters et al. 2006 [S]: P. acuspes is "largely absent from the adjacent North Sea".
  - Which Pseudocalanus species occurs at A17 is not resolved in SHARK (genus only) [D].
- Resting-egg banks: Madhupratap et al. 1996, Kiel Bight 1.8-7.4 x 10^5 eggs/m2 [F, ABS; winter_stock.md row 16]. Sichlau et al. 2011 [S]: Acartia, C. hamatus and Temora eggs in sediment, viable for decades.
  - No source for resting-egg banks at 100-300 m in the Skagerrak was found.

## E8. DVM of the resident community (settled in zooplankton_refuge.md)

- Shelf-sea summer copepods mostly do not migrate. Daro 1988: "In most areas, copepods did not vertically migrate". Irigoien et al. 2004 (93-130 m): "more than 60% of the plankton did not perform significant DVM". The significant migrants are Calanus, Pseudocalanus and chaetognaths [F].
- Refuge is a daytime depth/light refuge. Field daytime hiding share at floor 0.2 implied pi = 0.13-0.44 (zooplankton_refuge.md Q4) [D there].
- Food-limited copepods lose diel rhythms below ~200 mg C/m3 (Daro 1988) [F].
- Krill (Meganyctiphanes norvegica) migrates daily and is not dormant. Day-caught krill had less stomach pigment than night-caught (Plymouth PhD thesis 2004, doi:10.24382/4423 [F, ABS]); M. norvegica did not migrate through a 50-60 m thermocline in a fjord while Thysanoessa raschii did (Bergström & Strömberg 1997, J Plankton Res 19:255-261, doi:10.1093/plankt/19.2.255 [F, ABS]). Euphausiids are ~0 in the A17 0-25 m WP2 hauls (E1), which sample them poorly.

## E9. Model facts checked in code

- `lib/environments/ecosystem_env/movement.py` `apply_energy_costs`:
  - cost = biomass x action share x resting_metabolism x Q10 multiplier x action multiplier x impact factor.
  - The multipliers are rest = `resting_cost` (default 1.0, `state.py` line 221), eat = `feeding_cost` 4.0, move = `movement_cost` 1.2.
  - One temperature, the zooplankton layer `mean_0_50m`, applies to every action.
- `lib/environments/ecosystem_env/predation.py` ~line 220: visible = 1 - rest x (1 - floor).
  - The floor is per pair where set: planktivores -> zooplankton 0.5, mackerel -> zooplankton 0.5. Whiting and the water-column predators use the group's own 0.2.
- `lib/environments/ecosystem_env/natural_mortality.py` `build`:
  - V uses `env.dm_visibility_floor` (the group's own 0.2), not the 0.5 of the main predator pairs.
  - v_ref and d_ref are evaluated at hide_reference, so the annual-mean M1 equals natural_mortality only if the population hides hide_reference on average.
- `lib/environments/ecosystem_env/population_change.py` lines 314-327: below the maintenance level u = 0.5, biomass changes at starve_rate x (s - u) per tick. At an empty reserve this is -0.10/tick = -0.34/d [D].
- `fg_library.yaml` zooplankton values:
  - Energy: resting_metabolism 22.5, max_energy_reserve 675, maintenance_level 0.5, energy_content 4500, starve_rate 0.2.
  - Mortality and visibility: natural_mortality 0.015812, visibility_floor 0.2.
  - Temperature: metabolism_q10 2.0 with t_ref annual_mean.
