# Net displacement of harbour seals (grey seal as marked fallback) at 1, 6 and 24 h

Scope: straight-line (net) displacement of harbour seals (Phoca vitulina) at sea per 1 h, 6 h and 24 h, by behavioural state (transit, area-restricted search (ARS)/foraging, resting), plus transit speeds, path straightness and trip structure. The aim is to choose the transit jump k (cells of 1 km per step) and the local movement scale for the seal FG in the 60 x 60 km Skagerrak grid (57.8-58.3 N, 10.1-11.1 E), 6 h step (configurable 1-6 h). Today the cap is 1 cell per 6 h = 0.046 m/s.

Labels: values marked "(Own calc)" are my conversions or simulations. Everything else is quoted from the source. Regions are given for each source. GS = grey seal (fallback). No at-sea movement statistics exist for Skagerrak/Kattegat harbour seals (see Gaps). The closest region with state-resolved data is the North Sea (UK east coast, Dutch Delta, Wadden Sea).

## Q1. Step lengths / net displacement per 1 h, 6 h and 24 h, by state; transit vs foraging speeds

### Takeaway
North Sea harbour seals in the transit state cover a mean of 6.4 km net per 2 h (about 0.9 m/s, 3.2 km/h). In broad ARS they cover 2.7 km per 2 h (0.38 m/s), and in focussed ARS only about 0.07 km per 2 h. Simulating a seal that stays in transit gives about 17 km net per 6 h (median; 5-95 % 8-25 km). Real transit bouts last only about 7 h on average, though. The time-averaged net displacement over 24 h is therefore much smaller: median about 15 km, 95th percentile about 43 km. Grey seals move about 15-20 % further in every state.

### Cited Findings
- **Carter et al. 2025/2026 (Landscape Ecology 41(2), doi 10.1007/s10980-025-02281-z).** North Sea-wide GPS phone tags: 325 harbour and 116 grey seals; regions Orkney, Moray Firth, East UK, Southeast UK, Dutch Delta, Wadden Sea. 10,694 harbour-seal trips in the analysis. Methods:
  - "Cleaned location data were regularised to a constant 2 h time step using linear interpolation."
  - The HMM had states "focussed ARS", "broad ARS", "transit", "non-diving" and "unknown".
  - Step length is "the Euclidean distance travelled in a 2 h interval", Gamma-distributed; turn angles are wrapped Cauchy.
  - — [Carter et al., PMC12811352](https://pmc.ncbi.nlm.nih.gov/articles/PMC12811352/)
- Carter et al. state-dependent step parameters, grey vs harbour seal: "focussed ARS: 0.09 (0.09 SD) vs 0.07 (0.06 SD); broad ARS: 3.38 (1.76 SD) vs 2.70 (1.50 SD); transit: 7.63 (2.18 SD) vs 6.41 (2.19 SD)". The text does not print the unit. Step length is defined as the distance per 2 h interval, so km per 2 h is implied; see Inferences. — [Carter et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12811352/)
- Carter et al. turn-angle concentration (directional persistence), grey vs harbour: "focussed ARS: 0.07 vs < 0.01; broad ARS: 0.53 vs 0.46; transit: 0.81 vs 0.76". — [Carter et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12811352/)
- Carter et al. state switching per 2 h step:
  - "the probability of remaining in each state was high for both species (grey seals: >= 0.82, harbour seals: >= 0.74)".
  - Transit to broad ARS vs focussed ARS: "harbour seals: 0.25 vs 0.03" (grey 0.14 vs 0.004).
  - Into transit from broad vs focussed ARS: "harbour seals: 0.1 vs 0.02".
  - Focussed to broad ARS vs broad to focussed: harbour 0.21 vs 0.13.
  - — [Carter et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12811352/)
- **Bjorge et al. 1995 (ICES CM 1995/N:1; Froan, Central Norway; VHF plus depth-velocity telemetry).**
  - "Swimming speed of about 1.1-1.6 m/sec ... were typically recorded for transit and foraging dives."
  - "At the sea floor, the seals continued swimming, presumably searching for food, at their usual swim speed."
  - This is speed through the water measured by a velocity sensor, not net displacement.
  - — [Bjorge 1995, ICES CM](https://www.vliz.be/imisdocs/publications/ocrd/275455.pdf)
- **Vance et al. 2021 (German Wadden Sea, DTAG-4 with GPS and dead-reckoning; 8 seals, 12 offshore trips).** For dead-reckoning they assumed "1.4 m/s is the typical forward speed for harbour seals". — [Vance et al. 2021, Sci Rep](https://pmc.ncbi.nlm.nih.gov/articles/PMC7985366/)
- **Russell 2016, SMRU report to BEIS (southern North Sea).** GPS phone tags on 24 harbour seals from The Wash and 18 grey seals.
  - States were assigned on 2 h intervals using "speed (distance/time) over the two-hour interval and bearings". "Relatively slow, tortuous movements" were labelled foraging and "faster more directed movements" travelling.
  - The report gives no numeric speed per state.
  - Harbour-seal activity budget in geo-space: resting 0.30 (95 % CI 0.21-0.36), foraging 0.46 (0.24-0.63), travelling 0.25 (0.06-0.48). Grey seal: 0.28 / 0.42 / 0.26.
  - — [Russell 2016, SMRU/BEIS OESEA-15-66](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/586448/SMRU_2016_Activity_budgets-analysis_of_seal_behaviour_at_sea.pdf)
- Russell 2016 on the 6 h resolution: "On a six-hour resolution, there were issues in estimating separate foraging and travelling states for some tagged harbour seals". "In most regions of the UK, harbour seals forage close to the coast and thus are unlikely to engage in travelling periods that are long enough to result in an interval being assigned to travelling". At 2 h, foraging and travelling could be separated for about 90 % of individuals, against about 80 % at 6 h. — [Russell 2016](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/586448/SMRU_2016_Activity_budgets-analysis_of_seal_behaviour_at_sea.pdf)
- Upper speed bounds used in filtering:
  - "Seals can swim at speeds in excess of 2 ms-1 for limited periods". Argos data were "speed-filtered ... at 2ms-1" (UK, both species). — [Russell & McConnell 2014, SMRU report to DECC](https://tethys.pnnl.gov/sites/default/files/publications/Russell_and_McConnell_2014.pdf)
  - Sharples et al. used a 5.5 km/h (1.5 m/s) threshold speed for UK harbour-seal Argos data. — [Sharples et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3357409/)
  - Tougaard et al. used a 5 km/h cut-off for Danish Wadden Sea harbour seals. — [Tougaard et al. 2006, Horns Reef report](https://cpdp.debatpublic.fr/cpdp-eolien-en-mer/DOCS/DANEMARK/HARBOUR_SEALS_REPORT.PDF)
- GS, McConnell et al. 1999 (North Sea, Farne Islands and Abertay; 14 grey seals, Argos SRDL). As reported in a search-result summary of the abstract, not read in full: "During travel the seals moved at speeds of between 75 and 100 km day-1 (0.87 and 1.16 m s-1)". Return trips averaged 2.33 days and reached a mean 39.8 km from the haul-out. 88 % of trips returned to the departure haul-out. — [McConnell et al. 1999, J Appl Ecol 36:573-590 (record surfaced via search)](https://marine.gov.scot/sma/node/17251)

### Inferences
- (Own calc) Unit check on the Carter parameters. If they are km per 2 h:
  - Harbour transit is 6.41/7.2 = 0.89 m/s (3.2 km/h); grey transit is 7.63/7.2 = 1.06 m/s (3.8 km/h).
  - The grey value matches McConnell's independent 0.87-1.16 m/s grey-seal travel speed almost exactly.
  - Both values are below the 1.1-1.6 m/s swim speed (Bjorge 1995; Davis 1985 1.4 m/s), as they must be once diving and path wiggle are counted.
  - So km per 2 h is almost certainly the unit. The "SD" is read as the standard deviation of the fitted Gamma step distribution, following momentuHMM's mean/sd parameterisation. It is not a between-seal SD.
- (Own calc) Gamma percentiles of a single 2 h step, harbour seal:

  | State | p5 | p25 | median | p75 | p95 | mean (2 h) |
  |---|---|---|---|---|---|---|
  | Transit (shape 8.6) | 3.3 | 4.8 | 6.2 | 7.7 | 10.4 km | 6.41 km |
  | Broad ARS (shape 3.2) | 0.8 | 1.6 | 2.4 | 3.5 | 5.5 km | 2.70 km |
  | Focussed ARS | | | 0.05 | | 0.19 km | 0.07 km |

  - Focussed ARS is effectively stationary at a 1 km resolution.
  - Grey seal, transit: median 7.4, p95 11.5 km; broad ARS: median 3.1, p95 6.7 km.
- (Own calc) Net displacement over 6 h and 24 h. Method: Monte Carlo correlated random walk with the Carter Gamma steps and wrapped-Cauchy turns, 200,000 tracks, staying in one state the whole time. "Straightness" = net / path length.

  | Harbour seal | 6 h median (p5-p95), mean | 6 h straightness | 24 h median (p5-p95), mean | 24 h straightness |
  |---|---|---|---|---|
  | Transit | 17.2 (7.9-24.5), 16.9 km | 0.88 | 49.5 (14.9-75.4), 47.8 km | 0.62 |
  | Broad ARS | 5.9 (1.9-11.0), 6.1 km | 0.76 | 13.3 (3.7-26.1) km | 0.43 |
  | Focussed ARS | 0.12 (p95 0.30) km | | 0.26 (p95 0.57) km | |

  | Grey seal | 6 h median (p5-p95) | 24 h median (p5-p95) |
  |---|---|---|
  | Transit | 21.1 (10.7-28.3) km | 65.6 (20.9-91.7) km |
  | Broad ARS | 7.8 (2.7-13.8) km | 18.1 (5.0-34.2) km |

  - The 24 h pure-transit case is not realistic, because transit bouts are short (next bullet).
- (Own calc) State persistence. Harbour seals leave transit with probability at least 0.25 + 0.03 = 0.28 per 2 h step, so they stay with probability at most 0.72. That gives an expected transit bout of 1/(1-0.72), about 3.6 steps, about 7 h. The paper also says "remaining in each state ... >= 0.74", which is slightly inconsistent with 0.72; the figure may round, or the matrix may be conditioned differently.
  - So a typical transit leg lasts about one 6 h model step. Over 24 h a seal mixes states.
- (Own calc) Mixed-state simulation for harbour seals. Method: the three diving states only, with transitions transit {0.72, 0.25, 0.03}, broad ARS {0.10, 0.77, 0.13} and focussed ARS {0.02, 0.21, 0.77}, renormalised; resting omitted. The stationary shares are transit 0.20, broad ARS 0.50 and focussed ARS 0.31.
  - Starting from the stationary mix: net 6 h has median 5.3 km, mean 6.6, p90 15.3, p95 18.4 km. Net 24 h has median 14.5 km, mean 17.6, p90 35.9, p95 43.4 km.
  - Starting in transit (the k-jump case): net 6 h has median 14.0 km, p95 23.1 km. Net 24 h has median 22.5 km, p95 53 km.
  - The roughly 30 % resting share (non-diving, about zero active displacement) would lower the 24 h figures further, by up to about 30 %.
- (Own calc) 1 h scale. The HMM has no sub-2 h resolution. Transit straightness at 2 h is about 0.9, so linear scaling is reasonable.
  - Transit: about 3.2 km per h (harbour), 3.8 km per h (grey).
  - Broad ARS: between linear and square-root scaling, about 1.3-1.9 km per h.
  - Focussed ARS: under 0.05 km per h.
- (Own calc) Ratio of transit net speed to swim speed: 0.89 / 1.4, about 0.64. Transit "speed over ground" is about two thirds of swim speed because transit is done diving (V-shaped transit dives, Bjorge 1995) and the path is not perfectly straight.

### Gaps
- Carter et al. do not print the unit of the step parameter, and its figures were not readable. The km per 2 h reading rests on the definition of step length plus the cross-checks above.
- No published HMM step-length distribution at a 6 h resolution was found. Russell et al. 2015 (Oikos) used 6 h states but, from the reports read, gives no step parameters.
- The non-diving (resting) state in Carter et al. and Russell 2016 is set a priori from dive data. No step distribution is given for it, so drift while resting at sea is unquantified.
- McConnell 1999 travel speeds come from a search-result summary of the abstract. I could not open the full text.

## Q2. Foraging trip structure: distance to patches, trip duration, time to reach a patch

### Takeaway
Most harbour-seal trips are short and local: half reach no more than about 25 km, and in Skagerrak-Kattegat haul-outs are typically within 25 km and at-sea use mostly within about 40 km. Offshore North Sea trips are a separate mode, reaching 50-100+ km over several days to 1-2 weeks. Reaching a patch 20 km away takes about 6 h of transit; the Wadden Sea outbound legs took about 24 h.

### Cited Findings
- **Sharples et al. 2012 (UK, 118 seals, Argos/GPS, October-May; trip distance = "the distance covered on a foraging trip").**
  - Mean trip distances were 11-21 km for Shetland, Orkney and the Thames, 86 km (SD 111) for The Wash, and the Moray Firth was highest.
  - "The average duration of trips ranged from 4.5 days in the Moray Firth to less than 1 day in the Thames."
  - "The relationship between trip distance and duration was approximately linear."
  - — [Sharples et al. 2012, PLoS ONE](https://pmc.ncbi.nlm.nih.gov/articles/PMC3357409/)
- **Vance et al. 2021 (German Wadden Sea).**
  - Mean complete offshore trip lasted 7 days (SD 2.3).
  - "The predominantly straight-line movements away from/to shore tended to last about 24 h for all animals".
  - Seals foraged "almost continuously throughout the trip including on the outbound and return journeys".
  - Resting time at sea rose by 36.9 (SE 6.6) min per day over the trip.
  - — [Vance et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7985366/)
- **Tougaard et al. 2006 (Danish Wadden Sea / Horns Reef, satellite tags).**
  - Three tracked seals "spent 1-2 weeks at sea, 50-100 km from the haulout site, returned for up to one week of haulout and then took off on a new foraging trip".
  - The area used reached about 100 km offshore. In 2002, three yearlings "moved 200-300 km offshore on several foraging trips".
  - One adult male made 11 trips in 94 days, with foraging concentrated in a small area about 10 km north of Vovov.
  - — [Tougaard et al. 2006](https://cpdp.debatpublic.fr/cpdp-eolien-en-mer/DOCS/DANEMARK/HARBOUR_SEALS_REPORT.PDF)
- **Bjorge et al. 1995 (Central Norway, rocky archipelago, the habitat closest to the Skagerrak skerries).**
  - "Harbour seal foraging trips of up to 20 km were recorded". Foraging habitats were "few hundred meters" to 20 km from haul-outs.
  - One male "travelled regularly about 20 km from his haul out site ... His foraging trips were approximately 8 to 22 hours long".
  - Another foraged 3-6 km from the haul-out on trips of 3 to 8 hours. A third foraged "within distances of 5 km".
  - — [Bjorge 1995](https://www.vliz.be/imisdocs/publications/ocrd/275455.pdf)
- **Skagerrak-Kattegat**, from the sibling notes in `Parametervarden for salar i Skagerrak`:
  - Riaz et al. 2026 (Norwegian Skagerrak, 26 GPS-phone-tagged seals): haul-outs "typically occurred within 25 km" of the tagging site (mean 16.2 km, range 0-260 km); mean haul-out duration 5.8 h; at-sea locations not analysed. — [Riaz et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12796850/)
  - Andersen et al. 2013 (Anholt, Kattegat, 8 seals): "almost all at-sea locations within 40 km of the haul-out". — [openpolar record](https://openpolar.no/Record/crwiley:10.1002%2Faqc.2393)
  - Dietz et al. 2013 (Kattegat, 27 seals): maximum 249 km from the tagging site. — [Dietz et al. 2013](https://portal.findresearcher.sdu.dk/da/publications/movements-and-site-fidelity-of-harbour-seals-phoca-vitulina-in-ka/)
  - Cunningham et al. 2009 (W Scotland): 50 % of trips within 25 km of a haul-out. — [Cunningham et al. 2009](https://openpolar.no/Record/crwiley:10.1002%2Faqc.983)
- Carter et al. excluded trips shorter than 8 h, because near-haul-out movement can reflect "waiting for tidal haulouts to become available" (3.9 % of grey and 6.1 % of harbour-seal intervals). — [Carter et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12811352/)

### Inferences
- (Own calc) Time to reach a patch at the transit rate of 3.2 km/h net:
  - 10 km takes about 3 h; 20 km about 6 h; 50 km about 16 h.
  - This matches the roughly 24 h Wadden Sea outbound legs if those offshore sites lay about 50-80 km out. Vance does not give the distance, so this is a consistency check only.
- In this grid the haul-outs sit just outside the east edge (sibling notes). The grid's eastern 10-25 km covers the normal trip range. A seal crossing the whole 60 km grid needs about 19 h of pure transit, so cross-grid excursions are rare and multi-day.

### Gaps
- No trip-duration statistics exist for Swedish Bohuslan or Norwegian Skagerrak harbour seals; Riaz 2026 reports haul-outs, not at-sea trips.
- Sharples' "trip distance" may be path length or maximum range; the text does not say clearly. Its 86-100 km means therefore cannot be converted directly to a net displacement rate.

## Q3. HMM / state-space step-length distributions per state

### Takeaway
The single directly usable source is Carter et al. (2025/2026). It reports Gamma step distributions per 2 h for focussed ARS, broad ARS and transit, for 325 harbour and 116 grey seals in the North Sea (tabulated under Q1). The SMRU state-space work (Russell et al. 2015; Russell 2016) used 2 h and 6 h states but publishes activity budgets, not step parameters. The 6 h resolution was found too coarse for harbour seals.

### Cited Findings
- Carter et al.: momentuHMM multistate HMM with a 2 h step.
  - Gamma step-length mean/SD per state: harbour 0.07/0.06 (focussed ARS), 2.70/1.50 (broad ARS), 6.41/2.19 (transit); grey 0.09/0.09, 3.38/1.76, 7.63/2.18.
  - The two-ARS-state model was strongly preferred over one ARS state. With a single ARS state there was "strong overlap in state-dependent distributions for step lengths between the ARS and transit states ..., particularly for harbour seals".
  - The two-scale model gives harbour seals on average 16.3 % more ARS intervals (IQR 11.2-22.7 %, maximum 46.4 %).
  - — [Carter et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12811352/)
- Carter et al. regional ARS probabilities:
  - For harbour seals, broad ARS was the dominant foraging state (probability 0.53-0.64), except in the Moray Firth and East UK. There, focussed ARS was higher (0.56-0.57; 0.18-0.27 elsewhere).
  - The rest of the probability space is transit.
  - — [Carter et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12811352/)
- Russell 2016 (The Wash, 2 h, Bayesian SSM). Two diving states were separated by speed and bearing. Resting was set when 44.4 % or less of an interval was spent diving: seals can be under water at most 88.8 % of a 2 h period. — [Russell 2016](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/586448/SMRU_2016_Activity_budgets-analysis_of_seal_behaviour_at_sea.pdf)
- Russell 2016: in tidal areas geo-space speeds can misclassify behaviour. Harbour-seal foraging share of diving intervals was 0.64 in geo-space vs 0.80 in hydro-space (tidal currents removed). Yet "99% of intervals which were defined as foraging in geo-space were also defined as foraging in hydro-space". — [Russell 2016](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/586448/SMRU_2016_Activity_budgets-analysis_of_seal_behaviour_at_sea.pdf)
- GS, a southern North Sea GPS study (3 grey seals, 1 h regularised steps, 2-state HMM, foraging vs travelling) gives no step parameters in the text. Foraging was at 50-70 km from the coast or about 200 km from the haul-out, and also within 20 km for two seals. — [Frontiers in Marine Science 2022, 10.3389/fmars.2022.818635](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.818635/full)

### Inferences
- The southern North Sea is strongly tidal, and some Carter transit steps are tide-assisted. The Skagerrak study area is micro-tidal (this is an inference; I found no current data for the box). Ground speeds there are therefore probably closer to the hydro-space speeds, which means slightly slower transit and more ARS.
- Russell's finding that harbour seals rarely fill a 6 h interval with travel supports the transit-jump design. In a 6 h step, the "travelling" outcome should be rare, and k should reflect a partly travelling step, not 6 h of pure transit.

### Gaps
- No HMM step parameters for Skagerrak, Kattegat or Wadden Sea seals at 6 h were found.
- No sex-, age- or season-specific step parameters were found. Sharples and Dietz show season matters (longer trips in winter, February-March).

## Q4. Tortuosity / straightness of transit vs foraging

### Takeaway
Transit is strongly directed: harbour seal wrapped-Cauchy concentration 0.76 per 2 h, about 0.9 straightness over 6 h. Broad ARS is moderately tortuous (0.46; 0.76 over 6 h, 0.43 over 24 h). Focussed ARS is undirected (concentration < 0.01) and stationary at the km scale. At fine scale (tens of metres), even foraging seals move with a straightness of about 0.65 between prey captures. Tortuosity is only a weak indicator of foraging, because seals also feed during transit.

### Cited Findings
- Carter et al. turn concentration (harbour / grey): focussed ARS < 0.01 / 0.07; broad ARS 0.46 / 0.53; transit 0.76 / 0.81. — [Carter et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12811352/)
- Vance et al. (Wadden Sea, dead-reckoned tracks):
  - "The median distance made good between PCAs (i.e., the straight-line distance between PCA locations) across all animals was 44.2 m. The median distance travelled between prey encounters ... was 67.9 m". PCA = presumed prey capture attempt.
  - Tortuosity vs PCA rate offshore: "significant but weak ... (30 min interval: p = 0.0001, marginal R2 = 0.04 ...) suggesting that tortuosity is a weak predictor of foraging behaviour".
  - Foraging rates during the 24 h travel phases "were within the range of median foraging rates during days spent offshore".
  - — [Vance et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7985366/)
- Bjorge et al. 1995 described transit as "directed travel" with V-shaped dives. In foraging (U-shaped) dives, "the swimming direction frequently changed during and between dives". — [Bjorge 1995](https://www.vliz.be/imisdocs/publications/ocrd/275455.pdf)

### Inferences
- (Own calc) Straightness index (net / path length) from the simulation in Q1, harbour seal:

  | State | 6 h | 24 h |
  |---|---|---|
  | Transit | 0.88 | 0.62 |
  | Broad ARS | 0.76 | 0.43 |
  | Focussed ARS | 0.64 | 0.33 |

  - Grey seal transit: 0.90 (6 h), 0.68 (24 h).
  - Focussed ARS values are on a sub-km path and do not matter at the 1 km grid scale.
- (Own calc) Vance's fine-scale straightness: 44.2 / 67.9, about 0.65.

### Gaps
- No published straightness index computed at 6 h or 24 h for harbour seals was found. The values above are model-derived.

## Q5. Synthesis: transit reach k and local scale for the model

### Takeaway
At a 6 h step, set the harbour-seal transit jump to k of about 15 km (range 12-20). The local scale should allow several km per 6 h, not 1 cell, because broad-ARS foraging alone covers about 6 km net per 6 h. The time-averaged 6 h displacement is a median of about 5 km. At a 1 h step, k is about 3 km and the local step is 1 cell.

### Cited Findings
- Inputs for the synthesis:
  - Transit ground speed is about 0.9 m/s for harbour seals and about 1.06 m/s for grey seals ([Carter et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12811352/)).
  - Swim speed through water is 1.1-1.6 m/s ([Bjorge 1995](https://www.vliz.be/imisdocs/publications/ocrd/275455.pdf)) or 1.4 m/s ([Vance et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7985366/)).
  - Activity budget in the southern North Sea: about 30 % resting, 46 % foraging, 25 % travelling ([Russell 2016](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/586448/SMRU_2016_Activity_budgets-analysis_of_seal_behaviour_at_sea.pdf)).
  - Typical trip range in Skagerrak-Kattegat is within 25-40 km ([Riaz et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12796850/); [Andersen et al. 2013](https://openpolar.no/Record/crwiley:10.1002%2Faqc.2393)).

### Inferences
All of the following are (Own calc), derived from the Q1 simulations.

- **Transit jump k, 6 h step:**
  - Harbour seal, pure 6 h transit: median 17 km, mean 17 km, p95 25 km.
  - A 6 h step that begins in transit (with realistic switching): median 14 km.
  - Recommended k = 15 cells (range 12-18). For grey seals: k = 20 (range 16-24).
  - For comparison, swim speed x 6 h would be 1.4 m/s x 21,600 s = 30 km. That is an upper bound, not a net displacement, because it ignores diving and wiggle.
- **Local movement scale, 6 h step:**
  - Broad ARS (the most common foraging state) gives a median net of 5.9 km and p95 11 km per 6 h.
  - Focussed ARS and resting give about 0.1 km, i.e. stay in the cell.
  - So the realistic local spread per 6 h is 0-6 cells. A 1-cell local step at 6 h under-represents foraging movement by about a factor of 5-6. Either allow a local step of a few cells, or accept that broad-ARS movement is lumped into occasional transit jumps.
- **Time-averaged net displacement over all states, harbour seal, resting excluded:**
  - 6 h: median about 5 km, mean 6.6, p90 15, p95 18 km.
  - 24 h: median about 15 km, mean 18, p90 36, p95 43 km.
  - Including about 30 % resting at sea or hauled out would lower these by up to about 30 %.
  - Today's cap of 1 km per 6 h (4 km per day) is about 4-5 times below the median and about 10 times below the p95.
- **Scaling to a 1 h step:**
  - Transit k = 3 cells (range 2.5-4; 0.9 m/s = 3.2 km/h), since transit is nearly straight below 2 h.
  - Local step = 1 cell: broad ARS covers about 1.3-1.9 km/h and focussed ARS under 0.05 km/h.
  - Linear scaling is valid for transit up to about 6 h (straightness 0.88). It is not valid for 24 h (0.62) or for ARS. For ARS, net displacement grows roughly with sqrt(t) beyond a few hours: 2.7 km per 2 h becomes 5.9 km per 6 h and 13 km per 24 h.
- **How often to transit:**
  - The expected harbour-seal transit bout is about 7 h, and transit is about 20-25 % of diving time.
  - So about a quarter of seal biomass "should" take a transit jump in a given 6 h step at a 2 h resolution. At 6 h resolution the share is lower, because transit bouts are diluted within the step.
- **Caveats:**
  - Region and season: North Sea values, possibly inflated by tidal assistance. Skagerrak seals use a rocky skerry coast more like Bjorge's Norwegian sites, with shorter trips.
  - The 60 km grid wall means k = 15 crosses the grid in 4 jumps.

### Gaps
- No Skagerrak-specific at-sea movement data exist to calibrate k regionally.
- Seasonal variation in k (winter versus breeding and moult) is not resolved by the state parameters.
- I found no source for along-shore current drift in the box (Baltic outflow / Jutland current), which could add to geo-space displacement while resting.
