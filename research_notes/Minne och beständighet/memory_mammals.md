# Memory side of the neighbour weight: harbour porpoise and harbour seal (2026-11-14)

Research subagent. Nothing in code or library changed.

Scope: the MEMORY half of the "memory x persistence" reading of the neighbour weight,
w = sum over lags j of P(last visit j ticks ago) x rho(j), truncated at the memory span, for the two
mammal observers. The persistence side rho (how long the target's distribution in a cell holds) is
not reviewed here, except where a source mixes the two (section 1.3, flagged).

Grades: [F] measured (field or lab; [F/A] read as abstract or snippet only), [S] secondary (review,
model documentation, or earlier project note), [D] derived here from measured or documented inputs,
[J] judgement. Read first and not redone: `Observabilitetsparen/mammals.md`, `design.md` 1,
`Nettoförflyttning per funktionsgrupp/porpoises.md` and `seals.md` (o-umlaut).

Papers (metadata and PDFs in `papers/`): Iorio-Merlo et al. 2022 (PDF read), Chudzinska et al. 2021
AgentSeal (PDF read, plus its TRACE and NetLogo code from GitHub), Johnston et al. 2005 (PDF read),
Sharples et al. 2012 (PDF read), Sveegaard et al. 2011 (PDF read), Stalder et al. 2020 (PDF read),
Sørensen et al. 2018 (PDF read). DEPONS: TRACE of Nabe-Nielsen et al. 2018 (JASSS 23(2)7 S6, read) and
the Java source at commits a4c81e9 (master, Nov 2025) and 941a5b2 (v2.0), read. Lidgard et al. 2020
read as Europe PMC full text. Nabe-Nielsen et al. 2013 Oikos not reached (gap).

**Headline: no study measures a memory span for either species.** Every number in the final table is
either a model parameter calibrated by pattern-oriented modelling (DEPONS, AgentSeal), a revisit or
consistency pattern that memory is inferred from (Iorio-Merlo 2022, Johnston 2005, Lidgard 2020),
or an unpublished remark (the porpoise "return after about two days"). The memory spans are
therefore [S]/[J]: order-of-magnitude bounds, not measurements.

## 1. Evidence of spatial memory in foraging

### 1.1 Porpoise

- **DEPONS fine-scale memory: the mechanism is the caller's formula** [S, code read].
  `OriginalRefMemTurn.refMemTurn` (DEPONS master a4c81e9) keeps the porpoise's last positions
  (`posList`, a circular buffer) and the food level found at each (`storedUtilList`). The buffer
  holds `SimulationConstants.MEMORY_MAX = 120` half-hour steps; the code comment reads "Maximum number
  of half-hour steps the amount of food can be remembered (120 steps is 2.5 days)". The attraction
  vector is sum over ii = 1..119 of `storedUtil[ii] x RefMem.getRefMemStrength(ii) / distance x unit
  vector`, with `refMemStrength` a fixed list indexed by LAG (`ii` = steps ago), not by place.
  So DEPONS memory is lag-indexed, weighted by a fixed kernel, and hard-truncated at 120 steps =
  60 h = 10 ticks of 6 h. That is the form `w = sum_j P(last visit j ago) x kernel(j)` with M = 10
  ticks. In DEPONS the kernel multiplies remembered FOOD, and replenishment is modelled separately
  (rU, below), so the kernel is memory only, not persistence.
- **The kernel shape depends on the version** [S code; D conversions]. `calcArray` builds it as
  m(0) = 0.999 and m(i) = m(i-1) - r m(i-1) (1 - m(i-1)), a logistic decay that stays near 1 and
  then drops. `PorpoiseSimBuilder` (line 145) calls `RefMem.initMemLists(getRS(), getRR())` at
  model start, so the parameter values replace the hard-coded arrays:

  | version | rR (reference) | strength at 24 / 48 / 60 h | effective span (sum of strengths) | rS (satiation) | effective span |
  |---|---|---|---|---|---|
  | TRACE 2018, model v1.1 (Table 2; also the hard-coded array) | 0.10 | 0.91 / 0.077 / 0.007 | 36 h = 6.0 ticks (half-strength at 36 h, < 0.1 at 47 h) | 0.20 | 19 h = 3.1 ticks (half-strength 19 h) |
  | code v2.0 (941a5b2, 2019) | 0.04 | 0.99 / 0.96 / 0.90, then cut | 59 h = 9.8 ticks (the 120-step cap) | 0.04 | same |
  | code master (a4c81e9, Nov 2025) | 0.03 | 1.00 / 0.98 / 0.97, then cut | 59 h = 9.9 ticks (the cap) | 0.03 | same |

  In the code defaults memory is effectively a flat, hard-truncated 2.5-day window. **The 2.5 days
  is a software constant, not a measurement.** rS and rR were "calibrated using pattern-oriented
  modeling ... to ensure that animals developed movement tracks that closely resembled those observed
  for satellite-tracked animals in the inner Danish waters, i.e. with the same home range sizes, mean
  residence times and displacement distances (Nabe-Nielsen et al. 2013b)" (TRACE 2018, section 3.2)
  [S]. The TRACE adds that "these movements are unlikely to be representative for animals in all
  parts of the North Sea" [S]. The pattern values themselves (residence times etc.) are in
  Nabe-Nielsen et al. 2013 Oikos 122:1307, doi 10.1111/j.1600-0706.2013.00069.x, and its appendix,
  neither reached [gap]. Why the later versions use 0.03-0.04 is not documented in what I read [gap].
- **Satiation memory** (rS) sets E = sum of workMemStrength(i) x food found i steps ago (Porpoise.java,
  `veTotal`), the food the animal "should expect to find if taking an undirected CRW step" (TRACE).
  When E is low, the memory vector dominates and the animal heads back to remembered food [S]. It is
  the giving-up side (section 5).
- **Large-scale persistent memory (PSM)** [S]: for dispersal, DEPONS keeps the energy intake rate of
  every 2 x 2 km cell group visited "since it was born". The rationale: "satellite tagged animals tend
  to return to the same general part of the landscape after having been elsewhere for several weeks or
  months" (TRACE 2018, 2.7.2). This is landscape memory at weeks to lifetime scale; it steers
  dispersal targets at a preferred distance of 210-350 km, not 1-km neighbours.
- **"Return to the same area after approximately two days"** [S, unpublished]: TRACE 2018 section 3.1:
  "Euse was calibrated to ensure that the population reached a dynamic equilibrium size, assuming
  that food recovered after approximately 2 days. This is based on the observation that satellite
  tagged porpoises in the inner Danish waters often return to the same area after approximately two
  days (J Nabe-Nielsen, unpubl. data) ... we take such repeated returns to the same area as an
  indication that food has recovered." **Flag: this is the only Danish-waters revisit number, it is
  unpublished, and DEPONS uses it on the PERSISTENCE side** (food recovery, together with rU = 0.1 per
  day logistic regrowth), not as a memory span. If the parent also takes ~2 d as the memory span, the
  same datum enters both factors of memory x persistence. Read literally it says: an area is revisited
  after ~2 d and is then worth feeding in again. That bounds memory from below (the animal finds the
  place again) and says the resource renews, not that its distribution persists [J]. "Area" is at
  Argos resolution (positions every 1-3 days in the Skagen data), so km-scale, not 1 km [J].
- **Field site fidelity at 5-10 km, days to weeks** (Bay of Fundy) [F]: Johnston, Westgate & Read 2005
  (MEPS 295:279, doi 10.3354/meps295279; 6 Argos porpoises, PDF read): monthly ranges 7738-11289 km2
  but "concentrated their movements in small focal regions (August mean = 294 km2, September mean =
  252 km2)". PTT7235 spent 45 d in the Bay (2 Aug-20 Sep): "a minimum of 1 position within 5 km of
  the center of the Long Eddy on 32 (71%) of these days and at least 1 position within 10 km of Long
  Eddy on 42 (93%) of these days". Both tracked animals "made occasional forays away from their core
  areas, followed by movements back into the core region". The authors cite Read & Westgate 1997:
  individuals "inhabit relatively restricted areas for days to weeks ... and then make rapid
  movements over periods of hours to days across larger scales" [S].
- **North Sea and Skagerrak, coarse scale** [F]: Stalder et al. 2020 (MEPS 648:207, doi
  10.3354/meps13412; 57 Argos animals, 24 h SSM): 80.9 % of daily positions AR; "Animals remained in
  areas where their movements were classified as AR for extended periods of time (Fig. S1)". No bout
  length is tabulated [gap]. The authors list "memory of previously visited foraging patches" as an
  unmodelled source of individual variation [S]. Sveegaard et al. 2011 (MMS, doi
  10.1111/j.1748-7692.2010.00379.x, PDF read) maps high-density areas from 64 tracks, the tip of
  Jylland and three smaller areas along the Norwegian Trench for the Skagerrak group, at kernel scales
  of tens of km. Shared use by many animals, not individual memory at 1 km.

### 1.2 Harbour seal

- **Iorio-Merlo et al. 2022** (Proc R Soc B 289:20212261, doi 10.1098/rspb.2021.2261; Moray Firth;
  PDF read) [F]. Two months of GPS (29 seals in the memory model; the Discussion says 31). Memory
  proxy: per seal, the proportion of dive batches classified as ARS in each 1 km x 1 km cell during
  the PREVIOUS MONTH, used as a covariate on the HMM transition to ARS. No decay was modelled. Results:
  - "Individuals were highly consistent in the areas they visited in April and May" (50 % UDs,
    Bhattacharyya affinity well above a null of randomly paired seals; values in their table S3, not
    read).
  - Adding the memory map to the HMM: AIC 330,017 -> 329,759 (5-dive time step; 29 seals). In the
    5-seal accelerometer subset, memory + prey encounters had AIC 53,657; memory alone 53,780; prey
    alone 53,909. Both raise P(start ARS); no interaction.
  - "no correlation (Kendall tau = 0.14) between the prey encounters detected and the memory maps".
  - Trips 38.65 h mean (SD 34.79), longest 6.36 d; 10 (SD 5.6) trips per month in April-May. ARS steps
    165-207 m per 5-dive batch, transit 894-1049 m.
  - Caveat [J]: last month's ARS map predicts this month's ARS onset, but a persistent habitat
    preference that the seal re-detects on arrival would do the same. The authors say so: "without
    making any assumptions about what features the animals might be using to recognize the areas".
    It shows that information at the 1 km scale stays useful for about a month. It does not show that
    remembered knowledge of the prey distribution survives that long.
- **AgentSeal** (Chudzinska et al. 2021, Ecol Model 440:109397, doi 10.1016/j.ecolmodel.2020.109397;
  PDF, TRACE and NetLogo code read) [S, model]. Harbour seals, east Scotland. Memory of visited 5 x 5 km
  squares (not 1 km), with the mean energy intake obtained there. Decay is logistic as in DEPONS:
  M(t) = M(t-1) - r M(t-1)(1 - M(t-1)), M = 0.99 on each visit, deleted below 0.01. r = 0.009 per
  15-min step, "Calibrated" (TRACE Table 9) by POM against trip extents, activity budgets and the
  overlap of consecutive trips. Calibration range: r 0.003-0.01, i.e. "from when mseals remember most
  of the patches within the entire 1-month model situation (r = 0.003) to having complete memory loss
  after a week". TRACE: "seals forget a square if not revisited for approximately 10 days". The code
  applies the decay ONCE PER DAY at 08:00 with coefficient r x 96, although the TRACE text says "at
  each time step". From the code [D]:

  | r per 15 min | coefficient per day | M < 0.5 | M < 0.1 | M < 0.01 (deleted) |
  |---|---|---|---|---|
  | 0.009 (final) | 0.864 | 7 d | 9 d | 11 d |
  | 0.0074 (an experiment in the file) | 0.707 | 9 d | 11 d | 13 d |
  | 0.0044 (an experiment in the file) | 0.418 | 13 d | 18 d | 22 d |

  Memory is used to choose the target square for the next trip at the haul-out (TRACE, "CHOOSE NEXT
  TARGET PATCH"); the code also recalculates the attraction at two other points (nlogo lines ~1481 and
  ~1539), whose context was not read. Within a trip, movement is a biased CRW towards that target. So this is
  between-trip memory at 5-km resolution. Removing memory made the modelled seals travel further
  offshore and lose at-sea site fidelity [S]. Haul-out memory does not decay. (A comment in the code
  about "0.21 ... approx 4.6 days" refers to the Saimaa ringed seal model and to haul-outs. Not used
  here.)
- **Context** [S]: AgentSeal's introduction says harbour seals "usually haul-out in large groups but
  appear to forage alone". Foraging trips "rarely extend more than 50 km from the haul-out sites,
  lasting usually < 2 days", and "Individual seals show high site fidelity to their foraging sites"
  (citing Cunningham et al. 2009; Thompson et al. 1996, 1998). Sharples et al. 2012 (PLoS ONE 7:e37216,
  doi 10.1371/journal.pone.0037216, PDF read) [F]: mean trip duration per region from < 1 d (Thames)
  to 4.5 d (Moray Firth); mean trip distance 11-21 km (Shetland, Orkney, Thames) to 100.6 km (Moray
  Firth); "marked fidelity to the local haul-out region". Patch revisits not quantified.
- **Offshore counter-example** [F/S]: Vance et al. 2021 (Sci Rep 11:6514, doi 10.1038/s41598-021-85376-2;
  German Wadden Sea; read through WebFetch, not the PDF): mean offshore trip 7 d (SD 2.3); foraging
  rates in transit "comparable to offshore rates"; prey found "sporadically across the entire foraging
  range"; high-value-prey encounter rates "inconsistent between consecutive foraging trips"; and "a
  resting deficit rather than patch depletion may influence trip length". The authors suggest
  "continual patch assessment" rather than returns to known hotspots.

### 1.3 Grey seal (proxy only, flagged)

- Lidgard et al. 2020 (Movement Ecology 8:11, doi 10.1186/s40462-020-0196-y; 81 GPS seals, Sable
  Island; Europe PMC full text) [F]: T-LoCoH hulls. A "return visit" means the seal was outside the
  hull for more than 12 h. Hulls with the most return visits and the longest stays had the highest
  P(ARS) and lay "on or close to offshore banks". They report "intra-annual site fidelity at presumed
  foraging locations", which the authors link to predictable benthic prey. No return-interval
  distribution is given [gap].
- McConnell et al. 1999 (J Appl Ecol 36:573, doi 10.1046/j.1365-2664.1999.00429.x) [F/A, via
  `seals.md`]: North Sea grey seals made repeated trips to discrete offshore areas (sandeel
  habitat); trips 2.33 d mean; 88 % returned to the departure haul-out.

## 2. A memory span usable at a 6 h tick

No direct measurement exists for either species. The spans below come from model calibration or are
bounded by revisit patterns.

**Porpoise** [S/D/J]:
- DEPONS fine-scale memory: 36 h effective (6 ticks) in the TRACE 2018 calibration, and the 60 h cap
  (10 ticks) in the code defaults since 2019. Both are calibrated to Danish satellite tracks
  (home range, residence time, displacement) [S].
- Unpublished field return interval: ~2 d = 8 ticks (same area at Argos resolution) [S].
- Porpoises carry little reserve ("unable to carry large energy stores", Johnston 2005 citing
  Koopman 1998) [S]. Information older than a few days about a transient school is of little use to
  one [J].
- Central: about 2 d (8 ticks); range 1.5-2.5 d (6-10 ticks) [J]. Weeks to months for persistent
  features (Johnston's 45-day focal use; DEPONS PSM), but that is a habitat map, not a memory of the
  current prey distribution [J].

**Harbour seal** [S/F/J]:
- Between trips: AgentSeal calibrates to half-strength at ~7 d and forgetting at ~10-11 d (5-km
  squares; calibration range one week to one month) [S]. Iorio-Merlo shows that last month's 1-km ARS
  map still predicts ARS onset [F]. That gives a lower bound of ~1 month for usable site information,
  but it is confounded with persistent habitat (1.2) [J].
- Within a trip: a trip lasts 1.6 d mean (Iorio-Merlo, inshore Moray Firth) to 7 d (Vance, offshore).
  No study tests memory within a trip [gap].
- Revisit interval between trips: a seal leaves on another trip after a haul-out of ~6-7 h (Riaz et
  al. 2026 Skagerrak 5.8 h; AgentSeal 7.4 h, via `seals.md`) [S]. With trips of 1-2 d inshore, an
  individual returns to its foraging areas every ~1.5-3 d [D].
- Central: about 7 d (28 ticks); range 2 d to 1 month (8-120 ticks) [J]. The low end is the
  within-trip and between-trip revisit scale; the high end is Iorio-Merlo's month, which mostly
  reflects persistent sites.

**Kernel shape** [S/J]: both models use a logistic memory curve that stays near 1 and then falls
fast (DEPONS from 0.91 at 24 h to 0.08 at 48 h at rR 0.1; flat until the cap in the current code).
A hard truncation at M ticks, as in `scripts/memory_weight.py`, is a fair stand-in for both. An
exponential memory kernel would put more weight on the most recent ticks than either model does.

## 3. Prey patches or persistent habitat features?

- **Porpoise**: the strongest field case is habitat. In Johnston 2005, 3 of 6 animals used "predictable,
  tidally induced fine-scale oceanographic features" (island and headland wakes; porpoise density 9.59
  per km2 on flood vs 1.79 on ebb; acoustic prey aggregations along the fronts) [F]. Stalder 2020
  relates AR to salinity, chlorophyll anomaly, current velocity and steep slopes [F]. At 870 km2,
  porpoises co-occur with herring (Sveegaard 2012 MEPS, partial Mantel 0.254; `mammals.md`) [F]. That
  is resource tracking at a scale too coarse to say whether a 1-km patch is remembered. DEPONS models
  food patches as FIXED places that regrow (rU), so its memory is memory of persistent places by
  construction [S]. No study shows a porpoise returning to a transient school [gap].
- **Harbour seal**: inshore seals return to the same 1-km areas month after month (Iorio-Merlo), and
  grey seals to benthic banks (Lidgard; McConnell: sandeel grounds). These are persistent places,
  mostly benthic. Offshore Wadden Sea seals find prey sporadically and with inconsistent success
  between trips (Vance) [F/S]: no persistent prey patch to remember.
- **The box** [J]: offshore, deep (median bed 143 m), microtidal (Skagerrak tidal range is small; not
  looked up here), and seal density falls westward from haul-outs east of the box (`mammals.md` 3). Inshore tidal-front memory, which
  dominates the porpoise and seal field cases, has little to latch onto here. The persistent features
  are bathymetric (the Norwegian Trench slope, banks). The seal case in the box resembles Vance
  (offshore, sporadic prey) more than Iorio-Merlo (inshore, repeatable sites). In the model, the
  remembered quantity for a neighbour is the target group's current density. For pelagic prey
  (herring schools) that is a transient aggregation, so persistence rho, not memory span, will set w.
  For bed-associated prey over shallow beds it is closer to the persistent-habitat case.

## 4. Collective or social information

- **Porpoise** [F]: Sørensen et al. 2018 (Sci Rep 8:9702, doi 10.1038/s41598-018-28022-8; 6 DTAGs, 96 h,
  Danish waters; PDF read): conspecific clicks were audible on the tags in 9.9-58.8 % of one-minute
  intervals (">10% of the time"). The longest gap without a conspecific was 22-180 min. Porpoises
  produce high-repetition "calls" distinct from foraging buzzes (independent evaluators judged 91 % of
  the candidate calls distinct from buzzes and 97 % of the buzzes associated with prey echoes or
  jerks), and calls co-occur with conspecific calls. This shows
  acoustic contact and probable communication over a few hundred m (T-POD detection 260-400 m,
  `mammals.md`). **No study shows porpoises eavesdropping on conspecific buzzes to find prey**
  [gap; absence of evidence]. DEPONS has no social structure ("Social structure is not included",
  TRACE 2.4.10) [S]. Implication [J]: a neighbour cell's content could reach a porpoise from a
  conspecific's buzzes at < 0.5 km, but this is unquantified. The 1-(1-P)^N pooling of `design.md`
  1.3 has no empirical support for porpoises.
- **Harbour seal** [S]: "appear to forage alone" (AgentSeal, citing the literature). Seals can follow a
  conspecific's hydrodynamic trail in the lab (Schulte-Pelkum et al. 2007, `mammals.md`) [F]. No field
  study of social foraging information was found [gap]. Haul-out groups could in principle act as
  information centres. Untested [J].

## 5. Giving-up: how fast a depleted patch is abandoned

- **Porpoise** [S, model]: in DEPONS, a fine-scale animal whose remembered recent food E falls turns
  back towards remembered patches (satiation memory, effective ~19 h at rS 0.2). If its energy has
  declined for `t_todisp` = 3 days it abandons the area and disperses (TRACE 2.7.2). That makes 3 d an
  area-scale giving-up time in the model. In the field, no patch-leaving rule is measured. Porpoises
  forage near-continuously (Wisniewska et al. 2016, `mammals.md` E7) and cannot buffer a long drought
  [S]. Johnston's animals made forays out of their core areas and came back [F].
- **Harbour seal**: Vance 2021 [F/S]: trip length is set by a resting deficit (resting +~37 min/d over a
  trip), not by patch depletion. Iorio-Merlo [F]: prey encounters raise P(ARS) at the 5-dive scale,
  so a seal stays in ARS while it encounters prey and switches to transit when it does not. Thompson &
  Fedak 2001 (Anim Behav 61:287) and Sparling et al. 2007 (Anim Behav 74:207; grey seals left
  low-density patches earlier) [F/A] act at the dive scale (minutes). That is the wrong scale to bound
  how stale 6-h information can be; cited for completeness only.
- **Bound on staleness** [J]: both species re-assess a spot within the tick (dive-scale and
  ARS-scale decisions in minutes to hours). Information remembered from earlier ticks is therefore a
  prior on where to go, and the current tick corrects it on arrival. In the model this means memory
  sets which neighbours are known at all, while the value read is the current one only for the
  neighbour actually entered. That is a reason to keep rho, not memory, as the binding factor.

## 6. Table

| observer | memory span central (range) | in 6 h ticks | basis | grade |
|---|---|---|---|---|
| porpoise, fine-scale patch memory | 2 d (1.5-2.5 d) | 8 (6-10) | DEPONS: rR 0.1 gives an effective 36 h (TRACE 2018); the code defaults since 2019 (rR 0.03-0.04) give a flat 60 h cap (120 half-hour steps, a software constant); both calibrated by POM to Danish satellite tracks; the unpublished "return after ~2 days" | S (model), D (conversion), J (central) |
| porpoise, satiation / expected-food memory | 19 h (19 h to 2.5 d) | 3 (3-10) | DEPONS rS 0.2 (TRACE) vs 0.03-0.04 (code) | S, D |
| porpoise, site or landscape memory | weeks to months | > 100 | Johnston 2005 (45 d focal use, 71 % of days within 5 km of a front); DEPONS PSM (lifetime, 2 x 2 km) | F (pattern), S |
| harbour seal, patch memory between trips | 7 d (2 d to 1 month) | 28 (8-120) | AgentSeal calibrated r 0.009: half at 7 d, forgotten at 10-11 d, 5-km squares (calibration range one week to one month); trips 1.6 d (inshore) to 7 d (offshore); Iorio-Merlo: last month's 1-km ARS map predicts ARS onset (no decay modelled; confounded with habitat) | S (model), F (pattern), J (central) |
| harbour seal, within-trip memory | unknown (at most the trip length, 1.6-7 d) | at most 6-28 | no study | J / gap |
| grey seal (proxy) | weeks to the season | > 50 | Lidgard 2020: intra-annual fidelity to banks, return = > 12 h out of a hull; McConnell 1999: repeated trips to discrete sandeel grounds | F (pattern), F/A |

Reading the table [J]: for the box, memory spans of 1-2 weeks or more mainly carry persistent
habitat, which the model's channels do not separate from the current density. The memory that
carries information about the target's current distribution is short: ~2 d for porpoises (an
animal with no reserve, revisiting about every 2 days) and, for seals, the trip and revisit
cadence of ~2-7 d. For mobile pelagic targets, rho will be the binding factor whatever span is
chosen.

## 7. Gaps

- No field revisit-interval distribution for porpoises at 1-10 km. The ~2 d is unpublished (J
  Nabe-Nielsen) and DEPONS uses it for food recovery, i.e. for persistence, not memory.
- Nabe-Nielsen et al. 2013 Oikos and its appendix (the residence times and displacements rS and rR were
  calibrated to) not reached; the reason the code moved from rR 0.1 to 0.03-0.04 not documented.
- Stalder 2020 and Sveegaard 2011 give the persistence of AR only as "extended periods" (Fig. S1), no
  number.
- No within-trip memory study for seals. AgentSeal memory works at 5 km and mainly picks the next trip's target.
  Iorio-Merlo's month-lag map has no decay and is confounded with habitat. Their BA overlap values
  (table S3) not read.
- No porpoise or seal study of social foraging information (buzz eavesdropping, following at sea).
  Sørensen 2018 shows acoustic contact, not information transfer.
- No harbour-seal tracking at sea in the Skagerrak (Riaz 2026 gives haul-outs only).
- Johnston 2005 and the tidal-site literature are from macrotidal coasts; their transfer to the
  microtidal, deep box is [J].
