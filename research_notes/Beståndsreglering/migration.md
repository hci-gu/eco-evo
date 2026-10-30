# WP2 Migration across the box edge: immigration and emigration of each group's members (2026-10-29)

Question (plan.md WP2): direction, timing and size of the seasonal and ontogenetic movements of each
group's members into and out of the 60 x 60 km box (57.81-58.31 N, 10.07-11.15 E, northern
Skagerrak, NS-IBTS roundfish area 8, rectangles 44G0/45G0), whether the net movement is density
dependent, and a minimal edge-flux form that could take over the part of the regional K
(`r_max B_region / K`) that stands in for net emigration. No engine change.

Grades on every number: **D** computed from data (here or by a literature sub-agent, marked "D,
sub-agent"), **F** stated in a source, **A** analogy, **J** judgement. Sources are marked
[full] full text read, [abs] abstract only, [snip] search snippet, [2nd] cited inside another source.

Files:
- `scripts/mig_extract_cpue_age.py` - DATRAS NS-IBTS CPUE-at-age per haul (getCPUEAge, 1991-2025
  Q1/Q3, 1991-97 Q2/Q4; nine species) -> `data/cpue_age_all.csv.gz` (raw XML in `data/cpue_age/`,
  untracked, untrusted).
- `scripts/mig_cohort_rates.py` -> `tables/mig_density_ratio_by_age.csv`,
  `mig_cohort_net_emigration.csv`, `mig_share_elasticity.csv`, `mig_emigration_vs_density.csv`.
- `scripts/mig_stage_rates.py` -> `tables/mig_quarterly_presence_1991_96.csv`,
  `mig_q1_over_q3_biomass.csv`, `mig_stage_rates.csv`.
- `scripts/mig_group_summary.py` -> `tables/mig_group_summary.csv`.
- Literature downloads in `data/lit_pelagic/`, `data/lit_demersal/`, `data/lit_predators/` (untracked).

Builds on, does not redo: `../Den pelagiska poolens reglering/migration_scale.md` (Ecosim I - eB form,
pelagic Q1/Q3 index 0.34, box exchange time 4-18 d), `stock_composition.md`,
`../Nettoförflyttning per funktionsgrupp/*.md` (speeds; Svedäng 2006 shares),
`../Trålning och uppväxtområden/nursery_areas.md`, `../Årsungarnas andel/summary.md`,
`../Fiskeridödlighet i Skagerrak idag/ices_pelagic_f.md` (HAWG WBSS share table).

---

## Takeaway

- **Every fish group in the box is an open population, and the survey shows it directly.** Relative
  to the stock area (NS-IBTS roundfish areas 1-9), the Skagerrak's density ratio falls with age in
  every gadoid and in herring: Q1 ratio at age 1 -> 4 herring 1.65 -> 0.17, haddock 0.98 -> 0.13,
  whiting 1.17 -> 0.39, cod 4.8 -> 1.5, saithe 2.6 -> 0.4 (SKA50, D). The Skagerrak is a nursery that
  exports its young fish at **0.4-0.8 /yr** (log-slope of the ratio over ages 1-4, D).
- **Cohort rates (net emigration relative to the stock, same gear, same season, D, 1991-2025):**
  herring 0.5 /yr (age 1->2) and 0.8 /yr (2->3); haddock 1.1-1.25 /yr in the first 1.5 years, ~0
  after; cod 0.4-0.5 /yr at every age 1-5 (box: ~0 at 1->2, the box gains coastal 1-groups, then
  0.2-0.4); whiting 0.7-0.9 /yr at age 2->3, 0.2-0.4 otherwise; Norway pout 0.6-0.7 /yr; saithe
  0.9-1.4 /yr at ages 1-3, ~0 for adults 3-8.
- **Seasonal visitors (relative Q1/Q3 biomass presence, D):** water-column predators 0.12-0.18 of
  the August level in February (saithe 0.07-0.16, mackerel adults 0.06-0.07, the IESSNS/landings
  calendar Jun-Oct), Norway pout 0.2, planktivores as a group ~0.6 (herring 0.5-0.65; WBSS adults
  Jun-Sep), benthivores 0.6-0.85 (haddock 0.4-0.7, cod 0.4-1.0), whiting ~1 (0.7-1.4, no consistent
  swing).
- **Density dependence.** (i) Inflow: herring and sprat 1-ringers fill the Skagerrak and the box
  more than proportionally in strong year classes (share elasticity b 1.3 [1.0, 1.6] SKA, 1.9 box;
  the NSAS annex states it qualitatively); cod, haddock, whiting are proportional or below
  (b 0.6-1.1). This is stock-scale spillover, exogenous to the box. (ii) Outflow: at each species'
  ontogenetic emigration age the cohort leaves faster when it is relatively crowded in the
  Skagerrak, slope **c ~0.3 per e-fold of relative density (range 0-0.7)**, p < 0.05 in herring,
  cod age 2, haddock (box), pout (box), and the same sign in all five species (D, here and an
  independent sub-agent test). No density dependence of adult emigration. The evidence is real but
  weak; it cannot by itself carry the compensation K carries today.
- **Recommended minimal form (J):** per stage, `dB_J/dt = I_J(t) - e_J B_J`,
  `dB_A/dt = I_A(t) - e_A(t) B_A`, with (1) an imported recruit flux I_J independent of box biomass
  (sized by WP3), (2) constant ontogenetic emigration e_J, e_A from the cohort table (benthivore e_J ~1.1,
  haddock-dominated), optionally
  `e + c ln(B/B_ref)` on the stage that holds the leavers, and (3) for the
  seasonal visitors an absolute arrival flux in spring and a per-capita departure in autumn from a
  monthly presence table. Regulation then comes from export of locally grown biomass plus an inflow
  that does not depend on local biomass. Test per group: annual per-capita export (ontogenetic e_A
  plus seasonal turnover of non-homing visitors) must exceed the group's net local rate; with the
  library's closed-recruitment bound r_max - F it passes clearly for predators and planktivores,
  with little margin on the adults for benthivores (helped by a haddock-dominated juvenile export
  e_J ~1.1) and fails for whiting (central e 0.3 < 0.41).
- **Confidence:** directions and ages high (survey + tagging + genetics agree); annual rates
  moderate (factor ~1.5, 90 % intervals in the tables); seasonal amplitudes moderate for saithe,
  mackerel, pout, low for herring/cod/haddock (trawl depth and catchability confounds); monthly
  shapes mostly J; density dependence low.

---

## 1. Method: cohort net emigration relative to the stock [D]

- Data: DATRAS NS-IBTS `getCPUEAge` (numbers per hour at age per haul, DATRAS ALK per roundfish
  area), GOV hauls, 1991-2025 Q1 (Jan-Feb) and Q3 (Jul-Aug), 1991-97 Q2/Q4. Nine species only:
  herring, sprat, cod, haddock, whiting, saithe, Norway pout, mackerel, plaice. Nothing for blue
  whiting, long rough dab, dab, witch, rays, hagfish, spurdog, horse mackerel, hake.
- Units: **BOX** = hauls inside the grid (median 5 hauls in Q1, 4 in Q3 per year; all area 8,
  43-253 m; haul mean); **SKA** = roundfish area 8 (Skagerrak, 27 hauls per survey; mean of
  rectangle means); **SKA50** = SKA hauls at >= 50 m (the box's depth range; Q3 has twice the Q1
  share of < 50 m hauls); **STOCK** = areas 1-9 (North Sea + Skagerrak + Kattegat, ~360 hauls per
  survey). Q1 and Q3 Skagerrak hauls are taken by the same vessels (Argos, then Svea and Dana), so a
  ship effect between seasons is small.
- Estimator: `E_rel = -[ln(D_u(dest)/D_u(orig)) - ln(D_STOCK(dest)/D_STOCK(orig))]/dt` along each
  cohort; stock M, F and age-specific catchability cancel. E_rel is the unit's loss rate beyond
  stock-wide Z = net emigration (emigration minus immigration) if M, F and q are the same in the
  unit and the stock. Pooled over cohorts as ratio of sums (weights = cohort size), 90 % bootstrap
  over cohorts. Transitions: annual Q1->Q1 and Q3->Q3 (same season both ends: robust to seasonal
  catchability), summer Q1->Q3 and winter Q3->Q1 (dt 0.5 yr; carry any seasonal q change).
- The per-age **density ratio** D_u/D_STOCK (pooled over years) and its log-slope over age are the
  most robust summary of ontogenetic export.

## 2. Per member

Effort weighted by biomass share. Rates per year; "SKA50 | BOX" where both are given.

### 2.1 Herring (planktivores 0.79)

**Movements [F].**
- NSAS: larvae drift 3-4 months from North Sea spawning grounds (730-1030 km) into the Skagerrak-
  Kattegat nursery (NSAS stock annex [full]). 3.a holds "mainly 2+ ringers of [WBSS] and 0-2-ringers
  from the NSAS" (annex B.5.1 [full]). NSAS "leave the study area at the age between 1 and 2 years"
  (Clausen et al. 2015 [abs]); "When the herring start to mature in the summer of their third year,
  they leave the nursery areas" (Corten 2001 thesis p.198 [full]). So departure starts in the second
  summer and is complete by the third (J).
- HAWG 2026 Table 3.2.8 (NSAS catch numbers in 3.a, 1993-2021): ages 0-2 are 97.6 % (D, sub-agent);
  cohort catch ratios give apparent loss 1.42 /yr (1->2) and 2.20 /yr (2->3), which include Z and
  selectivity (D, sub-agent).
- HERAS July: immature NSAS in 3.a = 10.6-11.4 % of the stock's immature numbers (2024-25), of which
  Skagerrak strata 1.9-7.4 %; mature NSAS 1.3-1.6 % (WGIPS 2026 Table 5.14 [full]; D, sub-agent).
- WBSS adults: "The majority of 2+ ringers migrate out of the area during the 2nd quarter", "the
  younger individuals migrate up into Kattegat and Skagerrak and the older fish ... into the eastern
  North Sea"; "Towards the end of summer [they] aggregate in the eastern Skagerrak and Kattegat"
  before wintering in the southern Kattegat, Sound and western Baltic (WBSS annex 2023 A.1 [full]).
  WBSS share in SD20 (HAWG 2026 Table 3.2.4 [full]): age 3 Q1 0.35 / Q3 0.46, age 5 0.57 / 0.32;
  "NSAS dominated during the first quarter ... increased occurrence of WBSS in summer" (Berg et al.
  2017 [full]). Genetics: NSAS/Downs in HERAS largely juveniles, WBSS and Baltic components largely
  adults; WBSS 25 % of sampled fish (Bekkevold et al. 2023 [full, partial]).

**Survey [D].**
- Density ratio to stock, SKA50 Q1 age 1-4: 1.65, 1.08, 0.41, 0.17; Q3 age 0-4: 3.3, 3.2, 1.7, 0.83,
  0.30. BOX Q3: 1.31, 1.70, 0.77, 0.25, 0.11 (box Q1 low, 0.49 at age 1: winter 1-ringers are in
  < 50 m water, relative density 3.4 there against 0.26 at 100-200 m). Log-slope 0.78 /yr (SKA50 Q1
  and Q3), 0.52-0.94 (box).
- Cohort Q1->Q1, SKA50: 1->2 0.51 [-0.24, 1.18], 2->3 0.77 [0.02, 1.59], 3->4 0.48 [-0.22, 1.22];
  SKA 0.57, 0.93, 0.50. BOX ~0 at 1->3 (too few hauls, wide intervals). Q3->Q3 0->1 -0.03 [-0.67,
  1.0]: no net emigration in the first year. Q3->Q3 at ages 1-3 is ~0 because WBSS summer adults
  (not NSAS cohorts) enter the Q3 numbers.
- Seasonal: relative Q1/Q3 biomass 0.65 (SKA50), 0.50 (box), ages 1+.
- Consistent with migration_scale.md s3.2 (cross-sectional 0.3-0.7 at 1->2, 1-1.7 from age 2).

**Rates for the model.** NSAS juveniles e ~0 in the first year (to Q3 age 1), then **0.5-0.8 /yr**
(range 0.3-1.4) in years 2-3 (D/F). WBSS adults: a summer pulse, about one third of adult biomass
in August is seasonal (relative Q1/Q3 0.5-0.7, D); timing May/Jun arrival, Sep departure (F). Mature
NSAS are rare (1-2 % of the stock's mature in 3.a, F). Residence: NSAS juveniles ~1.5-2.5 yr in
the nursery in total (first year ~0 export, then 1/e 1.3-2 yr; departure complete by the third
summer, F/D); WBSS adults 3-4 months (Jun-Sep, F).

### 2.2 Haddock (benthivores 0.386)

**Movements [F].** Skagerrak haddock belong to the Northern Shelf stock; "The haddock found in the
Skagerrak consist mainly of younger fish"; juveniles occur "along the continental shelf at around
200 m and continuing round to the Skagerrak"; "Adult haddock are thought to be more sedentary"
(stock annex had.27.46a20 s.1.1 [full]). Albert 1991 [full]: 0- and I-group deeper in the trough
than older fish, autumn catches 65-98 % 0-group. Albert 1994 [abs]: shallow in summer, deeper in
winter, mature fish "probably leaving the area in winter". No tagging rates exist for North Sea
haddock (here and gadoids.md).

**Survey [D].** Ratio to stock SKA50 Q1 age 1-5: 0.98, 0.32, 0.32, 0.13, 0.11; Q3 age 0: 3.0 (box
1.53), age 1: 1.07. Cohort: Q3->Q3 0->1 **1.25 [0.92, 1.65]** (box 1.18), Q1->Q1 1->2 **1.10 [0.79,
1.46]** (box 0.84), older ages ~0 (Q1) or -0.4 to 0.7 (Q3, noisy). Seasonal: relative Q1/Q3 0.43
(SKA50), 0.68 (box); summer Q1->Q3 at ages 2-5 is net immigration (-2 to -3.5 /yr over half a
year), winter net emigration 1.9-2.8 /yr: adults come in for the summer, leave in winter
(spawning in the northern North Sea, Feb-May; F stock annex, timing A).

**Rates.** Juveniles **e_J ~1.2 /yr** (0.8-1.7): most 0-group that settle in the Skagerrak leave
before age 2. Adults annual e_A ~0-0.2 (D), with a seasonal swing of about half the summer biomass
(J from D). Residence: haddock juveniles < 1.5 yr (1/e ~0.8 yr, D); adults years, seasonal
absence of about half in winter.

### 2.3 Cod (benthivores 0.314)

**Movements [F].** "The Skagerrak and northern Kattegat appear to be a nursing ground for Viking and
Dogger cod, with most cod in the Skagerrak being Viking cod"; "many Viking juveniles settle in the
Skagerrak and subsequently make a return migration prior to spawning"; North Sea cod "migrate out
of the region to spawn" (WKNSCodID 2020 p.6-8, 30 [full]). Offshore Skagerrak cod are North Sea
ecotype: 62-63 % of archival-tagged deep-Skagerrak cod assigned to the North Sea reference (André
et al. 2016 Table 1 [full]); the North Sea ecotype "was the only type found in offshore North Sea",
35 % of coastal 0-group, 13-62 % by region and < 20-54 % by year (Knutsen et al. 2018 [full]).
North Sea larval drift supplies 1-10 % of each Norwegian fjord's 0-group (Stenseth et al. 2006
[full, via summary]); coastal juveniles move offshore in their second winter at 30-50 cm (Pihl &
Ulmestrand 1993 [abs]). Adults (> 40 cm, Svedäng et al. 2006 CM Q:06 [full]): 40 % resident plus
7 % in the deep Skagerrak, 20 % recaptured west of 10.5 E, 20 % went west and came back, mean 140 d
at liberty (D, sub-agent); migrants older (3.4 vs 2.9 yr) and more mature (54 vs 22 %). The 10.5 E
split runs through the box, so these are not box-edge rates (J).

**Survey [D].** Ratio to stock is high (box Q1 4.3 at age 1, 1.85 at age 5; SKA50 4.8 -> 0.9): the
box is cod-rich relative to the depleted southern North Sea. Cohort Q1->Q1 SKA50: 0.48 [0.27,
0.71], 0.43 [0.20, 0.68], 0.23, 0.47 at ages 1->5; BOX -0.13 [-0.53, 0.24] at 1->2 (gains coastal
1-groups, as Pihl & Ulmestrand), then 0.2-0.56. Q3->Q3 0->1: -0.02 (SKA50), 0.40 (box). Independent
sub-agent estimate on whole rectangles: retention 0.60-0.68 /yr at every age 1-5, e ~0.3-0.4 (D).
Seasonal: relative Q1/Q3 0.43-0.54 (SKA50) but ~1 in the box (box retains its cod in winter); the
1991-96 quarterly box index is low in Feb (0.51) and Nov (0.55), high in May (1.85).

**Rates.** e_J ~0 (Skagerrak; the box gains); **e_A 0.3 /yr (0.1-0.5)** at all adult ages (D/F);
spawning-season export Jan-Apr with about half returning in spring (F, Svedäng) - a homing loop
that by itself carries no compensation (section 4). Residence: coastal nursery 2-4 yr (Henriksson,
F), offshore box cohort 1/e ~3 yr (2-10, D), adult Skagerrak residents years with a Jan-Apr
excursion of ~0.4 yr (Svedäng, F).

### 2.4 Flatfish, rays, hagfish (benthivores ~0.30)

Plaice: North Sea juveniles and adults "migrate into Skagerrak for feeding in summer and autumn"
(Ulrich et al. 2017 [abs]); survey cohort rates in the box ~0 (-0.6 to 0.3), box Q1/Q3 1.0-1.1 (D)
but SKA50 0.3 (summer feeding influx in shallower Skagerrak water). Long rough dab: no study (J:
resident at the box scale). Starry ray: stay within 30-50 nmi after initial dispersal (Walker et
al. 1997 [snip]). Hagfish: very sedentary (DFO, FishBase [snip]). **e ~0, no seasonal term (J).**

### 2.5 Whiting (whiting group)

**Movements [F].** whg.27.3a: "a paucity of information on the population structure of whiting in
3.a", no genetic or otolith studies (stock annex [full]); Skagerrak catches dominated by 1-group
(FishMap [full]); adult displacement 53-123 km, non-philopatric (Tobin et al. 2010 [abs]).

**Survey [D].** Ratio SKA50 Q1 age 1-5: 1.17, 1.07, 0.52, 0.39, 0.24 (log-slope 0.40); box 2.08 ->
0.20 (0.70). Cohort Q1->Q1: 1->2 ~0 (-0.04 SKA50; sub-agent: net gain, 1-groups still arriving),
**2->3 0.66 [0.38, 0.91]** (box 0.87), 3->5 0.2-0.4. Q3->Q3 0->1 0.18 (box 0.26). Seasonal: box Q1/Q3
1.43 (0-group whiting arrive in the box Aug->Feb, winter rate -2.0 /yr; the box is a wintering area
for 0-1 group), SKA50 0.70.

**Rates.** e_J ~0.2 (-0.2 to 0.6); **e_A 0.3 /yr (0.1-0.6)**, concentrated at maturation (age 2-3,
L50 21.5 cm, F annex); no robust seasonal swing. Residence: 1/e ~3 yr (1.7-10, D), shorter around maturation (1/e ~1.5 yr).

### 2.6 Saithe (water-column predators 0.651)

**Movements [F].** West-coast Norwegian juveniles leave the coast "when it is about three years old,
at a length of about 35 cm", nearly all gone two years after tagging; of recaptures outside the
tagging area 79 % were in the North Sea region, "along the eastern side of the plateau, west and
southwest of the Norwegian Deep", most at 57.5-59 N, with "little seasonal variation" (Jakobsen 1978
CM G:32 [full]); "After the age of 4-5 years, most of the surviving fish seem to have left" (Jakobsen
1981 [full]); "Saithe migrate from nursery areas to the North Sea within the ages of 2-5", larger
fish earlier; "Smaller/younger fish are found in the Skagerrak" (pok.27.3a46 annex [full]).
Adults: spawning Jan-May along the shelf edge, earliest in the south (FishMap [full]); "Density in
the northeastern North Sea is markedly higher during summer than during winter" (FishMap [full]);
Norwegian adults move farthest in Feb-Mar (Homrum et al. 2013 [full, via summary]). The Skagerrak
coastal nursery itself was never tagged.

**Survey [D].** Box Q3 numbers: age 1 11.5 %, 2 3.7 %, 3-4 55.6 %, 5-6 27.9 % (D, sub-agent). Cohort
Q1->Q1 SKA50 at 1->3: 0.86, 0.92 [0.10, 2.1]; ages 3->8 -0.2 to 0.3 (~0); 8->9 1.1 (large fish
leave, or lower q). Seasonal: relative Q1/Q3 biomass **0.16 (SKA50), 0.07 (box)**; numbers in the
box 0.037 (sub-agent). In Q1 saithe sit deepest (relative density 1.9 at 200-300 m against 0.4 at
100-200 m), so part of the winter dip is descent below the trawled depth (box to 533 m; the Trench
axis lies north of the box) - the swing is an upper bound.

**Rates.** Adults annual net e_A ~0 (0-0.3); arrivals of 3-4 year olds from the coast (I_A); a
strong seasonal visit: monthly table below. Departure of the nursery leavers is size-dependent,
not density-dependent (annex; F). Residence: adults years at the stock scale (annual e ~0) but
only ~6-8 months a year in the box (May-Nov, D/J).

### 2.7 Mackerel (predators 0.128)

Skagerrak coastal landings 1987-2008: arrival (10 % of cumulative landings) median 12 June (18 May -
30 June), departure (90 %) median 14 October (25 Sep - 1 Nov); large fish stay 38 days longer;
arrival 15.6 +/- 12.1 d earlier per deg C (Jansen & Gislason 2011 [full, in thesis]; dates read off
Fig. 7, D sub-agent). Left the northern North Sea east of 4 E by October 2009; adults winter along
the western Norwegian Trench edge, outside the box (Jansen et al. 2012 [full]). Today's North Sea
catches are "predominantly juvenile ... 1- and 2-year-olds" (Coastal States WG 2022 [full]).
Survey: adults Q1/Q3 0.06-0.07 relative (D). Skagerrak (area 8) holds 6.8 % of the NS-IBTS Q3 index
on 5.5 % of the area; the two box rectangles 0.34 % on 1.1 % (D, sub-agent; GOV-biased).
**A pure summer visitor, Jun-Oct: residence ~4 months (12 Jun - 14 Oct medians, D).**

### 2.8 Minor predators (~0.22)

Spurdog: pregnant females in W Norway stay within 40 km over 82-364 d (Klöcker et al. 2024 [full]);
box Q1/Q3 biomass 0.12 (D, repo) - a summer influx. Horse mackerel: largest fish reach the northern
North Sea "by September" (hom annex 2025 [full]); Q1/Q3 0.02 (D): present ~Aug-Nov (J). Hake: North
Sea spawning Jul-Sep (hke annex [full]); Q1/Q3 0.20 (D). Anglerfish largely resident (Q1/Q3 0.80),
ling Q1/Q3 0.66 (D). **Seasonal visitors except anglerfish; no ontogenetic rate (J).**

### 2.9 Small planktivores: Norway pout 0.075, blue whiting 0.105, sprat 0.03

- Norway pout: in the eastern Skagerrak "only to a very small degree a self-contained stock and
  adults migrate out ... to spawn"; larvae drift in and "return mainly during the latter part of
  their second year of life before becoming mature"; spawning mid-Feb along the 120 m isobath
  Norway-Scotland (WKBWNG 2026 WD3 p.33 citing Poulsen 1968 [full]). Survey: cohort Q1->Q1 0.68
  (1->2), 0.56 (2->3), Q3->Q3 0->1 0.76 [0.30, 1.13] (SKA50, D); relative Q1/Q3 0.20-0.24 (D); Q1
  deeper. **e_J ~0.6-0.8, e_A ~0.6, strong winter absence.**
- Blue whiting: spawns Mar-Apr west of the British Isles; juveniles drift into the Norwegian Deep in
  strong years, "settle in autumn in deeper shelf areas", stay "during their first winter or
  longer", join the adults over 2-3 years (Coastal States WG 2024 s.2.1 [full]). No box data (J:
  imported juveniles leaving at age 2-3, e ~0.4).
- Sprat: one North Sea-3.a stock with partly independent coastal/fjord populations (HAWG 2026
  s.11.11.1 [full]; Glover et al. 2011 [abs]); offshore Skagerrak sprat groups with the North Sea
  (Saltalamacchia et al. 2022 [snip]); relative Q1/Q3 0.21 (SKA50, D). A summer visitor offshore.

### 2.10 Harbour seals and harbour porpoises (brief)

- Seals: branded Skagerrak seals were never resighted beyond 32 km of the branding site; adult
  females strongly site-faithful (Härkönen & Harding 2001, via dissertation summary [full summary]).
  No annual exchange rate exists. Keep the existing K (metapopulation regulated off the grid,
  `../Bärförmåga för säl och tumlare/seals.md`) (J).
- Porpoises: box in the North Sea assessment unit; autumn-winter "gradual net movement" west into
  the NE North Sea, Trench slope stays high-density (Sveegaard et al. 2011, via repo notes). No
  exchange rate. No change recommended (J).

## 3. Density dependence

**3.1 Inflow vs stock year-class strength [D]** (`mig_share_elasticity.csv`; slope b of ln D_unit on
ln D_STOCK over 35 years, 90 % bootstrap):

| species, age (quarter) | SKA | BOX |
|---|---|---|
| herring 1 (Q1) | **1.28 [0.99, 1.63]** | **1.92 [1.45, 2.62]** |
| herring 0 (Q3) | 1.27 [0.84, 1.68] | 1.83 [0.50, 3.24] |
| herring 2-4 (Q1) | 0.07-0.19 | -0.09-0.44 |
| sprat 1 (Q1) | 1.90 [1.46, 2.44] | 1.89 [1.24, 2.51] |
| cod 0 (Q3) / 1 (Q1) | 0.87 / 1.02 | 0.99 / 0.93 |
| haddock 0 (Q3) / 1 (Q1) | 0.76 / 0.85 | 0.93 / 0.77 |
| whiting 0 (Q3) / 1 (Q1) | 0.70 / 1.12 | 1.13 / 1.41 |
| saithe 1 (Q3) | 1.20 [1.02, 1.44] | 0.86 |
| mackerel age 1+ (Q3) vs NEA SSB | 2.1 (SE 0.65) | 2.5 (sub-agent; confounded by temperature trend) |

Herring and sprat juveniles spill over into the Skagerrak in strong year classes, matching the NSAS
annex ("higher proportions in the Skagerrak and Kattegat when the year class is good", [full], no
numbers) and Corten & van de Kamp 1992 [snip]. For gadoids the share is proportional or below
(also the sub-agent's year-to-year slope test, 0.77-0.94 at age 1: no spillover). Adult herring in
winter in the Skagerrak are independent of the North Sea stock (b ~0.1): they are WBSS/local fish.
Saithe adults: area-8 CPUE against SSB slope -2.0, no spillover (sub-agent). In the model these are
properties of the imported flux I (it scales with the stock's recruitment as R^b), not of box
biomass.

**3.2 Emigration vs local crowding [D]** (`mig_emigration_vs_density.csv`; per-cohort annual E
against the same cohort's relative share in the earlier Q3 survey - an independent sample, so no
ratio artefact; Spearman with permutation p; slope c = dE/d ln share):

| leaving age | herring 1->2 | cod 2->3 | haddock 1->2 | whiting 1->2 | pout 1->2 |
|---|---|---|---|---|---|
| SKA rho (p) | 0.37 (0.04) | 0.45 (0.01) | 0.22 (0.22) | 0.33 (0.06) | 0.25 (0.16) |
| SKA c [90 %] | 0.44 [0.09, 0.89] | 0.66 [0.32, 0.96] | 0.34 [0.06, 1.0] | 0.61 [0.20, 0.91] | 0.41 [0.06, 0.75] |
| BOX rho (p) / c | 0.17 (0.43) / 0.16 | 0.18 (0.31) / 0.16 | 0.37 (0.05) / 0.30 | 0.18 (0.32) / 0.23 | 0.48 (0.01) / 0.43 |

- The sub-agent's independent test on whole rectangles gave the same sign: elasticity of retention
  to the cohort's prior Skagerrak density -0.16 (cod 1->2), -0.33 (cod 2->3, haddock 1->2), -0.58
  (whiting 1->2) (D).
- Adult ages: no consistent sign (rho -0.4 to 0.4, mostly n.s.; saithe 6 and 8 positive on 11-22
  cohorts). Against stock-wide year-class size, E shows no dependence (rho -0.4 to 0.3).
- Caveats: ~10 relevant tests, 3-4 at p < 0.05; a persistent cohort-specific catchability or
  distribution anomaly that relaxes would give the same pattern (which is still a relaxation of
  local excess toward the stock mean). BOX values are attenuated by box sampling noise (4-5 hauls).
- Literature: juvenile cod spread out at high abundance and contract to optimal habitat at low
  abundance (Blanchard et al. 2005 [abs], ideal free distribution); Chen et al. 2005 [full] found a
  positive effect of North Sea abundance on Skagerrak cod at ages 3 and 5 (8 of 32 tests); haddock
  1962 and 1967 year classes expanded the distribution (FishMap [snip]); juvenile haddock on the
  Scotian Shelf fill marginal sub-areas at high abundance (Marshall & Frank 1995 [snip], A). Saithe
  and herring departure is scheduled by size and maturation (pok annex; Corten 2001) [full]. No
  MacCall-basin test exists for North Sea herring juveniles.
- **Verdict (J):** a weak density-dependent emigration of the cohort at its ontogenetic leaving age
  is supported, c ~0.3 (0-0.7) per e-fold of relative density; adult emigration and seasonal
  visits are not density dependent; the inflow scales with stock year-class strength (b ~1, herring
  and sprat 1.3-1.9). The emigration term can carry part of the compensation, not all of it.

## 4. Recommended minimal form and parameters (J, from the D and F above)

Per group g, stage J (juvenile compartment: from settlement to the Q3 survey at age 1) and A, per
year (divide by 1460 per 6-h library tick; kept outside the derived growth budget, like K today):

```
dB_J/dt|edge = I_J(t)                      - [e_J + c_J ln(B_J/B_J,ref)] * B_J
dB_A/dt|edge = I_A(t) + I_vis(t)           - [e_A + c ln(B_A/B_A,ref) + d_vis(t)] * B_A
```

1. **Imported recruits I_J(t)** (t/km2/yr): absolute, independent of box biomass, in the months the
   library's `recruitment` weights already use (planktivores May-Sep, benthivores Jun-Sep), scaled
   as `I_J,ref (R_stock/R_ref)^b`. Its size is WP3's box-share problem; the survey gives the box's
   age-0 density relative to the stock in August (herring 1.3 box / 3.3 SKA50, cod 8.1 / 7.7,
   haddock 1.5 / 3.0, whiting 0.8 / 1.5, pout 1.8 / 2.4; D). Local spawning by box adults is then
   only the box's own share of these recruits (WP3).
2. **Ontogenetic emigration e_J, e_A** (constant; emigrants take reserve with biomass, as M1).
3. **Optional density dependence** c (c_J or c) on the stage holding the leavers (planktivores and
   whiting: A; haddock-dominated benthivores: J), with `B_ref` = today's start biomass; default c = 0, sensitivity 0.3.
4. **Seasonal visitors**: arrivals `I_vis(t) = dp/dt * B_A,peak` in rising months (absolute,
   stock-driven), departures `d_vis(t) = -d ln p/dt` per capita in falling months, from the monthly
   presence table p(m) (fraction of the August level). Only non-homing arrivals regulate: a homing
   loop (cod: about half of the westward spawning migrants return, Svedäng) returns what left and is
   neutral. At the box scale (0.6 % of the North Sea) returning visitors are drawn from the stock-wide
   pool, so treat arrivals as independent of box B except for cod (return share ~0.5, J).
5. Fishing: the library F is the stock rate; with explicit emigration part of it is taken after the
   fish have left (migration_scale.md s6). Keep F as is only if the box stands for the stock's average
   fish (J).

**Parameter table** (per year; central [range]; grade). Weighting rule: e_A and p_min by the members'
biomass shares; e_J by the composition of the JUVENILE compartment itself (age-0 biomass, from
`../Årsungarnas andel/summary.md`: benthivore age 0 is ~90-95 % haddock, planktivore age 0 mostly
herring), not by the adult shares. Residence = 1/e or the stated stay.

| group (members) | e_J | e_A | c | Feb / Aug presence p_min | inflow b | evidence |
|---|---|---|---|---|---|---|
| planktivorous_fish (herring 0.79, blue whiting, pout, sprat) | 0.1 [0, 0.5] (herring ~0, pout 0.7) D | **0.6 [0.3, 1.0]** D/F | 0.3 [0, 0.6] D | 0.6 [0.5, 0.65] D/J | 1.3 [1.0, 1.9] D/F | survey cohorts, HAWG, Corten, Clausen |
| benthivorous_fish (haddock 0.386, cod 0.314, flatfish/rays/hagfish 0.30) | **1.1 [0.8, 1.3]** (J stage ~90-95 % haddock) D | **0.17 [0.05, 0.3]** (cod 0.3, haddock 0.1, flatfish 0) D/F | 0.3 on J D | 0.73 [0.6, 0.86] D/J | 0.9 [0.75, 1.0] D | survey, WKNSCodID, Svedäng, Albert |
| whiting | 0.2 [-0.2, 0.6] D | **0.3 [0.1, 0.6]** D | 0.3 [0, 0.6] D | ~1 [0.7, 1.4] D | 1.1 [0.7, 1.4] D | survey, stock annex |
| predatory_fish (saithe 0.651, mackerel 0.128, others 0.22) | - (no juveniles in the box) | **0.1 [0, 0.3]** (saithe adults ~0) D | 0 F/J | **0.09 [0.07, 0.18]** D/F/J | n/a (saithe <= 0, mackerel 2 D) | survey, Jakobsen, Jansen & Gislason, FishMap |
| seals, porpoises | - | 0 (keep seal K) | - | porpoise winter westward shift, unquantified | - | F |

**Monthly presence p(m), fraction of the August level** (J, anchored on D Feb/Aug ratios and F
timing; group = member-weighted; Jan-Feb and Jul-Aug cells are D, others J):

| group | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| saithe | 0.10 | 0.05 | 0.10 | 0.30 | 0.60 | 0.90 | 1 | 1 | 1 | 0.90 | 0.60 | 0.30 |
| mackerel | 0.01 | 0.01 | 0.02 | 0.03 | 0.10 | 0.50 | 1 | 1 | 0.90 | 0.50 | 0.10 | 0.03 |
| predatory_fish | 0.12 | 0.09 | 0.13 | 0.29 | 0.54 | 0.83 | 1 | 1 | 0.99 | 0.85 | 0.54 | 0.29 |
| herring | 0.60 | 0.60 | 0.60 | 0.65 | 0.80 | 0.95 | 1 | 1 | 0.85 | 0.70 | 0.60 | 0.60 |
| planktivorous_fish | 0.60 | 0.60 | 0.61 | 0.67 | 0.81 | 0.95 | 1 | 1 | 0.86 | 0.72 | 0.62 | 0.61 |
| benthivorous_fish | 0.73 | 0.71 | 0.71 | 0.78 | 0.91 | 0.96 | 1 | 1 | 1 | 0.96 | 0.88 | 0.81 |
| whiting | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

(Predators: saithe and mackerel rows plus the minor predators at 0.25 in Jan-Feb rising to 1 in
Jul-Sep. Planktivores: pout and sprat 0.2 in winter, blue whiting flat 1. Benthivores: haddock 0.5
and cod 0.75 in Jan-Mar, flatfish flat.)

**The regulation test** (J): with imported recruits and stock-driven arrivals the box biomass is
bounded when the annual per-capita export (e_A, plus the visitor turnover -ln p_min for non-homing
visitors) exceeds the adults' net local rate. The library's closed-recruitment bound r_max - F is
the conservative comparison:

| group | export: e_A + (-ln p_min) | r_max - F (library) | verdict |
|---|---|---|---|
| predatory_fish | 0.1 + 2.4 = 2.5 | 0.374 - 0.24 = 0.13 | regulated by the visit; behaves as a window on the stock (migration_scale.md D2) |
| planktivorous_fish | 0.6 + 0.5 = 1.1 | 0.37 - 0.10 = 0.27 | passes |
| benthivorous_fish | 0.17 + ~0.2 (cod loop half neutral) = ~0.35 | 0.31 - 0.17 = 0.14 | passes, with little margin on the adults; the juvenile export e_J ~1.1 (haddock) removes most locally settled recruits before they reach A |
| whiting | 0.3 + 0 = 0.3 | 0.49 - 0.08 = 0.41 | fails at central values; needs c > 0, a lower local recruitment (WP3), or K kept |

The real local rate is lower than r_max - F once local recruitment is replaced by I_J (r_max is a
whole-life-cycle rate whose recruitment happens outside the box), so the test is conservative; the
synthesis should redo it with the stage-structured local rate, not with model output.

## 5. Caveats

- E_rel is relative to the stock area: any extra Skagerrak mortality (fishing, predation) or a
  different catchability counts as emigration, so the rates are upper bounds on true emigration
  (cod F in 3.a.20 is mid-range per WKNSCodID, F).
- The GOV fishes <= 280 m in the Skagerrak; the box reaches 533 m. Saithe, pout and haddock sit
  deeper in winter, so the winter absence is partly descent out of the trawled layer (upper bound on
  the seasonal swing). Pelagic schools (herring adults, mackerel, sprat) are under-sampled by the
  GOV; the seasonal pelagic numbers are indices.
- BOX has 4-5 hauls per survey and no hauls below 253 m; SKA50 is the more reliable unit, BOX
  shows where the box differs (cod and whiting retained or gained in winter).
- Pooled ratios weight strong year classes; per-cohort medians differ (tables). Q1->Q1 and Q3->Q3
  annual rates differ where the leaving is seasonal (haddock 1.10 vs ~0 at 1->2): they bracket the
  timing; the stage rates above use the transition that matches the engine's stage boundary.
- 1991-96 quarterly presence (`mig_quarterly_presence_1991_96.csv`) rests on 5-6 years and few Q4
  Skagerrak hauls; it is used only qualitatively (cod low in Feb/Nov in the box).
- WBSS adults, mackerel and saithe are partly exogenous visitors whose summer biomass is set by the
  stock and by the route choice (Corten's site fidelity at the route scale, temperature for mackerel
  arrival); the monthly tables are climatological.
- No published fraction of an NSAS or Viking-cod year class that grows up in the Skagerrak, no
  tagging residence times for juveniles, no rates at all for long rough dab, dab, witch, blue whiting
  in the box, seal or porpoise exchange.

## Sources

| Source | Used for | Read |
|---|---|---|
| ICES DATRAS NS-IBTS getCPUEAge 1991-2025 (datras.ices.dk) | all D survey rates | data |
| ICES 2022 Stock Annex her.27.3a47d, figshare 18622589 | NSAS nursery, year-class sentence, B.5.1 | full |
| ICES 2023 Stock Annex her.27.20-24, doi 10.17895/ices.pub.23898678 | WBSS seasonal migration | full |
| ICES 2026 HAWG, Sci. Rep. 8:16, figshare 31424315 (Tables 3.2.4, 3.2.8; s.3.13.1, 11.11.1) | WBSS shares, NSAS catch ages, sprat | full |
| ICES 2026 WGIPS Suppl. Doc. 3 (HERAS 2025), doi 10.17895/ices.pub.33406126 | NSAS shares in 3.a | full |
| Clausen et al. 2015, J. Fish Biol., doi 10.1111/jfb.12571 | NSAS leave at age 1-2 | abs |
| Corten 2001, Herring and Climate (PhD), vliz.be ocrd/221866.pdf | departure at maturation, site fidelity | full |
| Corten & van de Kamp 1992, ICES Mar. Sci. Symp. 195:402 | strong year classes in Skagerrak | snip |
| Berg et al. 2017, PLoS ONE 12:e0187374 | NSAS/WBSS by quarter | full |
| Bekkevold et al. 2023, ICES JMS 80:173, doi 10.1093/icesjms/fsac223 | genetic stock ID | full (partial) |
| Saltalamacchia et al. 2022, Fish. Res. 245:106131; Glover et al. 2011, ICES JMS 68:2145 | sprat structure | snip; abs |
| ICES 2026 WKBWNG Norway pout WD3 (citing Poulsen 1968) | pout migration, spawning | full |
| Coastal States WG blue whiting 2024; mackerel 2022 (assets.publishing.service.gov.uk) | blue whiting drift; NS mackerel juveniles | full (parts) |
| ICES 2020 WKNSCodID, Sci. Rep. 2:89, doi 10.17895/ices.pub.7499 | cod substocks, Skagerrak nursery | full |
| ICES 2023 WKBCOD (archimer 00841/95303) | cod substock assumptions | full |
| Knutsen et al. 2018, Evol. Appl. 11:1527, doi 10.1111/eva.12640 | cod ecotype shares | full |
| André et al. 2016, BMC Res. Notes 9:63, doi 10.1186/s13104-016-1878-9 | deep-Skagerrak cod assignment | full |
| Henriksson et al. 2023, ICES JMS 80:145, doi 10.1093/icesjms/fsac220 | offshore ecotype, coastal nursery 2-4 yr | full (preprint) |
| Stenseth et al. 2006, Proc. R. Soc. B 273:1085, doi 10.1098/rspb.2005.3290 | larval drift share | full (via summary) |
| Knutsen et al. 2004, Proc. R. Soc. B 271:1337 | larval drift 2001 | abs |
| Pihl & Ulmestrand 1993, ICES JMS 50:63, doi 10.1006/jmsc.1993.1007 | juvenile cod offshore move | abs |
| Svedäng, Righton & Jonsson 2006, ICES CM 2006/Q:06 | adult cod shares, time at liberty | full |
| Chen et al. 2005, MEPS 302:219, doi 10.3354/meps302219 | NS abundance effect on Skagerrak cod | full |
| Blanchard et al. 2005, CJFAS 62:2001, doi 10.1139/f05-109 | density-dependent habitat selection | abs |
| Albert 1991, ICES CM 1991/G:45; Albert 1994, ICES JMS 51:31 | haddock depth by age, season | full; abs |
| ICES Stock Annexes had.27.46a20 (figshare 29356478), whg.27.3a (figshare 32791251), pok.27.3a46, dgs, hom, hke | stock structure, migration | full |
| Tobin et al. 2010, Mar. Biol. 157:1063 | whiting displacement | abs |
| Ulrich et al. 2017, J. Sea Res. 120:13 | plaice feeding migration | abs |
| Walker et al. 1997, ICES JMS 54:797; DFO / FishBase hagfish | ray, hagfish residency | snip |
| Marshall & Frank 1995, CJFAS (thesis) | haddock range vs abundance (A) | snip |
| Jakobsen 1978 ICES CM G:32, 1981 G:35 | saithe coastal emigration | full |
| Homrum et al. 2013, ICES JMS 70:782, doi 10.1093/icesjms/fst048 | saithe adult displacement | full (via summary) |
| Otterå & Skilbrei 2014, ICES JMS 71:2484; Bjordal & Skar 1992 ICES CM O:35 | saithe residence | full (via summary); full |
| ICES FishMap saithe, mackerel, haddock factsheets | seasonal density | full / snip |
| Jansen & Gislason 2011, Cont. Shelf Res. 31:64, doi 10.1016/j.csr.2010.11.003 | mackerel arrival/departure | full (thesis) |
| Jansen et al. 2012, PLoS ONE 7:e51541 | mackerel overwintering | full |
| Klöcker et al. 2024, Mov. Ecol. 12:62 | spurdog residency | full |
| Olafsdottir et al. 2019, DSR II 159:152 | mackerel expansion with stock size (A) | abs |
| Härkönen & Harding 2001 (via Harding dissertation summary) | seal site fidelity | full (summary) |
| Sveegaard et al. 2011 (via repo notes) | porpoise seasonal shift | 2nd |
