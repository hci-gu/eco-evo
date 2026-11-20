# Fine-scale predator-prey overlap inside a cell: literature

Can metre-to-km structure inside a 1 km cell and a depth band explain the gap between encounter-theory a and
field intake (~x0.05-0.3 on a)? Covered here: prey avoidance, anticorrelation, voids, and patches the predators fail
to track. All numbers are also in `tables/overlap_raw.csv`. Arithmetic is in `scripts/ov_factors.py`, which prints
only.

Tags:
- [F] full text read.
- [A] abstract only.
- [S] secondary, or held in a sibling note.
- [D] derived here.
- [J] judgement.

PROXY marks species outside the model's member list. Every number states its spatial scale.

## 0. How to read a number as a factor on a [D]

1. **The factor is a covariance ratio.** The model's encounter in a cell is a x mean(P) x mean(N) inside the band.
   The realised encounter is a x mean(P N). The factor on a is therefore
   `F_ov = mean(P N) / (mean P mean N) = 1 + r CV_P CV_N`.
   - This is Lloyd's/Hurlbert's interspecific index taken at the scale below the cell.
   - A bare correlation, coherence or phase sign is not a factor until the two CVs are known. Such rows are filed as
     `correlation_scale`.
   - The local index of collocation (LIC) and Schoener's overlap are normalised indices, not ratios to random.
2. **What it takes to reach x0.3** (`ov_factors.py` item 4):
   - at CV 1 for both: r = -0.70;
   - at CV 1.5: r = -0.31;
   - at CV 0.5: impossible.
   - So x0.05-0.3 means near-complete segregation inside the cell. F_ov >= 0 always.
3. **Only structure below ~1 km counts.** Haul-scale (GOV tow ~3.7 km), 3 km-grid and rectangle-scale (~55 km)
   numbers are between-cell structure that the grid can carry. They are labelled as such.
4. **Patchiness without tracking does not lower a.** If the prey is patchy and the predator is distributed
   independently of it, then mean(P N) = mean P mean N and F_ov = 1.
   - The loss is on SATURATED intake only (Jensen). With prey occupancy p and in-patch density N/p, intake falls by
     `R = (1 + x)/(1 + x/p)`, where x = a h N (item 3):
     - mean-field saturation S 0.2: p 0.1 -> 0.36; p 0.3 -> 0.68; p 0.8 -> 0.95;
     - S 0.5: p 0.1 -> 0.18; p 0.3 -> 0.46.
   - In the linear regime R -> 1.
   - This is kept apart from "factor on a" throughout. Tracking moves both toward or above 1.
5. **Overlap with existing model terms.** Each item below says whether it is already covered by one of these:
   - the schooling factor F (dilution, confusion, one meal per school met);
   - the visibility/hide floors (vigilance);
   - the vertical band overlap.

## 1. Measured predator-prey correlation and overlap at m-km scales

**Fish predators on fish prey**
- **Cod x capelin, Newfoundland (PROXY prey). Rose & Leggett 1990, Ecology 71:33 [A]** (abstract from the held
  OpenAlex record). Acoustic transects, 1985-86. This is the original of the held "anti-correlate below ~3.5 km" note.
  - "At large scales (>4-10 km) densities were in phase (positively correlated), but at scales less than aggregation
    dimensions (3.5 km), densities were out of phase (negatively correlated). At the smallest scales measured
    (2-3 km), densities exhibited maximum negative correlation in both years."
  - "When actually feeding, cod achieved coherent, in-phase distributions with capelin at scales as small as 3.5 m."
  - Scale: the negative band is 2-3.5 km, two to three model cells and so between cells. The only sub-cell statement
    is positive and comes from feeding cod.
  - No CVs are given, so it is not convertible to F_ov [D].
- **Same system, finer analysis. Horne & Schneider 1997, Phil Trans B 352:633 [A]** (abstract rebuilt from OpenAlex,
  `data/horneschneider1997_openalex`; full text not read).
  - "Spatial variance patterns in cod density were not due to aggregative responses by the predator to
    concentrations of capelin - there was no association, on average, at resolution scales from 20 m to 10 km."
  - Variance peaks at 20 m-10 km over short time scales. The spectral breaks are at 90 m for cod and 400 m for capelin
    (f^-1.08 and f^-1.1 below them, f^-0.18 and f^-0.20 above), so structure exists inside a 1 km cell.
  - The authors: "rates of interaction between predator and prey will be underestimated if local observations are
    averaged over the spatial scale of the population". That refers to population-scale averaging, not to a 1 km
    cell.
  - Reading [J]: no association means F_ov ~1 on average at 20 m-1 km. The CVs and the coherence are not in the
    abstract.
- **Barents Sea cod x capelin, spawning hotspot (PROXY prey). Pednekar et al. 2024, Commun Biol 7:1386 [F].** OAWRS
  acoustics, 15 m range resolution.
  - Diffuse cod were already "roughly 10-30 m apart" under near-surface capelin groups before dawn.
  - At dawn, capelin shoals formed above 1.4 capelin/m2, about 3x the diffuse mean. A cod shoal (critical
    0.06 cod/m2) "engulfed" them.
  - The consumption figure (10.6 million capelin in ~4 h, 3.32 million/km2, 0.1-0.2 % of the stock) is
    model-assumed: "all new cod entering the resolution cell prey on the available capelin". It is not measured.
  - Capelin in the densest regions had the highest survival.
  - Positive co-location at 10 m-10 km. A spawning hotspot, not a typical feeding state [J].
- **Cod, haddock, saithe x herring schools. Pitcher et al. 1996, ICES JMS 53:449 [A, held]** (abstract via search
  index, sibling note "Bottenfiskens och vattenpelarens golv", E16). High-resolution sonar.
  - Cod and haddock accompanied feeding herring schools. Their attacks modified the schools without dispersing them.
  - A fast saithe approach caused steep dives.
  - Metre-scale positive co-location. The school-level protection is what the model's F already carries.
- **Saithe x krill swarms x planktivores, Norwegian Sea. Kaartvedt et al. 2005, MEPS 299:1 [F, held copy].**
  - "Underneath most swarms was a 'stack' of fish that could extend almost 100 m below the krill." The fish were
    large saithe of 60-110 cm.
  - "krill swarms govern small-scale patchiness of large piscivores."
  - A piscivore collocates with its prey's food patch, which is positive for water column -> planktivores.
- **Whiting x juvenile cod. Temming et al. 2007, Ecosystems 10:865 [S, via Fauchald 2009 F].**
  - An aggregation of ">50 million juvenile cod within an area of only 18 km2 was entirely wiped out in 5 d by
    predatory whiting".
  - An aggregative response at ~4 x 4 km. The paper is not OA.

**In-region and survey-scale indices (between cells)**
- **Box GOV, cod >= 20 cm x whiting 10-28 cm [S, held; D in "Vitlingens uteblivna predation"].**
  - Hurlbert L: Q1 1.03 and Q3 1.59 in the box.
  - Within 0.1 deg cells, L_within is 1.08-1.54. The one exception is 0.55, on 20 hauls.
  - No avoidance down to the tow (~3.7 km). This is the only in-region ratio-to-random. It sits at and above the
    cell scale.
- **Celtic Sea. Trenkel et al. 2005, MEPS 299:257 [S, held F].** LIC (Bez & Rivoirard; 1 = proportional,
  0 = never together), haul scale:
  - whiting x Trisopterus 0.51 / 0.59 (spring / autumn), read as an aggregative response;
  - cod x Trisopterus 0.29-0.32;
  - "general independence between predator and prey distributions".
  - The LIC is not convertible to F_ov without the CVs.
- **North Sea cod and whiting on juvenile gadoids. Kempf et al. 2008, MEPS 367:295 [S, held F].** Schoener overlap
  per ICES rectangle (~55 km). Diet share drops fast when overlap < 0.1 together with relative prey abundance < 0.1.
  This is between-cell structure.

**Seabirds (PROXY predator) - the scale pattern**
- **Fauchald et al. 2000 (via Fauchald & Erikstad 2002) [S].** Murres x capelin: overlapping patches at ~50 km and
  > 300 km, but "similar but non-overlapping patches with a characteristic scale of a few kilometers".
- **Fauchald & Erikstad 2002, MEPS 231:279 [F].** Concordance rises with capelin patchiness. The data are aggregated
  at 9.26-37 km, so there is nothing below 9 km.
- **Fauchald 2009, MEPS 391:139 [F].** Text re-extracted without column layout.
  - "Many studies find weak positive or even negative spatial relationships between seabirds and prey".
  - "many studies find that the relationship becomes stronger at increasing measurement scales".
  - On local enhancement: "intense predation will be concentrated to a few resource patches while many patches will
    be free from predators. This type of spatial distribution of seabirds and prey has typically been found on small
    spatial scales when prey consists of pelagic schooling fish or krill".
  - This is the aggregated-predator, empty-patches pattern, not matching.
- **The seabird proxy leans weak or negative at small scales.** It combines non-overlapping few-km patches, the
  empty-patches pattern, and correlation that strengthens with scale. Seabirds detect prey from the air at km
  ranges and use flocks as cues. Transferring this to fish predators is a judgement [J].

**Planktivores x zooplankton**
- **Anchovy x zooplankton, Humboldt (PROXY). Bertrand et al. 2014, Nat Commun 5:5239 [F].** Acoustics resolve
  structures from 100 m to 100 km; internal waves dominate at ~300 m.
  - Inside structures, densities were higher by 8.8 % for zooplankton and 92 % for fish. Column biomass was higher by
    150 % and 950 %.
  - "30-45% of zooplankton biovolume and fish biomass were embedded within structures smaller than O(1 km)".
  - Two-zone arithmetic on the volumetric densities (item 2): F_ov = 1.007-1.013 at volume shares 0.1-0.5 [D].
    It captures only the in-structure vs out-of-structure covariance, not structure inside a structure.
  - The column-biomass upper bound is 1.54-1.61 [D]. It includes vertical compression, which the band overlap
    already holds.
  - The sign is positive (co-location), not avoidance.
- **Age-0 walleye pollock x zooplankton patches (PROXY). Swartzman et al. 1999, ICES JMS 56:545 [A, held OpenAlex].**
  - "When the plankton biomass was low, fish tended to remain close to existing plankton patches, while at high
    plankton biomass there was no consistent small-scale proximity relationship."
  - Tracking at low food raises F_ov above 1.
- **Herring x copepods.** I found no within-km acoustic overlap index.
  - Basin-scale positive association with overwintering Calanus (Norwegian Sea, search-result summary, [S]).
  - The coordinator's "Swalethorp" and "Maravelias 2001" hints were not verified within budget.

**Mackerel x plankton or krill.** No fine-scale source found. The only result is horizontal overlap between pelagic
stocks at survey scale (Norwegian Sea 1995-2006, [S] search summary), which is not relevant at sub-km.

## 2. Prey anti-predator spatial behaviour inside a layer

**Krill**
- **Voids beside schooling fish. Onsrud et al. 2004 [S, held F]** (Oslofjord).
  - By day: "The 'voids' in the krill recordings in the proximity of deep fish assemblages ... suggest that the krill
    performed instantaneous escape reactions."
  - Dispersed gadoids (whiting, pout) were "distributed together with the krill". No void is reported for them.
  - Vessel caveat: about 8x fish biomass beneath the anchored ship.
- **Herring schools. Christiansen et al. 2024, ICES JMS 81:1764 [S, held F in "Krillens flykt och energibudget" E9].**
  - Under schools > 10 m high, krill density fell by > 55 % in > 75 % of daytime encounters (median -68 %).
  - Under schools of 5-10 m the median fall was ~12 %. Schools < 5 m "did not induce any measurable change".
  - At night changes "centred around 0". Voids "typically lasted 5-15 min".
  - The local factor of 0.32 applies only to daytime encounters of herring in large schools.
  - Scaled to the planktivore group (herring 0.55 of biomass; schooled share 0.3-0.9 [J]; daytime share of encounter
    0.33-0.5 [J]): **F_ov 0.83-0.96 on the whole planktivores -> krill a** [D] (item 5).
    - This treats every schooled herring as sitting in a > 10 m school, so the true value lies nearer 1.
    - The sibling note "Krillparen mot vintermagar" gives x0.6-0.85 on the daytime a alone.
- **Single demersal fish.** No measured krill vigilance or avoidance distance (sibling "Aptit eller fångst i
  krillskiktet", "not found") [S].

**Herring and other schooling clupeids**
- Fast saithe approach -> steep dive. Cod and haddock alongside schools change school shape only (Pitcher 1996 [A]).
- Handegard et al. 2012 (Curr Biol 22:1213): predators hunt in coordinated groups, and prey schools fragment under
  attack. Numbers were not read: the PDF returned 403 [S, search summary].
- **Axelsen et al. 2001, Can J Zool 79:1586 [S, held A]** ("Makrillens antipredatorbeteende" E11, analog). Juvenile
  herring against puffins and predatory fish show the repertoire: bend, vacuole, hourglass, split.
- No avoidance distance in metres was found for herring facing fish predators. Sibling notes were checked:
  "Makrillens antipredatorbeteende", "Planktivorernas golv", "Fangst och stim i fiskparen", "Stimskydd i motes-a" and
  "Vilans avkastning/lit_activity_risk.md".
  - They place vigilance in the capture term (q 0.8 pre-strike escape) and schooling in F.
  - Their only foraging-vigilance evidence is a guppy analog, direction only [S].
- All of this is school-level dynamics at metres, which is what F (dilution, confusion, k meals per school met) and q
  stand for. It is not an additional overlap factor [J].

**Copepods**
- **Bollens & Frost 1991 [S, held F]** (Acartia in 3 m enclosures, PROXY). With fish, the noon-midnight mean depth
  differed by only ~0.5-1 m. Migration stopped within a day without fish and resumed in 12-24 h.
  - Vertical and sub-band. If anything it belongs in a band profile.
- **Calanus overwinters below visual depth (Kaartvedt 2000 [S, held]).** This is the band overlap.
- **Hirst & Batten 1998 [S, held].** The CPR correlation between herring and Calanus DVM is not significant once true
  day and night are used.
- No horizontal within-layer avoidance of planktivorous fish by copepods was found.
- Copepod escape jumps at mm-cm belong in the capture probability, not in overlap.

**Bed prey.** Nothing found on infauna or epifauna avoiding demersal fish at metre to km scale. Refuge and exposure
are already model terms.

## 3. Patch structure and tracking

**Direction.** Tracking raises encounter above the mean field. Untracked patchiness leaves a unchanged and lowers
only saturated intake (section 0, item 4).

- **Gadoids do track dense fish-prey patches.**
  - Whiting on juvenile cod over 18 km2 [S].
  - Cod engulfing capelin shoals at 10 m-10 km resolution [F].
  - Cod in phase with capelin at 3.5 m when feeding [A].
  - Whiting tracking Trisopterus, LIC 0.51-0.59 [S].
  - Saithe stacked under krill swarms, hunting planktivores [F].
- **On average there is no association** (Horne & Schneider 1997 [A]), with positive peaks during feeding. Rose &
  Leggett [A] and Horne 1994 [S, held: coherence "episodically concentrated at any single spatial scale"] agree.
- **Juvenile pollock** stay near plankton patches only when plankton is scarce [A].
- **Juvenile haddock vs benthos** (Rincón & Kenchington 2016, PLoS ONE 11:e0163374 [S], fetch-tool summary of the
  PMC full text).
  - Haddock were ~10x denser at High than Low sites, 10-30 km apart.
  - "the link between juvenile haddock abundance and benthic prey distribution was not consistent".
  - No fine-scale fish data.
- **Patchy schooling prey, saturating predator.** Rindorf & Gislason 2005 [S, held F].
  - Whiting's fish intake saturates at 6-17 % of the maximum ration.
  - The authors: "predators which overlapped with the prey became saturated while predators in other areas fed on
    other prey".
  - Sibling "Stimskydd i motes-a" (E18, the same paper): North Sea whiting showed no aggregative response.
  - **F does not hold this.** F is built as a factor on a: schools met at pi R_s^2 v_rel n/N, k prey per school, h
    unchanged, so intake = lambda_s k / (1 + lambda_s k h).
    - That is the linear-regime encounter with single schools.
    - It does not cap saturated intake. "Stimskydd" says so: "A window caps the saturated intake ... which a factor
      on a cannot express".
  - The plateau has two rival readings, and neither is in the model:
    - a twilight vulnerability window ("Stimskydd": K 0.10-0.17 equals 2.4-4.1 h per day);
    - clusters of schools at 100 m-km that the predators do not track (item 4: at S 0.1-0.2 and occupancy
      p 0.05-0.3, R = 0.2-0.8).
  - Both act on intake, not on a. Which one holds is not settled by any source found here.
- **Krill** (sibling "Krillens fläckighet", held [S]).
  - Acoustic occupancy at 1 km x 6 h is 0.77-0.9, giving x0.93-0.97 on whiting's potential intake.
  - Stomach occupancy is 0.1-0.3, giving x0.64-0.74.
  - This mixes occupancy with not eating krill that is present.
  - No evidence that gadoids track krill at km scale.
- **Fall & Fiksen 2020** [held, not re-read]: cited by the coordinator for the patchy-prey intake reduction.

## Not obtained (budget or paywall)

- Rose & Leggett 1990 full text (JSTOR, closed).
- Fauchald et al. 2000 full text (closed).
- Temming et al. 2007 (closed).
- Carroll et al. 2019, Global Ecol Biogeogr, a review of predator-prey overlap metrics (closed).
- Handegard et al. 2012 (403).
- Koslow 1981, Fish Bull 79, anchovy-school wake depletion of zooplankton: the NOAA URL served HTML.
- Maravelias 2001, Swalethorp, Hjellvik, Nøttestad & Axelsen: not searched within budget.

## Best estimate: residual fine-scale factor on a, per pair type

The residual is what remains after F, the hide floors and the band overlap.

| model pair type | low | central | high | basis | specific to |
|---|---|---|---|---|---|
| fish predator -> schooling clupeids (herring, sprat) | 0.7 | 1.0 | 1.5 | Horne & Schneider "no association" 20 m-10 km [A]; positive while feeding (Rose & Leggett 3.5 m [A], Pitcher [A], Pednekar [F]); low end from the seabird proxy's empty patches [S/J]; metre-scale school avoidance is already in F and q [J] | none on a. The intake-side cap (Rindorf plateau) is a separate, unassigned mechanism |
| fish predator -> pout / juvenile gadoids | 0.8 | 1.1 | 1.6 | box GOV L 1.03-1.59 (haul scale, [S/D]); whiting-Trisopterus LIC 0.51-0.59 aggregative [S]; whiting hotspot on juvenile cod [S]; nothing sub-km [J] | none; evidence leans >= 1 |
| herring / planktivores -> krill | 0.8 | 0.9 | 1.0 | Christiansen voids under schools scaled to the group, 0.83-0.96 [D on S]; voids only around schools, by day [S] | krill around fish SCHOOLS by day: the one measured within-layer avoidance |
| gadoids (whiting, water column, benthivores) -> krill | 0.8 | 1.0 | 1.2 | dispersed gadoids "distributed together with the krill", no void [S]; saithe stack under swarms [F]; no tracking evidence [S] | none measured |
| planktivores (and mackerel) -> copepods | 0.8 | 1.0 | 1.2 | anchovy-zooplankton volumetric co-location gives 1.01 [D, PROXY]; pollock track patches at low food [A, PROXY]; copepod fish-avoidance is vertical and sub-metre [S] | none |
| benthivores -> bed prey | 0.7 | 1.0 | 1.3 | no within-km measurement; haddock-benthos link inconsistent at 10-30 km [S] | pure [J] |

Closing reading [J]:
1. **No measurement supports a sub-cell overlap factor of x0.05-0.3 on a for any pair type.**
   - Reaching x0.3 needs within-cell segregation at r ~ -0.7 with CVs ~1 (section 0).
   - What exists for fish predators on fish prey is about 1 on average (Horne & Schneider) and positive while
     predators feed. The negative Rose & Leggett band (2-3.5 km) lies above the cell.
   - The only measured within-layer void is krill escaping large herring SCHOOLS by day. Diluted over the planktivore
     group and the diel cycle, it gives x0.83-0.96. It is specific to krill and schooling herring, not to dispersed
     gadoids.
   - The seabird proxy leans weak or negative at a few km: non-overlapping patches, many patches empty. It is a far
     proxy, but it is the one line of evidence that points below 1.
   - **No sub-km data at all:** benthivores -> bed prey, planktivores and mackerel -> copepods (only a PROXY anchovy
     figure), mackerel -> krill. Their rows are [J] around 1, not measurements.
2. **Patchiness can lower realised intake, not a.**
   - It needs patchy prey that predators do not track, or partial overlap with clusters of schools (Rindorf &
     Gislason's own explanation).
   - It cuts saturated intake to x0.2-0.8 at S 0.1-0.2 and occupancy 0.05-0.3, and does nothing in the linear regime.
   - It is NOT inside F, which acts on a. For whiting on clupeids it competes with the twilight-window reading as the
     cause of the 6-17 % plateau.
   - For krill, the cell-scale acoustic occupancy (0.77-0.9) makes it small: x0.93-0.97.
3. **For fish prey the evidence leans upward on a.** Aggregative responses and hotspots raise encounter. Within-cell
   overlap is an unlikely home for the a error. A saturated-intake cap from untracked school clusters or twilight
   windows is still open. So are the other terms of the a product (capture, search time, T, appetite).
