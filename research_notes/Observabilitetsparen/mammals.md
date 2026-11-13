# Observation pairs of the mammals: own cell, neighbour cells, both or neither (2026-11-13)

Research subagent. Nothing in the code, library, manifest or resume was changed. Scope: the two
mammal OBSERVERS (porpoises, seals) and the targets they observe today (`fg_library.yaml`
`<obs>_observes_<target>`, lines ~3878-3935). The reverse direction (fish sensing mammals) belongs to
another note; what was found on it is in section 6.

Grades: [F] measured (field or lab; [F/A] read as abstract or snippet only), [S] secondary (review or
earlier project note), [D] derived here from measured inputs, [J] judgement. Paper metadata in
`papers/*.json`; PDFs read: Villadsgaard et al. 2007 (`papers/10.1242_jeb.02618.pdf`) and Sveegaard et al.
2012 MEPS (`papers/10.3354_meps09959.pdf`).

Settled elsewhere and only cited here (not redone):
- movement per 1/6/24 h: research_notes/"Nettoförflyttning per funktionsgrupp" (o-umlaut), `porpoises.md`
  and `seals.md`;
- rest, hide floors, what fish sense of a mammal, bed depths D1-D6: "Sal och tumlare golv" (summary.md,
  evidence.md E1-E27);
- the coupling of hiding and observation (option A kept): "Gom och sinnen";
- porpoise diet and prey windows: "Tumlarens byten"; seal senses and light: "Sälars jakt i ljus och mörker"
  (a-umlaut, o-umlaut; `seal_senses.md`); seal diet and range: "Parametervärden för sälar i Skagerrak".

The two readings of a neighbour channel (both documented, neither chosen):
- **A** = the animal senses the neighbour cell from where it is: a cue that reaches 0.5 km (the nearest
  edge of a neighbour) to 1.5 km (its far edge); centres are 1 km away.
- **B** = during the 6 h tick the animal ranges through the neighbourhood and samples it in passing; then
  the "neighbour" value is information it gathered with short-range senses on its own track.

## 1. Movement per 6 h (the basis of reading B)

### Porpoises [S, from "Nettoförflyttning"/porpoises.md, which cites the primary sources]
- GPS path speed 0.65 m/s mean (0.15-2.8), mean turn 43 deg, 15-min fixes, Skagerrak + Wadden Sea
  (van Beest et al. 2018) [F]. Held for 6 h that is a 14 km path [D].
- Time budget at 24 h resolution (57 Argos animals, Danish/North Sea waters; Stalder et al. 2020):
  81 % area-restricted (AR), 6 % directed, 13 % uncertain [F]; daily net displacement AR ~11 km
  (mode 3-4), directed ~57 km [F, digitised].
- Per 6 h [D/J]: AR net displacement ~4-7 km (memory-free CRW upper bound 6-9 km median), path
  11-19 km; directed ~14 km (12-18). Expected 6 h net ~6 km over the time budget (order of magnitude).
- Foraging is near-continuous day and night (Wisniewska et al. 2016 [F], with the caveat that the five
  animals had just left a pound net, Hoekendijk et al. 2018; "Sal och tumlare golv" E7, E7b); 62 % of
  diving time feeding in 20 animals (Rojano-Donate et al. 2024, E6) [F].
- Revisits: tagged porpoises "often return to the same area after approximately two days" (DEPONS
  TRACE, citing satellite tracks) [S].

So in a 6 h tick a porpoise's track normally crosses several 1 km cells, and in ~81 % of ticks it is
searching (AR) while doing so. Reading B: a porpoise samples its own cell and typically one or more of
the four neighbours in every tick [D].

### Seals (harbour seal) [S, from "Nettoförflyttning"/seals.md]
- Swim speed through water 1.1-1.6 m/s (Bjorge et al. 1995), 1.4 m/s typical (Vance et al. 2021) [F].
- North Sea HMM at 2 h (Carter et al. 2025/26; 325 harbour seals) [F]: step means per 2 h transit
  6.4 km, broad ARS 2.7 km, focussed ARS 0.07 km. Stationary shares of the diving states [D]: transit
  0.20, broad ARS 0.50, focussed ARS 0.31.
- Per 6 h [D]: broad ARS 5.9 km net (p5-p95 1.9-11), transit ~17 km, focussed ARS ~0.1 km; mixed
  median ~5 km, mean 6.6 km.
- Rest at sea ~14 % of the day (Vance et al. 2021) [F]; hauled out 10-30 % of the time outside the
  moult, outside the box (E1) [S].
- Seals use memory: GPS of 29 harbour seals (Moray Firth) showed individuals "highly consistent in the
  areas they visited over two consecutive months", and both spatial memory and prey encounters raised
  the probability of starting ARS (Iorio-Merlo et al. 2022, doi 10.1098/rspb.2021.2261) [F/A].

So for seals reading B holds in broad ARS and transit (~70 % of diving time), not in focussed ARS
(~30 %), where the seal stays inside ~0.1 km and samples only its own cell [D].

### Swept area per 6 h (reading B, quantitative core) [D]
Swept area = path length x 2 x detection radius (radii from section 2; assumes the animal searches the
whole swath, which a narrow sonar beam scanned by head movement or a seal's whisker field only
approaches) [D/J]:

| observer, target class | path / 6 h | radius | swept / 6 h |
|---|---|---|---|
| porpoise, single herring-sized fish | ~14 km | ~40 m | ~1.1 km2 |
| porpoise, 25-35 cm gadoid | ~14 km | ~45-65 m | ~1.3-1.8 km2 |
| porpoise, fish school | ~14 km | ~140-260 m | ~4-7 km2 |
| seal, trail / vision / whiskers | ~20 km in water | ~15-35 m | ~0.6-1.4 km2 |

The own cell is 1 km2 and own + 4 neighbours 5 km2. A porpoise in AR, or a seal in broad ARS, sweeps
roughly one cell-area of water per tick for single fish and the whole neighbourhood for schools (porpoise
only). These are horizontal swaths; for bed prey they count only where the bed is in reach (section 3).

## 2. Senses and ranges per observer

### Porpoise
- **Echolocation (dominant).** Wild click source levels 178-205 dB re 1 uPa pp @ 1 m, mean 191,
  centred 130-140 kHz; search inter-click interval ~60 ms (30-200) (Villadsgaard et al. 2007, PDF read)
  [F]. A 60 ms ICI leaves an unambiguous two-way range of ~45 m [D], consistent with the fish ranges
  below.
  - Measured: a captive porpoise detected a 7.62 cm water-filled sphere at 26 m and a 5.08 cm sphere at
    15.9 m (50 % threshold; Kastelein et al. 1999, via Villadsgaard 2007 and the Au review) at source
    levels 165-170 dB [F/S].
  - Modelled with captive source levels: Atlantic cod 15-27 m in quiet water (Au, Benoit-Bird &
    Kastelein 2007, doi 10.1121/1.2734487) [F/A].
  - Villadsgaard et al. 2007 Table 2 (sonar equation, wild SL 191): a fish of TS -40 dB ("similar to that
    of an adult herring") at 40 m, against 10 m for captive porpoises; "very coarse estimates" by the
    authors' own words [D in source].
  - Same equation (EE = SL - 56.7 - 40 log R - 0.08 R + TS, threshold 27 dB re 1 uPa2 s from the
    hooped captive animal), run here [D]:

    | target (TS at ~120 kHz, J) | SL 178 | SL 191 (central) | SL 205 |
    |---|---|---|---|
    | small fish 3-10 cm (-55 to -48) | 9-14 m | 19-27 m | 38-53 m |
    | adult herring 25 cm (-40) | 21 m | 40 m | 76 m |
    | gadoid 25-35 cm, swimbladder (-38 to -30) | 23-35 m | 44-64 m | 83-114 m |
    | mackerel 30 cm, no swimbladder (-55 to -45) | 9-16 m | 19-31 m | 38-61 m |
    | fish school, -40 + 10 log N, N 1e3-1e5 (-10 to +10) [D] | 86-178 m | 141-259 m | 220-362 m |
    | seal body (-15 to -5) [J] | 70-105 m | 118-167 m | 190-253 m |

    Cross-check: Au 2007's 15-27 m for cod at captive levels, raised by the ~20-25 dB of wild over captive
    source levels, lands on the gadoid row at SL 191 [D]. The swimbladder carries 90-95 % of a gadoid's
    echo (Foote 1980, E13) [F/A]; mackerel has none, hence its short range.
  - Prey pursuit is short-range: the terminal approach starts at 2-4 m and ends in a buzz (Verfuss et al.
    2009, doi 10.1242/jeb.022137) [F/A]; in the wild DTAG data the prey reacted at 0.5 m and prey were
    mostly 3-10 cm (Wisniewska et al. 2016, proof read) [F].
  - Bed: a buried flatfish lies inside the sonar's bed dead zone, a swimbladdered fish on the bed above it
    ("Sal och tumlare golv" D5) [D].
- **Passive listening.** Porpoise hearing is best at 16-140 kHz and falls ~10 dB per octave below
  16 kHz; 0.5 kHz is near its lower bound (Kastelein et al. 2002, doi 10.1121/1.1480835) [F/S]. Cod
  grunts (fundamental ~50 Hz, SL ~123-127 dB re 1 uPa @ 1 m; Stanley et al. 2017 [F], Zhang et al. 2024,
  Bioacoustics, sea-cage median 123.1 dB [F/A]) therefore give a porpoise no usable cue beyond metres [J]. Herring bubble chirps are broadband
  (1.7-22 kHz) but only 73 +- 8 dB re 1 uPa rms @ 1 m (Wahlberg & Westerberg 2003) [F/A]: metres for
  one fish; a school could be louder, unquantified [J]. No study found of porpoises localising prey by
  its sounds [gap].
- **Conspecific clicks.** A click logger (T-POD) detects wild porpoises at 260-400 m (Villadsgaard 2007,
  Table 2) [D in source]; a porpoise's own hearing peaks at the click band, so conspecifics are probably
  audible at a few hundred m [J].
- **Vision.** Functional but unquantified underwater; metres at most in the dark box below ~50 m [J].
- **Smell: none.** Toothed whales lack olfactory bulbs as adults and are thought anosmic; olfactory
  genes are inactivated in Phocoena (Kishida et al. 2015, doi 10.1186/s40851-014-0002-z, and the
  OMP gene work it summarises) [F/S]. The "smell" part of the user's blend is empty for porpoises by
  evidence, not by omission.

### Harbour seal
- **Vibrissae (dominant in the dark).** Blindfolded seals follow hydrodynamic trails up to 30-35 s old
  (Dehnhardt et al. 2001; Wieskotten et al. 2010) and find flatfish breathing currents of 3.7-4.2 cm/s
  (Niesterok et al. 2017) [F, lab, n 1-3; via `seal_senses.md`]. A fish at 0.5-1 m/s leaves 15-35 m of
  trail in that window, so the search radius is tens of metres [D]. They followed a conspecific's trail
  too (Schulte-Pelkum et al. 2007) [F].
- **Vision.** Rod-dominated, pure scotopic vision proven at ~0.9 lx (n 2); at 50 m in Skagerrak water
  summer noon gives ~4.5 lx, at 100 m ~2e-4 lx (`seal_senses.md`) [F/D]. In this box (beds 44-450 m,
  median 143 m) vision is metres-scale and only in the upper ~50 m by day [D/J].
- **Passive hearing.** Best underwater 0.5-40 kHz with a gradual low-frequency decline (Kastelein et al.
  2009, doi 10.1121/1.3050283) [F/A] - far better at low frequency than the porpoise. Cod grunts: SL 127
  dB re 1 uPa @ 1 m; with ambient octave levels ~80-100 dB and a 15 dB detection threshold, a single
  grunt is audible over ~4-40 m and a chorus of 10-100 calling fish (+10-20 dB) over ~13-400 m [D, R = 10^((SL - NL - DT)/20); the seal's masked
  threshold at 50 Hz is assumed fish-like, J]. For fish listeners the same sonar sum gave effective radii
  of 1.2-21.6 m at Stellwagen Bank (Stanley et al. 2017; measured ambient noise, modelled range) [D in source]. Cod spawn Jan-Apr; spawning aggregations in
  this offshore box are unmeasured [gap]. Seals learn to home in on 69 kHz fish-tag pings (grey seals;
  Stansbury et al. 2015) [F/A] - proof of passive localisation, not of natural fish sounds. No study
  quantifies seals finding untagged herring or gadoids by sound (`seal_senses.md` Q3) [gap].
- **Porpoise clicks.** Harbour seal thresholds rise steeply above ~64 kHz (111 dB re 1 uPa at 64 kHz in
  one seal; Terhune 1988, Can. J. Zool., doi 10.1139/z88-230, via a search snippet [F/A, source not
  re-read]); seals detect tones up to 180 kHz only at high levels (Cunningham & Reichmuth 2016, doi
  10.1016/j.heares.2015.10.002) [F/A]. A 130 kHz click is therefore audible to a seal only close to the
  porpoise and near its beam axis, tens of m at most [J].
- **Smell.** Used in air; seals dive with closed nostrils, so underwater olfaction is not a prey sense
  [J, no source read].

## 3. Vertical reach (conditions "same cell")
- Porpoises: Skagerrak maximum dives 132-152 m (Teilmann 2007, van Beest 2018; "Tumlarens byten" E23)
  [F/S]; sonar adds the ranges above. Bed prey are within reach where the bed is shallower than
  ~150-200 m [D/J]. Box beds: min 44 m, median 143 m, 23 % of cells <= 100 m ("Sal och tumlare golv" D1)
  [D]: roughly half the cells.
- Seals: forage "at or close to the sea bed" (Bjorge et al. 1995, sites from kelp to 150-200 m basins)
  [F/A]; the Mareld EIA gives 40-50 m as normal coastal foraging depth [S]; "can dive to a couple of
  hundred metres" [S]. Their density decays westward from haul-outs east of the box; the western two
  thirds is 180-450 m deep ("Parametervärden för sälar" distribution note) [S/J]. Bed prey are in reach
  where the bed is shallower than ~150-200 m, mostly the eastern third [J].
- Pelagic prey (herring, sprat by day 50-150 m above the bed; Norway pout near the bed) are the mirror
  case: within a porpoise's dive plus sonar range in most cells; for seals only where the school is within
  dive depth [J; "Sal och tumlare golv" A].

## 4. Table: observer x target

Same cell = can the observer sense the target's biomass in its own 1 km2 cell within a tick. A = sensing a
neighbour from where it is (0.5-1.5 km). B = sampling the neighbours while ranging in the tick.

### Porpoises

| target (today) | same cell | sense(s), range | grade | neighbour A | sense, range | grade | neighbour B | reason |
|---|---|---|---|---|---|---|---|---|
| planktivores (on) | yes | sonar: single fish ~40 m, schools ~140-260 m (to ~360 m at max SL); hearing of chirps metres; vision metres | F (SL, sphere tests), D (ranges) | no; marginal for large schools near the cell edge | sonar on schools 0.15-0.36 km < 0.5 km | D | yes | AR track 11-19 km path, 4-7 km net per 6 h; swept 1-7 km2; herring is the prey porpoise distribution follows at 870 km2 scale (Sveegaard et al. 2012 MEPS, POR-HER partial Mantel 0.254, p < 0.0001) [F] |
| benthivores (on) | partial: beds < ~150-200 m (about half the cells); buried flatfish only at close range (dead zone) | sonar ~45-65 m for 25-35 cm gadoids; no useful hearing of grunts | D/J | no | nothing reaches 0.5 km | D | yes where beds are reachable | same track; samples bed prey only over reachable beds |
| whiting (on) | partial, as benthivores but whiting is partly off the bed | sonar ~45-65 m | D/J | no | - | D | yes where reachable | as above |
| water column (on) | partial (cue yes, use low): saithe and hake are strong swimbladder targets; depth decides | sonar ~45-110 m | D/J | no | - | D | yes | cue yes; but only 0.054 of the group is in the porpoise size window ("Tumlarens byten") |
| mackerel (on) | cue yes (weak target, ~20-30 m); no use | sonar ~19-31 m (no swimbladder) | D | no | - | D | cue yes, use none | not eaten: "rarely consumed by harbour porpoises" (Sveegaard 2012, citing Aarefjord 1995, Börjesson 2003) [F/S]; no co-distribution with porpoises (MAC-POR partial Mantel -0.127, p 0.996) [F] |
| seals (on) | cue yes (sonar would see a seal at ~120-170 m; vision near the surface); no documented use | sonar (TS J), vision | D/J | no; marginal at the cell edge | sonar 0.12-0.25 km | J | cue yes, use none | harbour seals are not known to prey on porpoises (the predator is the grey seal, not in the model; Leopold et al. 2015) [F/A, absence of records for harbour seals]; no behavioural response of porpoises to seals found [gap] |
| porpoises (self, on) | yes | conspecific clicks, sonar | D/J | yes, marginal (few hundred m) | clicks heard at ~0.26-0.4 km (T-POD analogue) | D/J | yes | not asked; listed for completeness |

### Seals

| target (today) | same cell | sense(s), range | grade | neighbour A | sense, range | grade | neighbour B | reason |
|---|---|---|---|---|---|---|---|---|
| benthivores (on) | partial: beds < ~150-200 m (east third, where seals are); buried flatfish found at cm by breathing currents | vibrissal trails tens of m; breathing currents cm; vision metres (dark below ~50 m); hearing cod grunts ~4-40 m single, ~13-400 m chorus (Jan-Apr) | F (lab senses), D (ranges) | no; marginal Jan-Apr for a spawning chorus near the edge | hearing 0.01-0.4 km, spawning only | D/J | partial (~70 % of diving ticks) | broad ARS ~6 km net / transit ~17 km per 6 h sample neighbours; focussed ARS (~30 %) stays within ~0.1 km; swept ~0.6-1.4 km2 |
| whiting (on) | partial, as benthivores | trails, vision, (hearing: whiting sound production not reviewed) | F/D/J | no | - | D | partial | as above |
| water column (on) | partial (cue yes, use low) | trails, vision | F/D/J | no | - | D | partial | 0.029 of the group in the seal window ("Tumlarens byten" open point 5; floors note) |
| planktivores (on) | partial: where schools are within dive depth | trails of schools, vision by day near the surface, chirps metres | F/D/J | no | - | D | partial | as above; schooling prey is attacked by seals (Bowen et al. 2002, E15) [F] |
| mackerel (on) | cue yes (trail, vision); no use | trails, vision | J | no | - | D | cue yes, use none | "seals do not take ... fast mackerel" (Harding 2022 presentation, in "Parametervärden för sälar" diet note) [S]; not booked as prey |
| porpoises (on) | cue weak | clicks audible only close (seal thresholds rise steeply above ~64 kHz, click at 130 kHz); trail and vision tens of m | F/A, J | no | - | J | cue yes, use none | no predation, no competition interaction documented between harbour seals and porpoises [gap] |
| seals (self, on) | yes | trails, vision, vocalisations (male roars 78-1300 Hz in summer, E20) | F/J | no | - | J | partial | not asked |
| seabirds (on, group muted) | - | - | - | - | - | - | - | inert while seabirds are muted |

## 5. Pairs with no cue or no use (candidates for "off", judged on cue and range only)
1. **porpoises -> mackerel**: cue weak (no swimbladder), use none: not eaten in Kattegat/Skagerrak
   stomachs [F/S], and no spatial association at the 870 km2 scale (MAC-POR partial Mantel -0.127,
   p 0.996) [F]. Both own cell and neighbours.
2. **seals -> mackerel**: cue at metres, use none: not taken by harbour seals [S], not booked.
3. **seals -> porpoises**: no predation or competition behaviour documented; porpoise clicks largely
   above the seal's hearing [F/A, J].
4. **porpoises -> seals**: a sonar cue exists, but no use is documented (harbour seals do not kill
   porpoises; no avoidance found) [F/A, gap].
5. **Neighbour channels under reading A, all fish targets**: no mammal sense reaches 0.5 km for a single
   fish. Marginal exceptions only: porpoise sonar on large schools (0.15-0.36 km, edge of the neighbour
   cell) and a seal hearing a spawning cod chorus (Jan-Apr). Under reading A the neighbour channels of
   both mammals are near "off"; under reading B they are "on" (porpoises always, seals in about 70 % of
   diving ticks).

Not candidates for "off": the water-column pairs have a strong cue and a small use (size windows
0.054 / 0.029), "cue yes, use low"; the bed pairs are depth-conditional, not absent.

## 6. Reverse direction (noted only)
Fish do not hear porpoise clicks (cod no response at 204-213 dB; herring no ultrasound up to 185 dB) and
harbour seals are nearly silent outside male summer roars ("Sal och tumlare golv" E8-E12, E20). The
porpoise click is narrow-band high-frequency, thought to be acoustic crypsis against killer whales (E19).
Nothing new found here.

## 7. Open points and disputed evidence
- **Sonar ranges rest on one captive threshold.** The 27 dB re 1 uPa2 s threshold is from a hooped
  porpoise in a quiet pen (Kastelein 1999); Villadsgaard et al. call their own 40 m "very coarse". Wild
  source levels span 27 dB (178-205), i.e. a factor ~3.5 in range. TS at 120-140 kHz for the model species
  was not looked up (values in the table are [J] around the 38 kHz relations); school TS by incoherent sum
  is [D] and assumes the school fills the beam.
- **Seal body TS** is [J]; no measurement found.
- **Sub-daily displacement is derived, not measured** (6 h values come from 2 h HMMs and 24 h SSMs; see
  the movement notes' gaps). Skagerrak-specific at-sea seal tracks do not exist.
- **Swept area assumes full-swath search.** A porpoise beam is ~10-16 deg wide and scanned by head
  movement; a seal's whisker field is narrower than its trail radius. The swept numbers are upper bounds
  [J].
- **Whiting sound production** not reviewed; haddock knocks (SL 119-125 dB, unpublished values used by
  Stanley 2017) and cod grunts are spawning-season sounds; spawning in this offshore box is unmeasured.
- **No quantified seal use of natural fish sounds** (only tag pings, Stansbury 2015); no porpoise use of
  prey sounds.
- **Memory and fronts - a third route** that is neither A nor B: porpoises return to areas after ~2 days
  (DEPONS TRACE) and concentrate at fronts and tidal features at hundreds of m to a few km (Johnston et al.
  2005, doi 10.3354/meps295279 [F/A]; Bailey & Thompson 2009, porpoise distribution related to habitat
  at 1-2 km grids but not at 4 km, via Sveegaard 2012 [S]); seals revisit the same areas month to month
  and memory raises ARS (Iorio-Merlo et al. 2022) [F/A]. Information about a neighbour can therefore come
  from earlier ticks. The model's policy is feed-forward; whether that route belongs in the observation
  is a design question, not settled here.
- **Co-distribution is not detection.** Sveegaard 2012 (870 km2 cells, June-August) shows porpoises where
  herring are at a scale 30x the 5 km2 neighbourhood; it says nothing about how they find them.
- **Harbour seal - porpoise interaction**: absence of evidence, not evidence of absence; the search was
  one targeted query plus the floors note.
- **Seal dive depth offshore Skagerrak** unmeasured; the "beds < ~150-200 m" condition is [J] from
  Bjorge 1995 and the EIA.
