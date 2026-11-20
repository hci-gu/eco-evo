# Mammals: does a resting fish face less risk from harbour seals and porpoises? (2026-11-20)

Literature subagent trail. No model runs, no code or library edits. Question: the activity-risk ratio
r = risk to an active (foraging, swimming) fish / risk to the same fish at rest, per mammal and fish prey
class, with its sensory mechanism. A pair floor f is risk at rest / risk active, so f = 1/r.

Tags: [F] full text read by me (journal page or PDF; JEB pages read through a web fetch that returns
verbatim quotes), [A] abstract or database record only, [S] secondary (search snippet, review, or an
earlier project note not re-read), [D] derived here (`scripts/mammal_activity_risk.py`, stdout
`tables/mammal_activity_risk_stdout.txt`), [J] judgement. PROXY = not the model's predator or prey species.
Raw numbers: `tables/mammals_raw.csv`.

Prior project notes reused (not redone): research_notes/"Sal och tumlare golv" (summary.md, evidence.md:
E1-E27, the current floors), "Salars jakt i ljus och morker" (a-umlaut, o-umlaut;
seal_senses.md, diel_foraging.md, prey_refuge.md), "Tumlarens byten", "Daggdjurens attackrater". Their
contents enter here as [S] unless I re-read the primary source.

Citation fix: the two Niesterok et al. 2017 J Exp Biol papers are cited by DOI only below. The earlier
notes give the "2017a/b" labels and the pages 220:174 / 220:2364 inconsistently: doi 10.1242/jeb.148676
is 220:174 (detection and localisation, 3 seals); doi 10.1242/jeb.158055 is 220:2364 (threshold, 2 seals).

---

## 1. Harbour seals

### 1.1 Trail following of moving objects (vibrissae)

**Dehnhardt et al. 2001, Science 293:102-104, doi 10.1126/science.1060514** [A,
Europe PMC record]. Blindfolded harbour seals detected and accurately followed hydrodynamic trails left by a
miniature submarine; the abstract states such cues could support long-distance prey location. Trail lengths
"as long as 40 m" are given for this work in Wieskotten et al. 2010a's Introduction [F there].

**Schulte-Pelkum et al. 2007, J Exp Biol 210:781-787, doi 10.1242/jeb.02708** [F, journal page]. Two seals: one followed the swim path of the other (biogenic trail).
- Success "444 out of 495 trials" (90 %), 83.7-94.5 % per course; start-to-end 10.5-15 m; the follower
  "never deviat[ed] by more than a body length" from the generator's path; mean trail generation 13.42 s.
- Seal wake: in the first 5 s it "spread laterally mainly to about 80 cm to each side"; wake "at least 2 m
  wide"; velocities "could exceed 200 mm s-1" in the first 5 s, ~45-60 mm/s, then ~30 mm/s after 30 s.
- Fish wakes, citing Hanke et al. 2000 [S]: "The trail of a goldfish of 10 cm body length has been shown to
  last for more than 3 min under laboratory conditions."
- Discussion: trail following "might bring the seal close to its prey where it can be hunted down
  visually"; the undulating search could track "fleeing, zigzagging prey fish". No statement on stationary
  fish.

**Wieskotten et al. 2010a, J Exp Biol 213:2194, doi 10.1242/jeb.041699**
[F, journal page]. One blindfolded seal (Henry) judged the moving direction of a fin-like paddle (227 mm2,
~0.2 m/s).
- Correct choices by trail age: 5 s 93.7 %, 10 s 91.3 %, 20 s 87.5 %, 25 s 73.8 %, 30 s 75 %, 35 s
  70.6 %, >= 40 s not above chance; 35 s called "a conservative estimate".
- Wake width ~20 cm at 5 s, ~30 cm at 20 s, ~50 cm at 35 s, ~60 cm at 50 s; mean velocities > 6 mm/s at
  5 s, ~3 mm/s at 20 s, ~2 mm/s at 35 s; jet flow > 60 mm/s at 5 s, > 10 mm/s at 50 s.
- Introduction: herring, saithe and cod sustain ~0.9-1 m/s.

**Wieskotten et al. 2010b, J Exp Biol 213:3734-3740, doi 10.1242/jeb.047134**
[F, journal page]. One seal (Sam) followed submarine trails, continuous ("burst") or with the motor
stopped after 3-5 m ("glide").
- Following success burst / glide: 5 s 93.2 / 95.4 %, 10 s 79.5 / 78.2, 15 s 62.3 / 43.1 (P < 0.05),
  20 s 45.5 / 32.9, 25 s 33.3 / 21.1, 30 s 20.6 / 22.9 %.
- Glide trails weaker and narrower: ~80 mm/s at 5 s, < 40 mm/s over the next 10 s; width < 30 cm vs up to
  50 cm for burst trails.
- Authors: results "are consistent with the hypothesis that glide phases in fish ... might also constitute a
  strategy to avoid piscivorous predators through the reduction of a constant trackable hydrodynamic
  trail"; a live gliding fish should be harder still to follow than the gliding submarine.
- [D/J] This is the one direct lab test of "less locomotion, less trackable": stopping the propulsor cut
  trail-following success at 15-25 s age to ~0.6-0.7 of the continuous trail.

### 1.2 Stationary prey: breathing currents

**Niesterok et al. 2017, J Exp Biol 220:174-185, doi 10.1242/jeb.148676**
[F, journal page]. Three seals swam over a 2 x 4 m platform with eight nozzles emitting artificial flatfish
breathing currents.
- Stimulus matched to flounders: "maximal flow velocities of 20-25 cm s-1" and "a spatial extension of
  1-3 cm" at vibrissa level; nozzle mouths 2-3 cm below a mesh grid 30 cm above the platform (confirmed in
  jeb.158055's Discussion: the grid is now "25-26 cm (previously 2-3 cm) above the nozzle"), so detection
  here was at a few cm.
- Detection, constant current, blindfolded: 161/216, 224/280, 193/216 trials (chance 1/8); pulsed,
  blindfolded: 198/216, 203/224.
- Swimming speed in successful trials 20-130 cm/s (by seal), lower with pulsed currents. Background flow
  0.5-7.2 cm/s.
- Discussion: "up to 20 cm s-1 in a small flounder of 18 cm total body length"; "Fish of this size can
  produce flow velocities of up to 6 cm s-1 when performing successive tail strokes" (referent of "this
  size" not verified in the fetched text); breathing currents
  "are especially strong in flatfishes"; "Fish can respond to aversive stimuli by holding their breath for
  several seconds." Live-flatfish decay data are cited to a conference abstract (Hanke, Bublitz &
  Niesterok 2015) [not obtained].

**Niesterok et al. 2017, J Exp Biol 220:2364-2371, doi 10.1242/jeb.158055** [F, journal
page]. Two blindfolded seals; the grid kept the snout >= 23 cm (Discussion: 25-26 cm) above the nozzle; the
current crossed the grid 36 cm from the opening.
- Thresholds (peak water velocity) 4.2 cm/s (Henry) and 3.7 cm/s (Luca); signal-to-noise at threshold
  1.4-1.5.
- A 16 cm flounder's current "would drop to 2-2.5 cm s-1 at the new, increased distance (extrapolated
  values)", i.e. below threshold; the imitated fish would be "24-32 cm"; seals hunting "at some distance
  over the sea bottom rely on breathing currents of larger flounders".
- Successful near-threshold trials were slower than unsuccessful ones (Henry 19-51 cm/s, Luca 33-77 cm/s;
  P = 0.0003, 0.0131): the seal must slow down to sense a still fish.
- [D] A still flatfish of 16-30 cm is sensed only within ~0.1-0.4 m of the vibrissae, at a search speed of
  ~0.2-0.7 m/s. No measurement for gadoid opercular currents was found; flatfish currents are called
  "especially strong", so gadoids at rest are likely sensed at shorter range [J].

### 1.3 Field: what seals do with resting / cryptic vs active prey

**Bowen et al. 2002, MEPS 244:235-245, doi 10.3354/meps244235** [F, PDF read
from `research_notes/Sal och tumlare golv/papers/`, not re-downloaded]. Sable Island, 39 adult male harbour
seals, Crittercam, daylight only (05:30 to > 13:30 h), depth 30-37 m, 1094 capture attempts.
- Table 2 (p. 239; per 10-min video unit; mean +- SE): capture attempts cryptic sand lance 5.5 +- 1.1, flounder
  1.9 +- 0.3, schooling sand lance 9.2 +- 2.0; capture success 63.7 +- 5.2, 79.9 +- 10.6, 37.5 +- 6.5 %;
  search time 235, 180, 135 (column labelled min, values are seconds per unit [J: 2.8 min mean search per
  unit in the text]); pursuit 9.3, 7.2, 21.9 s; swim speed in pursuit 2.6, 2.2, 3.4 m/s.
- p. 239: Flounders "usually (14 of 15 cases) hidden ... within the sandy bottom and only became visible after the
  seal had attempted a capture"; "We could not determine what cues the seal used to detect flounders."
  Some were caught "as they moved out of the substrate to escape".
- pp. 239-240: Sand lance broken out of a school "immediately swam to the bottom, where they sat motionless or attempted
  to escape by swimming rapidly near the bottom. In either case, the seal broke off its pursuit of the school
  and pursued isolated individuals."
- pp. 239, 242-243: Seals hunted sedentary prey (cryptic sand lance, flounders) by cruising 1-2 m above the bed at about the
  minimum-cost speed, or by digging; no sit-and-wait tactic was seen.
- p. 240: foraging on cryptic vs conspicuous prey did not differ by time of day (P = 0.15, daylight
  periods only); p. 243: "different tactics might be used during darkness".
- [D] Captures per unit: cryptic sand lance 3.50, schooling sand lance 3.45, flounder 1.52. Success
  ratio schooling / cryptic 0.59, schooling / flounder 0.47; search-time ratio cryptic / schooling 1.74.
  Caveats: prey density unknown; schooling and activity are confounded (a school is both active and
  schooled); success is per attempt; one site, breeding-season males, daylight, 30-37 m.

**Davis et al. 1999, Science 283:993-996, doi 10.1126/science.283.5404.993** [A, PROXY Weddell seal].
Seals got "within centimeters" of Antarctic cod without startling them, and flushed hiding
*Pagothenia borchgrevinki* from ice crevices by blowing air into them. Hidden prey are a target, not a
refuge, for a phocid.

**Diel pattern** [S: research_notes/"Salars jakt i ljus och morker"/diel_foraging.md]. No harbour seal
study found lower foraging in darkness; benthic-feeding populations (Moray Firth, Norway) show little diel
variation in dive depth or type (Tollit et al. 1998, Bjorge et al. 1995, via Wilson et al. 2014); Svalbard
seals dived more at night in autumn (Blanchet et al. 2015). Iorio-Merlo et al. 2022, Proc R Soc B 289:
20212261, doi 10.1098/rspb.2021.2261 [F, PMC]: prey encounters on the bed in 69 % of dives, 51 586 / 78 441
encounters by two accelerometer methods, but no day/night split and no prey state. No study pairs a seal's
prey captures with the prey's activity state.

### 1.4 Derived seal numbers [D] (script parts D1, D2)

- Gas-model encounter (random swimming, both directions): ratio active/rest = <|v - u|>/v. With a seal
  searching at v 1.0-1.9 m/s and a cruising fish at u 0.2-0.5 m/s: 1.00-1.08 (3D), 1.00-1.06 (2D). Only a
  slow searcher (0.5 m/s, as near the breathing-current threshold) and a fast fish (1 m/s) give ~2.1.
  Motion alone, without a trail, barely matters.
- Trail term (2D bed plane; a still fish sensed within R_b, a moving fish's trail of length L = u T sensed
  within R_b + a, randomly oriented): detection width ratio 2.8 (low: R_b 0.4 m, u 0.2 m/s, T 10 s),
  9.4 (central: 0.25 m, 0.3 m/s, 20 s), 59 (high: 0.1 m, 0.5 m/s, 35 s). 3D (pelagic): 4.7 / 25 / 320.
  These are UPPER BOUNDS on the encounter term: 1-3 animals, still pools (background 0.5-7 cm/s), no
  competing trails; natural turbulence and the shear near the bed shorten trail life; a seal must still
  follow the trail (90 % for short fresh trails, 70 % at 35 s, 21-33 % at 25-30 s in the 2010b seal).
- Capture given encounter: in Bowen, still/cryptic prey were caught 1.7-2.1x as often per attempt as
  schooling prey (0.47-0.59): resting raises capture success.
- Product: central encounter 9.4 x success 0.59 ~ 5.5 (lab-bounded); the field check (Bowen, daylight
  with vision at 30-37 m) gives captures per unit time ~1.0 for cryptic vs schooling sand lance. In the
  box (beds 100-450 m, dark) vibrissae carry more of the search than in Bowen's daylight, which moves the
  truth from the field ~1 toward the lab bound [J].
- D2's denominator counts only the gill current. It ignores the other ways still prey are found: Bowen's
  seals cruised 1-2 m above the bed (beyond gill-current range for a 16 cm flounder) and dug, and still
  found hidden prey by an unknown cue; touch while grubbing and dim vision at the box's shallowest beds add
  more. Each lowers the ratio, which is why the central r below is far under the D2 x success product.
- Flatfish: the gill-current range was measured on flatfish currents, so the encounter term applies to
  flatfish too; but their currents are the strongest ("especially strong"), so R_b sits at the upper end
  (D2 low case, ratio 2.8), and an active flatfish swims on or just above the bed in short bouts [J],
  leaving short trails. Product with the success ratio 0.47 (schooling / flounder, not a clean active-
  flatfish comparator): ~1.3.

---

## 2. Harbour porpoises

### 2.1 Does sonar need target motion?

**DeRuiter et al. 2009, J Exp Biol 212:3100-3107, doi 10.1242/jeb.030825** [F, journal page]. Captive
porpoises caught "dead, freely drifting fish" (herring 21.0 cm, capelin 15.1 cm, sprat 12.6 cm; "dead,
frozen then thawed"): 67 of 71 trials successful; buzzes began ~1-2 body lengths from the prey; with
eyecups capture took 19.6 s vs 15.9 s without (P = 0.000027). Motionless prey are found and caught by sonar
alone.

**Target detection range** [S: IOA proceedings review by W. Au, citing Kastelein et al. 1999]: 50 %
detection of a 7.62 cm water-filled steel sphere at 26 m, a 5.08 cm sphere at ~16 m (porpoise stationed in
a hoop; a stationary target). **Au, Benoit-Bird & Kastelein 2007, J Acoust Soc Am 121:3954-3962, doi
10.1121/1.2734487** [A]: cod detection range 15-27 m in a quiet environment for porpoises; largest
broadside, smallest tail-on (aspect, not motion). One web search for Doppler or target-motion use by
porpoises found none; **Vance et al. 2021, eLife 10:e68825, doi 10.7554/eLife.68825** [A]: wild toothed
whales adjust click rate to a prey's escape within 50-200 ms, trained porpoises tracking a moving target
alike - motion is tracked after detection, not needed for it. DeRuiter 2009's motionless dead fish carry
the claim that detection needs no motion [F]; the clicks are short (~44-175 us, E11 of the floors note [S]), a time-domain sonar.
Motion enters only as aspect (a fleeing fish shows its tail, the weakest aspect) and as echo amplitude
modulation by tail beats (2.2).

### 2.2 Prey escape and capture

**Wisniewska et al. 2016, Curr Biol 26:1441-1446, doi 10.1016/j.cub.2016.03.069** [F, accepted manuscript
at `research_notes/Tumlarens byten/data/lit/wisniewska2016_accepted.txt`, re-read].
- Fig. 3 legend: "The prey appears to respond to the approaching porpoise at a distance of 65 cm from the
  sound source (50 cm from the anterior rostrum) and oscillations in the echo level thereafter indicate tail
  strokes of the escaping fish"; prey BL < 5 cm, escape speed ~1.4 m/s.
- Video S1 (night): porpoise hunting "about 2-3 m above the seafloor"; 24 buzzes in a 75-s dive, "all 24
  prey pursuits probably ended with captures".
- Text: forage "nearly continuously day and night"; near-surface, pelagic and benthic dives by day,
  "primarily pelagic dives at night". Caveat: four juveniles and one adult just out of a pound net
  (Hoekendijk et al. 2018 [S]).
- [J] The fish began to flee only at 0.5 m, after detection; before that it was not escaping. Detection did
  not depend on the prey's flight; escape enters capture p, and capture is > 0.9 either way.

### 2.3 Passive listening

**Kastelein et al. 2002, J Acoust Soc Am 112:334-344, doi 10.1121/1.1480835** [A]: best sensitivity
~33 dB re 1 uPa at 100-140 kHz, within 10 dB over 16-140 kHz, falling ~10 dB/octave below 16 kHz.
[D] Extrapolated: ~83 dB at 1 kHz, ~93 dB at 500 Hz, ~103 dB at 250 Hz.
**Gadoid sounds** [S, search snippet of a 2017 Sci Rep paper, doi 10.1038/s41598-017-14743-9; dosits.org]:
cod grunts ~53 Hz fundamental; haddock knocks peak 184-356 Hz (mean 258); haddock spawning rumble 30-400
Hz; sounds tied to spawning and courtship. [J] Below the porpoise's sensitive band and seasonal; a
listening channel for resting vs active gadoids is not supported.
**Wilson, Batty & Dill 2004, Proc R Soc B 271 (Suppl 3):S95-S97, doi 10.1098/rsbl.2003.0107** [A]:
Pacific herring "Fast Repetitive Tick" sounds, broadband 1.7-22 kHz, 0.6-7.6 s, mostly at night, more per
capita at higher density; Atlantic herring too (bubble release from the anal duct). [J] Inside porpoise
hearing; a night cue of herring that are not feeding (night) - if used, it makes inactive-period herring
MORE detectable.
**Gannon et al. 2005, Anim Behav 69:709-720, doi 10.1016/j.anbehav.2004.06.020** [A/S, PROXY bottlenose
dolphin]: dolphins use passive listening extensively in the search phase; soniferous fishes ~6 % of
available prey but over half of the diet (Wells et al. 2009, via snippet [S]). No porpoise equivalent found.

### 2.4 Diel foraging and what the prey were doing

- Rojano-Donate et al. 2024 [S, floors note E6]: feeding 37 % pelagic + 25 % bottom of dive time; feeding
  dives 65 % nocturnal; buzz rates day 48 /h, night 142 /h (Daggdjurens attackrater summary [S]).
- **Schaffeld et al. 2016, MEPS 547:257-272, doi 10.3354/meps11627** [S, search snippet of the abstract]:
  western Baltic, five sites, one year of acoustic data; detections diel at 4 sites, but "foraging behaviour
  showed only nocturnal rhythms"; hypothesis: benthic prey in shallow water by day, pelagic in deeper water
  at night.
- [J] At night herring and sprat are dispersed and not feeding (visual feeders), i.e. in their
  inactive state; porpoise foraging peaks then. This is confounded with prey dispersal, vertical position
  and light (for the prey's own vigilance); it supports at most a sign: porpoises do not avoid inactive
  prey, and the night peak suggests the inactive period is the riskier one (r <= 1) for pelagic prey.

### 2.5 Buried and bed-bound prey

- Schneider von Deimling et al. 2023 [S, floors note E14]: > 40 000 seabed pits attributed to porpoises
  flushing sandeels in 20-40 m (inference from pit morphology).
- Rossbach & Herzing 1997, Mar Mamm Sci 13:498-504, doi 10.1111/j.1748-7692.1997.tb00658.x [S, PROXY
  bottlenose dolphin, via secondary descriptions]: "crater feeding" on fish buried in sand with buzz click
  trains; whether echoes alone find the buried fish is debated.
- Floors note D5 [S]: sonar bed dead zone ~3-13 cm plus 1-5 cm beam footprint at 1-5 m; a gadoid on the
  bed stands above it, a buried flatfish is inside it.

---

## 3. What "rest" means per prey class [J]

- Bed gadoids (cod, haddock): hovering or lying on/near the bed with fins still; ventilation continues.
- Flatfish: buried or lying flat; ventilation continues (breathing current strong, may pause briefly).
- Whiting: loose aggregations near the bed, swimming slowly; does not bury.
- Water column (saithe, hake): saithe keep swimming in schools; hake rest near the bed by day.
- Planktivores (herring, sprat; Norway pout near the bed): herring and sprat swim continuously in schools
  whether feeding or not; their non-feeding period is the night (dispersed). Pout as the gadoids.

## 4. Activity-risk ratio r (active / rest) per mammal x prey class [J unless tagged]

| pair | r central (range) | f = 1/r | mechanism | load-bearing sources |
|---|---|---|---|---|
| seal -> bed gadoids | 2 (1-5) | 0.5 (0.2-1) | moving fish leave a trail sensed 10s of m away; a still one only by its gill current within ~0.1-0.4 m; resting raises capture p | Wieskotten 2010a,b [F]; Niesterok jeb.158055 [F]; Bowen 2002 [F]; D2 |
| seal -> flatfish | 1.3 (0.6-4) | 0.75 (0.25-1.7) | trail vs gill current applies, but flatfish currents are the strongest (largest R_b) and an active flatfish leaves short trails [J]; hidden flounders caught at 80 % per attempt (cue unknown, fewest attempts 1.9/unit, density unknown) | Bowen 2002 [F]; Niesterok jeb.148676, jeb.158055 [F]; D2 low case |
| seal -> whiting | 1.5 (1-4) | 0.7 (0.25-1) | as gadoids, but a resting whiting still swims slowly (part of the trail stays) | as gadoids; [J] |
| seal -> saithe / hake | 1.3 (1-3) | 0.75 (0.33-1) | saithe swim in schools at rest (trail persists); hake rests on the bed (as gadoids) | Schulte-Pelkum 2007 [F]; [J] |
| seal -> planktivores | 1.1 (0.8-2) | 0.9 (0.5-1.25) | rest itself: schools swim continuously, the trail does not switch off; schooling, not rest, lowers capture (0.59 of cryptic). Today's 0.35 is, per research_notes/"Planktivorernas golv" summary 4, a stand-in for SCHOOLING protection while the seal a has no schooling factor F - a different quantity; this row does not argue against it | Bowen 2002 [F]; [J] |
| porpoise -> bed gadoids | 1.1 (1-1.5) | 0.9 (0.67-1) | sonar finds still targets; only the gas-model term (1.0-1.1) and bed proximity | DeRuiter 2009 [F]; Au 2007 [A]; D1 |
| porpoise -> flatfish | 2 (1.2-3.5) | 0.5 (0.3-0.8) | buried flatfish sit in the bed echo's dead zone; partial flushing by grubbing | floors note D5, E14 [S]; [J] |
| porpoise -> whiting | 1.05 (1-1.3) | 0.95 (0.77-1) | off-bed, swimbladder, sonar motion-blind | DeRuiter 2009 [F]; Wisniewska 2016 [F] |
| porpoise -> saithe / hake | 1.05 (1-1.3) | 0.95 (0.77-1) | as whiting | as whiting |
| porpoise -> planktivores | 1 (0.6-1.2) | 1 (0.8-1.7) | sonar motion-blind; night (non-feeding) herring dispersed, possibly ticking (FRT 1.7-22 kHz), and porpoise foraging peaks at night | Wilson et al. 2004 [A]; Schaffeld 2016 [S]; Rojano-Donate 2024 [S] |

Notes on the table:
- Seals -> benthivores as one pair: at the seal-window weights 0.57 gadoids / 0.43 flatfish (floors note
  D6 [S]), f ~ 0.57 x 0.5 + 0.43 x 0.75 = 0.61 (range ~0.2-1.3) [D/J], against today's 0.9.
- Seals -> bed gadoids central r 2 is a [J] between the lab-bounded product (~5.5) and the daylight
  field value (~1). What is new since the 0.9 of 2026-11-11: the glide experiment (Wieskotten 2010b:
  stopping propulsion cuts trail following to ~0.6-0.7 at 15-25 s) and the measured gill-current range
  (jeb.158055: a 16 cm flounder's current is below threshold at ~0.36 m). The 11-11 note judged rest on
  the bed against burial only.
- Seal gadoid/whiting values rest on lab sensory physics (1-3 seals, still water, artificial trails) set
  against one daylight field study with confounded comparisons; the range is wide on purpose.
- r < 1 (rest riskier) is supported only weakly: seals on buried flatfish (capture success) and porpoises
  on night herring (diel timing). Neither is a clean rest-vs-active comparison.

## 5. Not found, not obtained, not searched

Not found (searched):
- A measure of seal or porpoise capture RATE on the same fish species resting vs active (two web searches,
  both on seals: video / accelerometer field studies; none on porpoises beyond the diel sources).
- A seal's detection distance for a live swimming fish's wake (all trails were from paddles, submarines or
  seals); Wieskotten 2010b explicitly calls for live-fish trail experiments.
- Gadoid (non-flatfish) breathing-current velocity vs distance; live-flatfish decay curves (cited only to a
  2015 conference abstract).
- Porpoise use of Doppler or target motion for detection (one search; none); porpoise passive listening
  to fish sounds (one search; none).
- Seal diel prey-capture rates in the North Sea or Skagerrak (Vance et al. 2021 Sci Rep open data could give
  it; not analysed).

Not searched: target strength of the same fish swimming vs still (only Au 2007's aspect dependence [A]).

Task-named or cited sources, status:
- Read [F]: Schulte-Pelkum 2007, Wieskotten 2010a, 2010b, both Niesterok 2017, Bowen 2002, DeRuiter 2009,
  Wisniewska 2016, Iorio-Merlo 2022.
- Abstract or record only [A]: Dehnhardt 2001, Davis 1999, Au 2007, Kastelein 2002, Wilson 2004, Vance 2021
  eLife.
- Secondary only [S]: Gannon 2005, Schaffeld 2016, Rojano-Donate 2024, Kastelein 1999 (26 m sphere),
  Hanke et al. 2000 (goldfish wake > 3 min), Ydesen et al. 2014 (doi 10.1242/jeb.100016; lab jerk
  validation, one seal, no prey-state content, via seal_senses.md), Rossbach & Herzing 1997, gadoid sound
  frequencies (2017 Sci Rep snippet).
- Not obtained: Hanke & Bleckmann 2004 (doi 10.1242/jeb.00922; no OA copy), Wieskotten et al. 2011 (doi
  10.1242/jeb.053926; OA record saved, PDF refused, not read - size/shape discrimination, judged marginal).
- Not searched: Kuhn (seal stomach-temperature / foraging), Volpov (pinniped video), Linnenschmidt et al.
  2013 (doi 10.1111/j.1748-7692.2012.00592.x; OA record in the floors note says closed).

Bibliography: first authors, years, volumes and page ranges checked against Crossref for jeb.148676,
jeb.158055, jeb.02708, jeb.047134, meps244235, jeb.030825; co-author lists not verified, so given as et al.
