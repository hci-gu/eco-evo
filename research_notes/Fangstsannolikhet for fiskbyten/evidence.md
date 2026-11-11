# Capture probability on fish prey - evidence (2026-11-11)

Grades: [F] field, [L] lab (full text read), [A] abstract or search summary only, [S] secondary
(cited by another source or another note), [D] derived here, [J] judgement. PPR = prey : predator
total length. "Per strike" = captures / strikes (attacks); "per sequence" = captures / attack
sequences on one prey; "per encounter" = captures / prey detected (entered the detection disc).

## Model facts (from the repository, read only)

- M1. Encounter form of every section 220 / 269 fish pair (research_notes/"Attackrater ur
  motesteori" (o-umlaut) scripts/pred_encounter_a.py, `a3d`): a = scan x A(R, H) x v_rel x T x N_t x
  p x f / H; there is no pursuit, abort or attack-choice term. p_fish = (0.1, 0.2, 0.36) cited as
  "Turesson & Bronmark 2004: single prey 0.20, pursuit 0.12, sit-and-wait 0.36; pikeperch 0.085";
  benthivores p = 0.3 (0.1-0.6) [J] (benthivores_encounter.md).
- M2. R in section 269 is rule A: R_sat = min(image-area visual range, 2 L_pred), R_nv 0.3 m on fish
  (research_notes/"En synregel for fiskparen", "Synregel i vattenpelaren"): a DETECTION distance,
  calibrated against salmonid reactive-distance data (distance of first orientation). Light acts on R,
  not on p (Synregel item 3).
- M3. Engine: fish attack every prey they meet (section 265, p_attack = 1); appetite is the
  reserve-room cap (section 240/244.10), saturation the Holling h. Schooling is a separate factor on
  the schooled share (research_notes/"Stimskydd i motes-a": k prey per school met replaces p).
- M4. Members' shares of today's A_band a (research_notes/"Overlapp i fiskparen"
  tables/band_pairs_stdout.txt): WC fish pairs saithe 0.89-0.92 (adult 43 cm 0.59-0.61, large 70 cm
  0.29-0.33), hake 0.04-0.05, ling 0.02, anglerfish / spurdog / pollack <= 0.02; WC -> mackerel
  spurdog 0.76, hake 0.22; whiting adult 1.00; mackerel 0.96, horse mackerel 0.04. Benthivore fish
  pairs are cod-weighted (one_rule_pairs.py SCORE_P cod 0.3/0.7/0.9, haddock 0.1/0.25).
- M5. Prey-size windows and realised ratios (research_notes/"Arsungarnas andel" (A-ring)
  evidence_prey_size.md): saithe mean 0.19 (pollock, Scharf 2000), Mehl 2005 median 0.27, window p95
  0.30 (0.20-0.40); whiting 20-30 cm p50 0.26-0.31, p95 0.38-0.46; cod 45-65 cm p50 0.16-0.21, p95
  0.30; mackerel window ~10 cm (p95 0.31-0.33).
- M6. budget_gate before this note: planktivore prey side 0.69 (Stimskydd); our `library` run is
  byte-identical to Stimskydd's `budget_gate_library.txt`.

## Sources

E1. Turesson H, Bronmark C 2004. Foraging behaviour and capture success in perch, pikeperch and pike
and the effects of prey density. J Fish Biol 65:363-375. doi:10.1111/j.0022-1112.2004.00455.x [L]
(PDF: research_notes/"Attackrater ur motesteori"/data/schools_capture/turesson_bronmark2004_jfb.pdf)
- Pools 2.44 m diameter, 0.35 m deep, bare; daylight 1000-5000 lx (perch, pike), dusk 500 -> < 1 lx
  (pikeperch). Predators 18-23.5 cm; roach PPR 0.27 (perch), 0.30 (pike, pikeperch).
- CS = captures / attacks; only attacks preceding the first capture counted.
- Perch 35/199 = 0.176; single prey 34/168 = 0.20; groups of >= 2 prey 1/31 = 0.03.
- Pike 0.151 (pursuit 17/144 = 0.12, sit-and-wait 8/22 = 0.36); pikeperch 0.085.
- Per attack SEQUENCE: perch 35/47 = 0.74 ("many attacks per attack sequence"; "seven perch attacked
  and captured prey in all 35 trials"); pikeperch 9.3 % ("rarely more than one attack per chase");
  pike 25 % (text "93 and 25%": the PDF text drops decimal points, cf. "34/168 = 020"; 9.3 % is
  consistent with CS 0.085 and ~1 attack per sequence) [D reading].
- The authors: "it may be more relevant to measure hunting success as the proportion of successful
  attack sequences".
- Follow-up probability after a miss implied by p_seq = p_s / (1 - r (1 - p_s)): perch r 0.93, pike
  0.47, pikeperch 0.09 [D, scripts/p_ppr.py].

E2. Juanes F, Buckel JA, Scharf FS 2002. Feeding ecology of piscivorous fishes. In Hart & Reynolds
(eds) Handbook of Fish Biology and Fisheries vol 1, ch 12. doi:10.1002/9780470693803.ch12 [S]
(PDF: research_notes/"Attackrater ur motesteori"/data/schools_capture/juanes_buckel_scharf2002_chapter.pdf)
- "Usually about half of piscivore attacks on average are successful compared with 70-80% for
  planktivorous fishes. Capture success in piscivores has also been shown to be directly related to
  relative body sizes of prey and predator, and generally declines linearly as the prey size to
  predator size ratio increases."
- "Laboratory tanks ... are generally too small to yield any realistic estimates of encounter rate,
  as all prey are usually within the visual field of a given predator."
- Sequence: "located, pursued, captured, manipulated or handled, and finally digested".
- This is the source of the "review 0.45" in research_notes/"Synregel i vattenpelaren" item 3 and
  "En synregel for fiskparen": a PER-ATTACK figure, rounded down from "about half".

E3. Ellis T, Gibson RN 1997. Size-selective predation of 0-group flatfishes on a Scottish coastal
nursery ground. Mar Ecol Prog Ser 149:83-90. doi:10.3354/meps149083 [L] (PDF:
research_notes/"Attackrater ur motesteori"/data/benth_validation/ellis_gibson1997.pdf)
- 0-group cod 6.5-14.7 cm on 0-group dab and plaice 2.3-4.7 cm, PPR 0.20-0.49; arena 60 x 33 cm,
  26 cm deep, lit from above; cod fed daily (standardised hunger).
- "Capture efficiency i.e. the reciprocal of number of attacks prior to ingestion"; first prey only.
  Fig. 3A (read by the earlier note): PPR 0.20-0.33 most trials 1.0, the rest 0.33-0.5; PPR 0.35-0.45
  0.05-0.3; r_s -0.67 (dab), -0.36 (plaice).
- "cod pursued flatfishes with a rapid acceleration and would often continue to pursue prey that
  escaped after the initial pursuit." 16 flatfishes were pursued but not ingested after capture.
- Authors: "in small, well lit, bare laboratory tanks. In the field, in low light intensities, where
  escape distance will not be so confined ... escape of prey during pursuit and ingestion" should be
  higher.
- Reading [D]: per strike on prey that is eventually eaten; per sequence ~1 in the arena.

E4. Staniland IJ 1995. PhD thesis, Univ. Leicester (figshare 10142354), ch. 4-5 [L] (PDF:
research_notes/"Attackrater ur motesteori"/data/gap_vision/staniland_whiting_thesis.pdf)
- Whiting 23.5-25.5 cm (small) and 27-30 cm (large) on shrimp 5.9-6.5 / 7.5-8.0 cm (PPR 0.2-0.34),
  lit video tank.
- Behaviours: orientation -> pursuit -> (touch) -> strike -> capture / miss; "If the fish failed to
  capture the shrimp it would either rapidly reorientate, or simply turn away."
- "The probability of attack was calculated by dividing the number of strikes by the number of
  shrimps orientated to"; "The small fish struck at almost every shrimp they orientated towards when
  they were empty".
- Probability of success per strike (model fit to Fig. 4.16): 0.8 when empty, falling with stomach
  content (slopes -0.17 large, -0.10 small shrimp per g); large whiting more successful; success rose
  with experience. The earlier note's "0.4 on large shrimp" is the large-shrimp model value s(1).
- "In their natural environment the whiting do not tend to encounter barriers such as the walls of
  tanks".
- No count of re-orientations after a miss is given in the text (flow diagrams only).

E5. Juanes F, Conover DO 1994. Piscivory and prey size selection in young-of-the-year bluefish:
predator preference or size-dependent capture success? Mar Ecol Prog Ser 114:59-69.
doi:10.3354/meps114059 [L] (papers/10.3354_meps114059.pdf, Unpaywall)
- Bluefish 90-120 / 130-160 / 170-200 mm TL, groups of 3, on Atlantic silverside 20-40 / 50-70 /
  80-100 mm, tank 121 x 64 x 58 cm.
- Attack = orientation plus S-start towards the prey; attack distance 2-48 cm (0.2-4 BL).
- "Prey items were rarely ignored"; "Unsuccessful attacks were not usually followed by pursuits;
  instead, the bluefish would temporarily (until the next attack) return to the school." Attacks
  "appeared to be directed towards isolated prey (stragglers or fringe individuals)".
- Per-attack success, single-size trials: medium bluefish 63.1 / 40.8 / 22.9 %, large 86.8 / 51.2 /
  50.4 % on small / medium / large silversides. Choice trials (Table 2, successful/unsuccessful):
  small 11/4, 6/20; medium 23/3, 9/27, 15/5, 6/15; large 3/2, 4/2, 13/3, 13/14, 8/15.
- "These results suggest that bluefish attack all prey sizes upon encounter but capture primarily
  small prey."
- Line through 17 points [D]: p_s = 0.973 - 1.230 PPR (residual SD 0.11); count-weighted choice
  trials 1.033 - 1.432 PPR. At PPR 0.20 / 0.25 / 0.30 / 0.40: 0.73 / 0.67 / 0.60 / 0.48.

E6. Scharf FS, Buckel JA, Juanes F, Conover DO 1998. Predation by juvenile piscivorous bluefish: the
influence of prey to predator size ratio and prey type on predator capture success and prey
profitability. Can J Fish Aquat Sci 55:1695-1703. doi:10.1139/f98-057 [A]
- "For each prey species, bluefish capture success declined linearly and handling time increased
  exponentially with increasing length ratios between 0.20 and 0.65"; lower success on juvenile
  striped bass than on Atlantic silverside; capture-success functions with field length
  distributions predicted diet length ratios well. No OA copy (Unpaywall).

E7. Wahl DH, Stein RA 1988. Selective predation by three esocids: the role of prey behavior and
morphology. Trans Am Fish Soc 117:142-151. [A]
- Muskellunge, northern pike, tiger muskellunge 150-225 mm, 700 L tank, 15 "optimally sized" prey.
- Mean captures per strike: fathead minnow 0.67, gizzard shad (a clupeid) 0.78, bluegill 0.14.

E8. Webb PW 1982. Avoidance responses of fathead minnow to strikes by four teleost predators.
J Comp Physiol 147:371-378. doi:10.1007/BF00609671 [A]
- 75-90 % of minnows responded to strikes by rainbow trout, smallmouth bass and rock bass, 28 % to
  tiger musky; 54-94 % of responses evaded the strike. Implied capture per strike ~0.3-0.6 for the
  three, ~0.8 for tiger musky [D, 1 - response x evasion].

E9. Scharf FS, Buckel JA, McGinn PA, Juanes F 2003. Vulnerability of marine forage fishes to
piscivory: effects of prey behavior on susceptibility to attack and capture. J Exp Mar Biol Ecol
294:41-59. doi:10.1016/S0022-0981(03)00260-0 [A]
- Bluefish and striped bass on several estuarine forage fishes; species-specific ability to avoid
  attack AND capture; prey reaction distance to an approaching predator and prey activity explain much
  of the variation; capture success differences drive profitability. No OA copy; numbers not read.
- Relevance: the step between detection and strike (prey seeing the predator first) is measurable and
  prey-specific - it belongs in a per-encounter p (q below).

E10. Herbert-Read JE et al. 2016. Proto-cooperation: group hunting sailfish improve hunting success
by alternating attacks on grouping prey. Proc R Soc B 283:20161671. doi:10.1098/rspb.2016.1671 [F]
(papers/10.1098_rspb.2016.1671.pdf)
- 51 captures / 210 attacks = 0.24 on sardine SCHOOLS; prey injured in 95 % of attacks; capture speed
  rises with school injury. A school figure (k domain of Stimskydd), not a dispersed-prey p.

E11. Langbehn TJ et al. 2019. Light comfort zone in a mesopelagic fish emerges from adaptive
behaviour along a latitudinal gradient. Mar Ecol Prog Ser 623:161-174. doi:10.3354/meps13024 [S]
(papers/10.3354_meps13024.pdf; the capture value is in Suppl. Table S1, cited by "Synregel i
vattenpelaren" 1l)
- Visual-encounter model with a capture probability 0.7 per encounter for a generic piscivore, a model
  assumption.

E12. McMahon TE, Holanov SH 1995. Foraging success of largemouth bass at different light
intensities: implications for time and depth of feeding. J Fish Biol 46:759-767 [A, via search
summary]
- Foraging success > 95 % from low daylight to moonlight, 62 % at starlight, ~0 in darkness (tank,
  combined detection + capture per trial). Light effect = detection; consistent with M2 (light in R).

E13. Howick GL, O'Brien WJ 1983. Piscivorous feeding behavior of largemouth bass: an experimental
analysis. Trans Am Fish Soc 112:508-516 [A]
- Reactive distance rises with light, prey size and motion; attacks before ingestion rise with
  bluegill size; "At high light, bluegills can spot modest-sized bass well before being spotted."
  -> pre-strike escape (q < 1) in bright light.

E14. Major PF 1978. Predator-prey interactions in two schooling fishes, Caranx ignobilis and
Stolephorus purpureus. Anim Behav 26:760-777 [A]
- Lone jacks were most successful on isolated anchovies, relatively unsuccessful on schooled ones;
  per cent captured fell with school size. Direction only (dispersed >> schooled).

E15. Andersen NG et al. 2017. MEPS 583:195 [F] (via research_notes/"Stimskydd i motes-a" E17): cod on
sprat, 87 % ingested at dusk/dawn, 1 of 194 by day. Timing evidence, not p.

Not obtained: Scharf 1998 and 2003, Webb 1982/1986, Wahl & Stein 1988, Christensen 1996 (Oikos
76:368) full texts (no OA copy via Unpaywall or not tried); Folkvord & Hunter 1986 (chub mackerel on
anchovy LARVAE, out of the size domain). No gadoid (cod, saithe, whiting, haddock, pollack, hake)
capture success on clupeids or other pelagic fish in open water or in a tank was found; no mackerel
capture success on fish was found.

## Derived quantities (scripts/p_ppr.py, scripts/gate_p.py)

D1. Per encounter: p = q p_s / (1 - r (1 - p_s)), q = P(strike | detected, hungry) = 1 - pre-strike
escape, r = P(another strike on the same prey | miss). Tank bounds: q ~1 (E4 empty whiting, E5
"rarely ignored"); r 0.09 (pikeperch) to 0.93 (perch) in pools where the prey cannot leave R (E1);
low in bluefish (E5); "reorientate or turn away" in whiting (E4); "often continue" in cod (E3).
Central open-water q 0.8 (0.6-1.0), r 0.3 (0-0.6) [J].

D2. Per-strike p_s at the pair PPR from the bluefish line (E5) for the central reading; the
cross-species bracket at PPR 0.25-0.33 is 0.2 (perch single prey; pike pursuit 0.12, pikeperch 0.085
lower) to 0.8 (cod E3, whiting E4, esocids on shad E7).

D4. Gut fullness (E4): p_s = 0.8 + s x stomach g (s -0.10 small, -0.17 large shrimp; 8 g maximum).
Read at a mean stomach of 0.22-0.53 of the maximum (the field feeding level quoted in AGENTS.md for the
benthivores): x0.77-0.47 (small shrimp), x0.62-0.11 (large). Central x0.6 [L one species and prey
type, J]. The engine has no fullness term on a (reserve_room fish: h = 1; Holling h = gut evacuation).

D3. Pair PPR (encountered prey inside the window) [D/J from M5]: WC fish pairs 0.25 (0.19-0.33),
WC -> mackerel 0.30 (0.20-0.40), whiting -> planktivores 0.28 (0.26-0.40), whiting -> benthivores
0.33 (0.27-0.42), mackerel -> planktivores 0.22 (0.15-0.31), benthivores -> planktivores 0.20
(0.16-0.30), benthivores -> whiting 0.22 (0.16-0.33).
