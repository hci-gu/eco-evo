# Addendum: vision, light and capture gaps in the encounter-derived attack rates

Follow-up to `summary.md`, `planktivores_encounter.md`, `benthivores_encounter.md` and
`predators_encounter.md` (2026-10-25). Limited budget (15 web searches, 25 fetches). Nothing in the
library, code, manifest, resume, notes or scripts was changed; every implied change in `a` below is
hand arithmetic on the factors those notes give (encounter a ~ R^2 x v x p x f x O / H, so a factor on
R changes a by its square). Downloads: `data/gap_vision/`. Tags: [F] full text/data, [A] abstract,
[S] secondary, [D] computed here, J judgement.

Sections in order of the work: gap 4 (verification), 1, 2, 3, 5; then a summary table and the list of what is still missing.

## Gap 4. Verification of two flagged items

### 4a. Beauchamp et al. 1999 reactive-distance equation (predators -> planktivores, day R)

- Eckmann et al. 2016 [F, `data/vision/eckmann_2016.pdf`, Methods eq. 3] quote, citing Beauchamp,
  Baldwin, Vogel & Gubala 1999 and Vogel & Beauchamp 1999:
  RD = 0.25490 x I^0.4747 (m, I in lx at depth) for I <= 17.8 lx, RD = RDmax = 1.012 m above.
  The note's "0.2549 I^0.4747" is copied correctly.
- Internal check [D]: 0.2549 x 17.8^0.4747 = 0.2549 x 3.923 = 1.000 m, so the cap and the power
  law meet at 17.8 lx; the equation is self-consistent.
- What it is: a lake-trout-on-fish-prey fit (Vogel & Beauchamp 1999, lab, salmonid prey), used by
  Beauchamp et al. for pelagic piscivores in lakes. Its form has no Ke: R^2 ~ I^0.95, i.e. R^2 is
  close to linear in light below 17.8 lx and flat above. At 0.01 lx it gives RD 0.029 m, at 0.1 lx
  0.086 m, at 1 lx 0.255 m [D]. It is not a gadoid value (the predator note already tags it [S/F]
  and uses it only for the light shape). Original not re-fetched (budget; see "still missing").
- Consequence: confirmed, no change to the predator note's day R (central 0.5 m, which the
  equation gives at ~4.4 lx [D: (0.5/0.2549)^(1/0.4747) = 4.1 lx]).

### 4b. Mackerel feeding light threshold (Macy, Sutherland & Durbin 1998, MEPS 172:89)

- Read from the rendered abstract image (300 dpi) of the local PDF
  (`data/mackerel/macy1998_meps172_089.pdf`) [F]: "Feeding rate did not change significantly
  between 1.6 x 10^0 and 2.0 x 10^-6 uE m-2 s-1, but decreased to nearly zero at 8.2 x 10^-8
  uE m-2 s-1, indicating a light intensity threshold for feeding of about 10^-7 uE m-2 s-1."
  The exponent -7 is confirmed. Light levels tested: 1.6, 2.7e-4, 1.6e-5, 2.0e-6 and 8.2e-8
  (lights off) uE m-2 s-1.
- Discussion [F]: "At this threshold, feeding rates are 50% of the feeding rate in full light. An
  absolute threshold, at which the feeding rate is zero, was not found." The fish appeared to be FILTER
  feeding on small/intermediate copepods in these trials.
- Caveat [D]: the 50 % point is interpolated, not measured: it lies somewhere in the 1.4-decade gap
  between 2.0e-6 (no effect) and 8.2e-8 (near zero). 1e-7 uE m-2 s-1 is ~5e-6 lx at ~50 lx per
  umol for white light (fewer lx per umol for blue-green water light), i.e. 2-3 decades below
  the herring biting threshold of 1e-3 lx (Batty et al. 1990 [S]) and which Macy et al. call "of the
  same order of magnitude" as the mackerel schooling threshold (1.8 x 10^-? uE m-2 s-1, Glass et
  al. 1986, via Macy [S]; exponent not legible in the OCR, ~1e-7 by that statement).
- Consequence: confirmed. The threshold shows that mackerel CAN feed (filter) at night near the
  surface, so the predator note's mackerel night factor 0.2 (Dahl & Kirkegaard 1986 diets) is a
  behavioural / overlap factor, not a visual one; keep it (field data outrank a lab threshold).
  Sensitivity only [D, from `tables/pred_encounter_stage_detail.csv`]: with the night factor at 1
  instead of 0.2, mackerel's krill term goes from (0.00393 + 0.00029)/2 = 0.0021 to
  (0.00393 + 0.00147)/2 = 0.0027 (x1.28); mackerel carry 23 % of the pair, so
  predators -> zooplankton a x1.06: 0.0015 -> 0.0016. Negligible.

## Gap 1. Herring reactive distance, capture success and light (planktivores -> zooplankton)

Found:
- **Varpe & Fiksen 2010 (Ecology 91:311) parameter table: NOT obtained.** Unpaywall lists only a
  Wiley "pdfdirect" bronze copy, which returns a bot-check page; the ESA full-text page is HTTP
  403; Varpe's 2007 thesis (local, `data/herring/varpe2007_thesis.pdf`) holds only the
  introduction, not the paper. The 4 mm vs 1.5 mm reference-prey conflict therefore stands:
  van Deurs et al. 2015 [F, local `data/herring/vandeurs2015.txt` l.195-196] state "1 body length
  when light is not limiting and prey is 1.5 mm copepods (Varpe & Fiksen 2010)" with contrast
  C = 0.3. No change; the k_v lever (x0.28-x3.5 on herring a) remains the largest single one.
- **Gibson & Ezzi 1985, 1990, 1992 and Batty et al. 1986, 1990: closed** (Unpaywall is_oa false
  for 10.1007/BF00397157 and 10.1007/BF01313419). Secondary values from Kulke 2018 (local
  `../Fiskgrupper i rutan/planktivores/data/ediss7589/Dissertation.pdf`) [S]:
  - Gibson & Ezzi 1992: 13-20 cm herring, maximum biting rate ~1.0 s-1 on Calanus finmarchicus and
    ~1.5 s-1 on Artemia nauplii, at up to ~1000 prey/L. This is the bite-rate ceiling (h side),
    consistent with the 0.85-1 bites/s already used in planktivores_encounter 2.5; it does not
    touch a.
  - Batty et al. 1990: particulate (biting) threshold ~0.001 lx on large prey (Calanus) and 0.01 lx
    on Artemia-sized prey (0.5-0.9 mm); herring were size-selective, taking the larger organism
    first. Same values as the planktivore note; still [S], not upgraded to [F].
  - Kulke: Gibson & Ezzi (1992) and Bernreuther et al. (2008) "supposed a type II response for
    herring", uncertain at low density; Kulke's own data: herring bite faster than sprat below
    ~40 copepods/L (Acartia). Supports the Type II form on the pair, no number for a.
- **Lake herring (Coregonus artedi, a coregonid proxy, not Clupea) reactive volume vs light**
  (Link & Edsall 1996, Hydrobiologia 332:131) [A, via USGS record]: 2-1500 lx on Limnocalanus
  macrurus at 10-13 C; reactive distance and volume "significantly smaller at 2-10 lx than at
  40-1500 lx"; they "forage most effectively at light levels above 10 lx".
  - Factor: herring Ke (light half-saturation of R^2), central 1 lx (0.1-10) J in the note. A
    planktivore that is still visibly below saturation at 2-10 lx suggests Ke nearer 10 lx than
    1 lx (J; the attack geometry, not a Ke fit, was reported).
  - Implied change [D, from the note's one-at-a-time table, Ke at its high end = x0.87]:
    herring a 0.0083 -> 0.0072; group a 0.0070 -> ~0.0062 (herring 93 % of the group a).
    Within the existing MC band. The night/day ratio and the -0.6 to -2.3 deg half-point would move
    slightly toward the day side (higher sun needed), i.e. further from the engine's -8 deg.
- **Adult herring search speed and attack geometry** (Thetmeyer & Kils 1995, MEPS 126:1)
  [F, `data/gap_vision/thetmeyer_kils1995_meps126_1.pdf`]:
  - 27.4 cm (22.8-30.1) Kiel herring in a 10-11 m3 tank at ~8300 lx cruised at 1.0 BL/s
    (SD 0.22, n 20), attack bursts 3.5 BL/s (max 5.6). Confirms v 1.0 BL/s (central) [F].
  - Attacks on mysids came from below at 47.5 deg (SD 9.7, n 30); juvenile herring (3.9 cm) on
    copepods in situ at 42.8 deg (n 261). Herring see prey against the downwelling light, from
    below (prey silhouettes; mysids poorly visible head-on). This supports the half-visual-field
    f 0.5 (or less) rather than f 1, i.e. it argues against the "k_v high + f 1" route to 0.04
    (planktivores_encounter 2.9). No change to central a.
  - No reactive distance was measured.
- Herring capture success p: no herring value found (central 0.6 J stands). Proxy:
  Thetmeyer & Kils give none; Kulke none for herring.

Net for the pair: central a 0.007 unchanged to slightly lower (0.006-0.007 with Ke ~10 lx);
the dominant uncertainty (reactive distance / reference prey, k_v) is NOT closed.

## Gap 2. Gadoid detection range and capture success on fish (benthivores and predators -> fish)

Found:
- **Meager et al. 2010 (juvenile cod R vs light) and 2005 (turbidity): closed** (Unpaywall
  is_oa false; the USC repository record links only the publisher DOI). Abstract numbers are
  already in `../Torskfiskars ljusberoende vid jakt/gadoid_vision_thresholds.md`. Two attempts,
  stopped.
- **Cod per-attack capture efficiency on fish prey** (Ellis & Gibson 1997, MEPS 149:83, Fig. 3A)
  [F, read from the figure of the local PDF `data/benth_validation/ellis_gibson1997.pdf`]:
  0-group cod 6.5-14.7 cm on 0-group dab and plaice 2.3-4.7 cm, small bare lit tanks.
  Capture efficiency = 1 / number of attacks before ingestion, first prey:
  - prey:predator length ratio 0.20-0.33: most trials 1.0, the rest 0.33-0.5;
  - ratio 0.35-0.45: 0.05-0.3 (Spearman r -0.67 dab, -0.36 plaice);
  - the authors: in the field, "in low light intensities, where escape distance will not be so
    confined", escape during pursuit and ingestion should be higher.
  - Factor: p on fish prey (benthivores -> planktivores / predators central 0.3 (0.1-0.6) J;
    predators -> planktivores single-fish 0.2 (0.1-0.36)). Box prey:predator ratios: herring
    20-25 cm vs cod 68 cm = 0.3-0.37; adult whiting 25 cm on 10-15 cm sprat/herring = 0.4-0.6.
  - Reading [D/J]: in bright, confined tanks a cod at ratio <= 0.33 needs ~1-2 attacks per prey
    (p 0.5-1); at ratios 0.4-0.6, which is where adult whiting (68 % of predators -> benthivores,
    46 % of predators -> planktivores) feed on clupeids, p falls to 0.05-0.3. Per ENCOUNTER (not
    every encounter is attacked; dim light; open water; schools) the value is lower still. The
    lab data bracket the notes' central values from above; they do not argue for a change of
    central p.
  - Upper-bound arithmetic: p 0.3 -> 0.6 on the benthivore fish pairs gives a x2
    (0.016 -> 0.032 and 0.022 -> 0.044); predators -> planktivores p 0.2 -> 0.3 gives x1.5
    (0.0056 -> 0.0084). These are ceilings, not proposals.
- **Whiting strike success on shrimp** (Staniland 1995, PhD thesis, Univ. Leicester; figshare
  10142354) [F, `data/gap_vision/staniland_whiting_thesis.pdf`, ch. 4-5]:
  - 23.5-25.5 cm ("small") and 27-30 cm ("large") whiting, shrimps 5.9-6.5 cm (small) and
    7.5-8.0 cm (large) (prey:predator length ratio 0.2-0.34), lit video tank.
  - Probability that a strike ends in capture (small whiting, empty stomach), as fitted for the
    thesis's dynamic model: 0.8 on small shrimp, 0.4 on large shrimp, falling by 0.05 per unit
    of stomach content; large whiting were more successful, and success rose with experience.
    The probability of attacking a prey orientated to was ~1 when the stomach was empty.
  - Detection: "The main methods used to detect the prey were probably sight and vibration as the
    response usually occurred to a moving shrimp."
  - Factor: predators -> zooplankton krill p 0.5 (0.3-0.7) J and predators/benthivores on
    epifauna p 0.3-0.4 J. A hungry whiting's 0.4-0.8 per strike on shrimp of 0.2-0.34 of its
    length supports the krill p 0.5-0.8 (krill 3 cm vs 25 cm whiting = ratio 0.12, easier
    still) and epifauna p at the upper part of 0.3-0.4.
  - Implied change [D]: krill p 0.5 -> 0.8 is x1.6 on the krill part of predators ->
    zooplankton, which carries most of the pair (krill a terms >> copepod terms in
    `tables/pred_encounter_stage_detail.csv`): pair a 0.0015 -> <= 0.0024. Epifauna p 0.3 -> 0.4
    moves the benthos pairs < 10 %; they are saturated either way.
- **Gadoid capture success on clupeid schools: not found** (one search). No per-attack rate for
  saithe, cod or whiting on herring schools surfaced; the school-yield k = 1 (0.3-3) J stands.
- **Lateral-line detection range for gadoids (R_nv = 0.6 L J): no gadoid value found.**
- **Lateral-line detection range (general, not gadoid)** [S, search-result level only]: adult
  oyster toadfish (33 cm TL) detected mobile prey only within ~40 % of body length (J Exp Biol
  2005, toadfish lateral-line study); an unattributed statement (two artificial-lateral-line
  patents) gives "one or two body lengths" for dipole-guided attacks.
  - Factor: R_nv = 0.6 L (0.3-1.0) J in the benthivore note; 0.4 L is inside that range.
  - Implied change [D]: with R_nv 0.4 L, q = (0.4/2)^2 = 0.04 instead of 0.09. The day visual
    term (0.36 = 0.09 + 0.91 x 0.297) becomes 0.04 + 0.96 x 0.297 = 0.325 and the night term
    0.09 -> 0.04, so the time-mean R^2/R_sat^2 goes 0.225 -> 0.183 (x0.81). Benthivores ->
    planktivores (prey mostly in the band by day) moves less, ~x0.9: 0.016 -> ~0.014;
    benthivores -> predators ~x0.85: 0.022 -> ~0.019. Not a recommendation; R_nv stays J.

## Gap 3. Light at depth in the Skagerrak (bottom light for the benthivore fish pairs)

Found (two searches failed to find a measured Skagerrak / Norwegian Trench PAR profile below the
pycnocline; the values below are from PDFs already on disk that the notes had not used):
- **Skagerrak surface-layer attenuation** (Arst et al. 1993, Proc. Estonian Acad. Sci. Ecol.
  3:38-48) [F, `data/benth_light_layers/Arst1993_Skagerrak_irradiance.pdf`]: pyranometer profiles
  of integral solar (shortwave) irradiance, RV Arnold Veimer, summers 1989-1990. Mean depth of
  1 % of the sub-surface irradiance, z(1%): **23 m in Skagerrak waters** (11 m Gotland Deep,
  5.5 m Kunda Bay).
  - [D] Mean K over 0-23 m = ln(100)/23 = **0.20 m-1** for total shortwave. K(PAR) is lower still
    over that layer (the near-infrared part is absorbed in the top metres), so 0.20 is an upper
    bound for Kd(PAR).
  - Factor: the 0-20 m Kd in the planktivore and benthivore notes, central 0.30. That value is
    itself [F]-based: `scripts/light_at_depth.py` takes Aas et al. 2014's regression
    Zq(1%) = 1.8 x Secchi with the Oslofjord-Skagerrak Secchi 8.3 +- 2.8 m, i.e. Kq = 4.6/14.9 =
    0.31 (0.23-0.46). The predator note used 0.15-0.20 near the surface.
  - **This is a conflict between two [F] sources, not a correction.** Arst's total-shortwave
    1 % depth (23 m) should be SHALLOWER than the PAR 1 % depth, yet it is deeper than Aas's PAR
    1 % depth (~15 m). Likely reasons (J): Arst is one July-cruise mean in open Skagerrak water;
    Aas's Secchi mean includes the darker Oslofjord / coastal stations. Spring-bloom Kd in the top
    20 m can reach 0.3-0.5, so an annual-mean surface K for an annual-mean `a` lies between ~0.20
    (Arst, summer, offshore) and 0.30 (Aas, area mean); 0.25 is a fair middle (J).
  - Implied change [D]: replacing 0.30 by 0.20 over 0-20 m multiplies the light everywhere below
    20 m by exp(0.10 x 20) = **x7.4**; by 0.25, exp(0.05 x 20) = **x2.7**. For the clupeids at 15 m (planktivores): the day R^2 factor
    is already 0.97, so herring a rises by < 5 % (the note's foraging-depth lever is x0.89-1.08).
    For the bottom (benthivores, median cell 142 m): central daytime bottom light 3e-3 lx ->
    ~0.02 lx, i.e. about Ke (0.02 lx). Rough arithmetic on the benthivore note's numbers
    (J for the saturation step): mean daytime E/(E+Ke) 0.30 -> ~0.55, day R^2/R_sat^2 0.36 ->
    ~0.59, time mean 0.225 -> ~0.34 (x1.5; x1.6 for the day-weighted planktivore pair):
    benthivores -> planktivores 0.016 -> ~0.025, benthivores -> predators 0.022 -> ~0.033.
    With K 0.25 (light x2.7): E/(E+Ke) ~0.30 -> ~0.42, day 0.36 -> ~0.47, time mean ~0.28,
    i.e. x1.2-1.3: 0.016 -> ~0.020, 0.022 -> ~0.027.
    A proper value needs the sibling script re-run with Kd(0-20 m) = 0.20; not done here
    (scripts are not to be edited or re-run under this task).
- **Clearest-case deep attenuation** (Kaartvedt, Røstad & Aksnes 2017, MEPS 574:259) [F,
  `data/benth_light_layers/Kaartvedt2017_weather.pdf`]: Red Sea, 700 m station, PAR (400-700 nm)
  profiles: K = 0.063 +- 0.001 m-1 over 0-100 m and **0.038 +- 0.002 m-1 over 100-275 m**.
  - This is oligotrophic, among the clearest ocean water; Skagerrak deep water (Atlantic water
    under the Norwegian Coastal Current) cannot be clearer. So K(> 50 m) ~0.04 (the benthivore
    note's low-K profile, and the Aksnes 2009 derived Skagerrak 0.024-0.037) is a floor, not a
    likely value.
- **Fjord deep K from isolume tracking** (Staby et al. 2011, MEPS 422:265) [F,
  `data/benth_light_layers/Staby2011_MEPS422_265.pdf`]: Masfjorden, the slope of Maurolicus
  isolume depth vs ln(surface irradiance) implies K = 0.054-0.073 m-1 (March) and 0.047-0.071
  m-1 (September) in the depth range of the layer. With Røstad et al. 2016 (0.060 > 50 m, already
  in the note), this brackets the central deep K 0.06 (0.04-0.08) well.
- Net [D/J]: the deep K is now bracketed 0.04 (clearest-ocean floor) to ~0.07 (fjord), central
  0.06 stands; the surface-layer K is in conflict, 0.20 (Arst, summer offshore) vs 0.30 (Aas,
  area mean), annual central ~0.25 (x2.7 more light at depth than the notes' central). This pulls
  the benthivore fish-pair light lever (0.74 decades) toward the low-K side: a x1.2-1.3
  (x1.5 if 0.20 holds all year) on both benthivore fish pairs, and the effective light threshold
  for vision at the median bottom stays >= 0 deg (daytime bottom light ~8e-3 lx at K 0.25, ~0.02
  lx = Ke at K 0.20; vision at the median bottom stays dim-light limited all day). Still no measured PAR at 100-200 m in the
  Skagerrak.

## Gap 5. Flatfish reaction distance to infaunal prey and siphons

- **Holmes & Gibson 1983 (Anim Behav 31:1244): not obtained.** Not in OpenAlex's open set; one
  search found no copy or abstract with numbers.
- **de Groot 1971 (Neth J Sea Res 5:121): not obtained.** OpenAlex marks it open
  (research.wur.nl record), but the WUR record carries no file, only the DOI.
- **Holmes 1979 thesis** (local, `data/benth_flatfish/holmes1979_thesis_stirling.pdf`) [F],
  re-read for a reaction distance: none is reported. It summarises de Groot 1971: 1, 2, 4 and
  8 cm black balls, with and without chemical stimuli, offered to six flatfish species; turbot and
  brill did not respond to any ball size and chemical stimuli did not help, but they responded
  to shrimp and fish models. Plaice movement elements: FORWARD < 0.5 body length, SHUFFLE more
  (with the eyes moving), a SKIM lower limit ~15 cm/s; stalking <= 1 cm/s. (Holmes 1979 restricts
  publication of derived information; internal use only.)
- Consequence: the flatfish search width w = 2R, R 0.6 L (0.25-1.2) J stands. The benthos pair is
  saturated at any a in range (benthivores_encounter 1d), so this gap does not change intake; it
  sets only the depletion floor.

## Summary of implied changes to a (hand arithmetic; none applied)

| Pair | Note central a | Item | Implied a | Status |
|---|---|---|---|---|
| planktivores -> zooplankton | 0.0070 | herring Ke ~10 lx (lake-herring proxy, [A]) | ~0.0062 | proxy, within MC |
| planktivores -> zooplankton | 0.0070 | surface Kd 0.20 not 0.30 (Arst [F]) | < +5 % | negligible |
| benthivores -> planktivores | 0.016 | surface Kd 0.25-0.20 (Arst [F] vs Aas [F]) | ~0.020-0.025 | rough, needs script re-run |
| benthivores -> predators | 0.022 | surface Kd 0.25-0.20 (Arst [F] vs Aas [F]) | ~0.027-0.033 | rough, needs script re-run |
| benthivores -> fish pairs | 0.016 / 0.022 | R_nv 0.4 L (toadfish [S]) | ~0.014 / ~0.019 | sensitivity |
| benthivores -> fish pairs | 0.016 / 0.022 | p 0.3 -> 0.6 (cod lab ceiling [F]) | <= 0.032 / 0.044 | ceiling only |
| predators -> planktivores | 0.0056 | p 0.2 -> 0.3 (cod lab ceiling [F]) | <= 0.0084 | ceiling only |
| predators -> zooplankton | 0.0015 | krill p 0.5 -> 0.8 (whiting strike success [F]) | <= 0.0024 | upper end |
| predators -> zooplankton | 0.0015 | mackerel night factor 1 (Macy threshold [F]) | 0.0016 | negligible; keep 0.2 |
| predators -> planktivores | 0.0056 | Beauchamp RD equation verified [F via Eckmann] | unchanged | confirmed |

The pelagic-prey pairs of planktivores and predators stay 5-25x below the library 0.04, as in
`summary.md`. The benthivore fish pairs move from 0.4-0.55x of 0.04 toward 0.5-0.8x (still near
the knee, 1/(a h) ~0.15-0.2 t/km2), pending a re-run of the benthivore light script with the
surface K at 0.20-0.25. That surface-layer Kd is the largest change found here.

## Still missing

1. Herring reactive distance (any juvenile/adult measurement; one direct search found only larval
   work) and the Varpe & Fiksen 2010 table
   (reference prey 1.5 vs 4 mm: the x3.5 k_v lever). Wiley blocks the bronze copy; needs library
   access.
2. Gibson & Ezzi 1985/1990/1992 and Batty et al. 1986/1990 full texts (closed Springer/Wiley):
   herring capture success per bite at low density, and an [F] upgrade of the 0.001 / 0.01 lx
   thresholds. A measured herring Ke.
3. Meager et al. 2005 / 2010 full texts (cod R vs light, chemical vs visual range); any gadoid
   lateral-line detection range; any per-attack or per-encounter capture rate of cod, saithe or
   whiting on clupeid schools.
4. A measured Kd(PAR) or PAR profile below 50 m in the Skagerrak / Norwegian Trench (e.g. CTD-PAR
   casts from IMR or SMHI cruises); the benthivore light script re-run with Kd(0-20 m) 0.20.
5. Beauchamp et al. 1999 / Vogel & Beauchamp 1999 originals (equation confirmed only through
   Eckmann 2016 and its internal consistency; one OpenAlex search found no open copy).
6. Holmes & Gibson 1983 and de Groot 1971 (flatfish reaction distance to siphons).

Budget used: 10 web searches, 25 page/API/PDF fetches (Unpaywall, OpenAlex, Semantic Scholar and
figshare API calls counted as fetches).
