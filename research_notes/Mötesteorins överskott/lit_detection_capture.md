# Detection distance and capture success in situ vs the encounter chain (2026-11-20)

Literature trail for "Mötesteorins överskott". Research only; nothing in the model, library or project files was
changed. Raw rows: `tables/detection_capture_raw.csv`. Scripts: `scripts/dc_pdftext.py`, `scripts/dc_grep.py`,
`scripts/dc_openalex.py`, `scripts/dc_turbidity.py`, `scripts/dc_write_csv.py`. Downloads (untrusted, not to be
committed) in `data/upw1/`, `data/scharf2003/`, `data/dukas_kamil2000/`, `data/scott2017/`, `data/openalex1-2/`
(`data/meager2005/` is empty; the server returned an HTML page and it was deleted).

Tags: [F] full text read, [A] abstract only, [S] secondary (search snippet, held note, or citation), [D] derived
here (arithmetic stated), [J] judgement. PROXY = species not in the model's member list. Lab / in situ stated.

Bottom line first: no in situ detection distance and no in situ capture success was found for ANY model species
(cod, haddock, whiting, saithe, hake, herring, pout, sprat, blue whiting, mackerel, flatfish, spurdog). What exists
are lab values for proxies and a handful of in situ per-strike values for reef and pelagic proxies. With that,
neither term can be shown to carry the x0.05-0.3 the chain needs. Detection looks ~x1 (x0.85-0.9 at most from
turbidity, and only on krill). Capture on fish prey could be x0.3-1, but only by proxy and [J]. On copepods,
krill and bed prey it is unconstrained around 1.

## Q1. Reaction / detection distance in situ vs lab, turbidity, attention, scan share

### Turbidity

- **Meager et al. 2005, CJFAS 62:1978 [A]**: juvenile cod on mysids, lab, beam attenuation "0.4 to 17 m-1" (the
  only verbatim number held). In paraphrase of the abstract: chemical-cue reactive distance was longer than visual
  and unaffected by turbidity; visual and visual+chemical distances were only slightly reduced; search time tended
  to rise; predation on free-ranging mysids was not reduced by turbidity but was lower in darkness. The authors
  propose that cod combine chemoreception with close-range vision. A held note already carries a related value
  ("Torskfiskars ljusberoende vid jakt"/prey_side_and_benthic.md [S]: foraging rate little affected up to 28 m-1).
  The fish are juveniles and the setting is lab.
- **De Robertis et al. 2003, CJFAS 60:1517 [A]**, lab, PROXY: "Comparatively low (5-10 NTU) turbidity decreased
  both the rate at which sablefish pursued prey and the probability of successful prey capture". Planktivorous
  walleye pollock (a gadoid PROXY) and chum salmon were "much less sensitive". Their feeding was reduced at high
  light but not at low light, over 0-40 NTU.
  - This is an empirical capture effect, and it sits beside the [D] result below that turbidity does nothing on
    fish-prey pairs. No NTU-to-beam-attenuation conversion is held here, so the two cannot be compared and do not
    contradict.
  - For the Skagerrak offshore water the relevant NTU range is unknown here [J].
- **Utne-Palm 2002, Mar Freshw Behav Physiol 35:111 [A]**: a review. Turbidity cuts long visual ranges (adult
  piscivores) more than short ones (planktivores, larvae), and at short range it can raise prey contrast.
- **Aksnes & Utne 1997, Sarsia 82:137 [A]**: goby (PROXY) on copepods, lab. Measured reactive distances at high
  light were "much lower than those predicted" by the earlier model, hence a saturation parameter. The chain
  already uses the revised saturated form.
- **[D] What turbidity can do inside the chain** (`scripts/dc_turbidity.py`). The chain's rule A solves
  R^2 exp(c R) = R00^2, where R00 = sqrt(|C0| A) k_v is the c = 0 range ("En synregel for fiskparen"/scripts/
  one_rule_pairs.py, r0_c). It then caps the result at R_sat = min(R0, 2 L). Output (R in m; ratios are to the
  chain's c 0.25 /m):

  | R00 (c=0) | c 0.1 | c 0.25 | c 0.5 | c 1.0 | R ratio 0.5/0.25 | R^2 ratio | R ratio 1.0/0.25 | R^2 ratio |
  |---|---|---|---|---|---|---|---|---|
  | 0.30 | 0.296 | 0.289 | 0.280 | 0.263 | 0.967 | 0.935 | 0.909 | 0.826 |
  | 0.50 | 0.488 | 0.471 | 0.447 | 0.408 | 0.949 | 0.900 | 0.865 | 0.748 |
  | 0.80 | 0.770 | 0.730 | 0.676 | 0.594 | 0.925 | 0.856 | 0.814 | 0.662 |
  | 1.20 | 1.134 | 1.052 | 0.947 | 0.803 | 0.900 | 0.810 | 0.763 | 0.583 |
  | 2.00 | 1.826 | 1.631 | 1.407 | 1.134 | 0.863 | 0.744 | 0.695 | 0.484 |

  - Krill (25 mm, |C0| 0.4): R00 = sqrt(0.4) x 100 x sqrt(0.75 x 0.025 x 0.2 x 0.025) = 0.61 m. Doubling c from
    0.25 to 0.5 /m gives encounter (proportional to R^2) x0.86-0.90.
  - This is an UPPER bound on the turbidity effect for krill. For small predators (pout, sprat) the 2 L cap binds
    below R0, and at depth the light term g dilutes R_sat toward R_nv; both mute any change in c.
  - Fish prey (15 cm, |C0| 0.7): R00 = 0.837 x 100 x 0.058 = 4.9 m, far above the 2 L cap (0.5-1.0 m), so turbidity
    changes nothing. The held note "Vitlingens skydd på grunt vatten" makes the same point: c must exceed
    1.7-3.7 /m before contrast loss pulls R0 below the cap [S].
  - Also, the chain's c 0.25 /m is itself [J]. No in situ Skagerrak beam attenuation was found here, so the
    plausible c range (0.1-0.5 /m offshore) is [J].
- Light-weighted R_eff on fish prey (0.3-0.8 m) is set by the light term g and the 2 L cap, not by turbidity. In
  the dark it falls to R_nv. The cod data (Meager) suggest chemoreception holds R in turbid water, which argues
  against a smaller R_nv, not for one [J].

### Attention, divided attention, scan share

- **Dukas & Kamil 2000, Behav Ecol 11:502 [F]**, blue jay (PROXY, bird), lab operant task. Peripheral targets were
  detected at 38 % with an easy central task and 14 % with a hard one (0.37x [D]). The paper cites Milinski
  (1984) on sticklebacks choosing low-density Daphnia swarms after a predator model. The paper measures a
  predator-detection analogue, so it gives direction only and sizes nothing for prey detection.
- **Dukas & Kamil 2001, Behav Ecol 12:192 [S]** (search snippet; full text not obtained). Jays searching for two
  cryptic prey types at once detected fewer than when focused on one; switching between types did not cost
  detection. No number held, no fish datum. This is the only source that bears on "several prey types present". It
  would be a term shared across a predator's prey, and it acts on cryptic prey (a search-image effect), so it
  matters least for conspicuous prey within R_sat.
- **O'Brien, Evans & Browman 1989, Oecologia [S]**: saltatory search. Planktivores search during pauses and not
  during the move phase (white crappie; PROXY). A scan or search-mode share applies to all prey of the predator.
  The chain's scan 0.5 (Varpe & Fiksen 2010, Langbehn 2019, Kulke 2018 sprat field clearance [S held]) already
  stands for this. No new number was found that moves it.
- **Luecke & O'Brien 1981 [S]**: bluegills sometimes located prey beyond the measured reactive distance, so lab
  reactive distance can understate the search volume (direction: R up, not down).
- **Viitasalo et al. 1998, MEPS 175:129 [F]**, prey side: copepods detected a stickleback at 0.24 cm and a mysid at
  0.45 cm. Detection by the prey is part of capture (q), not of R.
- Not obtained or not searched: Gregory & Northcote 1993, Milinski 1984 (cited only via Dukas & Kamil), Godin &
  Smith 1988, any in situ ROV, video or acoustic reaction distance of a model species. No in situ measured R
  for a gadoid, clupeid or mackerel was found.

## Q2. Capture / attack success in situ for fish predators

### Lab, per attack (the chain's p_s currency)

- **Scharf et al. 2003, JEMBE 294:41 [F]** (doi 10.1016/S0022-0981(03)00260-0; listed "not obtained" in the held
  note "Fangstsannolikhet for fiskbyten"; now read). Bluefish (PROXY) per-attack capture success ("the number of
  attacks resulting in prey ingestion"), in 475-L tanks. It declines linearly with relative prey size x (Table 3;
  the minus signs follow from the text "declined linearly"):

  | prey | line | at x = 0.25 [D] |
  |---|---|---|
  | bay anchovy | 1.01 - 0.98x | 0.77 |
  | Atlantic silverside | 0.70 - 0.94x | 0.47 |
  | Atlantic menhaden (clupeid) | 1.12 - 1.66x | 0.70 |
  | age-0 striped bass | 0.70 - 1.34x | 0.37 |

  - Striped bass as predator (Table 4): anchovy 0.77 - 1.29x, silverside 0.49 - 1.15x, giving 0.45 / 0.20 at
    x 0.25 [D].
  - The prey species differ by ~2x at equal relative size, a PREY-SPECIFIC term. The authors tie it to prey
    behaviour: "variable reaction distances to approaching predators and activity levels of prey". Anchovy and
    menhaden "displayed short reaction distances" and were caught more often.
  - Tank size: "capture success in 215-l tanks was significantly higher than in 475-l tanks ... or 900-l tanks
    ..., and ... capture success was not different between 475- and 900-l arenas (Tukey q = 0.26; p>0.50)". The
    authors call this preliminary.
  - The chain's bluefish line (Juanes & Conover 1994, held E5) used a 121 x 64 x 58 cm tank = 449 L [D], on the
    475-900 L plateau, so tank VOLUME is not a correction by this evidence. Whether arenas of thousands of litres
    or open water lower it further is untested.
  - **[D] Level check, same predator and prey.** The chain's line is p_s = 0.973 - 1.230 PPR (held E5, bluefish on
    silverside); Scharf 2003's 475-L silverside line is 0.70 - 0.94x.

    | PPR | chain line | Scharf silverside | ratio |
    |---|---|---|---|
    | 0.20 | 0.727 | 0.512 | 0.70 |
    | 0.25 | 0.666 | 0.465 | 0.70 |
    | 0.30 | 0.604 | 0.418 | 0.69 |

    A second study in a tank of the same size puts the same pair ~0.7x lower: a between-study level spread.
  - **Choice of prey analogue moves p_s the other way.** At PPR 0.25 the clupeid (menhaden) line gives 0.705 and
    anchovy 0.765, which is 1.06 and 1.15 x the chain's 0.666. The silverside line gives 0.70x and striped bass
    0.55x.
  - Lab per-strike evidence therefore brackets the chain's p_s at x0.55-1.15, central ~1 for clupeid-like prey
    [D on F, J on the analogue]. Neither the tank nor the analogue gives a large factor.
  - Scharf also writes that "prey behavior can strongly affect the probability of predator attack" (Eklov &
    Persson 1995). That would be a prey-specific attack-refusal term.

### In situ, per strike (all PROXY)

- **Scott et al. 2017, Sci Rep 7:12966 [F]**, coral trout (a grouper; reef stalker), diver focal follows: "of the
  278 strikes ... 47 were considered successful, giving an overall strike success rate of 17%". Success did not
  vary with body size or season, but it differed between the two locations (strike-rate units in the extract
  are inconsistent; not used).
- **Feeney et al. 2012, Coral Reefs [S]**, dottyback, in situ: ambush strikes 26 % successful, pursuit 5 % (search
  snippet).
- **Herbert-Read et al. 2016, Proc R Soc B 283:20161671 [A]**, sailfish on sardine schools, in situ video: "only
  24% of attacks result in prey capture" (51 of 210), and prey are injured in 95 %. The authors say success
  rates of other marine predators attacking schools are of the same order. These are attacks on SCHOOLS, already
  carried in the chain's F; do not stack them on p.
- **Parrish 1993, Ethology 95:233 [A]**, four reef piscivores on schools of flat-iron herring, in situ. Solitary
  stalkers made most attacks but "capture success was low". Fast pelagic group hunters were "extremely proficient";
  the group-size effect was positive but not significant. No number held. Also a school (F) effect.
- **Handegard et al. 2012, Curr Biol 22:1213 [S]**: the predator was spotted seatrout (PROXY), NOT saithe on
  herring as the task's parenthetical suggested; the prey identity is not confirmed here. Predators attacked in
  line formation, which fragmented schools and raised the estimated per-capita risk for schools under ~2 m2.
  Distance to prey was the success proxy, with no capture rate.
- **Pitcher et al. 1996, ICES JMS 53:449 [S]** (attribution likely, not confirmed: the snippet named neither
  authors nor year), Norwegian Sea sidescan sonar, 14 herring schools (MODEL species). Individual cod and haddock accompanied the schools and attacked, modifying but not dispersing school
  structure; saithe approaches made herring dive. No attack outcome is reported. This is the only in situ
  gadoid-on-herring observation found, and it has no success rate.
- **Kaartvedt et al. 2005, MEPS 299:1 [F]**: large saithe (60-110 cm) with blue whiting "patrol" krill swarms at
  200-300 m (acoustics + trawl). It gives no capture rate.
- Held, lab, model species: cod p ~1 up to prey:predator 0.33 (Ellis & Gibson 1997 [S held]); whiting on shrimp
  0.8 per strike when empty, 0.4 when fed (Staniland 1995 [S held]).
- Held, other: Turesson & Brönmark 2004 [S held] (percid per-strike and per-sequence success; the source of r in
  D1). Fall & Fiksen 2020 [S held]: a visual encounter model of cod on capelin gives fuller stomachs than observed.
  That is a whole-chain over-prediction for a model species, and it cannot be split into R or p. Rindorf &
  Gislason 2005 [S held]: whiting's fish-prey plateaus sit at 0.07-0.29 of the model ceiling, a whole-chain field
  constraint, not a p or R datum.

**[D] Comparing in situ with the chain.** Put the in situ per-strike values into the held D1, p = q p_s /
(1 - r (1 - p_s)), with q 0.8 and r 0.3:

| in situ p_s | per-encounter p | ratio to chain 0.52-0.63 |
|---|---|---|
| 0.17 (coral trout) | 0.8 x 0.17 / (1 - 0.3 x 0.83) = 0.18 | 0.29-0.35 |
| 0.26 (dottyback ambush) | 0.8 x 0.26 / (1 - 0.3 x 0.74) = 0.27 | 0.43-0.52 |

Three caveats keep this from being a correction:
- **Reef refuges.** Reef prey dart into coral, a refuge the open-water North Sea prey lack. Those prey have
  schooling, which F carries.
- **Fullness.** Field strikes come from fish of mixed fullness. The fullness effect on strike success (Staniland
  x0.6) is the engine's Holling h, so part of the field-vs-lab gap is already in the engine.
- **No model species.** No measurement exists for any model species.

**Attack refusal (a detected prey not attacked, apart from satiation).** Tanks show it near 0 for hungry fish:
bluefish "Prey items were rarely ignored" (Juanes & Conover 1994 [S held]), and empty whiting struck almost every
shrimp (Staniland [S held]). No in situ refusal share was found. The chain's q 0.8 (pre-strike escape) is the only
refusal-type term.

## Q3. Copepod / krill capture by planktivores

- **Viitasalo et al. 1998, MEPS 175:129 [F]**, stickleback (PROXY) on copepods, lab 3-D video, per interaction (the
  per-encounter currency):
  - E. affinis: 0.53 captured (41 % at first attack + 12 % after pursuit); 60 of 137 escaped before the first
    attack (q 0.56 [D]); 38 of those were pursued and 15 caught (0.39 [D], approximate).
  - The scan is OCR-degraded: 15/137 is 11 %, not 12 %, so the count may be 16. Treat the split as approximate;
    the abstract's 53 % total is clean.
  - T. longicornis: 0.92 (no attack failed; 1 of 12 escaped before attack).
  - The difference is prey-specific (1.7x) and follows the copepods' flow-sensitivity thresholds (4x).
  - Sticklebacks caught a copepod only when they approached slowly to within a strike distance of <0.1 cm.
- **Viitasalo et al. 2001, MEPS 216:191 [F]**: siphon capture success, a measure of escape ability, spans ~10x
  among Baltic copepods: E. affinis 9 %, Acartia 33 %, Pseudocalanus 86 %, Bosmina 100 %. Yet field herring
  selectivity (Baltic herring, MODEL species) does NOT follow escape rank. "The evasive Eurytemora affinis was
  strongly positively selected", and the non-escaping Bosmina was rejected. The large escape differences seen in
  the lab are not visible in field herring diets. So the chain's single copepod p (herring 0.6, pout/sprat 0.7)
  is not contradicted, and no copepod-specific correction is indicated [J on F].
- **Ryer & Olla 1999, MEPS 181:41 [F, abstract part]**: juvenile walleye pollock (PROXY) consumption vs light is
  sigmoid, and they were "capable of detecting and capturing prey in darkness". This is a light term, already in
  the chain.
- **Krill**: no fish-on-euphausiid capture success, lab or in situ, was found (searched once). The chain's 0.5
  stays [J]. The held record carries krill voids under fish schools (Onsrud 2004; Christiansen 2024) as the only
  in situ escape evidence.
- **Field clearance vs theory**: Arrhenius & Hansson 1999 [S held] found that theory over-predicts YOY herring
  field clearance by 1.5-2x with R proportional to length, and 9-49x with size-free R. That is a whole-chain
  ratio, not one term. No further field-vs-theory clearance for herring, sprat or mackerel was found.
  Mackerel on zooplankton is not encounter theory in the chain: it is measured clearance (Macy et al. 1998 [S
  held], lab), so R and p corrections do not apply to that pair.
- Not obtained or not searched: Gibson & Ezzi 1985/1992 bite success (searched, no numbers), Batty et al. 1990,
  O'Brien 1987 krill escape, Kils 1981, Mehner, Axelsen et al. 2001 (bird predator), Nøttestad (whale and
  herring sonar work; no fish-predator outcome seen in the snippets), the Pitcher & Parrish full text, Holmes &
  Gibson 1983 (known unobtainable).

## Already in the chain: do not double count

- Contrast loss at c 0.25 /m is in R. Turbidity can only act through a different c, which binds on fish prey only
  above c ~1.7 /m.
- The fullness effect on strike success (Staniland) is Holling h (held note), not p.
- Pre-strike escape is q 0.8, already inside the per-encounter p 0.52-0.63.
- Schooling: the in situ success on schools (sailfish 24 %, Parrish) belongs to F (0.51-0.68), not to p.
- The scan share is 0.5. A saltatory or attention term is the same kind of term; a further cut would need its own
  evidence.
- Light: g = E/(E + 0.02 lx), with night/day windows. The lab light thresholds (Ryer & Olla; Meager darkness) are
  this term.

## Best estimates (multipliers on the chain; evidence vs [J])

(i) In situ R relative to chain R. The R column holds ratios of R; the next column is that ratio squared, the
factor on a, since bed pairs (a swept width) scale with R and visual pairs with R^2.

| group | prey | R central (low-high) | on a: R^2 central (low-high) | basis | shared / prey-specific |
|---|---|---|---|---|---|
| benthivores (cod, haddock, flatfish) | bed (chemo-tactile width) | 1.0 (0.7-1.1) | width ~ R: 1.0 (0.7-1.1) | [J]; cod keep R by chemoreception in turbidity (Meager [A]) | shared over bed prey (search mode) |
| benthivores | fish | 1.0 (0.9-1.0) | 1.0 (0.8-1.0) | [D] at the 2 L cap turbidity is inert; low end [J] for attention | shared |
| benthivores | krill | 0.95 (0.9-1.0) | 0.9 (0.8-1.0) | [D] x0.86-0.90 on R^2 per doubling of c (upper bound) | shared (turbidity), binds only on krill |
| whiting | fish (cap binds), krill, epifauna | 1.0 (0.9-1.0) | 1.0 (0.8-1.0) | [D]/[J] | shared |
| water column (saithe, hake) | fish, krill | 1.0 (0.9-1.0) | 1.0 (0.8-1.0) | [D]/[J] | shared |
| planktivores (herring, pout) | copepods, krill | 0.95 (0.8-1.0) | 0.9 (0.65-1.0) | [J]; low end = a large share of the herring whole-chain over-prediction (1.5-2x) put on R; most of it may sit in other terms | shared over the copepod and krill pairs |
| mackerel | fish, krill (zoo is measured clearance) | 1.0 (0.9-1.0) | 1.0 (0.8-1.0) | [J] | shared |

(ii) In situ capture per encounter relative to chain p.

| group | prey | central | low-high | basis | shared / prey-specific |
|---|---|---|---|---|---|
| benthivores | fish | 0.8 | 0.4-1.0 | [J]; lab cod p ~1 at small relative prey size (held); in situ proxies 0.3-0.5 [D] but reef and fullness caveats | prey-specific (prey escape, Scharf ~2x across prey) |
| benthivores | bed prey | 1.0 | 0.5-1.0 | [J]; no datum; exposure e and structure (Persson [S held]) are encounter terms | prey-specific |
| whiting | fish | 0.8 | 0.4-1.0 | [J], same as above | prey-specific |
| whiting | epifauna (shrimp), krill | 1.0 | 0.6-1.2 | [J]; Staniland 0.8 empty (lab) vs chain | prey-specific |
| water column | fish | 0.8 | 0.3-1.0 | [J]; open-water pursuit; in situ stalker 0.17 per strike (proxy) | prey-specific, partly shared (hunting mode) |
| water column | krill | 1.0 | 0.6-1.4 | [J]; no datum | prey-specific |
| planktivores | copepods | 1.0 | 0.8-1.2 | lab 0.53-0.92 (proxy) brackets the chain's 0.6-0.7; field herring selectivity shows no escape effect [F] | prey-specific in the lab, invisible in the field |
| planktivores | krill | 1.0 | 0.6-1.4 | [J] | prey-specific |
| mackerel | fish | 0.8 | 0.4-1.0 | [J] | prey-specific |
| mackerel | krill | 1.0 | 0.6-1.4 | [J] | prey-specific |

**Reading.** Detection: the [D] turbidity bound is x0.86-0.90 on a, on krill pairs only. The [J] lows in table (i)
reach x0.65-0.8 on a and stand for attention or search mode, for which no fish number exists. Capture: the
in situ proxies give ~x0.3-0.5 per encounter on FISH prey; the lab level check gives x0.55-1.15 on p_s, central ~1
for clupeid-like prey; central 0.8 [J]. The only in situ values are for reef and pelagic proxies. Both are far from the needed x0.05-0.3 on a, and the gap is largest where they apply least:
- The benthivores need x0.03-0.05 and eat mostly bed prey, which have no capture datum.
- Capture corrections are prey-specific (escape ability; Scharf ~2x across prey at equal size, Viitasalo 1.7x
  across copepods). The sibling M1 finding needs a term reaching every prey of each predator.
- The shared candidates here (turbidity, attention, scan or search mode) have either no evidence of size or a
  derived bound near 1.

Neither detection distance nor capture success, as measured or bounded here, can carry the excess. A modest
fish-prey p correction (x0.5-0.8 [J]) remains possible but has no model-species evidence.
