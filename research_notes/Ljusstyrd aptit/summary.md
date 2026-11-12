# Light-cued appetite (2026-11-12)

Research only, by a research subagent; saved by the main agent. Nothing in the library, code, tests, manifest,
resume or AGENTS.md was changed. No training or probe was run, and no model output set or selected a value.
Trail with quotes: `evidence.md`. Arithmetic: `scripts/light_appetite_numbers.py` ->
`tables/light_appetite_numbers_stdout.txt`. The day-length and krill-flow effect sizes are the main agent's
`scripts/effect_size.py` -> `tables/effect_size_stdout.txt`. Papers: `papers/` (not to be committed).

Tags: [F] full text, [L] lab, [A] abstract only, [S] secondary (an earlier note or a review), [D] arithmetic,
[J] judgement.

## Question

Background: `research_notes/"Krillparen mot vintermagar"`. In winter, field herring, Norway pout and whiting
meet krill but eat about maintenance, 0.1-0.5 of their Cmax. Neither encounter nor ceiling explains this.

Is a light-cued appetite the mechanism? Four sub-questions:
- Form: a defended reserve level s_T(L), or a rate Cmax x f(L)?
- Driver and phase: L, dL/dt, a clock, temperature, or maturation? What lag?
- What does a fish below the target do in winter?
- Does it apply to each member species?

## Verdict

### Form: a set point, not a rate cap [L/J]

Salmonid analogues only; no member species has been tested.

- **The winter low is not a capacity limit.** Under excess food, salmon parr lose appetite from September
  (Metcalfe & Thorpe 1992 [A]: the appetite index fell 96 % from early September to late December).
  - Fish whose fat was cut by 3 weeks without food ate more, "made up their lost fat within 4 weeks", and their
    appetite then "decreased to the level of control fish" [A].
  - Bull 1996 [F]: control intake was "much lower than that predicted ... relating food intake by salmonids to
    temperature".
  - A rate cap Cmax x f(L) cannot give eating above control followed by a return to it. A defended level can.
- **The defended level falls through the winter.** It is a trajectory, not a constant.
  - Bull [F]: control lipid went from 5.32 % BW in September to 2.63 % in April "despite food always being in
    excess" [D: x0.49].
  - After a 3-week deprivation, appetite rose above controls only in early winter (October), for 14 days. Fat
    restored during refeeding: +0.49 % BW after October, -0.05 after December, -0.37 after February.
  - Bull's reading: the fish respond "not to their current reserve level, but to their projection of whether
    they would be above or below the target level at the end o[f] the winter".
- **The top of the target.** Tveiten et al. 1996 [A], charr at a constant 4 C with liberal food: "The fish
  appeared to reduce feeding once a 'threshold condition' of 1.4-1.5 had been attained". From December to April
  only 20-60 % of the fish had fed at each check.
- **Below the target in winter:** extra appetite (hyperphagia) early in the winter, then anorexia late in the
  winter even when below the earlier level. In model terms: intake is capped at the room up to s_T(t), and
  s_T(t) declines from autumn to a spring low [J].

### Driver and lag: an endogenous clock that photoperiod entrains [L/J]

- **Endogenous.** Sæther, Johnsen & Jobling 1996 [A]: charr at a constant 4 C under simulated natural light or
  a constant 12:12 LD both showed "distinct seasonal cycles", "highest in mid-summer and lowest in autumn". Food
  was given in excess during a fixed 5-h window every day. The authors: "circannual rhythms ... exist in the
  absence of seasonal changes in these environmental cues".
- **Photoperiod sets the phase.** Bull ch. 3.5 [F]: three groups had the same ambient temperature, the same
  24-h ad-lib feeders and the same deprivation dates, but different light.
  - After the January deprivation, the group shown April light "did not restore losses".
  - The group shown October light, and the controls, did restore.
  - Nordgarden et al. 2003 [A], salmon post-smolts fed in excess: continuous light from January to June
    advanced the seasonal growth pattern. The seasonal SGR was "independent of temperature and fish size".
- **Temperature modulates the rate, not the phase.** The salmon parr's anorexia "commences in early autumn
  independent of water temperature or food availability" (Bull, citing Metcalfe et al. 1986 [S]).
- **Lag.** The onset is in early autumn (late summer in the slow-growing parr, from midsummer), on falling L,
  about 1-3 months before the solstice. The low is reached at the end of winter.
- **Consequence for an engine form [J].**
  - s_T(L) alone is ambiguous: at the box (58.15 N) the day is 12.3 h in mid-September and 11.5 h in mid-March
    [D, main agent's table], yet the biology runs opposite ways on those dates.
  - A physical-state form needs (L, sign or size of dL/dt). At a fixed latitude that pair is equivalent to the
    day of year.
  - This is the project-rule tension: the driver is physical, but its information content is the calendar. I
    name it and do not resolve it.

### Herring (0.554 of the planktivores)

**Berg, Seljestad & Folkvord 2024 [F, L].** Natural and 6-month offset light, at a constant 7 or 10 C, for 3.5
years.

Seasonal swing of condition (ln weight-at-length residual, GAM lines in Fig. 6) [D]:

| group | swing (ln units) | % of weight at length | phase |
|---|---|---|---|
| 10 C natural | 0.080 | 8.3 | minimum winter, maximum late summer-autumn |
| 7 C natural | 0.035 | 3.6 | minimum winter, maximum summer |
| 7 C offset | 0.075 | 7.8 | minimum one quarter later (spring, in experienced light) |
| 10 C offset | 0.045 | 4.6 | "not conclusive" (authors) |

- The phase follows the light in 3 of 4 groups.
- Individual residuals span about +-0.3, so the seasonal swing is ~1/8-1/4 of the individual range.
- Sampling was quarterly, at solstices and equinoxes, so the phase is known to +-1.5 months.

Confounds, recorded as found:
1. **Feeding hours: not stated anywhere** in the paper or supplement (checked by grep). Excess dry feed could
   have been given in light hours only, which would make the result feeding time, not appetite.
2. **Ration:** "fed in excess based on previously observed growth rates". This is an allowance computed from
   expected growth; no intake or leftover was measured. The paper cannot separate appetite from allocation.
3. **Maturation is a small term by arithmetic.**
   - The sister experiment (dos Santos Schmidt et al. 2022 [F]; same lab, 8.9 C, natural vs offset light) had
     natural-group GSI of 1.05-1.92 % in February and 2.72-3.63 % in May.
   - Spring-spawners' winter gonads would raise winter whole-body condition, not lower it.
   - Berg's first winter (315 days post-hatching, ~10 cm SL) is immature.
   - The authors still do not separate maturity in the condition analysis.
4. **Size is a plus.** The tank-stunted fish (SL ~10-19 cm) match the box's 1-2-ringers.

**Sister experiment: dos Santos Schmidt et al. 2022 [F].**
- "individuals gathered relatively high body reserves throughout the year, even during the winter period".
- No difference in condition between the light regimes.
- The fish were "fed ad libitum" but "received the same amount of food independent of day length". The wording
  is contradictory and cannot be resolved from the text.
- Kn by experienced season (Table 1, n-weighted) [D]:

| season | Kn | n |
|---|---|---|
| winter | 0.976 | 21 |
| spring | 1.022 | 86 |
| summer | 0.944 | 10 |
| autumn | 1.001 | 60 |

- So there is no winter low. But n is 2-9 per sex and sample and the SD is 0.05-0.25, giving an SE of ~0.03-0.05
  for a sample of 10 [D]. The experiment can neither show nor exclude Berg's swing.

**Size of Berg's swing in fill units [D].**
- 3.6-8.3 % of weight converts to a fill change of 0.04-0.08 if the lost mass is wet tissue at 5 MJ/kg, or
  0.28-0.66 if it is pure lipid.
- The field planktivore fill drops 0.48 from December to April (start_fill 0.68 -> 0.20).
- Herring replace lost fat with water, so weight-at-length understates fat loss. The bracket is therefore wide
  on the low side.

**Phase gap (open).** Berg's natural-light minimum is at the winter-solstice sample. The field fill minimum is in
April (0.20). Either the field recovery is food-limited after the clock has turned, or the set point lags more
than Berg resolves.

**Other herring sources.**
- Varpe & Fiksen 2010 [A]: the spring condition rise of Norwegian spring-spawning (NSS) herring is predicted by a
  visual ingestion model with irradiance and prey; it is "closely linked to the return of longer days in
  spring". That is mechanism (a), feeding time and visibility, on Calanus. The model's daylight multiplier and
  vision rule already carry it. Do not count light twice.
- Slotte 1999 [A]: wintering NSS adults are "non-feeding". These are migratory adults over deep fjords, not the
  box's juveniles.
- No lab measurement of herring appetite or intake against photoperiod was found.

**Verdict for herring [J].** Herring condition follows photoperiod at constant temperature with food available.
This is the only member-species evidence for a light-entrained seasonal state. The form (appetite, hours or
allocation) is not established, and the size is 0.04-0.66 fill.

### Norway pout (0.35): a light-cued winter appetite contradicts the data [S/J]

- Albert 1994 (Norwegian Deep incl. Skagerrak, January-April) [S: F in `Planktivorernas vinterkostnad`]: pout
  feed all winter on euphausiids (79-81 % by mass). "The mean dry weight of the stomach contents relative to fish
  weight ... was consistently higher in winter than in summer".
- Contents are not a ration, since evacuation is slower in winter, but intake is clearly not near zero.
- Their "pronounced reduction in condition factor and liver weight during the winter season" coincides with
  first spawning: "close to 100% matured for the first time as II-group", spawning from mid-February into April,
  with most fish dying after it [S].
- Raitt 1968 [S]: juveniles showed "no marked variations in feeding intensity" from September to April.
- **A winter appetite suppression would contradict pout's data.** The mechanism could apply at most to herring
  (0.554) or herring + sprat (0.575) [D]. Blue whiting is not on the bank in winter.

### Gadoids (whiting, cod, haddock, saithe): it does not fit [A/J]

- Taranger et al. 2006 [A], cod: continuous light raised growth "mainly as a result of the weight loss of the
  natural-light groups during spawning". Maturation "results in loss of appetite and weight during spawning".
- Hansen/Taranger et al. 2006 [A], cod in sea cages: natural-light fish "decreased in body weight during the
  spawning season (February-April)".
- Davie et al. 2007 [A]: the direct growth stimulation by continuous light "is transitory"; "the photic
  inhibition of maturation realized a far more significant growth stimulation".
- Imsland et al. 2007 [A]: 3 months of continuous light in early juveniles gave a 1-9 % larger harvest size.
- Juvenile cod on demand feeders (Glasgow MSc 2009 [A]): intake was lowest in early summer, "the opposite result
  to that expected".
- **Field fills:** benthivores 0.76 in January-February and 0.33 in May-July; whiting 0.69 and 0.30 (the haddock
  liver series). Gadoids draw down from February to June, which is spawning (liver to gonads), not winter
  appetite.
- **Verdict:** the gadoid phase is the missing gonad term. Light-cued winter appetite does not fit.
- Whiting's winter krill excess (Nov-Mar 0.035 of the five-group krill flow, main agent's table) is not a
  light-appetite question.
- Hansen et al. 2001 and Trippel & Neil 2003 were not obtained.

### Mackerel: no evidence found [J]

- No photoperiod or appetite study was found.
- In the box mackerel is a summer migrant at a season factor of 0.1 in winter, so a winter appetite term barely
  touches it.

### Feeding hours (a) versus appetite (b)

- Bull's fish had automatic feeders running 24 h and still went anorexic, so this is (b) [F].
- Sæther's charr had a fixed 5-h window all year and the cycle persisted, so this is (b) [A].
- Krill are eaten by day and night (Onsrud et al. 2004 [S]); pout feed mainly at night (Albert 1994 [S]). (a) is
  weak for krill.
- (a) for Calanus-feeding NSS herring (Varpe & Fiksen) is already in the engine.

## Effect sizes on the model (main agent's `tables/effect_size_stdout.txt`) [D]

| case (Nov-Mar, at the field krill ration) | five-group krill flow |
|---|---|
| whole planktivore group | -24 to -33 % |
| herring share only (pout as realised) | -12 to -19 % |
| whiting | -1.5 to -2.9 % |

## Candidate forms for the user (none applied)

1. **Set point s_T(t) inside `reserve_room`.**
   - Intake is capped at the room up to s_T, i.e. (s_T - s) B ME instead of (1 - s) B ME.
   - s_T is high in summer and autumn and declines from the onset (falling L, ~September) to a low at the end of
     winter. A fish below s_T eats up to it.
   - Needs (L, dL/dt) or an entrained clock. At a fixed latitude this is the calendar (rule tension).
   - Herring only (0.554), or with sprat (0.575); not pout, not gadoids.
2. **Rate Cmax x f(L): not supported.** It contradicts the eating above control after deprivation.
3. **Circularity warning [J].** An s_T built from the planktivores' start_fill would reproduce the 247
   calibration by construction. The independent test is whether the winter rations come out at the field
   0.45-1.0 %BW/d (krill 0.10-0.46).

## Open items / not obtained

- Winter vs summer voluntary intake in %BW/d at constant temperature: Sæther 1996 and Tveiten 1996 full texts
  are closed (abstracts only).
- Full texts not obtained: Nordgarden 2003 (blocked), Huse & Ona 1996 (blocked), Hansen 2001, Trippel & Neil
  2003, Varpe & Fiksen 2010, Metcalfe & Thorpe 1992.
- Any herring intake measurement against photoperiod.
- Berg's feeding hours (could be asked of the authors).
- The 3-4-month phase gap between Berg's minimum and the field April minimum.
- Whether the herring signal is appetite, feeding hours or allocation.
