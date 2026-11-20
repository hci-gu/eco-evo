# Evidence trail: what limits a piscivorous gadoid's intake rate while foraging (2026-11-20)

Research only. Nothing in the library, manifest, code, tests, AGENTS.md or the resume was changed. No value
is chosen from model output. The resume 289 decomposition (frozen fix105) is the yardstick only.

Tags: [F] full text read, [A] abstract only, [S] secondary (an earlier note, a search summary, or a citing
paper), [D] derived here, [J] judgement.

Files:
- Arithmetic: `scripts/intake_arithmetic.py` -> `tables/intake_arithmetic_stdout.txt`.
- Table: `scripts/make_table.py` -> `tables/mechanisms.csv`.
- Unpaywall helper: `scripts/unpaywall.py`. It reads the address from the environment only.
- `data/`: downloads, untrusted, not to be committed:
  - `rindorf2003/`, `rindorf_gislason2005/` (copied from research_notes/Observabilitetsparen/papers);
  - `ellis_gibson1997/`, `jeschke2002/`, `fall2020/`;
  - `upw_*` Unpaywall records.
- The Staniland 1995 thesis is read in place: research_notes/"Attackrater ur motesteori" (o-umlaut)
  `data/gap_vision/staniland_whiting_thesis.pdf`.

## E0. The yardstick [D] (`intake_arithmetic.py` s1)

Uncapped intake while foraging is `T x X/(1 + X)`.
- X = sum over prey of a h B_vis (shared handling, fixed attack).
- T is the share of ticks spent foraging.
- Inputs are from resume 289:
  - whiting: T 0.905, saturation 0.845, krill 0.82 of handling time;
  - water column: T 0.97, saturation 0.78, krill 0.46.

| | whiting | water column |
|---|---|---|
| X (krill / other prey) | 5.45 (4.47 / 0.98) | 3.55 (1.63 / 1.91) |
| uncapped potential | 0.765 | 0.757 |
| need at fill 1 | 0.388 | 0.435 |
| potential with every non-krill pair at 0 (fish and epifauna) | **0.74** (0.72-0.76) | 0.60 |
| potential with krill at 0 | 0.45 | 0.64 |
| generic factor on every a for potential 0.50 / 0.45 / 0.40 | x0.23 / 0.18 / 0.15 | x0.30 / 0.24 / 0.20 |
| krill-only factor for 0.50 / 0.45 | x0.057 / 0.002 | impossible |
| factor on a time share instead | x0.52-0.66 | x0.53-0.66 |

Consequences:
1. No fish-prey-only mechanism can bring whiting below 0.72: krill alone holds it there.
2. For the water column, neither a fish-only nor a krill-only factor reaches 0.5.
3. A factor that acts on a (encounter) has to be ~x0.15-0.3 on every pair, because saturation damps it. A
   factor on the time share is proportional and has to be ~x0.5-0.66.
   - Equal-cell reference at the 5 t/km2 krill mean: `tools/probes/budget_gate.py`, run here read-only on
     the live library, gives saturation 0.95 for whiting -> krill (X ~19) and 0.85 for the water column.

## E1. Pursuit and handling time per item

Sources:
- **Staniland 1995** [F], whiting 23.5-30 cm on Crangon 5.9-8 cm, lit tank, video.
  - Median hunting time (orientation to capture) 1.94 s (large fish), 2.54 s (small fish), Table 4.20.
    Outliers "over four seconds" and once "over twelve seconds" in inexperienced fish.
  - Median handling in the mouth 0-0.99 s (large), 4.26 s (small), Table 4.21.
  - Handling rose with stomach fullness.
- **Ellis & Gibson 1997** [F], 0-group cod 65-147 mm on 0-group dab and plaice, PPR 0.20-0.49.
  - Pursuit 1.02 / 1.40 / 1.99 s and pursuit + ingestion 15.1 / 20.3 / 30.3 s for the first three prey.
  - "If the first attempt at capture was unsuccessful, cod often chased prey"; the cod are pursuers.
- **Fall & Fiksen 2020** [F]: in their cod model, handling runs from 3 s to a maximum of 10 min
  (`h = 0.05 + 5 (L_i/L_p)^3 + 5 (v_i/v_p)^3` min). They describe these as "reasonable handling times
  compared to those actually measured for juvenile cod".
  - In their words: "Digestion times of large prey are usually longer than handling times, and gut
    processing limits feeding rate at high prey density."
- **Jeschke, Kopp & Tollrian 2002** [F], the steady-state satiation (SSS) equation.
  - The asymptote is 1/max(b, c), not 1/(b + c): "digestion is a 'background process'".
  - "Based on available data, we suggest that most predators are digestion limited."

Derived [D] (s2). The digestion time per item at the model ceiling is c = E_item / ceiling. With
handling b of 4 / 30 / 600 s per capture (misses included), the intake lost in the additive disc form
(an upper bound; SSS gives ~0) is:

| predator | prey | c per item | loss (b 4 / 30 / 600 s) | b costing 10 % of intake |
|---|---|---|---|---|
| whiting 25-30 cm | herring, sprat, juvenile whiting (3-9 g) | 15-80 h | < 0.01 / < 0.1 / 0.2-1.1 % | 1.6-9 h |
| whiting 25-30 cm | Meganyctiphanes, 0.12 g (Rindorf 2003 mean) | 33-53 min | 0.1-0.2 / 0.9-1.5 / 16-23 % | 4-6 min |
| saithe 45 cm | Meganyctiphanes | 8.5 min | 0.8 / 5.5 / 54 % | 0.9 min |
| saithe 45 cm | Thysanoessa-size, 0.03 g [J] | 2.1 min | 3 / 19 / 82 % | 13 s |

**Verdict: not a limit on fish prey.** At 30 s per capture the loss is under 0.3 % for every predator. At 10 min
it is under ~1 % for whiting and 1.2-4.2 % for saithe and hake (s2).
- On krill it matters only if a capture costs far longer than the tank seconds:
  - for whiting, more than 4-6 min per Meganyctiphanes captured;
  - for saithe, more than ~1 min (Meganyctiphanes) or ~13 s (Thysanoessa-size).
- Not in the model, which has no external handling term. It is not needed for fish prey. For saithe on
  small krill it is an open, second-order term.

## E2. Field capture success and pre-strike detection

Findings:
- **No field (open-water) capture success for any gadoid on fish or krill was found.**
  - This holds for tags, video and acoustics; it confirms the gap in "Fangstsannolikhet for fiskbyten".
  - The CODFUN survey (Skaret, Johannesen, Fall & Fiksen 2016, IMR report 41-2016) [S] set out "to quantify
    attack rates of cod on capelin using acoustics, and to estimate proportion of successful attacks using
    stomach samples".
    - The report host could not be resolved from this machine, and no follow-up paper was found.
    - This is the one programme that measured item 2 and item 3 directly in the field.
- **Gut fullness lowers strike success** (whiting, Staniland 1995: about x0.6 at field fullness) [S via
  Fangstsannolikhet D4]. Under Jeschke's derivation this belongs inside the model's Holling h.
  - The SSS equation assumes a success rate proportional to hunger: a(x) = alpha h(x) with h = 1 - c y.
  - With b = 0 it reduces to the disc equation with c (digestion) in place of b [F, eqs 8-13].
  - The engine's Holling term with h = E/Cmax is that limit [D on F; it holds for a linear hunger term], so a
    fullness factor on p would count satiation twice.
  - This supports the user's 2026-11-11 convention ("a carries the hungry rate") on mechanistic grounds.
- **Pre-strike escape, q.**
  - It is in p at q 0.8 (0.6-1.0) [S]. An open-water value of 0.6 is the low end.
  - Fish -> krill p is 0.5 (0.3-0.7), with no fish-euphausiid measurement ("Krillens flykt") [S].

**Verdict:** in the model (p, q). The field corrections that are sized are x0.6-1.4 on a. Under saturation
these move the potential by a few hundredths.

## E3. Search efficiency in the field

Direct comparisons of encounter theory with field feeding all point the same way: the theory over-predicts.
- **Fall & Fiksen 2020** [F]: a visual encounter model of adult cod with 8 h of foraging, R 0.25-2 m and
  v 0.5 m/s.
  - It predicts gut-limited feeding at the low-year average capelin density (4.4e-4 ind/m3). "The prey
    densities at the intersection ... much lower than the average low capelin density".
  - The modelled stomachs were "high, but reasonable" against observed fullness. The authors note that the
    result "could also imply that cod's true encounter rate with capelin is lower than predicted since the
    modelled stomachs were fuller than seen in the data".
  - The observed diet was closer to the non-selective model.
- **Krill pairs** ("Krillparen mot vintermagar" [S]):
  - Whiting and pout in the Oslofjord krill layer (Onsrud et al. 2004 [F]) eat 0.1-0.7 of field Cmax.
  - The field-implied a is 1/50-1/260 of the library a. This is a JANUARY figure for planktivores and whiting
    on 100-150 m beds; applying it to the annual X and to the water column is extrapolation. In the
    authors' words, eating the field ration "needs only 0.02-8 % of the time in the layer".
  - **Conflict.** That note's verdict is (C): the fish met krill and took about maintenance, and encounter
    is NOT the field limit. Its fish sat in the krill layer day and night with half-full stomachs. So for krill
    the "low rate while foraging" is inside the layer, not a matter of finding the layer.
    - The patch-finding reading below fits fish prey, not krill.
    - Whether the in-layer krill limit is appetite (winter) or vigilance and escape (krill tail-flip, no
      fish-euphausiid capture datum) is open.
  - Onsrud et al. 2004 [F, read here through the publisher page; whiting lengths 21-28 cm are from
    "Krillparen", not from the page]:
    - krill 82 % of whiting stomach mass;
    - 1.4-5.9 krill per stomach;
    - 6-27 % empty;
    - fullness 2.2-3.3 of 5;
    - "no systematic day/night trend".
- **Rindorf & Gislason 2005** [F], whiting 20-35 cm, five North Sea locations, early September.
  - Fish intake "reached a saturation level for all types of fish prey, none of the levels corresponded to
    the maximum digestive capacity".
  - The plateau is 0.06-0.22 Wmax per fish-prey group, i.e. 0.07-0.29 of the model ceiling.
  - "Other food" stayed at 8-22 % of Wmax and varied at most x3 between locations, while fish intake varied
    17-36 fold.
  - There was no aggregative response; predator and prey catches were slightly negatively correlated.
  - "the predator population was thus never limited by gastric processing rate even though prey density was
    very high at some locations."
- **Essington, Hodgson & Kitchell 2000** [S: R&G 2005 and the abstract via search]: largemouth bass
  "consumed only 50 % of maximum daily ration in their natural environment though prey density was very
  high".
  - The abstract concludes that satiation is "probably not a major constraint" in the field and that "prey
    behavior may be a more important component".
- **Counter-case** (Pednekar et al. 2024 [S: news and abstract]; the gut-limited reading is [J], since cod
  and capelin masses were not checked). When a spawning capelin shoal was met,
  ~2.5 million cod ate ~10.5 million capelin within a few hours, about 4 capelin per cod.
  - That is a full meal in hours: gut-limited inside a dense patch.
  - So the field limit is the rarity of such encounters (the patch and school scale), not the rate inside a
    patch.

**Search speed is NOT at the in-situ value.**
- The chain uses whiting 0.8 BL/s (0.5-1.3) and saithe 1.0 BL/s (0.5-1.4): "Attackrater ur motesteori"
  `scripts/pred_encounter_a.py`, `v_whiting`, checked here.
- In-situ whiting in the krill layer swam ~0.6 BL/s, and at night ~0.7 of the day speed (Onsrud et al. 2005
  via that note's addendum [S]). The addendum computed the effect but the chain kept 0.8.
- So the field correction is x0.75 (night x0.525) on every whiting pair. This is new and sized.
- For saithe there is no in-situ speed; "in-situ gadoids mostly < 1 BL/s" gives x0.5-1 [S/J].

Apart from speed, no field datum separates reaction distance or search share. The field totals say only
that the product is far below the theory.

**Verdict:**
- The gap is real in each case named (cod-capelin, whiting and pout on krill, whiting on fish, bass).
- It is far larger for krill (x50-260 on a in January) than any sized component explains.
- The speed part (whiting x0.75) is sized and not in the model; the rest is not in the model either.

## E4. Prey availability windows

Sources:
- **Rindorf & Gislason 2005** [F]: "fish was ingested almost exclusively during dawn and dusk"; "The
  ingestion around midday was low".
  - The authors read this as the "limited time in which fish prey was vulnerable to predation", "related to
    increased vulnerability of the prey at moderate light levels".
  - Norway pout peaked once, around 15:00.
- **Rindorf 2003** [F], the same whiting.
  - Bottom-dwelling prey were eaten at night (4 of 5 locations).
  - Krill were eaten "around dawn and in the afternoon and early evening hours"; "The intake around midday
    was limited, perhaps indicating that krill is more vulnerable during the period of ascent and descent".
  - "Krill appeared to be a patchily distributed prey as some periods were categorised by complete lack of
    krill in the stomach contents".
  - Conclusion: "diel intake patterns are regulated by changes in availability of the prey rather than by
    size-dependent changes in the diel behaviour of the predator."
- Against the krill window: Onsrud et al. 2004 [F] found no systematic diel trend in Oslofjord whiting.

At 58.5 N a moderate-light window (sun between -12 and +6 deg) is 0.19-0.34 of the day [D, s3]. The window
definition is [J]: R&G's civil twilight is ~37 min at each end in September, and their "around dawn and
dusk" peaks are broader.

Already in the model:
- The vision rule gives fish pairs night/day 0.14-0.64.
- F cuts the schooled daytime share.
- Pairs eaten at night (bed prey) carry no light term.

So the twilight pattern is partly reproduced as a low night a (vision) times a protected day (F).
- The residual is the low midday intake on prey that school.
- The x0.25-1 (central 0.5) on fish pairs and x0.5-1 (central 0.75) on krill are [J]; no source sizes them.

Gut buffering keeps a window from limiting the daily ration when the within-window rate is high
("Mattnad hos rovfisk" lit_foraging_limits). The fish plateaus show the within-window rate is not high, so
the window acts with the low rate, not instead of it.

**Verdict:** supported as a pattern for fish prey and, weaker, for krill. Partly in the model. The residual
is unsized.

## E5. Satiation-independent limits: meals, inter-meal intervals, appetite

Sources:
- **Rindorf 2002** [S: the abstract via search; the text was blocked by Cloudflare]. Whiting stomachs (20-36
  cm, five North Sea locations, September) were less likely to hold fresh food when partly digested food was
  present.
  - Fall & Fiksen 2020 read it as "not related to absolute gut limitation", and suggest a predation-risk
    trade-off.
  - Under SSS this is the hunger term itself: search probability falls with gut content. In the model it is
    the Holling h.
  - A steeper-than-linear hunger dependence would lower the curve before the ceiling. No gadoid fit of its
    shape was found.
- **Meals**: Seyhan et al. 1998 [S]: captive whiting resume feeding at a stomach fill of 0.40, every ~21 h.
  - Prior result: gut buffering of 1-4 days lets a daily window reach 0.6-0.9 of Cmax [S].
  - Maximum field intake 3.1-6.2 % BW/d (R&G 2005 [F]).
- **Essington 2000** [S]: 13 % of bass diets were above the maximum sustainable fullness. Bass "could
  opportunistically forage" past it, so appetite does not stop foraging at the sustainable level.
- **Dempster 2011** [S]: wild saithe eat 0.43-0.65 of what farm-associated saithe eat; appetite is not the
  summer cap.
  - Winter is different: "Krillparen mot vintermagar" found fish in the krill layer taking about maintenance
    (Onsrud 2004 [F]; Berg et al. 2024: herring condition follows day length with food in excess).
  - So the satiation-independent appetite question is seasonal and is not settled for gadoids.

**Verdict:**
- Meal dynamics and hunger-dependent search: supported, and already in the model's Holling h (SSS).
- A lower field appetite: contradicted in summer (Dempster), suggested in winter (krill stomachs).
- No mechanism-free size.

## E6. Other

- **Interference.**
  - The engine carries w 0.1 for the predators (no field evidence) [S].
  - Rindorf & Gislason found no aggregative response and slightly negative predator-prey catch correlations.
    That is prey avoidance or patchiness at the haul (km) scale, not interference.
  - Size [J]: x0.9-1 on a.
- **Kleptoparasitism**: nothing found for gadoids.
- **Cost of pursuit making encounters not worth attacking.**
  - Fall & Fiksen's contingency and digestive-rate models give partial preferences.
  - Ignoring prey lowers intake only when the gut, not the encounter, limits. So it cannot hold a fish below
    its ceiling while the encounter is high.
  - Their observed cod diets were closest to the non-selective model.
  - The engine's fish attack every prey met (section 265).
  - Verdict: not a cause.

## E7. Forward combination [D] (s5)

Each factor is sized independently above; nothing is back-solved.

| mechanism | fish-prey factor (low / central / high) | krill factor (low / central / high) |
|---|---|---|
| 1 pursuit/handling | 1 / 1 / 1 | 1 / 1 / 1 |
| 2 capture beyond p | 0.75 / 1 / 1 | 0.6 / 1 / 1.4 |
| 4 window residual [J] | 0.25 / 0.5 / 1 | 0.5 / 0.75 / 1 |
| 5 satiation-independent | 1 / 1 / 1 | 1 / 1 / 1 |
| 6 avoidance | 0.9 / 1 / 1 | 0.9 / 1 / 1 |
| product before speed | **0.169 / 0.5 / 1** | **0.270 / 0.75 / 1.4** |

Mechanism 3, in-situ speed, applies to every pair of a group: whiting 0.525 / 0.75 / 1, water column
0.5 / 1 / 1.

Resulting potential (low / central / high):
- **Whiting: 0.38 / 0.67 / 0.80** (need 0.388).
- **Water column: 0.27 / 0.67 / 0.78** (need 0.435).

The central estimate removes ~0.09-0.10 of the ~0.3 excess. The all-low corner overshoots into starvation,
so the corners are not confidence limits.

The empirical field-implied krill a ("Krillparen", library/field 50-260) is not a mechanism, and it is a
January figure. Applied to krill alone all year it gives whiting 0.45-0.47 but leaves the water column at
0.64.

## E8. Double counting [J]

1. **Gut fullness on p vs Holling h**: the same process (SSS). Never stack them.
2. **Window residual vs the vision rule and F**:
   - the night half is the vision rule;
   - the daytime protection of schooled prey is F;
   - only the midday-on-dispersed-prey part is new.
3. **Field-implied a vs capture p, q and speed**: the field-implied a contains all of them. If it were used,
   no sized factor could be stacked on it.
4. **Per-prey plateaus ("Mattnad hos rovfisk") vs a factor on a**:
   - a plateau acts on the ceiling side, per pair;
   - a factor on a acts on the encounter;
   - the R&G plateaus are density-independent, which a factor on a cannot reproduce. One or the other.
5. **Time share T vs a**: a non-foraging share (rest that pays) acts proportionally. It overlaps with the
   window only if the policy rests outside the window.

## Email check

No file in this directory, data/ included, contains the Unpaywall address (checked with grep before
handback).
