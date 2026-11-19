# Benthivores on bed prey: is intake food-limited? (2026-11-07)

Written by the research subagent, saved by the main agent. Research only. The full trail is in
`evidence.md`. Scripts are `scripts/pair_K_bounds.py`, `scripts/kattegat_fishfood_patchiness.py`
(needs xlrd, run from a venv) and `scripts/openalex_abstract.py`; their output is in `tables/`.
Do not commit the ICES slides PDF (`data/ices_pres_hiddink/`, no licence) or the WUR/CSIC landing
HTML (`data/README.md`). Tags: [F] stated in the source, [D] derived here, [J] judgement, [A]
abstract only.

## Verdict

At today's prey densities the benthivores' TOTAL intake of bed prey is saturated: the field feeding
level is set by foraging time. The intake of each prey separately is density-dependent, through
the shared handling the engine has had since 242. Confidence is moderate that the model's a is not
1-3 orders too high, and low on anything finer.

- **The exposure note's a (infauna 0.036) is supported.**
  - The Irish Sea trawling gradient is the only prey-density gradient with stomach-intake data.
  - It gives a field lower bound on a 6.5-19x below the model value, at sigma 0.3-0.45. Sigma is
    the between-station SD of ln mean fullness, a judgement because Johnson 2015 does not report it.
  - Nothing found contradicts a 0.036.
- **The feeding-level note's inversion is circular.**
  - At x >> 1 the ration equals foraging share x ceiling, so the ration says nothing about a unless
    tau is known. Setting tau = 1 assumes the answer.
  - Its a values (0.00015-0.0025) are disfavoured by likelihood ratios of 73-6700 at sigma 0.3,
    7-50 at 0.45 and only 3-9 at 0.6. This is weak if sigma is large.
- **Third reading** (that of "Mattnad hos bottenbyten", a-umlaut, now built into the engine):
  total bed intake saturated, each prey's share responding to its density.
  - Ratio-dependence (intake set by prey per consumer, as Hiddink 2016 claims for the Kattegat)
    remains open. Their stomach data are not public.
  - The public condition data at Hiddink's own fish-food size class give the opposite sign.
- Also cited: "Tathetsberoende och interferens hos torskfiskar" (a-umlaut): no significant
  density-dependent growth in North Sea cod, haddock or Norway pout (Lorenzen & Enberg 2002).

## Key evidence

- **Irish Sea, Johnson et al. 2015** [F].
  - 15 stations; common plaice prey fell with trawling (R2 0.51), over 2.4-18.5 t/km2.
  - Plaice fullness did not change (F1,13 = 0.004, p 0.949), nor did stomach energy (p 0.981); dab
    fullness p 0.76.
  - Diet composition shifted (p 0.007): diets track prey.
  - Predicted intake ratio lowest/highest station [D]: model a 0.70 (reachable 0.2) or 0.92 (share
    1); feeding-level a's 0.14-0.23.
  - Caveat: the prey:fish ratio stayed constant, so ratio-dependence also predicts flat fullness.
    The test rules out only the prey-dependent form, which is the model's form.
- **Hiddink 2011** (abstract only): plaice condition falls with lower infauna production, while
  Johnson found fullness flat in the same species. Condition is not an intake response.
- **Kattegat, Dryad reanalysis** [D].
  - Fish food 11.6-78 t/km2; plaice condition does not track it (r2 0.00; long rough dab 0.05).
  - Against food per fish the slopes are negative (r2 0.15-0.19).
  - Model a predicts an intake ratio of 0.92 between the extremes, the low a's 0.18-0.48. Weak
    support, not a test.
- **Patchiness** [D].
  - CV between 0.1 m2 grabs within a station: 0.35 prey, 0.50 fish food.
  - CV between stations at km scale: 0.38-0.42.
  - CV <= 0.5 costs 0-6 % of intake, so the 1 km cell mean is a fair stand-in for local density.
- **The model with shared handling** [D].
  - X_bed 178 (197 with fish and krill).
  - Own-density elasticity: infauna 0.58-0.62, epifauna 0.49-0.54, Nephrops 0.94; whole-menu
    elasticity 0.006.
  - At 0.1x its own biomass a pair keeps 0.11-0.18 of its intake.
- **Supply** [D]: field consumption takes 0.20 /yr of the exposed infauna (18.8 t/km2), against a
  refuge exchange of 2.5 /yr. Box-mean depletion does not limit intake.

## Per pair (model's prey class, box mean)

| pair | field K (t/km2) | implied a = 1/(K h) | model a | model / field | tau | sources |
|---|---|---|---|---|---|---|
| infauna (h 112) | <= 8-24 total (<= 1.6-4.7 reachable at 0.2), sigma 0.3-0.45; none at 0.6 | >= 0.0019-0.0056 | 0.036 | >= 6.5-19x above the lower bound: compatible | any (a bound on a while foraging) | Johnson 2015 [F]; Hinz stations [D] |
| epifauna (h 258) | 0.0065-0.18 accessible (lab, scaled ~1/L [J]) | 0.022-0.60 | 0.15 | 0.25-7x | 1 within a lab trial | Persson 2012 [F]; Smith 2013 [F] (total intake flat over 5-250x epifauna contrasts) |
| Nephrops (h 286) | none | - | 0.44 | - | - | gap; minority prey, elasticity ~0.94 for any a within ~10x |

- Hvingel & Kingsley's near-linear stock-scale response of cod to Pandalus is what the shared
  denominator gives a minority prey, not evidence of a low a [J].
- Extrapolated to the box (infauna 94 t/km2): X_infauna >= 4-12, so >= 0.8 of the ceiling while
  foraging.
- Caveats:
  - The bound is on plaice and the model a is a group value: order of magnitude only.
  - The Irish Sea muds are 4-40x poorer than the Skagerrak.

## Model-side consequence [J]

Every fish group has `appetite: reserve_room` (no hunger gate). At X ~180 the benthivores' intake is
the learned eat share x the ceiling, so the field level 0.22-0.53 of Cmax can only come from the
policy's foraging share. Cutting a by 1-3 orders to produce it would contradict the Irish Sea
gradient. The check is the realised benthivore feeding level in the long-run probe, not a parameter
change.

## Gaps

- Station-level sigma of fullness (Johnson's raw stomach data).
- Full texts of Hiddink 2011 and 2016 and the Kattegat stomach data; blocked (Cloudflare or 403),
  not circumvented.
- Any functional response of gadoids or flatfish on infauna; an adult-scale one on epifauna; any
  on Nephrops.
- Local depletion of the exposed infauna at home-range scale.
- A Skagerrak gradient with stomach data.
