# Mackerel's accessible share of the planktivore pool (2026-10-29)

Question (user): derive the prey-side `accessible_fraction` f of the planktivore pool for mackerel
(and horse mackerel), the number that decides whether a mackerel group could learn the field's
plankton -> fish switch (resume 223.31, "What a mackerel split would mean").

Script: `scripts/mackerel_planktivore_gape.py`; output `tables/mackerel_planktivore_gape_stdout.txt`,
`tables/mackerel_planktivore_f.csv`. No model output enters; nothing in the library was changed.

## 1. Gape: prey/predator length eaten [D]

SMS 2023 key run stomachs (North Sea 1981 + 1991; `summary_stom.out`), fish prey only, weighted by
stomach-content share x hauls:

| predator | records | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| mackerel | 94 | 0.20 | 0.31 | **0.33** | 0.56 | 0.56 |
| horse mackerel (N + W) | 27 | 0.24 | 0.29 | **0.31** | 0.34 | 0.34 |

The mackerel p99 / max 0.56 is one record (a 22 cm mackerel with 12.5 cm sandeel/sprat). Prey
eaten: sprat, sandeel, Norway pout and herring at 7.5-13.5 cm (the Q3 herring are 8.5 cm, 0-group).
Box predator sizes (NS-IBTS GOV, Q3 biomass): mackerel mean 30.5 cm (25-30 cm dominant), horse
mackerel 32.5 cm. Gape at p95: ~10 cm for both.

## 2. The pool below the gape [D]

Share of each species' biomass below L (box, GOV 2001-25 long run, depth-stratified; HERAS RV Dana
late Jun-Jul pelagic trawls inside the box polygon (not the 44G0/44G1 rectangles), 156 herring
hauls, catch-weighted):

| L cm | herring Q1 | herring Jun-Jul | herring Aug | sprat Aug | pout Aug | blue whiting |
|---|---|---|---|---|---|---|
| 10 | 0.001 | 0.003 | 0.011 | 0.020 | 0.105 | 0 |
| 12 | 0.007 | 0.005 | 0.088 | 0.747 | 0.127 | 0 |
| 14 | 0.052 | 0.012 | 0.186 | 0.967 | 0.169 | 0 |

The herring inside a mackerel's gape is the 0-group: absent in Jun-Jul (0.5 % below 12 cm), 9 %
below 12 cm in August. Pool weights (params_planktivorous_fish.md, J on the season split): Q3
herring 0.78, blue whiting 0.15, pout 0.04, sprat 0.03; Q1 0.88 / 0 / 0.11 / 0.01.

## 3. f [D]

Predator-biomass-weighted, gape = ratio x predator length:

| | Q1 | Jun-Jul (HERAS) | Aug (GOV Q3) | summer mean |
|---|---|---|---|---|
| mackerel p95 | 0.018 | 0.014 | 0.035 | |
| horse mackerel p95 | 0.010 | 0.015 | 0.037 | |
| **group (a-weighted 0.80 / 0.20), p95** | 0.016 | **0.014** | **0.036** | **0.025** |
| group, p90 | 0.012 | 0.010 | 0.024 | 0.017 |
| group, p99 | 0.145 | 0.172 | 0.259 | 0.215 |

Central f = **0.025** (range 0.017-0.036 over gape percentile and season; 0.21 only with the single
p99 record). Presence weighting (mackerel 95 % in Q3) gives the same 0.024. The field mechanism the
user described shows in the numbers: f rises x2.5 from Jun-Jul to August as the 0-group grows into
the gape. A constant f cannot carry that rise.

## 4. What it means for a mackerel group [D, on the frozen fix80 u550 state]

Share of the ceiling (a 0.0391 / 0.00226 per group tonne, ceiling 35.25; zoo / herring):

- f 0.025: zoo wins Feb-Oct at the grid means (Jun 0.71 / 0.35, Aug 0.43 / 0.34); herring only in
  Nov-Jan, when zoo is ~0.3 t/km2 and real mackerel are absent. On the old water column's herring
  cells August is 0.43 / 0.47, a marginal switch.
- f 0.036 (August's own value): herring wins Jul-Oct on the herring cells, Sep at the grid mean.

So a mackerel group with the derived f would be a zooplanktivore with at most a weak late-summer
move to herring cells. That agrees with the SMS stomachs (other food 0.81 of the Q3 diet, herring
0.066), less with a strong coastal switch. Winter: 0.05-0.15 of the ceiling on either prey - a
resident group needs a fat cycle or seasonal absence.

## 5. Side result: the pooled pair today

The current `predatory_fish_preys_on_planktivorous_fish` has f 1. A member-weighted f as in 223.31
(share of the pair's a x the member's prey-side share) with mackerel and horse mackerel at 0.025
(0.61 of the a) and saithe's 19 cm gape (herring below 19 cm ~0.4-0.45, pout ~0.95; 0.33 of the a)
would be roughly 0.2. Not derived here (saithe, hake and spurdog need the same treatment).
Expected consequence: herring is saturated in summer (x 8-25), so nothing changes then, but in
Dec-Jan, when it hides (x ~2, return 14.2-14.5 MJ/t/tick), f 0.2 gives x ~0.4 and ~6.4 MJ/t/tick -
level with the benthivore pair (4.7-6.0). More learned winter eating on the benthivores is the
likely result (the chain-1 gap of 223.31). The same holds for a split: the saithe group keeps only
0.39 of today's herring a.

## Caveats

- SMS stomachs are North Sea wide, 1981 and 1991, coarse prey length classes; 94 + 27 records.
- GOV catchability of small pelagic herring is unknown (no Walker q for herring); HERAS is the
  pelagic check and agrees for Jun-Jul.
- Sandeel, a large mackerel fish prey in Q2 (0.37), has no pool in the model. The split's mackerel
  diet (planktivores 0.30, M2 0.031) includes it; at f 0.025 a mackerel group would realise little
  of that booking - read it as a booking, not a target. Its zoo booking (0.0235) it would realise.
- Section 4 uses the old water column's spatial state, not a trained mackerel group.

## 6. Stage access of a mackerel group on the juvenile compartments (2026-10-30) [D]

Question (user): derive the mackerel group's f_J now that the planktivores and benthivores carry a
juvenile compartment J (resume 223.32; J = below the length at age 1 in Q3: herring 18.2, pout 14.9,
sprat 10.9 cm). Script `scripts/mackerel_stage_access.py` imports `../Årsungarnas andel/scripts/
stage_access.py` (same stage length tables, predator length mixes, quarter mean); output
`tables/mackerel_stage_access_stdout.txt`, `tables/mackerel_stage_access.csv`. Member weights = the
members' shares of today's pooled pair a, renormalised (mackerel 0.80, horse mackerel 0.20). The
pooled water column -> planktivores pair reruns to [0.319, 0.109], the library value (check).

| pair | f_J | f_A | f_J Q1 / Q2 / Q3 / Q4 | effective f at J/B 0.34 / 0.38 / 0.45 |
|---|---|---|---|---|
| **mackerel group -> planktivores** (p95 0.33 / 0.31) | **0.079** | **0.004** | 0.046 / 0.042 / 0.126 / 0.104 | 0.030 / **0.033** / 0.038 |
| gape p90 (0.31 / 0.29) | 0.056 | 0.003 | 0.032 / 0.034 / 0.088 / 0.070 | 0.021 / 0.023 / 0.027 |
| gape ICES-DB 0.28 | 0.035 | 0.001 | 0.021 / 0.025 / 0.056 / 0.039 | 0.012 / 0.014 / 0.016 |
| gape p95 + 0.04 (0.37 / 0.35) | 0.154 | 0.010 | 0.096 / 0.079 / 0.223 / 0.217 | 0.059 / 0.064 / 0.074 |
| rest water column -> planktivores (saithe, hake, spurdog, velvet belly) | **0.69** | **0.27** | 0.64 / 0.60 / 0.75 / 0.79 | 0.42 / 0.43 / 0.46 |
| mackerel group -> benthivores (gadoids) | **0.030** | **0** | 0.001 / 0.000 / 0.090 / 0.030 | 0.005 at J/B 0.15 |

Reading:

- Mackerel reach only the small end of J: the age-0 herring and pout in Q3-Q4 and small sprat; no
  adult in any quarter (f_A 0.004). f_J rises x3 from Q1-Q2 to Q3 (0-group into the gape), the same
  field rise as section 3; a constant f_J (the engine's form) carries it only through J/B, and the
  model's J/B moves 0.34-0.45 over the year.
- The effective f on the whole planktivore biomass, 0.033 (0.014-0.038 over gape and J/B), matches
  section 3's 0.025 (0.017-0.036): the compartment moves the herring the mackerel can reach into J,
  it does not add any. It stays below the 0.05-0.1 window in which a learned late-summer
  zoo -> fish switch appears on the frozen state (section 4; that state predates 223.32/223.37).
  The quarter weighting decides it: the table is the four-quarter mean (stage_access.py's
  convention), but real mackerel are present almost only in Q3 (IBTS Q3 share 0.95). Weighted by
  presence, f_J ~ Q3's 0.126 and the effective f is 0.052 / 0.060 at J/B 0.38 / 0.45 - at the
  window's floor; the annual mean gives 0.033 / 0.038, below it. The gape decides it too: p90 or the
  ICES-DB 0.28 ratio puts it below either way, p95 + 0.04 above. Which constant a resident group
  standing for a migrant should carry is the user's call. Verdict: marginal - at best a weak move to
  herring cells in August-September. The window itself is from the frozen fix80 state with f 1 on
  the pooled pair (before 223.32 / 223.37); a decision needs the share-of-ceiling check rerun on
  the fix83 state with the derived [f_J, f_A] and the model's monthly J/B.
- Validation: implied juvenile share of the take at the box J/B 0.38 = 0.92, SMS mackerel / horse
  mackerel 1.00. The rest water column: 0.61 vs SMS saithe 0.15 (box J/B is 2.7x the stock's; the
  nursery caveat of parameters_B.md section 4).
- For a split the remaining water column gets [0.69, 0.27] on the planktivores (effective 0.43;
  today's pooled 0.19 is diluted by mackerel's 0.61 of the a): saithe reaches age 0 + most of age 1
  and a quarter of the adults.
- Mackerel -> benthivores [0.030, 0] (age-0 gadoids in Q3 only; the sketch's 0.032 gape f).

Applied (resume 228): at the field composition of the group (research_notes/"Makrillens täthet": mackerel
0.80, horse mackerel 0.05 t/km2 -> a-shares 0.964 / 0.036) the same script gives [0.078, 0.0037]
(`tables/mackerel_stage_access_field.*`, run with `"mackerel=0.9642,horse mackerel=0.0358"`); that is the
library's `mackerel_preys_on_planktivorous_fish` stage access. The rest water column stays [0.69, 0.27].
