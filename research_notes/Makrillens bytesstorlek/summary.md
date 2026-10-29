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
