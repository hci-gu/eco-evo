# Herring and sprat winter profile in the fish -> planktivore pairs - summary (2026-11-11)

Research subagent; the main agent saved this text. Nothing in the library, code, tests, manifest, AGENTS.md or
resume was changed; no training; nothing set from model output; budget_gate was run as a check only, on library
copies in the session scratchpad. Per-source numbers: `evidence.md` (model facts M1-M5, sources E1-E16, derived
D1-D10). Grades: [F] field data or full text, [L] lab, [A] abstract only, [S] secondary, [D] derived, [J]
judgement.

Files (run from `scripts/`; stdout and tables in `tables/`):
- `gov_herring_season.py` -> `gov_herring_season_stdout.txt`: NS-IBTS GOV herring and sprat catch rates by
  quarter, bed class, length and sun elevation (box and Skagerrak), and the NSAS SAM catchability ratio Q1/Q3;
  reads the GOV data in place from `../../Sejens djuplage/data/gov` and `gov_q2q4_1991_97`.
- `seasonal_pairs.py` -> `seasonal_pairs_stdout.txt`, `seasonal_pairs.json`, `gates/budget_gate_*.txt` (12 runs):
  the band pair with a seasonal herring and sprat prey profile, the schooling factor F at the new p, the monthly a
  against the engine's daylight lever, and the gate. Execs `band_pairs_copy.py` ("Overlapp i fiskparen") and
  `school_factor_pout_copy.py` ("Vitlinglojans stim"); `combined_copy.py` and `pair_terms_copy.py` are reference
  copies; also uses p_ppr.json, gate_p.py's run_gate/parse and the combined central factors (combined.json).
- `flow_engine.py` -> `flow_engine_stdout.txt`: the presence-weighted annual a in the engine's frame.
- `papers/`: Kaartvedt et al. 2009, Andersen et al. 2017, Falkenhaug et al. 2022 PDFs (Unpaywall helper; not
  committed) and the Unpaywall records.

```
python3 -I gov_herring_season.py "../../Sejens djuplage/data/gov" "../../Sejens djuplage/data/gov_q2q4_1991_97"
python3 seasonal_pairs.py <repo> <scratch dir> ../tables        # ~8 min
python3 -I flow_engine.py ../tables/seasonal_pairs.json
```
(Directory names written without Swedish letters; on disk "Sejens djuplage" has an a-umlaut.)

## Question

1. Where do the box's herring and sprat sit by day and by night in winter (Nov-Mar), compared with July?
2. What winter profile phi_j follows, with a range and a season?
3. Recomputed with it (light at the winter depth, schooling factor F at the combined derivation's new p), what
   are the annual-mean a of water column (WC), whiting, mackerel and benthivores -> planktivores, and what does
   budget_gate say against the combined central?
4. Is an annual mean enough, or does the engine need a seasonal a?

## Model (checked)

- The herring and sprat phi_j is not seasonal (M1). By day f_b(bed) uniform in the bottom 10 m plus (1 - f_b)
  uniform 10-100 m; f_b is the HERAS July share (0.66 at 70 m, 0.04-0.06 over beds >= 120 m; box equal-cell mean
  0.127). By night uniform 0-50 m.
- The engine's only seasonal lever on a is a(t) = a m(t) (M2): m(t) from the sun elevation against a threshold,
  annual mean 1 - a threshold on day length, not on light intensity.
- Benthivores -> planktivores is the benth_encounter path: the herring band share there is a constant, and the
  schooling rows split the pair pout 0.85 / herring 0.15 (M3).
- Combined central (M4): a WC 0.02797, whiting 0.06357, mackerel 0.04579, benthivores 0.02905; gate planktivore
  prey total 0.89, WC -> krill diet ratio 0.97.

## Findings

### 1. Winter position of the box's herring [F/D]

No acoustic or depth-resolved winter profile exists for 1-2 ringer herring in 3.a (no Q1 acoustic survey of the
area in the ICES acoustic database; a literature sub-agent found no measured winter height above the bed). The
evidence is indirect; it agrees in sign and roughly in size:

| line of evidence | winter / summer near-bed availability | grade |
|---|---|---|
| NSAS SAM catchability, IBTS-Q1 age 1 vs IBTS-Q3 age 1 (same cohort, both delta-GAM GOV indices; HAWG 2023 Table 2.6.2.2) | 2.29 raw (1.82-2.88); 1.63-2.29 after the Walker et al. 2017 GOV length efficiency | [F/D] (E3, D1) |
| Box GOV over 150-250 m beds: Q1/Q3 kg/h 1.04 while the column presence falls to ~0.6 | ~1.7 (a lower bound, see Consequences) | [D] (E1, D2) |
| GOV over beds >= 100 m: midday (sun > 10 deg) vs low sun | Q1 x2.0, Q3 x1.2 | [D] (E2, D3) |
| Day > dark catches of 1-2 ring herring in IBTS Q1; the 1-ringer index uses daylight hauls only; small immature herring leave the bed at night | sign only | [F/S] (E4-E6) |
| Baltic in March and October: clupeids school close to the bottom by day, rise and disperse at dusk | sign only, analogue | [F/A] (E10, E11) |
| Oslofjord in Nov and Mar: herring in the deepest schools by day, feeding in the upper and middle krill layer > 70 m and on benthos; 0-50 m at night | depth, analogue | [F] (E7) |
| Winter krill from 90 m to the bed by day, Calanus at 85-145 m (Oslofjord); Calanus overwinters in the Norwegian Trench | the prey is deep in winter | [S/F] (E8, E14) |

- Near the bed over deep beds in winter: 20-25 cm NSAS 1-2 ringers (Q1 length mix 0.10 / 0.58 / 0.33 for < 20 /
  20-25 / >= 25 cm). In summer the near-bed herring there are WBSS adults (0.81 >= 25 cm). Herring under 20 cm sit
  over beds < 100 m.
- The 100-150 m class shows no rise (0.85).
- Night is the same in both seasons: the fish leave the bed at night (dark deficit -0.9 to -2.3 in log catch).
- Sprat is absent over beds >= 100 m in both quarters; over beds < 100 m Q1/Q3 0.03; sprat is 0.021 of the pool.
  The herring profile is reused for sprat; the choice is immaterial.

### 2. Winter profile [D on F; J]

| parameter | central | range | basis |
|---|---|---|---|
| k on the July bottom-10-m share f_b (cap 0.9) | 2.0 | 1.5-3 | D1-D3 cluster 1.6-2.3; top of the range for the deep-bed juvenile signal [D on F] |
| day midwater layer | 50-120 m (bed - 10 at most) | July 10-100 m to 70 m - bed | E7, E8 analogues, the box's krill day layer at 95-145 m [J on F] |
| night | 0-50 m | unchanged | E7, E2 |
| months | Nov-Mar | Dec-Feb to Oct-Apr | [J] |
| schooling sigma, herring and sprat | day 0.9, night 0 | unchanged | dense non-feeding wintering layers belong to adult stocks outside the box ("Planktivorernas golv") |

At k 2 the box mean f_b by day goes 0.127 -> 0.227.

### 3. Recomputed pairs [D]

Reproduction: with the winter profile equal to July every pair's equal-time a equals band_pairs.json A_band to
1e-6; F reproduces the combined central exactly for whiting, mackerel and benthivores; for WC F 0.542 vs 0.543
(0.2 %, R_eff weighted over spurdog's two seasons).

Herring day term, winter vs the rest of the year, per unit of daytime (central): WC overlap x1.76, R_eff 0.87 ->
0.81 m; whiting x1.78, 0.47 -> 0.45 m; mackerel x0.13, 0.52 -> 0.32 m (the deep layer leaves mackerel's 0-40 m).
The deeper layer lowers the light, but only ~10-15 % in R^2; the overlap dominates. Short winter days were already
in the null reading.

Annual-mean a, equal time (the engine convention), central p and F at the new p:

| pair | combined central | seasonal central | x | range over readings (k 1.5-3, layers, months) | night/day |
|---|---|---|---|---|---|
| WC -> planktivores | 0.02797 | 0.02833 | 1.013 | 1.005-1.022 | 0.114 (comb 0.116) |
| whiting -> planktivores | 0.06357 | 0.06477 | 1.019 | 1.000-1.030 | 0.334 (0.342) |
| mackerel -> planktivores | 0.04579 | 0.04369 | 0.954 | 0.927-0.994 | 0.16 (stomachs, kept) |
| benthivores -> planktivores | 0.02905 | 0.02925 | 1.007 | 1.000-1.012 | 0.236 (0.238) |

- Behind the 1-2 % moves in WC and whiting, F falls slightly (WC 0.542 -> 0.522): the herring day share of a rises,
  and herring are schooled by day.
- Benthivores: the herring day term rises x1.7 in Nov-Mar at k 2 (x2.1 at k 3); the pair +9 % in those months,
  +0.7 % over the year.
- Class shape (a by bed class / equal-cell a; < 100 / 100-150 / 150-250 / >= 250 m): WC 2.21 / 1.21 / 0.41 / 0.16
  -> 2.23 / 1.21 / 0.40 / 0.16; whiting 1.90 / 1.04 / 0.62 / 0.48 -> 1.96 / 1.02 / 0.61 / 0.47. The class tables
  barely move.

Gate (implied / booked M2, equal cells unless noted):

| | combined central | seasonal central | seasonal central, class tables | all other seasonal readings |
|---|---|---|---|---|
| planktivores (prey total) | 0.89 | 0.89 | 0.89 | 0.88-0.89 |
| WC -> planktivores (prey side) | 0.88 | 0.88 | 0.88 | 0.88-0.89 |
| whiting -> planktivores | 0.95 | 0.96 | 0.96 | 0.95-0.97 |
| mackerel -> planktivores | 1.38 | 1.32 | 1.32 | 1.29-1.37 |
| benthivores -> planktivores | 0.81 | 0.81 | 0.81 | 0.81-0.82 |
| WC -> krill, diet gate (field spread) | 0.97 (0.97) | 0.96 (0.97) | 0.96 (0.98) | 0.96-0.97 (0.97) |
| WC -> krill, prey side | 0.97 | 0.96 | 0.96 | 0.96-0.97 |
| benthivores / whiting (prey totals) | 1.06 / 1.03 | 1.06 / 1.02 | 1.06 / 1.02 | 1.05-1.06 / 1.02-1.03 |

OFF pairs: the same four in every run (WC -> mackerel, seals -> WC, whiting -> benthivores, whiting -> epifauna
at the field spread). The gate's "WC krill" figure is the diet-gate share ratio, not the "Overlapp" note's class-2
krill share. The runs `seasonal_central_flow` and `null_flow` scale a by a presence-weighted mean that ignores
m(t): upper bounds only (planktivores 0.93-0.94), not the presence reading.

### 4. Annual mean or seasonal a [D]

Derived monthly a after F vs the engine's m(t) at the pair's new night/day and threshold; residual = derived /
engine (1 = the daylight lever already carries the swing):

| pair, reading | Jan | Feb | Mar | Nov | Dec | Nov-Mar mean | Apr-Oct mean |
|---|---|---|---|---|---|---|---|
| WC, July profile | 0.66 | 0.90 | 0.98 | 0.93 | 0.93 | 0.88 | 1.05 |
| WC, seasonal central | 0.73 | 1.01 | 1.11 | 1.04 | 1.04 | 0.99 | 1.00 |
| whiting, July profile | 0.84 | 0.95 | 0.99 | 0.96 | 0.96 | 0.94 | 1.03 |
| whiting, seasonal central | 0.90 | 1.04 | 1.11 | 1.05 | 1.03 | 1.03 | 0.98 |

- The engine's day-length lever already carries the seasonal swing; the winter profile removes the residual winter
  deficit the July profile left (0.88 -> 0.99 for WC).
- The one month the engine misses is January (WC 0.73, whiting 0.90), already there with the July profile (0.66 /
  0.84): light intensity at depth on short dim days, which a threshold m(t) cannot express; not a herring-profile
  effect.
- Presence-weighted annual a in the engine's frame (a = a_eq x sum der w / sum m w, w = herring presence x predator
  presence): WC x1.007 at herring winter presence 0.6 or 0.3; whiting x0.997 / x0.993. The naive mean without m(t)
  would give x1.13 / x1.03, double-counting the summer weight m(t) already applies.
- Mackerel: the monthly and flow rows are not interpretable (rule A's own night/day ~0.83 builds the derived series,
  while the library keeps the stomach 0.16); the -5 % falls in months when mackerel's box presence is x0.01-0.1.

An annual mean is enough: the winter profile moves the annual a by -5 % to +3 % in every reading and brings the
monthly a closer to what the daylight lever produces. A seasonal a would be needed only if the swing reversed sign
against m(t) (a winter/summer ratio above ~1.5 for a resident predator; a per-pair 12-month multiplier with annual
mean 1, like outside_season). Nothing comes close: benthivores, whose herring term rises most, gain +9 % in winter
months.

## Consequences (flagged, not proposed)

- The WP2 herring winter presence 0.6 (M5), on which the planktivores' `outside_season` rests, is a GOV Q1/Q3
  ratio. If near-bed availability doubles in winter, the true winter column presence is nearer 0.3 and the library
  overstates the winter planktivores (the same circularity makes D2's x1.7 a lower bound). Immaterial for the a's
  (engine-frame factor x1.007 / x0.993 at presence 0.3); it may matter for the planktivores' own winter budget and
  the realised winter M2.
- The winter near-bed fish over deep beds are 20-25 cm 1-2 ringers; their meaning for the stage_access windows is
  out of scope.
- SMS Q1 fish-predator M2 on 1-ring herring is 1.6x the annual mean ("Planktivorernas golv" E21), which fits
  resident gadoids meeting juvenile herring near the bed in winter; the model's winter a stays at or below summer
  because of light. Not tested further.
- The July profile puts half of the day herring below ~55 m over deep beds, while HERAS July logs over beds >= 150 m
  have 0.26 below 50 m ("Vitlingens fiskbyten" step2): a separate summer-profile question.

## Gaps

- No direct winter measurement for the box's herring: no Q1 acoustics in 3.a, no height above the bed or
  bottom-5-10-m share for 1-2 ringers; k rests on GOV indices, which mix vertical position, horizontal shifts and
  length.
- The winter day midwater layer (50-120 m) comes from fjord analogues (Oslofjord herring, krill, Calanus); no
  open-Skagerrak winter depth for krill or Calanus.
- Vestheim et al. 2014, Woodhead 1963, ICES CM 1998/D:12 read by a sub-agent, not verified; Petrakis et al. 2001
  and Cardinale et al. 2003 abstracts only.
- f_m is predator occupancy, not search time ("Overlapp i fiskparen" caveat 1); saithe's winter position is the
  July profile; only the prey side was made seasonal.
- Norway pout's winter profile not examined (pout carries 0.52-0.75 of the fish pairs' planktivore a).

## Verdict

The box's NSAS 1-2 ringers very likely sit nearer the bed by day in winter than in July: the daytime bottom-band
share roughly doubles, k 2 (1.5-3) [D on F], from the SAM catchability ratio, the GOV midday concentration and the
deep-bed Q1 catches; by day likely deeper in midwater, 50-120 m [J on fjord F]; by night 0-50 m.

In the section 269 band pairs with the combined p and F this changes the annual-mean a by +1.3 % (WC), +1.9 %
(whiting), +0.7 % (benthivores) and -4.6 % (mackerel); full range -7 % to +3 %. The gate does not move
(planktivores 0.89, WC -> krill 0.96-0.97 vs the combined central's 0.89 / 0.97).

An annual mean is enough [D, robust over every bracket]: the engine's daylight lever already carries the derived
monthly swing and the winter profile narrows the residual. The only misfit is a January light-intensity deficit on
WC (~-25 %), unrelated to the herring profile. Keep the July profile in the library: the change is inside the
derivation's other uncertainties.

## User decisions

- 2026-11-11: keep the July profile in the library; the winter-presence question
  (0.6 vs ~0.3 in the planktivores' outside_season) stays flagged open.
