# Fish on fish prey, krill and zooplankton: is intake food-limited? (2026-11-07)

Written by the research subagent, saved by the main agent. Research only: nothing in code, library,
tests, manifest, AGENTS.md or the resume was changed. The full trail is in `evidence.md`, the
arithmetic in `scripts/field_K.py` (output in `tables/field_K_stdout.txt`), and licences in
`data/README.md`. Two downloaded PDFs are publisher-copyright and must not be pushed to the public
repo: `data/juanes2002/` and `data/saithe_growth/`. Tags: [F] stated in the source, [D] derived
here, [J] judgement, [A] abstract only.

## Verdict

1. **In the model, saturation is per predator, not per pair** [D].
   - With shared handling, the most saturating prey sets the denominator for all of a predator's
     prey:
     - krill for whiting (x = a h B_vis 34), the water column (20) and the benthivores (12);
     - zooplankton for the planktivores (6.2).
   - These predators therefore feed at T x ceiling, where T is the eat share, whatever x their
     fish-prey pairs have.
   - The field puts the plateau well below Cmax:
     - whiting feed at 0.25 (0.10-0.35) of Cmax overall;
     - Rindorf & Gislason 2005 give a separate plateau per prey group (clupeids 0.10-0.17, gadoid
       prey 0.06-0.12, other food 0.08-0.22), each independent of the others and of prey density
       from 20 to > 9000 herring per trawl-hour [F].
   - So the field constraint is on T per prey group (a window on the whole response), not on a or
     K.
   - A multispecies shared-handling fit was "very poorly" supported, and fish and other-food intake
     were uncorrelated. That is field evidence for per-group windows that add rather than compete
     (evidence, not a code proposal) [F/J].
2. **Fish-prey pairs are not "almost all saturated" once the visible shares are applied** [D].
   - Model x: whiting 1.6, water column 2.3, benthivores 8.1, mackerel 0.12.
   - Field, local scale (~1-6.5 km): saturated, with Type II close to the data.
   - Field, stock scale: clearly food-limited.
     - Whiting length-at-age and stomach contents tracked forage fish in 2000-07 (Lauerburg 2018
       [F]); the implied stock-scale x is 0.4-16 [D].
     - Saithe growth rises with Norway pout biomass (Cormon 2016 [F]).
     - Barents cod show the same against capelin.
   - Reconciliation: the half-saturation K grows with the averaging scale. Barents cod on capelin
     give ~0.1 t/km2 from a 2 km Holling fit, 0.4-1.0 where the main rise ends, 1.6 at ~100 km,
     and 2.5-10 where the stock curve levels out [D].
   - Confidence: medium on "limited at stock scale, saturated locally"; low on any per-pair a.
3. **Krill pairs: no field or lab functional response of any fish on euphausiids exists** (gap).
   - Whiting and pout follow the krill layer day and night (Onsrud 2004), so the twilight window
     does not apply: tau ~1 [J].
   - Krill dive away from passing fish schools (ICES JMS 2024 [A]), which acts on a.
   - Whether whiting are food-limited on krill in the field is unknown. The field feeding level
     says the plateau is not Cmax.
4. **Mackerel pairs are unsaturated in model and field.**
   - Model x: zooplankton 0.55, krill 0.9. Stock-scale growth is density-dependent (Olafsdottir
     2016, Homrum 2022 [A]).
   - Mackerel start feeding only at 4.4-14.7 Calanus/L (Macy 1998), 15-75x the box mean. They feed
     in patches, which is within-cell patch structure, not a or K [D].
5. **Where the capture window goes.**
   - Capture success per attack (piscivores ~0.5, planktivores 0.7-0.8; Juanes 2002 [F]) belongs in
     a only.
   - A multiplier on a alone cannot give a plateau below Cmax.
   - The plateau comes from twilight-only capture of schooling prey and partial predator-prey
     overlap (Rindorf 2005; Kempf 2008 [F]). That is a window on the whole response, which in the
     engine is the eat share T.
   - Inside the window the cap may instead be on the capture rate (a cut ceiling). The data cannot
     separate the two:
     - with the window on the whole response, the library a matches Rindorf at a trawl clupeid catch
       efficiency c ~0.04;
     - with the cap on the ceiling, at c ~0.2-0.4;
     - c is unknown in 0.05-1.

## Per pair (t/km2 box mean; visible prey upper bound, hiding ignored)

| pair | visible B | model K / x | field K lo / c / hi | implied a | model a | ratio model/implied | tau | sources |
|---|---|---|---|---|---|---|---|---|
| whiting -> planktivores | 0.37 (share 0.079) | 0.23 / 1.6 | 0.004 / 0.043 / 0.39 (12-14 cm clupeids) | 0.11 (0.012-1.3); 0.011-0.019 if the cap is on the ceiling | 0.0212 | 0.19 (0.02-1.7); 1.1-1.9 | 0.10-0.17 | Rindorf 2005 Table 2; Lauerburg 2018 |
| whiting -> benthivores (gadoid prey) | - | 0.033 / - | 0.018 / 0.13 / 0.80 | 0.054 (0.009-0.39) | 0.21 | 3.9 (0.5-24) | 0.06-0.12 | Rindorf 2005 |
| water column -> planktivores | 1.77 | 0.77 / 2.3 | no local fit | - | 0.006 | - | no window data | Cormon 2016 |
| benthivores -> planktivores (cod-capelin proxy, flagged) | 1.72 | 0.21 / 8.1 | 0.11 / 0.6 / 1.57 | 0.0070 (0.0027-0.038) | 0.0198 | 2.8 (0.5-7.4) | - | Fall 2021, MULTSPEC, WKCAPELIN 2023 |
| mackerel -> planktivores | 0.05 | 0.38 / 0.12 | none | - | 0.0266 | - | - | - |
| planktivores -> zooplankton | 9.7 | 1.58 / 6.2 | none (lab filter switch at 18 /L, ~50x box mean) | - | 0.0122 | - | - | Gibson & Ezzi [A]; Bachiller 2016, Debes 2012 |
| mackerel -> zooplankton | 9.7 | 17.5 / 0.55 | a from lab clearance | - | 0.00084 | - | - | Macy 1998 |
| whiting / water column / benthivores / planktivores / mackerel -> krill | 5 | 0.15 / 0.25 / 0.42 / 0.96 / 5.3; x 34 / 20 / 12 / 5.2 / 0.94 | none | - | 0.043 / 0.0241 / 0.013 / 0.018 / 0.0025 | - | ~1 for whiting and pout [J] | Onsrud 2004, Kaartvedt 2005, 2024 avoidance [A] |

## Gaps

- No field or lab functional response of any fish on krill.
- No absolute field K for herring or pout on copepods.
- The EXPO trawl's wing spread and clupeid catch efficiency c, which dominate the whiting K by
  ~20x (Rindorf 2002's full text was blocked with a 403).
- No local K for saithe, hake or cod on herring or pout in the North Sea. The benthivores rest on a
  Barents capelin proxy.
- No diel window data for the water column or the benthivores on fish prey.
