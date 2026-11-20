# The encounter-realisation factor per group (2026-11-20)

Derivation only (main-agent fork; saved by the main agent). Nothing in the library, code, tests, manifest, AGENTS.md
or the resume was changed, and nothing was committed. Every field number is reused from an earlier note; there was
no new literature search.
- **Trail:** `evidence.md` (E0-E8).
- **Arithmetic:** `scripts/anchors.py` -> `tables/anchors_stdout.txt`.
- **Sizing:** library COPIES (`scripts/make_copy.py`), frozen fix105 (`fill_decomp_all.py`) ->
  `tables/fill_decomp_copy_*`.
- **Gate:** `scripts/gate_on_copy.py` -> `tables/budget_gate_copy_a.txt`.

Factor = the field whole-product a / today's chain a under the anchor's own conditions. No model output, FIELD_QB
or frozen-model potential set a factor.

## Per anchor

| group, pair | anchor | field | chain, same conditions | factor central (range) | tags | basis |
|---|---|---|---|---|---|---|
| planktivores -> zooplankton | Arrhenius & Hansson 1999, YOY herring, Baltic (V1) | 46 L/g/h, daylight | 864 L/g/h, size-free R (437-2258); 78 with R ~ L | **0.36 (R ~ L, x summer q) or 0.04 (size-free)**; YOY raw 0.053 / 0.59 | F, D, J (adult transfer, R scaling) | stomach consumption vs measured plankton; independent of the box ration. R(L) and the factor cannot be separated |
| planktivores -> zooplankton | Kulke 2018, field sprat 12 cm (V2) | 0.5-1.0 m3/h | 0.86 m3/h | 0.58-1.17 (x q 0.45-1.0) | F, D | field functional response; NOT independent (scan 0.5 was chosen with it); sprat is 0.05 of the a |
| planktivores -> krill | winter stomachs: group, pout NS 1980, pout and herring Oslofjord | 0.15-0.75 %BW/d | Jan a_eff 0.0115 at D_lib 5 | **0.012 (0.007-0.037)**; x1.7 at D 3, x0.4 at D 20 | D on F, J | stomach-based; ICES DB rows share data with the booked ration. Winter only. On a x D |
| whiting -> krill | ICES DB 1981 Q1 (NS, Skagerrak), Skagerrak Apr 1991, Oslofjord Nov/Mar/Jun | 0.14-0.94 %BW/d | Jan a_eff 0.031 (Apr/Jun 0.028) | **0.0099 (0.0024-0.030)** | D on F, J | stomach-based; shares data with SMS / the booked krill M2. On a x D |
| whiting -> planktivores | Rindorf & Gislason 2005, clupeids | slope K Wmax / half-saturation 0.0178 km2/t/tick | 0.108 (library x Sep q at Rindorf's bed mix) | **0.16** (0.30 equal-cell; corners 0.011-8.7) | F, D, 3 J (gear, catchability) | stomach series + trawl, 1988-89; independent of the box ration [J] |
| whiting -> benthivores | Rindorf & Gislason 2005, gadoids | 0.0040 | 0.33 | 0.012 (corners 0.0013-0.27) | F, D, J; weak (prey include juvenile whiting, pout) | as above |
| water column | none (Fall & Fiksen 2020 cod-capelin: models "high, but reasonable" vs stomachs; no ratio) | - | - | **open** | F (PROXY) | - |
| mackerel | none (its zooplankton pair is the Macy lab clearance and is excluded) | - | - | **open** | - | - |
| benthivores | excluded (294.4) | - | - | - | - | - |

**Coverage.**
- Two groups have anchors, whiting and the planktivores, each on two of its prey types.
- The water column and mackerel are open. Transfers to them would be [J]: whiting's fish slope 0.16-0.30 to the
  water column's fish pairs, and krill ~0.01 to the water column's and mackerel's krill pairs.
- No anchor reaches whiting -> epifauna, the planktivores' small members, or any water column or mackerel pair.

**Not shared.** Within each predator the anchored factors differ by prey type by >10x: krill ~0.01, fish ~0.16,
zooplankton 0.04-0.36. One shared factor of x0.25-0.35 is not what the anchors show.
- Caveat 1: the krill factor is on a x D. The library's krill density (5 t/km2) is unmeasured at every site, so
  part of it may be density or occupancy rather than encounter realisation.
- Caveat 2: the krill rows are mostly winter.

## Frozen sizing (library copy a: anchored pairs only; fix105, seeds 0/1)

Copy a: whiting -> krill x0.0099, -> planktivores x0.164, -> benthivores x0.012; planktivores -> krill x0.0124,
-> zooplankton x0.36. Every other pair unchanged.

| group | uncapped, live -> copy | realised | fill |
|---|---|---|---|
| planktivores | 0.54 -> 0.25-0.27 | 0.26 -> 0.23-0.24 | 0.91 -> 0.70-0.72 |
| whiting | 0.74-0.75 -> 0.45-0.48 | 0.39 -> 0.37-0.38 | 0.98 -> 0.87-0.94 |
| water column, mackerel, benthivores | unchanged (no factor) | | 0.99 / 0.93 / 0.98 |

- **Planktivores.** Monthly fill is 0.14 in Feb, 0.99 in Jun-Aug and 0.55 in Dec, against the field's 0.36 / 0.96 /
  0.68 (minimum April). They are food-limited Nov-Mar. Krill falls to 0.03 of the potential.
- **Whiting.** The potential moves onto epifauna, an unanchored pair, at 0.66. The fill minimum falls in Oct-Nov,
  the reverse of the field, which is the sated placement artefact of 294.6.
- **Copy b** (planktivores -> zooplankton at the size-free reading, x0.042): planktivores uncapped 0.10, fill
  0.25-0.27. They are near starvation.
- **Gate on copy a.** The diet gate puts four new pairs OFF: planktivores -> krill (encounter share 0.018 vs booked
  0.398), whiting -> krill (0.06 vs 0.56), whiting -> epifauna (0.67 vs 0.17) and whiting -> zooplankton. Prey
  side: planktivores -> krill implied M2 0.10 vs booked 2.07. Factors on the anchored pairs alone break the booked
  diets, so either the unanchored pairs carry a similar excess or the krill cut is too deep.

## Contradictions between notes

1. **"Shared" vs prey-specific.** "Mötesteorins överskott" Q2 / resume 294.4 read the excess as "shared within
   each predator; krill residual <= ~1.5-2x". The whole-product anchors give krill ~0.01 against fish ~0.16 and
   zooplankton 0.04-0.36.
2. **Two whiting-krill factors that agree once corrected.** "Aptit eller fångst i krillskiktet" puts f 0.03-0.26
   on the yardstick X_eff 4.47. On the per-cell X (~24; 14-33 in potential_season) the same field rations give
   0.006-0.05. This agrees with E2 (0.002-0.03), so it is a units issue, not a conflict.
3. **Inconsistent model-side "needed" factors.** Resume 294.4 and 294.6 quote different values for whiting (0.15,
   0.114, ~0.03). All are model algebra and none is used here.
4. **Rindorf converted two ways.** The 220-era note converted Rindorf's half-saturation through the engine's h and
   read "passes" (a x1.28 today). The field's own low-density slope says x0.16-0.30. The note mixed the field
   plateau with the model ceiling.
5. **R scaling.** The planktivore note kept size-free R while stating that V1 favours R ~ L. Under the library's
   own convention, V1 gives x0.05, a reading copy b shows to be unviable.
6. **Whiting -> benthivores.** The gate flags the pair as an UNDER-encounter (share 0.28 of booked). Rindorf's
   gadoid slope says its a is ~80x high. The booked M2 0.1067 and the field slope disagree; Rindorf's gadoid set is
   mixed, so this is weak.

## Main agent's reading (2026-11-20)

- The krill factor (~0.01) is winter-only and on a x D with D unmeasured. Applied as an annual factor it takes the
  planktivores' krill to 0.03 of their potential while their booked krill M2 (2.07, from field diets) needs krill
  to be a main prey, so as an annual factor it contradicts the field diet the pair was booked from. The krill
  pairs' excess is real but its size and season are not settled by these anchors.
- The fish-prey (0.16-0.30) and zooplankton (0.36 at R ~ L) factors are the two field-sized, ration-independent
  results; both are single-study anchors with wide corners.
