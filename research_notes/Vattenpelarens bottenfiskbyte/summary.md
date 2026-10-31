# Water column -> benthivores: which pair factor is off (2026-10-31)

Research only: nothing in the library, the manifest, the code or the tests was changed, and the budget gate was not re-run. Written by the research subagent, saved by the main agent.

**Tags:** [F] field value or stated in a source, [D] derived here from data, [J] judgement.

**Where things are** (this folder):
- `evidence.md`: field diets, access bookkeeping, clearance tests and the overlap candidate.
- `evidence_literature.md`: layers, new diet sources and the H_juv check.
- `scripts/`: run with `python3 -I` from this folder; `rest_diet_ices.py` and `stage_access_check.py` take path arguments (header docstrings).
- `tables/`: their stdout.
- `data/lit/`: downloads, listed in `data/lit/README.md`.

No model output set any value.

## Verdict

**1. The answer depends on the convention, and here the two conventions give two different answers.**
- **Under the 220/223.26 eat-share convention**, a 0.0651 is not contradicted by the field.
  - In that convention a is defined inside the prey's layer, and the time spent there is the learned eat share.
  - On the time-allocation check of the whiting audit, the bed pair pays 0.57 of the ceiling per unit eat share at today's density, against herring 0.66, zooplankton 0.26 and whiting 0.17 (`tables/clearance_test_stdout.txt`).
  - Whiting's epifauna pair paid 0.97 against 0.44-0.76 on the same check. This pair does not out-pay the herring pair.
- **Under the overlap convention 232 applied to whiting's bed pair** (concurrent search, which is what the diet gate assumes), the pair is 7-12x too high.
  - That is its clearance relative to the planktivores, against five field diets. It is 9-22x relative to zooplankton.
  - The field needs x0.09-0.15 on a x f_eff.
  - The structural factor is that every member searches inside the 2-m layer of settled 0-1 group gadoids for all of its foraging time.
  - Saithe (0.77 of the a) is a 30-100 m benthopelagic and pelagic forager. Field-based estimate of its foraging time within ~2 m of the bed: 0.05 (0.02-0.15) [J].
- For whiting, both readings gave 0.006-0.024. Here they diverge.

**2. If the overlap value is adopted, the engine as built would starve the pair.**
- At a 0.0055 the pair pays 0.11 of the ceiling per unit eat share, against 0.57 for herring.
- The engine allocates eat share to one prey at a time, and each pair saturates on its own (232).
- A policy would then put about nothing on the bed, so the realised M2 would be about 0, not the booked 0.083 that sits in the benthivores' growth budget.
- The overlap value only makes sense together with either a shared-handling (multi-prey) response, or a decision that the concurrent-search meaning, not the eat share, defines the pair.

**3. Two observations that fit the concurrent-search reading.**
- The trained runs' 25-39 % benthivores is close to the concurrent-search share at the trained J/B 0.05-0.08, which is 0.28-0.33. So the trained policies behave like the gate's concurrent searcher. That is why the gate predicts them and the time-allocation check does not.
- Prey hiding pushes the trained runs further toward the bed. With the herring pair floor 0.25 against the benthivores' species floor 0.55, hiding moves c_ben/c_plk x2.2 toward the bed. At full hiding the library concurrent-search share is 0.60.

**4. The a and the stage access do not double-count a size restriction.**
- The a has no prey-size factor: prey share 1, and the capability score only gates which stages count, at a threshold of 0.1.
- Two bookkeeping faults in the access:
  - **Normalisation.** The window share was normalised over each member's diet-identity species only. The engine multiplies f_A by all adult benthivore tonnes, 30 % of which are flatfish, rays and gurnard. Fixing this moves f_A 0.027 -> 0.016.
  - **Length weighting ("aw").** The window was averaged over predator lengths by biomass. But a per member tonne scales as about L^-2 (the per-stage values confirm it: 2.55 against 2.65 predicted). So 40-55 cm saithe carry about 80 % of saithe's a and have 12-16 cm windows. Weighting the window by each length's share of the a gives [0.312, 0.0070]. f_eff at J/B 0.15 goes 0.088 -> 0.053.
- The length-weighting fault is common to the rest group's fish pairs (planktivores x0.68, whiting x0.64), so against the planktivore anchor this pair moves only x0.89.

**5. The box juvenile share does not explain the gap.**
- At the trained J/B 0.05-0.08, f_eff is 0.047-0.059 against 0.088 at J/B 0.15.
- The pair is still 4-7x the field ratios at that J/B. The realised M2 of 0.11-0.30 (1.3-3.6x booked) occurs with the lower access.

**6. Per member, every member is high on benthivores relative to planktivores** (aw access, `tables/member_test_stdout.txt`):

| Member | Model / field, c_ben/c_plk | Field source |
|---|---|---|
| saithe | 5.6-9.0 | CRR 164, CRR 219, SMS |
| spurdog | 8.5 | Ecopath 1991 |
| hake | ~270 | Skagerrak 2000+ ICES, 115 stomachs: benthivores 0.4 % |
| anglerfish | 17 | Ecopath; its planktivore a meets only pout and blue whiting |

- Saithe's whiting/planktivore ratio is also high (3-19x), while its benthivore/whiting ratio is near field (0.25-2.5). Both of saithe's near-bed pairs are inflated by a similar factor against two pelagic anchors that agree with each other. That is the signature of saithe's layer, not of the juveniles.

**7. The juvenile layer is confirmed and is not a lever.**
- H_juv 2 m (1-5) is supported in the literature [F]. Lough et al. 1989: "within a few centimeters of the bottom" by day, 3-5 m off at night.
- The section 220 MC lower end for the pair (x0.32) cannot reach the field's x0.09-0.15 without a structural factor.

**8. The seasonal shape is a level fault with a seasonal tail.**
- Saithe alone, model/field of c_ben/c_plk: Q3 5-8, Q2 5-8, Q4 12-13, Q1 infinite, because 1991 holds no haddock in Q1 (1981 winter held 3-10 %).
- The model's Q1 take is 0.28-0.55 of Q3.

## Parameters

| Pair / key | Current | Overlap convention (central, range): only together with shared handling or a decided concurrent-search meaning | Eat-share convention | Tag | Basis |
|---|---|---|---|---|---|
| rest -> benthivorous_fish `max_intake_rate` | 0.0651 | **0.0055** (0.0026-0.013) | 0.0651 unchanged | D (on J overlaps) | 0.0651 x sum of (member share x O_m). O_m: saithe 0.05 (0.02-0.15), spurdog 0.15 (0.05-0.30), hake 0.25 (0.10-0.50), velvet belly 0.10 (0.05-0.30), anglerfish 1. See note A |
| `stage_access` | [0.43, 0.027] | [0.437, 0.035] (low [0.488, 0.059], high [0.375, 0.020]): aw with the new member shares (saithe 0.45, spurdog 0.18, hake 0.21, velvet belly 0.05, anglerfish 0.12) | aw with today's shares: [0.312, 0.0070] | D | `scripts/proposal.py`, `scripts/stage_access_check.py` |
| f_eff at J/B 0.15 | 0.088 | 0.095 (0.073-0.124) | 0.053 (aw) | D | |
| f_eff at J/B 0.05 / 0.08 | 0.047 / 0.059 | 0.055 / 0.067 | 0.022 / 0.031 | D | |
| a x f_eff at J/B 0.15 | 0.0057 | 0.00053 (0.00032-0.00096) | 0.0034 (aw) | D | field needs 0.09-0.15 x library |
| `predation_mortality` | 0.0833 | unchanged | unchanged | F | field benthivore share of the checked diet 0.062-0.067 (CRR 164, CRR 219, group-weighted), booked 0.082 |
| `dark_ratio`, `handling_time` | 0.48, 151.1 | unchanged | unchanged | | not tested |
| hake's share of the new a | 0.07 | 0.21 (flagged) | 0.07 | F vs D | 269x its only in-area diet |

**Note A, on independence of the saithe overlap.** It is not independent of the diets: the literature argued the 0.05 foraging share partly from CRR 219's Q1 haddock zero. The independent layer-only alternative (saithe biomass within 2 m = 0.05-0.10, from H_saithe ~50 m (10-100) plus the near-bed ceiling of spawning aggregations) gives a of about 0.0055-0.008, inside the bracket.

**Note B, on the strongest-sounding source.** The 30-100 m best-match height in Aglen 1996 also holds for large haddock and cod. It is a trawl-acoustic calibration height, not a position. The saithe-specific evidence is:
- Skilbrei & Otterå 2016: depth tags spread over about 100 m.
- Kaartvedt 2005.
- Bed-referenced shares (Neilson 2003; DFO 2003/075), which exist only for aggregations sampled because they sat on the bed.

## Concurrent-search diet shares against field

Shares of the five checked prey (`tables/clearance_test_stdout.txt`, `tables/proposal_stdout.txt`):

| | plk | zoo | ben | wht | mack |
|---|---|---|---|---|---|
| library, all visible (gate) | 0.39 | 0.11 | **0.42** | 0.06 | 0.01 |
| library, trained J/B 0.05 | 0.49 | 0.14 | 0.28 | 0.08 | 0.01 |
| library, every prey at its floor | 0.25 | 0.06 | 0.60 | 0.08 | 0.02 |
| aw access only | 0.39 | 0.17 | 0.37 | 0.06 | 0.02 |
| overlap central (low-high), other pairs aw | 0.57 | 0.24 | **0.083** (0.052-0.143) | 0.08 | 0.02 |
| the same, plus saithe overlaps on plk 0.78, zoo 0.8, wht 0.25 [J] | 0.57 | 0.25 | 0.10 (0.065-0.17) | 0.04 | 0.03 |
| field V1 booked M2 x B | 0.53 | 0.33 | 0.082 | 0.046 | 0.012 |
| field V2 group with CRR 164 saithe (1980-82) | 0.51 | 0.39 | 0.066 | 0.026 | 0.013 |
| field V3 group with CRR 219 saithe (1991) | 0.72 | 0.16 | 0.067 | 0.040 | 0.015 |
| saithe CRR 219 alone, box length mix | 0.73 | 0.18 | 0.064 | 0.020 | 0.003 |

- Clearance relative to the planktivores, model/field, overlap central: 0.95 (V1), 1.14 (V2), 1.57 (V3).
- How the groups were built: saithe is 0.83 of the group's consumption. The other members come from Ecopath 1991 (spurdog, anglerfish) and the ICES stomach database for the Skagerrak 2000+ (hake, ling); velvet belly and pollack are [J].

**Caveat on the field diets:**
- Every saithe weight table is 1980-91, NW and northern North Sea, at prey densities of that time that we do not know.
- No in-area saithe diet by mass exists. The ICES database holds 408 northern North Sea saithe stomachs with 21 g of weighed prey.
- So the 7-12x is robust, but the +/-2x around the overlap value is not.

## Open points (user's call)

1. **Convention.** Eat share (a in-layer, time = policy) or overlap (a includes time in the layer, concurrent search). The overlap route needs a shared-handling (multi-prey) response or an accepted concurrent-search meaning, otherwise the pair goes unpaid. Whiting's 232 change took the overlap route on one pair; it should be applied the same way to every layer-separated pair.
2. **Consequence for rest -> whiting.** Under the same convention this pair is 1.7-3.1x high against the planktivores, with the same geometry (saithe inside the 10-m whiting layer). Saithe's overlap of 0.25 would cut it. 232 already reports its realised M2 falling 0.18 -> 0.03 in fix86. To be weighed, not proposed.
3. **Length-weighted windows ("aw") are a library-wide bookkeeping fault.** Every window pair from 223.32 and 230 averaged the window over predator lengths by biomass, while the a is dominated by small predators (a ~ L^-2). This covers the rest -> planktivores, whiting and benthivores pairs, cod -> planktivores, cod -> whiting, whiting -> planktivores and whiting -> benthivores. Effect here: f_eff x0.6-0.7 on the rest's fish pairs; about x0.89 relative to the planktivores, but it moves every fish pair against zooplankton. It is the same family as the whiting window fault 232 fixed.
4. **Hake.** Its benthivore a, at 0.07 of today's a and 0.21 after the overlap, is about 270x its only in-area diet (Skagerrak 2000+: plk 81 %, whiting 13 %, ben 0.4 %). Its windows (f_A 0.20 lib) and its night layer are both suspects.
5. **Seasonal tail.** In 1991 saithe ate no haddock in Q1 (spawning season, 26-50 % empty stomachs, deeper than the 1-group). The model's Q1 take is 0.3-0.55 of Q3. This is not settled by any change above.
6. **Planktivore hiding.** Prey at their floors push the trained diet toward the bed (x2.2). Field herring school too, so part of the realised overshoot is not the a.
7. **Missing sources.** No Skagerrak or Trench saithe diet, and no bottom-referenced saithe profile from the box. The SATS/NORACU acoustic-trawl cruise reports are the best future source. Bergstad 1991 and Du Buit 1991 were not accessible.
