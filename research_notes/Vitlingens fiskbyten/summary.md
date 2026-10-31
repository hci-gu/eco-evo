# Whiting's fish and bed prey: which pair factor is off (2026-10-31)

Research only; nothing in the library, manifest, code or tests was changed. Tags: [F] field / stated in a source, [D] derived here from data, [J] judgement. Evidence with quotes in `evidence.md`; scripts in `scripts/`, their stdout in `tables/`; downloads listed in `data/README.md`. Written by the research subagent, saved by the main agent.

## Verdict

1. **The epifauna pair is the outlier, by ~25-50x, and it is a geometry error.** Against every field diet, whiting's clearance on the epifauna relative to its clearance on zooplankton is 27-50x higher in the library than in the field (`tables/clearance_test_stdout.txt`). The four diets are: the library's booked M2 (V1), the 223.26 diet table (V2), SMS + Hislop North Sea stomachs weighted to the box whiting lengths (V3), and Skagerrak April 1991 ICES stomachs (V4).
   - Cause: the section 220/223.15 encounter keeps the foraging whiting inside the 0.5-m near-bed prey layer for all of its search time.
   - The same library places whiting in a 5-20 m layer above the bed (H_whiting 10 m, used when whiting is prey).
   - The field agrees with the library's layer, not the encounter: demersal whiting eat pout and herring (Pedersen 1999); whiting did not forage on the bottom at the deepest (137 m) North Sea site (Rindorf 2003); small whiting are pelagic.
   - Taking the vertical overlap H_bed / H_whiting gives a = 0.008 (0.005-0.023). That is within 0.8-1.5x of every field ratio [D].
   - The library's 0.27 gives a half-saturation of 0.11 t/km2 of a 3.78 t/km2 pool (x 35 at field density). That is why a trained whiting eats 99 % epifauna and empties it.
2. **The planktivore stage access is right in kind and slightly too tight.** Whiting eat only small planktivores: herring eaten have a median of 8.5 cm and a p95 of 12.5-13.5 cm, sprat and pout a p95 of 8.5-9 cm (SMS) [D]. So f = 1 (pre-223.32) overstates the pair 2.5-3x against the North Sea diets and ~28x against the Skagerrak one.
   - The 223.32 derivation has one double count. The a counts only whiting >= 20 cm (juvenile score 0.04 < 0.1), but the window was averaged over the whole whiting length mix.
   - Correcting it gives [0.16, 0.010], effective 0.066 instead of 0.050 at J/B 0.38 [D]. Weighting the species by encounter changes nothing.
   - The window ratio is a band [J]. 0.39 is the SMS all-fish p95 at 20-30 cm. 0.45 is the herring-specific p95 at 25-29 cm (0.49 at 20-24 cm), consistent with Rindorf & Gislason's 14-cm clupeid maximum in 25-35 cm whiting; it gives [0.26, 0.025].
   - The pout and sprat ratios (p95 0.28-0.38) do not test the window: those prey are 6-8.5 cm at every whiting size, so their ratio measures the prey's own size distribution.
3. **The data cannot settle the remaining planktivore question.** After the corrections, the planktivore clearance relative to zooplankton is 5-6x below the North-Sea-based diets (V1-V3). It is 1.9x above the only in-area sample (V4: krill 0.63, pout the only planktivore, no herring).
   - The box survey holds almost no herring <= 10 cm (0-3 % of the herring J biomass), the size whiting eat elsewhere.
   - Reading (i): box whiting eat few planktivores, and the booked M2 0.203 (SMS North Sea partial M2 scaled to the box, 223.37) is too high for the box.
   - Reading (ii): the box's small planktivores are under-measured; herring and sprat carry no GOV catchability correction.
   - **No in-area whiting stomachs support the booked M2 0.203.** This is an open point, not a derived value.
4. **Zooplankton (a 0.0023) is consistent** and is the anchor: no stage access, and the whiting layer lies inside the near-bed krill layer. **Benthivores (a 0.21, [0.049, 0])** are 0.4-1.6x the North Sea ratios; unchanged, see open point 3.

## Parameters: current vs proposed

| Pair / key | Current | Proposed (central, range) | Tag | Basis |
|---|---|---|---|---|
| whiting -> mobile_epifauna `max_intake_rate` | 0.27 | **0.008** (0.005-0.023) | D (mechanism) + F (test) | per-stage a (x e 0.39): juvenile 0.606, adult 0.157, at J weight 0.165 (section 230; was 0.247) x overlap H_bed/H_whiting. Adults 0.5/10, juveniles 0.5/30 (pelagic; Rindorf 2003 citing Bailey 1975, Robb 1981, Bromley & Kell 1995) [J on 10 and 30 m] -> 0.0082. Both stages at 10 m: 0.0116; H_whiting 20 / 5 m: 0.0058 / 0.023. Field ratio to zooplankton across V1-V4: 0.006-0.012. Time-allocation check (no pair far more profitable than the others at field density): 0.006-0.024 |
| whiting -> mobile_epifauna `accessible_fraction` | 0.29 | 0.29 | D (223.15) | pool-weighted time on the bed and small crabs only. e 0.39 was computed on the accessible composition (Pandalus 0.40, small shrimps/mysids 0.31, crabs 0.29), so no double count in the relevant direction |
| whiting -> mobile_epifauna `handling_time`, dark_ratio | 119.7, none | unchanged | J | Rindorf 2003: bottom prey eaten at night, shrimp by day; no net diel factor |
| whiting -> planktivorous_fish `stage_access` | [0.12, 0.008] | **[0.16, 0.010]** (band to [0.26, 0.025]; low end = library) | D (window over whiting >= 20 cm), J (ratio 0.39 vs 0.45) | `tables/planktivore_access_stdout.txt` |
| whiting -> planktivorous_fish `max_intake_rate` | 0.021 | 0.021 | D (220) | reproduced: adult a 0.0280 on the pool x adult share 0.753 = 0.0211. It has no prey-size factor inside, so the stage access is not counted twice on the prey side |
| whiting -> planktivorous_fish `predation_mortality` | 0.203 | unchanged; **flagged** | F (SMS) vs D (in-area) | implies a 0.42 planktivore share of the four pairs; in-area April stomachs 0.08; SMS North Sea box-weighted 0.26 of the whole diet |
| whiting -> zooplankton `max_intake_rate` | 0.0023 | 0.0023 (optionally 0.0020 at J weight 0.165) | D | consistent with every field ratio; juveniles carry 0.56 of it at weight 0.247 |
| whiting -> benthivorous_fish | a 0.21, [0.049, 0] | unchanged (open point 3) | D | 0.37-1.55x the North Sea ratios |

## What the proposed set implies

Concurrent search, all prey visible (`tables/clearance_test_stdout.txt`):

| Diet share of the four pairs | planktivores | zooplankton | epifauna | benthivores |
|---|---|---|---|---|
| library now | 0.011 | 0.069 | **0.913** | 0.007 |
| library with planktivore f = 1 (pre-223.32) | 0.178 | 0.057 | 0.759 | 0.006 |
| **proposed** (zoo B 9.72 / 7.0) | 0.12 / 0.15 | 0.59 / 0.51 | 0.23 / 0.28 | 0.06 / 0.07 |
| proposed, planktivores 25 % visible | 0.03 | 0.62 | 0.25 | 0.07 |
| field V1 booked M2 x B | 0.42 | 0.33 | 0.16 | 0.09 |
| field V2 223.26 diet table | 0.36 | 0.36 | 0.18 | 0.10 |
| field V3 SMS + Hislop, box length mix | 0.44 | 0.42 | 0.11 | 0.03 |
| field V4 Skagerrak Apr 1991 | 0.08 | 0.73 | 0.19 | 0.00 |

Single-pair returns at today's density (x = a f h B; share of the ceiling):

| Pair | now | proposed |
|---|---|---|
| epifauna | x 35.4 -> 0.97 | **x 1.05 -> 0.51** (half-saturation 0.11 -> 3.6 t/km2 of the pool) |
| planktivores | x 0.80 -> 0.44 | x 1.06 -> 0.51 |
| zooplankton | 0.76 | 0.76 |
| benthivores | 0.25 | 0.25 |

No pair then pays far more per allocated tonne than the others at field density.

Viability floor (a check that bounds the range from below; not a tuning criterion):
- The epifauna is an NDM and always visible, so its return is the whiting's floor.
- At a 0.006 that return is 0.44, at 0.008 it is 0.51, against whiting's cost/C 0.41 (223.26).
- Planktivores at 25 % visible fall to ~0.2 and zooplankton low-visible to 0.14, so the bed pair is what keeps a fed whiting fed.
- The budget gate was not re-run.

Caveat: the clearance ratios pair 1981-91 (and 1991) diets with today's prey densities, and the relative densities then are unknown. So the 27-50x is robust, but the +/-2x around the proposed value is not.

## Juvenile whiting (section 230) and the predator side

The section 230 compartment is prey-side. The whiting's own pair a values are biomass means over predator sizes, with the juvenile (< 20 cm) weight fixed at 0.247 by the section 220 stage table; the box target is 0.165 [D].

- Field: whiting < 20 cm eat 0.08 planktivores and ~0.6-0.8 crustaceans (krill, mysids, Crangon); whiting >= 30 cm eat 0.5 planktivores and ~0.9 fish (SMS, Hislop) [F/D].
- In the library, juveniles carry none of the planktivore and benthivore a (score < 0.1). They carry 0.56 of the zooplankton a and 0.56 of the epifauna a (0.43 at weight 0.165, 0.20 after the overlap correction with pelagic juveniles).
- So the J weight moves the crustacean pairs by about 14 % and leaves the fish pairs alone. It matters for the seasonal shape (J/B 0.13-0.19), not for the 25-50x fault.
- A predator-side stage (juvenile whiting on crustaceans, adults on fish) would move crustacean intake with the compartment through the year. That is engine design: noted, not designed.

## Open points (user's call)

1. **Booked whiting -> planktivores M2 0.203.** It rests on SMS North Sea partial M2 (sprat- and sandeel-rich shallow areas). The only in-area stomachs (Skagerrak April 1991, 106 whiting) are krill and pout. The 2000+ in-area records hold 95 g of prey in 539 stomachs, mostly 9 cannibalised whiting. A box-specific value needs in-area Q1/Q3 whiting stomachs.
2. **Box small-planktivore biomass.** Herring and sprat carry no GOV catchability correction, and 8-9 cm herring are almost absent from the box survey. If they are under-measured, f_J rises; if they are truly absent, point 1 applies.
3. **Benthivore pair geometry.** The same overlap applied to whiting -> benthivores (settled juveniles in H_juv 2 m, 1-5 m [J], in a 10 m whiting layer) gives x0.2 (0.1-0.5). The adult-only window with the benthivore-prey ratio 0.45 raises f_J from 0.049 to ~0.09-0.11. Net x0.3-0.5, which puts the pair 1.4-6x below the North Sea ratios. That is within the encounter MC spread, so it is not proposed.
4. **Convention.** The engine's eat share stands for time in the prey's layer (223.26 section 5). The overlap used here stands for how a whiting searching its own layer meets bed prey. Both readings give 0.006-0.024 here.
   - If the overlap convention is adopted, it applies to the planktivore pair's herring/sprat term too (86 % of a_plk). That term assumes the whiting is inside the 70-m day / 40-m night herring layer. The overlap is about H_whiting/H_herring = 0.1-0.25 by day and ~0 at night (herring in the upper 30-50 m over a 100-300 m bed).
   - That would move the implied species mix of the planktivore take toward the stomachs. Encounter x window now gives herring 0.60, pout 0.28; the SMS stomachs show pout 0.61, sprat 0.26, herring 0.12.
   - It would also cut a_plk to ~0.005 and widen the planktivore deficit against the North Sea diets to ~20x.
   - The convention [J] should be applied the same way to every bed and layer-separated pair if adopted.
