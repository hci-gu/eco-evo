# The levers on clumping (section 278.6)

The benthivores pack to Lloyd 30-90 in the trained world. The field value is 3.1. Section 278 traced the payoff to predator swamping, mainly against the water column (WC).

This note tests the predator-side levers left after 278.2-278.5. It is a diagnostic only; nothing in the library or the code changed.

Setup:
- Policies: frozen fix104 u1267, on a 30 x 30 grid.
- Library: the pre-279 copy (`git show HEAD:fgconfig/fg_library.yaml`), the one fix104 was trained on.
- Runs: seeds 0 and 1; snapshots on days 0, 91, 182 and 273 after the 60-day warm-up; one-day windows (4 ticks).

## Arms (scripts/cf_lever.py)

| Factor | Settings |
|---|---|
| WC placement | `trained`; or `field`, where each bed class is rescaled to the GOV saithe shares (0.043 / 0.127 / 0.782 / 0.048) and the pattern within each class is kept |
| WC -> benthivores pair | `lib`; or `dark`: no daylight anywhere, with depth multiplier 1, no diel swing, and a x 0.31 (the 278.5 R_nv-only variant) |
| Benthivores | `clumped` (as trained); `3.0` (Gaussian spread to Lloyd 3); `uniform`; or `field`: adults at the GOV class shares, J at the recruit shape, each spread evenly within its class (Lloyd ~1.2-1.3) |

## Result

Mean over the 8 snapshots of M2 on the benthivores per year (tables/lever_*). M2 is shown for the WC and for all predators, as WC / all. The booked M2 is 0.058 for the WC and 0.25 for all checked pairs.

| WC placement, pair | clumped | Lloyd 3 | uniform | field | saving at Lloyd 3 (all) | saving at field (all) |
|---|---|---|---|---|---|---|
| trained, lib | 0.116 / 0.148 | 0.327 / 0.396 | 0.413 / 0.569 | 0.462 / 0.579 | 0.25 | 0.43 |
| trained, dark | 0.080 / 0.112 | 0.222 / 0.291 | 0.260 / 0.416 | 0.284 / 0.401 | 0.18 | 0.29 |
| field, lib | 0.069 / 0.100 | 0.213 / 0.281 | 0.351 / 0.507 | 0.195 / 0.310 | 0.18 | 0.21 |
| field, dark | 0.058 / 0.089 | 0.185 / 0.253 | 0.304 / 0.460 | 0.146 / 0.262 | 0.16 | 0.17 |

- **What the levers achieve.** Spreading remains 2.6-2.8 times more costly than clumping in every arm. Field placement and the dark pair together cut the absolute saving from 0.25-0.43 to 0.16-0.17 per year. That saving is still 22-23 % of the benthivores' somatic production P (0.74 per year).
- **The ratio itself is not evidence.** A scalar on a and a flat depth table cannot move the spread/clumped ratio, so the constant ratio says nothing about those levers. The absolute saving is the readout.
- **The dark arm is not a bound.** Only this one pair was darkened; the WC's other pairs stayed visual. Encounter fell x 0.2-0.45 but the flow fell only x 0.65-0.87, least in winter. The benthivores may have become the WC's night prey. Read it as "not shown sufficient", not "ruled out".
- **J riding with the adults is not the cause.** The `field` benthivore arm places J apart from the adults, at the recruit shape. The saving stays at 0.17-0.43 per year.
- **The WC already follows the clumps.** The WC density met per benthivore tonne is 1.1-3.4 times the box mean; from the class shares alone it would be 1.0-1.7.
- **Every predator is swamped alike.** The all-predator saving has the same ratio as the WC's.
- **The "field" arm is not a pair-level overshoot.** Its WC M2 runs at 2.5-3.4 times the booking (0.146-0.195 against 0.058). Planktivores and whiting are still placed as trained, so the WC's diet shares are distorted there. The pair-level number stays budget_gate's ~2.

## Which cap binds (scripts/cf_cap.py, tables/cap_*)

| Predator | Limited by | Share of pre-cap demand that fits: benthivore cells | Elsewhere | Pre-cap demand / Holling ceiling | Fill |
|---|---|---|---|---|---|
| WC | reserve room | 0.38-0.52 | 0.59-0.72 | 0.8-1.2 | 0.98-1.00 |
| Whiting | reserve room | 0.42-0.95 | | | 0.92-1.00 |
| Seals, porpoises | hunger gate (not capped) | | | 0.24-0.55 | 0.65-0.80 |

So every predator of the benthivores is satiation-limited. Its intake is fixed by its stores, not by what it meets.

- Spread out, the benthivores take a larger share of each predator's fixed meal.
- Clumped, they are met only by the few predators in the clump cells, and those predators cannot eat more.
- This is satiation swamping, the textbook predator-swamping mechanism. It acts against every predator.

Rough estimate of hungrier predators (the 278.4 fill gap): lifting the room cap raises the clump cells' intake by ~1/0.40 and the spread cells' by ~1/0.50. The ratio falls about 20 %, but the absolute saving rises. It is not a lever.

## Reading

The predator side is exhausted. Placement, dark vision and the juvenile habitat each remove at most a third of the payoff, and the mechanism is generic.

What holds field benthivores near Lloyd 3 must then be on the cost side, which the model lacks:
- **Intake elasticity.** Intake per tonne on local density is -0.012 in the model (278.3). In the field it is -0.19 to -0.21 for Baltic large cod and flounder, at haul scale with depletion included. It is 0 for small cod, and haddock condition is higher where haddock are dense.
- **Break-even estimate (rough).** The saving of ~0.2 per year needs a growth loss of ~0.2 per year. At the growth-ration slope of ~0.3 that takes ~15 % of the ration. Clumps sit ~20 times more crowded than the field, so an elasticity of about -0.05 would offset the swamping.
- **Mechanism.** The field figure is four times that, but the evidence is mixed and the mechanism unknown. The bed pairs are saturated in the field too ("Fodobegransning"), so the cost cannot be plain bed depletion at today's half-saturations.

## How the water column's excess splits (scripts/cf_pair.py, tables/pair_*; section 278.9)

The same frozen world and snapshots, with the WC as trained. Arms:
- **Pair:** the WC -> benthivores attack rate x 1, x 0.5 or x 0.42. The last is about the pre-279 pair at its booking: the gate's equal-cell ratio 1.95 on today's library, times the pre-280 a 0.1162 / 0.09443. This is a counterfactual only, never a library value.
- **Planktivores:** as trained, or spread evenly.

Means over 8 snapshots; the day-0 snapshots weigh most.

| Pair | Planktivores | Saving at Lloyd 3 | Saving at field placement | WC M2 at Lloyd 3 | WC ration, MJ / booked |
|---|---|---|---|---|---|
| x 1 | trained | 0.25 | 0.43 | 0.327 | 1.26 |
| x 0.42 | trained | 0.16 | 0.29 | 0.211 | 1.26 |
| x 1 | even | 0.12 | 0.20 | 0.165 | 1.17 |
| x 0.42 | even | 0.07 | 0.12 | 0.089 | 1.17 |

- **The room cap refills the meal.** At x 0.42 the WC's flow on Lloyd-3 benthivores falls only x 0.65, an elasticity of about 0.5. The other prey take the rest; krill is 0.6-0.8 of the WC's tonnes, but the script does not read this out per prey. The pair at its booking takes the saving from 0.25 to 0.16, not to the ~0.1 estimated in chat after 278.8.
- **Per snapshot the ratios are stable.** Each figure is the share of the saving that remains:

  | Arm | Share of the saving left |
  |---|---|
  | Pair x 0.42 | 0.47-0.69 |
  | Planktivores even | 0.42-0.58 |
  | Both | 0.22-0.39 (up to 0.57 at field placement) |

- **Today's library.** Section 280 already cut this a by x 0.81, so the remaining pair error is about x 0.5, the x 0.5 arm, with a saving of 0.18.
- **The missing planktivores weigh more.** As trained they are 0.07-0.08 of the WC's tonnes; spread evenly, ~0.45. The WC stands over shallow beds and the planktivores over deep ones. Restoring them alone takes the saving to 0.12. This is trained placement of both groups. The even arm also removes the planktivores' own clumping. It restores the same overlap as the WC field placement above, so the two must not be added.
- **The ration is not the cause.** 3.75 t/t/yr is 1.44 times the booking in tonnes, but only 1.17-1.26 times in MJ, because krill is lean. With the planktivores back it is 1.17 times, about 1.1 times the field Q/B.
- **Both together leave 0.07 per year at trained fill**, 9 % of P. The rough break-even crowding elasticity there would be about -0.02. This is no closure:
  - with field fill (278.8, base 0.25 -> 0.42) the refill and the planktivore redirection change, and that combination is unmeasured;
  - the model has no crowding-cost mechanism.
- **The other predators.** Without the WC, the predators realise ~0.03 per year on clumped and ~0.12 on field-placed benthivores, against a booked 0.19. Clumping holds them down too, not only the pairs that are OFF on the prey side.

Of these factors only the pair derivation is a biology fix: the J encounter at the bed, about twice the field flow. It is worth a third of the saving. The rest is the trained overlap between the WC and the planktivores.

## The infauna refuge as a crowding cost (resume 282)

`scripts/refuge_clump.py`, `tables/refuge_<arm>_s<seed>_fix104_u1267.txt`. Frozen fix104 u1267 on the pre-279 library, 30 x 30, three years, seeds 0 and 1. The arms vary the infauna's `exposed_share` phi and `refuge_exchange_rate` k inside the ranges of "Refugium och utbyte i bentos". The elasticity is that of eaten intake per tonne on own density k. "Eaten" means demand after the reserve-room cap; in emptied cells it is an upper bound.

| Arm | Elasticity yr 1 | yr 2 | yr 3 | Benthivores yr 3 / start | Eaten k > 32 / k < 0.5, yr 3 |
|---|---|---|---|---|---|
| phi 0.2, k 2.5 (library) | -0.014 / -0.012 | -0.028 / -0.027 | -0.014 / -0.022 | 1.40 / 1.37 | 0.90 |
| phi 0.2, k 1.5 | -0.022 / -0.015 | -0.017 / -0.027 | -0.017 / -0.030 | 1.36 / 1.31 | 0.84-0.88 |
| phi 0.08, k 2.5 | -0.049 / -0.051 | -0.25 / -0.15 | -0.34 / -0.29 | 0.89 / 0.93 | 0.45-0.51 |
| phi 0.08, k 1.5 | -0.039 / -0.061 | -0.20 / -0.25 | -0.36 / -0.49 | 0.86 / 0.81 | 0.35-0.43 |

- **At phi 0.2, k does nothing.** The accessible infauna under the clumps stays far above the half-saturation of 0.35 t/km2.
- **At phi 0.08 the clumps return to cells they have emptied.** Accessible infauna there is ~1.3 t/km2, the 10th percentile 0.03, and the epifauna 0.01-0.03 of K. The grid as a whole holds 0.8 of K_A. Lloyd stays at 48-63, because frozen policies cannot respond. The run has not settled by year 3.
- **What this does not show.** It cannot say whether retrained fish would spread out or roam to unvisited cells. phi may move only on evidence of how deep fish reach in mud, never because it reduces clumping.

## Cost against benefit at phi 0.08, and fresh beds (resume 282.1)

`scripts/cf_refuge_spread.py`, `tables/cf_refuge_<lib>_s<seed>_fix104_u1267.txt`. The setup is the year-3 state of the runs above, four snapshots, each run for 4 days. The benefit is the all-predator M2 avoided by clumping. The cost is the realised intake that clumping gives up, converted at 278.6's rule of 0.3 x the change in ration (rough). Ranges cover seeds 0 and 1.

| Library | Arms compared | Benefit /yr (day 1 / days 1-4) | Cost /yr (day 1 / days 1-4) |
|---|---|---|---|
| phi 0.2 | Lloyd 3 vs clumped | 0.19-0.21 / 0.10-0.12 | 0.01 / -0.01..-0.03 |
| phi 0.08 | Lloyd 3 vs clumped | 0.19-0.22 / 0.11-0.12 | 0.37-0.91 / 0.27-0.62 |
| phi 0.08 | field vs clumped | 0.33-0.35 / 0.20-0.22 | 0.39-0.90 / 0.42-0.78 |
| phi 0.08 | fresh beds: Lloyd 3 vs clumped | 0.21-0.23 / 0.13-0.14 | -0.05..-0.09 / -0.04..-0.05 |

- **At phi 0.08 the frozen clumping costs more than it saves.** The whole cost comes from returning to emptied beds: the clumps meet A/K_A 0.03-0.22 while the grid holds 0.8.
- **On fresh beds the cost vanishes.** The same clumps eat as much as spread fish over the 4 days, longer than a typical stay, and keep the full swamping benefit.
- **Prediction.** Retrained fish at phi 0.08 would roam in clumps rather than spread out. The refuge penalises revisiting, not packing, so a lower phi does not explain the field's Lloyd 3.
