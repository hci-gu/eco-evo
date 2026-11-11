# The krill pool's supply (2026-11-10)

Main agent. Question: does the open krill pool supply the fish more than the field allows, and is
the supply mechanism itself the fault? Nothing in the library or code was changed. Script:
`scripts/krill_supply.py` (frozen fix96 policies, the probe world 30 x 30, seed 20260530, unfished,
the run's currents, two years); table `tables/krill_supply_pzbpgp248nsrpfix96.json`.
Tags: [S] library / note / resume, [D] measured or computed here, [J] judgement.

## Mechanism [S]

`open_pool.grow` after M1 and predation, per cell and tick:
B <- T + (B (1 + g) - T) exp(-k), T = 5 t/km2 x depth weight x season.
- g = 2.11 /yr, derived with r_max 0 to pay the booked fish M2 at the outside density (field P/B 2).
- k = 20 /yr (5-50): Rodhe 1987's 5-15 cm/s along the slope at the krill depths -> ~18 d in the
  60 km box, diel migration retaining krill only in the offshore half (library comment;
  `Krill i Skagerrak/exchange`).
- Local production is g B whatever the density (no K); the exchange imports where B < T and exports
  where B > T, in every cell directly.

## Flows in the probe world (t/km2 per cell and year) [D]

| bed class | B | T | B/T | local production | import | export | eaten | M2 /yr | gross import / eaten |
|---|---|---|---|---|---|---|---|---|---|
| < 100 m | 1.51-1.70 | 2.23 | 0.68-0.76 | 3.2-3.6 | 11.9-15.9 | 1.2-1.3 | 14.4-19.0 | 8.5-12.6 | 0.83-0.84 |
| 100-150 m | 4.40-4.44 | 4.99 | 0.88-0.89 | 9.3 | 15.3 | 3.5-4.1 | 21.1-22.7 | 4.8-5.1 | 0.67-0.72 |
| 150-250 m | 5.56-5.85 | 6.09 | 0.91-0.96 | 11.7-12.3 | 10.4-14.7 | 4.2-5.6 | 19.3-22.3 | 3.3-4.0 | 0.54-0.66 |
| >= 250 m | 6.38-7.02 | 7.08 | 0.90-0.99 | 13.4-14.8 | 9.3-18.0 | 4.0-8.1 | 18.8-27.5 | 2.7-4.3 | 0.49-0.65 |
| box | 4.43-4.58 | 5.00 | 0.89-0.92 | 9.3-9.7 | 13.0-14.7 | 3.2-4.5 | 20.2-20.8 | 4.4-4.7 | 0.64-0.71 |

(Ranges are year 1 - year 2.) The net import (8.5-11.5) is 0.42-0.55 of what is eaten; local
production pays the rest. Eaten by predator, t/km2/yr: planktivores 13.1-13.3 (64 %), mackerel 3.2,
whiting 1.5-2.3, water column 1.5-1.8, benthivores 0.3-0.7. Monthly box B/T 0.71-1.19: above 1 in
April-May, when the season factor drops T to 0.6 while g keeps growing the pool (export), lowest in
July (0.71).

## How hard the pool pushes back [D]

Steady state of the per-tick form, B/T = (1 - e^-k) / (1 - (1 + g - m) e^-k):

| k (/yr) | M2 2.11 (booked) | 3.0 | 4.7 (realised) | 7.0 | 10.0 | most it can deliver (~k T) |
|---|---|---|---|---|---|---|
| 5 | 1.00 | 0.85 | 0.66 | 0.51 | 0.39 | 25 t/km2/yr |
| 10 | 1.00 | 0.92 | 0.79 | 0.67 | 0.56 | 50 |
| **20 (library)** | 1.00 | 0.96 | **0.89** | 0.80 | 0.72 | 100 |
| 50 | 1.00 | 0.98 | 0.95 | 0.91 | 0.87 | 250 |

At the library's k, eating 2.2x the booked M2 lowers the pool by only 11 %, and doubling it again
by 28 %. The probe's 0.89 matches the formula.

## Is the supply the fault? [J]

- **The flushing evidence supports a fast exchange, not a slow one.** The current analysis
  (`Strömmen i rutan`) gives a surface-water first-exit time of 1.6-1.9 d (4.7 d at 25 m in spring)
  and 23 cm/s northward at 50 m on the Swedish side. The krill sit deeper by day (70-150 m), where no
  box statistics exist, and the literature check (`Krillens flykt och energibudget`, section 4) found
  nothing that moves the 18 d. If anything, the water says k >= 20, a stiffer reservoir.
- **So a box this well flushed cannot deplete krill it does not hold.** Its krill density is set by
  the Skagerrak population around it; the box's fish eat what the water brings. Capping the import at
  the local production, or slowing k to make the pool deplete, would contradict the physics. This
  withdraws the earlier suggestion (check 4b) to "cap it at the field production".
- **The excess is the fish side.** The pool delivers what the fish take; how much they take is set
  by their rations (food surplus plus the costs and growth they pay at a full reserve: checks 3 and 4,
  `Bottenfiskens budget`) and by their diet shares (the water column's krill pair against its fish
  pairs: check 2, resume 252.3-252.5). Those are where a source-based fix belongs.
- **Two caveats inside the pool.** (1) The exchange acts on every cell directly, as if each cell
  traded water with the outside at k; in reality water enters over the west and south edges and
  downstream cells receive water already grazed upstream. With a 2-18 d residence the box-scale
  error is small, but the most-grazed class (< 100 m beds, M2 8.5-12.6, B/T 0.68-0.76) is refilled
  from outside at the full rate. (2) The outside density T is the regional density under the
  REGIONAL predation (field M2 ~2 /yr). That stays valid as long as the box is a small part of the
  region and its fish eat at field rates; a box whose fish eat at twice the field rate per krill is
  a fish-side error that T cannot correct.
