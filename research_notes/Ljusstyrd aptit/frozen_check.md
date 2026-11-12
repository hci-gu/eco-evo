# Frozen check of the reserve set point (section 271, 2026-11-12)

Fix98's update-630 policies (frozen copy; training stopped by the user) were run in the long-run probe world on
the CPU engine for three years. The library was the 271 library with the planktivores' block, but with
`observed: false`, because fix98's policies have no room slot. The real block has `observed: true`.

This is case (a) of `design.md`: the cap acts, and the policy cannot see it. The comparison is the same run on the
270 library.

Scripts:
- `research_notes/Ransonen i energi/scripts/` (ledger, limits, analyse, seasonal, prey_timing);
- `scripts/growth_terms.py`.

Tables: `tables/*_setpoint_frozen.*`, `tables/growth_terms_frozen_{sp,nosp}.txt`. Without the set point:
`research_notes/Fix98 mot fix97/tables/`.

## Rations: the winter target is met

Planktivores, years 2-3, per tonne of mean biomass:

| %BW/d | Nov | Dec | Jan | Feb | Mar | Apr | May |
|---|---|---|---|---|---|---|---|
| ration, no set point | 1.74 | 1.59 | 1.49 | 1.54 | 1.62 | 1.79 | 2.07 |
| ration, set point | 1.23 | 1.34 | 1.11 | 0.95 | 0.98 | 1.97 | 2.91 |
| krill, no set point | 1.36 | 1.49 | 1.40 | 1.30 | 0.96 | 0.40 | 0.23 |
| krill, set point | 0.94 | 1.25 | 1.03 | 0.74 | 0.46 | 0.33 | 0.30 |

- **Winter.** The group's ration in January-March reaches the field 0.45-1.0 %BW/d, against 1.5-1.6 without the
  set point. November-December stays above it.
- **Spring.** May shows the refill at 2.91 %BW/d, the post-deprivation hyperphagia of the salmon data.
- **Fill.** Fill tracks the target: January 0.84, March 0.76, April 0.74 (targets 0.86, 0.79, 0.74), then 0.98 in
  June.
- **Annual ration.** 6.74 → 6.19 t/t/yr, energy 1.00 of the field Q/B (was 1.10).
- **Krill per planktivore tonne.** 3.27 → 2.51 t/t/yr (-23 %). At unchanged planktivore biomass the five-group krill
  flow would fall about 13 %. The other groups eat +3 % more krill, because more is left.
- **Rest share.** Unchanged (0.002 in December-February), as expected with frozen, blind policies.

## Production: falsified assumption

| /yr (tonnes grown per tonne) | year 1 | year 2 | year 3 |
|---|---|---|---|
| no set point | 0.57 | 0.58 | 0.58 |
| set point | 0.34 | 0.36 | 0.33 |

Losses are unchanged (0.55-0.58 /yr), so the planktivores decline. Mean biomass:
- with the set point: 3 343 → 2 969 → 2 765 t (-11 %, -7 % a year);
- without it: 3 570 → 3 537 → 3 508 t.

**Binding term of the material growth law** (`growth_terms.py`; year 2; biomass share of cells, MJ/t/yr):

| month | paid G, no set point | binds, no set point | paid G, set point | binds, set point |
|---|---|---|---|---|
| Sep-Dec | 2 140-2 220 | g ceiling 0.79-0.82 | 375-1 205 | surplus 0.58-0.92 |
| Jan-Mar | 2 155-2 514 | g ceiling 0.77-0.94 | 203-609 | surplus 0.84-0.96 |
| Apr | 2 517 | g ceiling 0.96 | 1 364 | g ceiling 0.59 |
| May-Aug | 2 258-2 499 | g ceiling 0.84-0.96 | 2 406-2 839 | g ceiling 0.87-0.96 |

The two terms:
- surplus = max(I - C, 0), section 244.10;
- g ceiling = B g (s - u) p.

**What the table shows:**
- While the target falls (September-March), the reserve is released and intake stays below cost.
- Under 244.10's rule (no net lean growth below maintenance), growth is then near zero.
- In summer the g ceiling binds. The x1.12 on g buys only 12 %.
- The derived g still pays P as if the fish grew all year, so the set point makes growth seasonal without the budget
  knowing.
- This holds whatever the policies learn: a learned winter rest lowers intake further.

## The growth window (271.12): options put to the user

The clock gives its own growth window, the days on which the target is not falling: 15 April to 30 August.
- It lasts 138 days (4.54 months), and its mean group target is 0.956.
- Paying the field P (0.66 /yr) inside it requires g = P / ((0.956 - u) 1460 x 4.54/12) = 0.00136 /tick. That is x2.78
  the no-set-point g (0.00049) and x2.47 today's 271 g (0.00055).
- It is payable: P_n = 1.75 /yr against Pmax 15.5 /yr, and the material need is 0.08 of the ideal intake against
  kappa_m 0.43.
- The field herring season at the site is May-September (section 184).

Two forms, both derivable from the block without a new library value:

1. **g rescaled only (x12 / n_window).** No new table. Growth then runs at the higher g whenever I > C, including a
   winter refill after a dip. The budget does not see that overshoot.
2. **g rescaled plus a growth multiplier from the same clock** (1 in the window, 0 outside, like
   `growth_temp_table`). Growth is clean, but it is a gate on growth from the clock. It is not a month key, but it
   is the kind of thing section 184.2 did with months and the user withdrew.

A third option: neither. Accept the lower P for now and revisit; the planktivores then decline about 7 % a year
under these policies.

Section 250 refused `feeding_season_months` for an open population ("somatic_production is an annual rate"). Either
form reopens that refusal for the clock's window only.

## Form 2 applied (user, 2026-11-12), frozen check again

The user chose form 2.
- Growth runs only while the target is not falling. The window lasts from 15 April to 30 August, 0.378 of the year.
- `set_point.growth_window_*`: the multiplier sits on the material law's g ceiling, in
  `movement.income_growth` and the GPU `income_growth`.
- The budget pays P inside the window: g = P / (1460 mean(w (s_T - u))) = 0.00136 /tick.
- Same frozen run as above: fix98's update-630 policies, `observed: false`, 3 years.
- Tables: `tables/*_form2_frozen.*`, `tables/growth_terms_frozen_form2.txt`. Note that the hook's "paid" column
  ignores the window multiplier; outside the window the engine pays 0.

| /yr | year 1 | year 2 | year 3 |
|---|---|---|---|
| grown, form 2 | 0.61 | 0.64 | 0.64 |
| grown, no set point | 0.57 | 0.58 | 0.58 |
| mean biomass, form 2 (t) | 3 735 | 3 731 | 3 653 |
| mean biomass, no set point (t) | 3 570 | 3 537 | 3 508 |

**Production and biomass.** Production is restored, the losses are unchanged (0.56-0.58), and the biomass is
stationary.

Planktivores, years 2-3, %BW/d:

| | Nov | Dec | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ration | 1.19 | 1.17 | 1.00 | 0.91 | 0.92 | 2.17 | 3.53 | 2.89 | 2.82 | 2.64 | 1.42 | 1.28 |
| krill | 0.90 | 1.09 | 0.93 | 0.69 | 0.41 | 0.34 | 0.35 | 0.34 | 0.66 | 1.07 | 0.76 | 0.81 |

Without the set point the ration was 1.5-1.7 in winter and 1.8-2.1 in summer.

- **Winter.** January-March reaches the field 0.45-1.0 %BW/d. November-December is 1.2, against field winter
  0.45-1.0.
- **Summer.** The ration rises (May 3.5, June-August 2.6-2.9). The fish now pay their growth inside the window.
  The field summer herring ration is not checked yet [open].
- **Annual ration.** 6.91 t/t/yr, energy 1.11 of the field Q/B. That is unchanged from no set point (1.10), and
  the need on the field fill cycle was 1.06. The intake has moved from winter to summer; it has not shrunk.
- **Krill.** Per planktivore tonne 3.27 → 2.58 t/t/yr (-21 %); the planktivores' total krill fell 16 % and the
  five groups' 9 %.
- **Zooplankton.** Per planktivore tonne 3.47 → 4.26 t/t/yr (+23 %), total +30 %.
  - Fix98's realised zooplankton M2 from the planktivores was 1.05 of the booking, so this reads about x1.3: inside
    the band, but a pressure to watch in the retrained probe.
- **Rest share.** Still ~0 in winter (blind policies). With the room channel a trained policy can rest instead,
  which lowers the winter ration further (case (b), 0.4-0.7 in the preview).
