# The ration in energy: where fix97's ration excess goes (2026-11-12)

Main agent. The question (user: "investigate the ration"): fix97's realised Q/B was above the field
value (mackerel x2.3-2.5, benthivores x1.4; `Fix97 som baslinje`). Is that an energy excess, and if it
is, which term takes it? Nothing in the library, code, tests or manifest was changed. No model output
set any value. Tags: [D] computed here, [S] an earlier note or section, [J] judgement.

## How it was measured

- `scripts/ledger.py` runs fix97's long-run probe world: its `progress/config.json`, 30x30, seed
  20260530, the inference start and no fishing.
  - It uses the 269 library (`git show 109d25f^:fgconfig/fg_library.yaml`, the same as 7153a6c) with
    fix97's frozen policies, for three years on today's code.
  - Per fish group and tick it books:
    - assimilated gain (`temp_energy_gains`);
    - action costs per action, from `apply_energy_costs` with the Q10 multiplier;
    - the reserve exchanged with the outside (`tick_exchange_reserve`);
    - the reserve carried out by losses (predation, M1, impact; at the tick's fill);
    - the reserve change and the grown tonnes.
  - Output: `tables/ledger_fix97.json`.
- The ledger closes [D]. The residual gain - costs + reserve imported - reserve lost - dR, divided by
  the grown tonnes, is the engine's tonne price: benthivores 7 335 MJ/t against ec + c (ec + ME) =
  7 326, mackerel 4 670 against ec 4 670. So there is no engine leak.
- The setup reproduces the run [D]. Its realised rations in years 2-3 match probe 68 of fix97
  (`Fix97 som baslinje` tables) within a few %:

  | | planktivores | benthivores | whiting | water column | mackerel |
  |---|---|---|---|---|---|
  | ledger, t/t/yr | 6.78 | 5.83 | 3.60 | 3.25 | 6.33 |
  | probe 68, t/t/yr | 6.77 | 5.99 | 3.39 | 3.28 | 6.25 |

  So running the 269 library on today's code matches the 269 world the run was trained in.
- `analyse.py`: annual budgets, and the ration in tonnes and in energy against FIELD_QB. The field
  energy is FIELD_QB x the booked diet's gain per tonne (`energy_gain_mat`). Output:
  `tables/analyse_fix97.txt`.
- `fieldfill.py` and `seasonal.py`: the need of a fish that follows the FIELD fill cycle. That cycle is
  `outside_fill`, else `start_fill`. The need uses the realised costs and grown tonnes:
  `gain = cost + G (price(f) + f ME) + B ME df/dt`. Losses and the exchange carry the reserve out and in
  at the same fill, so they cancel.
- `prey_timing.py`: each group's prey flows if its monthly intake followed that need, with the
  realised monthly diet.

## Findings (years 2-3) [D]

### 1. In energy, the rations are close to what the library needs on the field fat cycle

| group | ration t/t/yr (x field Q/B) | energy x field | realised / need on the field fill cycle |
|---|---|---|---|
| planktivores | 6.78 (x1.09) | x1.11 | 1.06 |
| benthivores | 5.83 (x1.31) | x1.15 (x1.04 against the box closure Q/B 4.94 of 267.4) | 1.08 |
| whiting | 3.60 (x0.86) | x0.86 | 1.07 |
| water column | 3.25 (x1.15) | x1.27 | 1.08 |
| mackerel | 6.33 (x2.30) | x2.32 | 1.07 |

**Benthivores.** The tonnage excess is mostly diet composition. Infauna is 0.70-0.79 of the eaten
tonnes against a booked 0.56, at a gain of 1 560 MJ/t against 2 197 for the booked mix. In energy the
group sits at x1.04 of the 267.4 closure.

**Mackerel.** About half of the intake leaves the box as fat in emigrants: 10.3 of 20.8 GJ/t/yr. That
is the field fattening, not a leak.
- The box holds mackerel while they fatten. Arrivals come in at fill 0.13-0.57 (May-July) and leave
  at about 1.0 (September-December).
- With the model's costs, the field cycle needs 18.5-19.6 GJ/t/yr, about 5.7-6.0 t/t/yr
  presence-weighted.
- There are three mackerel numbers, not two:
  - the booking: the pairs' M2 x B at the probe start, 1.87 t/t/yr (krill 562 t/yr);
  - the yardstick: FIELD_QB 2.75;
  - the need on the fat cycle: 5.7-6.0.
- SMS cannot pay the measured fattening, as `Makrillens ranson` found [S]. This ledger confirms it
  with a closed budget.

**Why `Makrillens ranson` got 4.99 (x1.23) and this note 5.7-6.0 (x1.07-1.13).** The difference is
reproduced [D]: `scripts/seasonal.py` against its variants `scripts/seasonal_fwd*.py`, with the same
conventions as `ration_vs_field.py`.
- That note left out the fat that the grown tonnes must carry, G x f x ME. A fish on the field cycle
  keeps fill f on its new tissue too. This accounts for about 1 900 MJ/t/yr, or 0.6 t/t/yr.
- It took the fill change as a forward difference, which puts May's fattening (0.13 -> 0.51) into May,
  when few mackerel are present. The difference lies between that convention and the centred one.
  The fills are monthly values, and the source does not say which convention is right.
- With both of that note's conventions, this ledger gives 16.6 GJ/t/yr, which is 5.1 t/t/yr at
  x1.26. That reproduces the earlier note.
- The range 5.7-6.0 (forward / centred) includes the fat of new tissue, which belongs in the need.

**Water column.** The costs alone (eat cost 2.14 RM, 10.9 GJ/t/yr) use the whole field energy ration
(11.0). Its need on the field fill corresponds to Q/B 3.33, inside the source range 2.6-3.6.
- The "x1.27 in energy" is partly prey fatness, not more eating. The eating is the tonnage ratio,
  x1.15. The fish prey (`prey_includes_reserve`) are eaten at fill ~1, while the yardstick's gain per
  tonne is at their reference fill. So the realised gain per tonne is 4 324 MJ against 3 903 for the
  booked mix.
- The same applies to every energy column with fish prey.

### 2. The full reserve costs 6-8 % a year, and most of it is a timing shift

Every fish group sits at fill 0.97-1.00 all year (`reserve_room`). The field fish's fill follows its
cycle. So the model's intake is flat while the field need swings (`tables/seasonal_fix97.txt`):

| group | months above the field need | months below it |
|---|---|---|
| planktivores | Dec-Mar x1.6-2.3; Oct-Nov x1.2-1.4 | May-Jun x0.6 (fattening) |
| benthivores | Feb-May x1.2-1.6 | Aug-Sep x0.75-0.85 |
| whiting | Feb-May x1.2-1.5 | Aug-Sep x0.8-0.9 |
| water column | Apr-Jul x1.2-1.3 (arrivals at 0.71 topped up to 1) | - |
| mackerel | Oct-Apr (a small winter remnant, as `Makrillens ranson`) | Aug-Sep x0.5-0.8 |

### 3. Effect on krill: the appetite explains about a quarter of the krill excess

Suppose each group's monthly intake followed the field need, with the realised monthly diet. Then
the five fish groups would take [D]:
- krill: 17 % less, 19 570 -> 16 290 t/yr. The planktivores take 23 % less, because their winter
  intake is krill-heavy.
- zooplankton: 12 % more.

The krill flow excess against the booking was x1.71 (x1.93 in M2 at the realised krill biomass, 0.89
of start). The appetite timing takes it to about x1.43. Two things are left:
- **The diet shares.** Fish prey were under-realised in 269, which 270 addresses (capture p and
  schooling). The krill share was 0.52 against 0.40 booked for the planktivores, and 0.54 against
  0.36 for the water column.
- **Mackerel's booking.** Mackerel is the second-largest krill consumer: 3 230 t/yr against 562
  booked. Its booked ration is 1.87 t/t/yr against a need of 5.7-6.0.

## Consequences [J]

- **Tonnes are a misleading ration yardstick when diets shift between prey of different energy
  density.** The benthivores' "x1.4" is mostly infauna. `probe_diets.py` could also print the ration
  in MJ against FIELD_QB x the booked diet's gain per tonne. That is a diagnostic change only.
- **The ration itself (energy over a year) is not the main lever.** At fill 1 it costs 6-8 %.
- **The timing of the full-reserve appetite is a lever of about 17 % on the krill flow.** It sits
  mostly in the planktivores' winter.
  - A season-keyed appetite is out: behaviour must be learned, with no season switches.
  - Anything here would have to change a payoff physically. Two candidates:
    - The hunger gate (a lipostat) instead of `reserve_room`. Section 240 moved every fish off the
      gate because no fish data supported it, and the zooplankton's chain collapsed at s* = 1. That is
      the known history.
    - A cap on the deposition rate: a maximum MJ/t/tick the reserve can take up, since lipid
      synthesis is rate-limited. It would slow the top-up of arrivals and the winter top-up, with no
      calendar and no set point. No value is proposed and no source was read.
  - Both are bounded by what was measured here: at most about 17 % of the krill flow and 6-8 % of
    the annual energy. Either is an engine change and needs retraining. The user decides.
- **Mackerel's booking is the largest single item on the krill side.** Re-booking its pairs on the
  energetically consistent ration (5.7-6.0 t/t/yr presence-weighted; booked today 1.87) would raise the
  booked M2 on krill, zooplankton and planktivores, and so their derived g.
  - It is a choice between SMS (0.9 % BW/d) and the measured fat cycle plus Bachiller's feeding-season
    ration. Neither source was re-read here.
  - The user decides.
- **Judge 270 with the same ledger on a frozen fix98 checkpoint** (copy it first; the trainer
  overwrites `policy_*.pth`), with LIB = today's library. Its fish prey changes may move the
  winter diet.

## Caveats

- One seed, the 30x30 probe world, frozen fix97 policies on today's code with the 269 library. The 270
  code paths are inert there: no `recruit_depth_table`.
- The need uses the realised costs, at eating shares of 0.93-0.97. A field fish that rests more in
  winter would cost less, so the winter excess is if anything larger.
- `outside_fill` / `start_fill` stand in for the box fish's fill cycle. The water column's is flat
  (0.71).
- The monthly need uses centred differences of mid-month fills.
- In `prey_timing.py` the mackerel need is floored at 0 in Jan-Apr. In those months the field fish
  draw on fat faster than they spend; there are few of them in the box.

## Addendum: what limits the intake (user: something must be missing if the fish can stay full all year)

`scripts/limits.py` -> `tables/limits_fix97.txt` [D]. It wraps `predation._reserve_room_cap` and runs the
same world for two years. Per month it compares the demand's energy before the reserve-room cap (pre)
with the energy after it (post).

- **The reserve room is the only binding limit, in every month, for every fish group.** The one
  exception is mackerel in April-June, while the arrivals are still lean.
  - pre/post by group: planktivores 1.7-3.1, benthivores 1.1-1.5, whiting 1.25-2.1, water column
    1.5-2.2, mackerel 2.5-3.4 in Aug-Feb.
  - gain/post is 1.00 in every month: the prey-side harvest caps never cut the intake.
  - So the fish meet more food than they can store all year, and no month is a hungry one.
  - In the field the fill falls in winter and spring (herring 0.5 -> 0.2 Jan-Apr, gadoid livers
    0.76 -> 0.33 Feb-Jun) because intake is below cost then. Herring fast in winter.
- **The winter food is krill.** Monthly realised diets, from `tables/ledger_fix97.json`:
  - planktivores: krill 0.94 of the tonnes in Jan and 0.95 in Dec;
  - whiting: krill 0.88-0.90 in Oct-Jan;
  - mackerel: krill 0.92-0.99 in Oct-Feb;
  - water column: krill 0.41-0.49 in Oct-Feb.
- **Krill is an open pool that eating cannot deplete.** In the last year it lost 19 200 t to predation
  (4.86 /yr of the mean biomass 3 960 t).
  - The local growth was 8 300 t (2.10 /yr).
  - The import was 14 900 t and the export 3 950 t: net 10 900 t/yr, 57 % of what was eaten.
  - The pool relaxes towards outside_density x season at k 20 /yr. The season is 1.0 / 0.6 / 1.1 / 1.3
    by quarter, so Q4 and Q1 are the high quarters.
- **The benthivores' infauna is a buffet too.** It is 0.72-0.79 of their tonnes all year at a realised
  M2 of 0.07-0.09 /yr, with an exposed part of about 19 t/km2 against 1.5 t/km2 of benthivores.

[J] What the model lacks is a hungry season, not a ration rule.
- With `reserve_room` and fish hide floors of 1.0 (270), resting gains nothing but the cost difference.
  So a fish stops eating only when it is full, and it is full whenever the encounter exceeds the cost.
- In winter the encounter is krill: from a pool held at its winter-high season level by import, and
  met by planktivores, whiting and mackerel at 2-3x the energy they can store.
- Candidates to check against field data:
  1. krill's winter level and season;
  2. the winter encounter, i.e. the krill depth against the fish depth in winter;
  3. whether the exchange (k 20 /yr) may supply what the box's fish eat.
- The gonads (+570 MJ/t/yr for the planktivores, `Planktivorernas energibudget`) are too small to
  explain a 1.7-3x surplus.

## Addendum 2: the three krill questions against field data (user: "do so")

Most of this was already in the sibling notes. Tags as above; [S] is a sibling note.

1. **The winter level is supported by data, not too high.** The only in-box krill data are
   winter-night IKMT catches (Boehle & Moksness 1991). They give a winter column of 7-9 t/km2, with a
   floor of about 3 (`Krill i Skagerrak`) [S]. The library's winter, 5.0 in Q1 and 6.5 in Q4, sits
   below that on purpose.
2. **The exchange is supported by the physics** (`Krillens tillfoersel`, o-umlaut) [S].
   - The box is flushed in 2-18 days, and nothing found moves k = 20 /yr.
   - A box this well flushed cannot deplete krill it does not hold. The outside density is the
     regional density under regional predation.
   - At the library's k, eating 2.2x the booked M2 lowers the pool by only 11 % (0.89 in fix96 and
     fix97).
3. **The winter encounter fails a field test.** The winter intake that the model's encounter allows is
   several times what the field planktivores eat at the same or higher krill level.

   | Dec-Mar | ration % BW/d (wet) | assimilated kJ/kg/d | source |
   |---|---|---|---|
   | herring, Baltic Jan-Feb | 0.4-1.0 | 12-25 | Lohmeyer & Hempel 1977 [F] in `Sillens vinterfoeda` |
   | herring, Bornholm winter | <= 0.5, about maintenance | <= 22 | Bernreuther 2007 [F], same note |
   | herring, Oslofjord Nov/Mar, krill only | 0.02-0.9 | - | Onsrud et al. 2004 stomachs: krill in 17-50 % of stomachs, 1.5-5 krill each; fullness 2.3-3.4 of 5, 0-6 % empty [F] |
   | Norway pout, winter (~80 % euphausiids) | 0.4-1.3 | - | Albert 1994 [F] in `Planktivorernas vinterkostnad` |
   | model planktivores, Jan (fix97), realised | 1.55 (1.45 of it krill) | 54 | `tables/ledger_fix97.json` [D] |
   | model planktivores, Jan, demand before the reserve cap | 2.8 | 96 | `tables/limits_fix97.txt` [D] |
   | model planktivores -> krill, Holling ceiling at the Jan Q10 | about 5.5 | - | 1/h = 1/57.8 t/t/tick [D] |

   - The pair is near saturation at the box krill density: a h B / (1 + a h B) = 0.84 at 5 t/km2
     (`Krillens flykt och energibudget`) [S].
   - So the model's herring and pout could eat 2-7x the field winter ration and still meet more krill
     than they can store. The field fish eat about their maintenance, with half-full stomachs.
   - Either the encounter a on krill is too high in the dark season, or the field fish meet the krill
     and do not take them. The stomach data cannot tell these apart, but both mean the model lacks
     what limits the field fish.
4. **The model's winter cost adds to it.** The model fish eat on 97 % of the ticks at 1.65 RM, which
   costs about 39 kJ/kg/d in January. The members' field routine cost is 22 kJ/kg/d (4.5-8.2 MJ/t/tick,
   `Planktivorernas vinterkostnad`) [S]. A fish with a full reserve pays the cost with more food,
   because resting gains it nothing.

[J] So the krill pool itself is not what is missing. The level and the exchange both rest on data.
The gap is on the fish side of the krill pairs: an encounter that saturates the planktivores and whiting
in winter, against field fish that eat at about maintenance.
- The planktivores -> krill a (0.023, section 220/245/269 encounter theory, p_cap 0.5, no swarm term)
  has not been tested against a field ration before.
- These stomach rations are an independent test, and the a fails it in winter by 2-7x.
- Whiting (krill 0.88-0.90 of its winter diet, demand 2x its room) needs the same test against whiting
  winter stomachs.
