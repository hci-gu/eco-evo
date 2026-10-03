# Zooplankton energy budget: why it collapses and what to change

Status (2026-10-03): options 1-4 are **applied** (sections 6-8). The loss accounting fix (M1 shown in the loss breakdown) is
implemented separately.

Provenance tags: **[V]** read in the source, **[S]** search snippet or abstract only,
**[C]** computed here, **[M]** measured in the model. Literature notes:
`research_notes/Zooplankton energibudget/copepod_energetics.md`.

## 1. What happens in the training run

Run `pzbpgp248nsrpfix39`: train_gpu, mass balance, mortality, migration and currents
on. The policies saved at 19:13 were replayed on CPU in the same world [M].

| configuration | zoo after 300 ticks | mean s | production /tick | M1 /tick | predation /tick |
|---|---|---|---|---|---|
| mass balance OFF | 1.11x | 0.68 | 0.0151 | 0.0134 | 0.0015 |
| **mass balance ON (training)** | **0.10x** | 0.56 | 0.0093 | 0.0134 | 0.0037 |
| greedy behaviour, mass balance ON | 0.04x | 0.59 | 0.0097 | 0.0134 | 0.0001 (starvation 0.026) |

- **The dominant loss was invisible.** M1 is the largest loss, 3-4x the predation,
  but the loss breakdown did not count it, so the live plot showed "100 %
  predation". Fixed in section 132.
- **Pelagic fish starve because zooplankton disappears.** Losses are 99 % starvation;
  gadoid predation on them is about 0.015 /yr, against a budget of 0.30 /yr.
- **The trained zooplankton policy feeds little.** It eats about 45 % of the time,
  rests about 27 % and moves about 30 %, which is useless at speed 0.02.
- **Greedy behaviour overshoots.** It grazes phytoplankton from 22 to about 2 t/cell
  within 100 ticks, so zooplankton then starves and pelagic fish follow. The cycles
  continue: zooplankton does not go extinct but swings between about 1,000 and
  15,000 t.

## 2. The budget does not close (analytic, one cell, saturating food)

Per tonne of zooplankton per tick, at a steady energy level s [C]:

    intake   = eat * f(B_phyto) * h(s) * E_phyto * 0.65
    costs    = resting_metabolism * feeding_cost
    growth   = g (s - u) * (energy_content + s * ME)     (mass balance)
    persist  if  g (s - u) >= M1 + M2_fish  ->  s >= 0.70

Model values: f(20 t/cell) = 0.238 t/t/tick, E_phyto 2000 MJ/t, resting metabolism
22.5 MJ/t/tick, feeding cost 2, ME 675 MJ/t, energy content 4500 MJ/t, g 0.0799,
u 0.5, M1 0.0135 /tick, fish predation about 0.0025 /tick.

| variant | s* | net growth /tick | max zoo a cell can feed (phyto r 0.067, K 24) |
|---|---|---|---|
| **now: h = 1 - s, phyto 2 kJ/g** | 0.656 | **-0.0035** | **none** |
| h = min(1, (1-s)/(1-u)), phyto 2 kJ/g | 0.759 | +0.0047 | 3.6 t |
| h = 1 - s, phyto 4.8 kJ/g | 0.785 | +0.0068 | 8.0 t |
| h = min(1, (1-s)/(1-u)), phyto 4.8 kJ/g | 0.869 | +0.0135 | 6.5 t |
| same + feeding cost x4 | 0.845 | +0.0116 | 5.5 t |
| h = 1 - s, phyto 4.8 kJ/g, feeding cost x4 | 0.746 | +0.0037 | 6.7 t |
| without mass balance, h = 1 - s | 0.811 | (positive) | - |

What the table shows:
- **No feeding behaviour can save zooplankton with today's parameters.** The
  appetite gate halves intake exactly where growth is needed: at s = 0.7, h is
  0.3.
- **Supply caps the stock.** The spawn of 35,000 t (9.7 t/cell) is above anything
  the phytoplankton can feed with E_phyto 2 kJ/g, so an overshoot is built in.
  With 4.8 kJ/g a cell can carry 6.5-8 t.

## 3. What the literature says

Copepod literature [notes; see verdicts there]:

| model parameter | literature | verdict |
|---|---|---|
| appetite h = 1 - s, halved at s = u | Hunger effects act on gut timescales: clearance rises 14-60 % after 6-14 h without food and is gone 1-3 h after feeding resumes (Tiselius 1998; Mackas & Burns 1986) [V]. Ingestion follows food concentration, saturated by gut capacity (Kiørboe et al. 2018) [V]. | **Not supported.** Appetite should stay about 1 until the reserve is nearly full. |
| phytoplankton energy 2 kJ/g WW | C/WW of about 0.042, i.e. large diatoms only. Flagellates 6-8 kJ/g WW; Ecopath convention 4.8. | **Low end.** |
| max ingestion 0.25 t/t/tick (0.44 body energy/day) | Small neritic copepods: median 0.44 body C/day at 10 C (0.75 without temperature correction); Calanus 0.13-0.5 [V/S]. | plausible to low |
| Holling half-saturation 1/(a h) = 1 t/km2 | 5-60 g WW/m2 (10-20 m feeding layer) [C from literature]. | **1-2 orders too low.** Zooplankton grazes phytoplankton down to very low density, which drives the overshoot. |
| implied max GGE about 0.56 | mean 0.26, median 0.22 (Straile 1997) [V]. | about 2x too high |
| feeding cost x2 | more than 4x (Kiørboe et al. 1985) [V]. | too low |
| resting metabolism 2 %/day of body energy | 1.5-3.5 %/day starved, 3-7 %/day routine | low end |
| assimilation 0.65 | 0.6-0.85 | OK |
| reserve 675 MJ/t (7.5 days at rest) | Acartia starves in 6-10 days [S] | OK for small copepods |
| zoo energy 4.5 kJ/g WW | small copepods 3.5-4.1, Calanus 5-6.8 | OK |
| M1 19.7 /yr | total 21-36 /yr at 8-12 C, non-predation 8-11 /yr (Hirst & Kiørboe 2002) [V]; invertebrates and larvae take about 53 % of predation on copepods (NS Ecopath) | plausible as invertebrate predation + non-predation |
| spawn 35,000 t = 9.7 t/km2 WW | Kattegat/Skagerrak summer about 3-10 g WW/m2 (seminar slides, not peer reviewed); North Sea Ecopath 16 g WW/m2; zoo:phyto in carbon 0.1-0.5 | within range |

## 4. Options

All are literature-anchored; none was chosen by tuning to the simulation.

1. **Appetite shape (mechanism, all DMs).** Replace h = 1 - s with an appetite that
   stays full until the reserve is nearly full, for example
   h = min(1, (1 - s)/(1 - u)) (full below maintenance, linear to zero at a full
   reserve) or h = 1 - s^k.
   - This is the change the literature supports most directly, and it removes the
     contradiction in section 2.
   - It changes every DM's ration at the maintenance level: h(u) goes from 0.5 to
     1.0.
     - Porpoises would then have about 2x the needed intake, and the feeding_cost
       1.1 workaround of section 130 could go back to 1.4.
     - The GUI energy gate and its tests move with it.
2. **Phytoplankton energy content 2 -> about 4.8 kJ/g WW** (Ecopath convention;
   flagellate-rich assemblage). It doubles zooplankton energy per tonne eaten and
   halves the tonnage needed, which also eases the supply limit. Phytoplankton K
   (wet weight) should be read in the same currency.
3. **Holling half-saturation for zoo -> phyto.** Lower `a`, keeping the ceiling 1/h,
   so that saturation happens at 5-60 t/km2. This gives the phytoplankton a refuge
   and damps the greedy overgrazing cycles. It is a pair parameter
   (max_intake_rate / handling_time).
4. **Then bring zooplankton efficiency back to the literature.** With 1-3 fixed the
   budget becomes generous, and the implied GGE (about 0.56) should be brought to
   about 0.26 via a feeding cost above 4x (Kiørboe 1985) and routine metabolism at
   3-7 %/day. This sets the zooplankton production level from the literature
   instead of from the gate.
5. **Leave alone:** M1 (19.7 /yr), the zooplankton spawn biomass, assimilation,
   reserve and zooplankton energy content. All are inside the literature ranges.

Recommended order: 1 + 2, check the closed phyto-zoo world and the trained-policy
world, then 3 and 4. Option 1 is a structural change for all decision makers; 2-4
are library values.

## 5. Caveats

- The analytic budget is one cell at steady state. Movement, currents and policy
  behaviour are ignored, and the greedy runs show the real system cycles.
- Several literature values are snippet-only. The Kattegat zooplankton standing
  stock comes from seminar slides, and the half-saturation conversion assumes a
  10-20 m feeding layer.
- Option 1 changes the budget of every decision maker, not only zooplankton.

## 6. Applied: options 1 and 2 (section 133 of the resume)

### What changed

- **Appetite.** h = min(1, max(0, (1 - s)/(1 - u))) for every decision maker:
  - in `FunctionalGroup.get_hunger`;
  - in `energy_balance.hunger_at(s, u)`, so the GUI gate evaluates h(u) = 1;
  - in the GPU engine.
- **Phytoplankton** `energy_content` 2000 -> 4800 MJ/t.
- **Porpoise** `feeding_cost` 1.1 -> 1.4. The 1.1 only compensated the halved
  appetite. The break-even ration on clupeids is now 7.5 %/day (Kastelein 4-9.5 %).

### Tests

The porpoise tests were re-posed for the new mechanism:
- the break-even ration lies in the Kastelein window;
- at Kastelein's maximum ration a pure gadoid diet cannot pay.

Suite: 780 passed.

### Effect

mareld2, 60x60, mass balance, mortality, migration and currents on, 1000 ticks
[M]. Values are biomass relative to the start:

| behaviour | t | zooplankton | pelagic fish | phyto (t/cell) | porpoises |
|---|---|---|---|---|---|
| policies trained before 133 | 100 | 0.05 | 1.02 | 0.8 | 0.41 |
| | 300 | 0.38 | 0.62 | 19.8 | 0.17 |
| | 1000 | **0.52** | **0.53** | 7.1 | 0 |
| greedy | 100 | 0.01 | 0.79 | 1.7 | 0 |
| | 1000 | 0.09 | 0 | 0.5 | 0 |

Before 133, the same policies ended at zooplankton 0.10x and pelagic fish 0.09x.

- **The energy budget now closes.** Zooplankton reaches s 0.6-0.7 and recovers. Its
  losses with the policies: natural 52 %, starvation 38 %, predation 11 %.
- **The next bottleneck is overgrazing.** Zooplankton now grazes phytoplankton from
  22 to 0.5-0.8 t/cell within 100 ticks, then crashes, and pelagic fish follow
  (greedy: extinct). That is option 3: the half-saturation of 1 t/km2 lets
  zooplankton feed at the ceiling at almost any phytoplankton density, so the
  phytoplankton has no refuge.
- **Gadoids still eat about 90-99 % benthos** (see the training-run analysis), and
  porpoises still starve.

Next: option 3 (zoo -> phyto half-saturation from the literature, 5-60 t/km2), then
option 4 (zooplankton efficiency to GGE about 0.26). Policies trained before 133
should be retrained.

## 7. Applied: option 3, half-saturation 20 t/km2 (section 134 of the resume)

### What changed

- `zooplankton_preys_on_phytoplankton` carries a pair-level attack rate
  `max_intake_rate: 0.0125`, with h = 4 unchanged:
  - ceiling 1/h = 0.25 t/t/tick, the same as the zooplankton `max_intake_rate`;
  - half-saturation 1/(a h) = **20 t/km2**, within the literature 5-60.
- **Mechanism.** The runtime now reads a pair-level `max_intake_rate`
  (`interactions.py`). `tick_time` already rescaled that key, but the matrix
  builder ignored it.
- **Energy gates.** The GUI energy gate, the live-project tests and
  `tools/probes/budget_gate.py` now use the Holling ceiling 1/h
  (`energy_balance.intake_ceiling`), not the attack rate. The two were equal
  under the old h = 1/a convention, which is also why every pair had a
  half-saturation of exactly 1 t/km2.

### Tests

`tests/test_half_saturation.py`. Suite: 783 passed.

### Effect

Same world and behaviours as section 6, 1000 ticks [M]:

| behaviour | t | zooplankton | pelagic fish | phyto (t/cell) | porpoises |
|---|---|---|---|---|---|
| policies trained before 133 | 100 | 0.58 | 1.03 | 3.0 | 0.41 |
| | 1000 | **0.66** | **1.45** | 5.0 | 0 |
| greedy | 1000 | 0.35 | 0.01 | 2.0 | 0 |

- **No crash any more.** Phytoplankton settles at 3-5 t/cell instead of crashing to
  0.5-0.8.
- **Zooplankton and pelagic fish persist with the policies.** Their losses are now
  mostly M1: 75 % and 93 %.
- **Greedy still loses pelagic fish (83 % starvation).** Greedy is a poor proxy: it
  keeps zooplankton at 0.35x.
- **Porpoises still starve in every run, and gadoids eat 92-98 % benthos.** These are
  the next problems, outside the zooplankton budget.

Remaining: option 4 (zooplankton efficiency to GGE about 0.26: feeding cost above 4x,
routine metabolism 3-7 %/day). The policies must be retrained.

## 8. Applied: option 4, zooplankton feeding cost (section 135 of the resume)

### What feeding_cost should contain (user decision)

The literature gives two bands, and they measure different quantities:

| definition | band | source |
|---|---|---|
| respiration only, fed / starved | 1.6-2.5 | Thor 2000; Svetlichny 2022; Koski 2017 (section 124) |
| **respiration + excretion, fed / starved** | **> 4** | Kiørboe, Møhlenberg & Hamburger 1985 [V abstract] |

The model has no excretion term (ammonium, DOC), so excretion has to be counted in
`feeding_cost`. Value: **4.0**, the lower bound of the Kiørboe band. This supersedes
the feeding-cost reasoning of section 124.2; the rest of 124 stands.

Kiørboe et al. 2018's mf/m0 = 6.7 also includes defecation. The model already
books defecation as 1 - assimilation (0.35), so 6.7 is not used as an upper bound.

### Cross-check (not the basis)

At steady state the assimilated energy pays for metabolism plus the growth that
covers M1 and fish predation (0.016 /tick). Gross growth efficiency:

| feeding_cost | metabolism while feeding (%/day of body energy) | steady-state GGE |
|---|---|---|
| 2.0 (before) | 4.0 | 0.40 |
| **4.0** | 8.0 | **0.29-0.31** |
| Straile 1997 | | mean 0.26, median 0.22, IQR 0.13-0.35 |

The metabolism while feeding, 7-8 %/day, sits at the top of the field routine
respiration band (3-7 %/day at 10 C). Resting metabolism stays at the fasting floor
(2 %/day).

### Effect

mareld2, mass balance, mortality, migration and currents, 1000 ticks, policies
trained before 133 [M]:

| feeding_cost | zooplankton | pelagic fish | phyto (t/cell) |
|---|---|---|---|
| 2.0 | 0.66 | 1.45 | 5.0 |
| **4.0** | **0.66** | **1.44** | **5.6** |
| 4.0, greedy | 0.32 | 0.00 | 2.3 |

- **Zooplankton persists.** Its stock is set by supply and losses, not by its
  efficiency. Phytoplankton ends slightly higher.
- **Energy gate:** the zooplankton ceiling of 780 MJ/t/tick far exceeds the feeding
  cost of 90, so it passes.
- Suite: 783 passed.
