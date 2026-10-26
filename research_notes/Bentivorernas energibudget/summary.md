# Benthivore energy budget: why the infauna M2 is ~5x the budget

**Question (resume 219.2, 223.3).** In frozen-policy probes of fix76 gen 10:
- Benthivores live on 93-99 % infauna at Q/B 7.6-7.9 by mass.
- Realised M2 on the infauna is 0.12-0.16 /yr against the budget's 0.029.
- The infauna falls to 0.78 of its start in 8 years.
- Benthivore starvation rises from 0 to 0.16 /yr in years 4-8, while their mean fill holds at 0.70.

Which side is wrong: the engine's energetics, or the budget's Q/B 3.48 and with it M2 0.029?

No library, manifest or code was changed. J marks a judgement. Literature tags and full citations are in
`tables/literature_evidence.md`. (Written by the analysis subagent; saved by the main agent.)

## 1. Engine mechanics (from the code)

**Intake** (`predation.py`)
- Demand per cell = B x eat x a N/(1 + a h N + w B) x hunger, with hunger = (1 - s)/(1 - u).
- Gains = intake x assimilation x E: 1820 MJ per t of infauna, 3000 per t of epifauna.

**Costs** (`movement.apply_energy_costs`)
- Each fraction pays RM x Q10(t) x its action multiplier: rest 1.0, eat 2.0, move 1.6.
- The eat multiplier replaces the resting cost, it is not added to it, and it is charged per eat tick
  whatever the amount eaten.
- The at_50m Q10 multiplier has an annual mean of exactly 1.000 (0.86 in April, 1.27 in October).

**Growth** (mass balance)
- dB = B g (s - u), debited at `energy_content` 4500.
- The new tonne also needs its reserve, s ME, so a tonne produced costs 4500 + 0.72 x 1800 = 5800 MJ in
  steady state.
- Starvation applies below s = u.

**Losses.** M1 0.29 /yr and the K loss (0.144 /yr) remove biomass and reserve together.

**Holling never binds on the infauna.**
- The visible half-saturation is (1 + wB)/(a h) = 0.04-0.2 t, i.e. 0.2-1 t/km2 of infauna, against beds
  of 50-130 t/km2; visible/half-saturation is about 250 (median) in fed cells.
- Intake is therefore the ceiling (15.2 /yr) x eat x hunger, and the reserve settles where intake pays
  costs plus production: 0.96 x 15.2 x 0.28/0.5 = 8.1 /yr.
- **The realised Q/B is set by the energetics; the budget's M2 never enters the dynamics.**

### 1.1 Accounting (probe year 1; mix eat 0.957 / move 0.041 / rest 0.002; fill 0.722)

| Item per t fish per yr | MJ assimilated | t infauna |
|---|---:|---:|
| A. Resting cost at 100 % rest (3.6 x 1460) | 5,256 | 2.89 |
| B. Action costs at the realised mix (x1.98) | 10,414 | 5.72 |
|   of which the feeding surcharge (eat x (2.0 - 1) x RM) | 5,030 | 2.76 |
| C. Production, g(s - u) = 0.708 /yr x 5800 | 4,108 | 2.26 |
|   C1. At the budget's gross 0.821 /yr (r_max 0.31 + M1 0.29 + M2 0.22) | 4,764 | 2.62 |
|   C3. Part of C: the reserve each new tonne carries (s ME) | 921 | 0.51 |
| **D = B + C** | **14,522** | **7.98** |
| Measured in the probe: gains (costs charged 10,407) | 14,623 | Q/B 7.57 |

What Q/B 3.48 delivers depends on the diet:
- **Budget mix** (infauna 0.511, epifauna 0.209, planktivores 0.084, predators 0.019; the other 0.177
  taken at the pools' mean quality, 2571 MJ/t): 8,946 MJ/t/yr.
  - The Q/B needed to pay B alone is 4.05; to pay B + C it is 5.65 at the realised P 0.708, or 5.90 at
    the budget's P 0.821.
  - The literature file values the "other" share as fish at 4750 x 0.8 and gets 2620 MJ/t and 9,120
    MJ/t/yr; the difference is under 2 %.
- **Engine diet** (99 % infauna, 1820 MJ/t): 6,334 MJ/t/yr.

### 1.2 Decomposition

Engine infauna intake is 7.98 t/t/yr against the budget's 1.78 (x4.5). Two factors multiply:

1. **Energetics, x1.62.** Engine need 14,522 MJ/t/yr against 8,946 delivered. On the budget diet this
   alone gives M2 = 5.65 x 0.511 x 1.53/93.6 = 0.047 (0.049 at P 0.821).
2. **Diet, x2.77.** The fish take 99 % infauna instead of 51 %: the epifauna is overeaten by the
   predators (223.3); the fish -> fish pairs are small; 17.7 % of the budget diet (sandeel, gobies,
   cephalopods, carrion, within-group) has no pool. In noepi the diet is still 85-88 % infauna (Q/B
   7.2-7.3, starvation 0.18 /yr in year 8).

The energy intake itself is not absurd: 14,600 MJ/t/yr assimilated is 1.5-1.6x what the published ration
on the mixed diet assimilates. The mass Q/B doubles because the engine diet is almost all low-energy
infauna.

### 1.3 Is SDA double-counted?

**Not with the assimilation factor.** In C = F + U + SDA + R + G the 0.7 nets only F (or F + U under the
Ecopath reading); neither convention nets SDA. The engine has no intake-proportional SDA, so
`feeding_cost` is the only place SDA lives.

**But it is too large and has the wrong form.**
- RM 3.6 is already SMR 3.0 x 1.2 (routine). The surcharge adds a whole RM on 96 % of ticks: 5,030
  MJ/t/yr.
- SDA is 0.10-0.15 of ingested energy (Tirsgaard et al. 2015; Fonds et al. 1985; Fish Bioenergetics 4
  cod: 0.17 of (C - F), Hansson et al. 1996). At the field ration (about 0.3 of Cmax) that is 1,200-2,300
  MJ/t/yr.
- The fixed surcharge equals SDA only at 0.85-1.25 x Cmax; the fed/unfed 2.4 of Fonds et al. 1985 was
  measured at satiation.
- **Total cost while eating is 2.4 x SMR.** Field metabolism, SDA included, is 1.5-2.5 x SMR, central
  2.0 (J). Wild cod measure 1.9-3.5 x SMR (Chung et al. 2019, Commun. Biol. 2:24), and a FB4 North Sea
  cod run at an SMS ration gives about 7.9 MJ/t/tick. The level is plausible for cod (31 % of the group)
  and high for haddock and flatfish.

**The per-tick form also matters in depleted cells.** The fish there gain 9,360 and pay 10,652 MJ/t/yr:
the full surcharge on an empty bed. With routine cost plus intake-proportional SDA they would pay about
7,300.

### 1.4 Budget premise notes

- **s* = 0.758 and g come from the best checked pair, the planktivores** (5700 MJ/t), which are 8 % of
  the budget diet and 0-1 % of the realised one. Flag only.
- **Production double charge (J).** `energy_content` 4500 is the whole body, liver included, yet the
  engine charges each new tonne its reserve on top, so an engine tonne holds about 5,800 MJ against a
  measured 4.5-4.75 kJ/g. Planktivores and predators use lean energy plus reserve; the benthivores do
  not. Literature production is about 2,900 MJ/t/yr (Z 0.64 x 4500); the engine charges 4,108-4,764, of
  which ~920 is the reserve charge and the rest P/B 0.71-0.82 vs 0.64.
- **`budget_gate.py` only tests feasibility at the ceiling.** Benthivore line: ration 0.01043 t/t/tick
  (4.2 %/day, 15.2 /yr), cost 7.2, need_q 690, worst quality 1820, verdict "any diet". Infauna sat_min
  0.379 means break-even on infauna alone needs Q/B 5.8 at 100 % eat, before any production. The gate
  never compares this with the budget's Q/B.

## 2. Literature (details and citations in tables/literature_evidence.md)

- **Q/B (stock level, mixed diet): 3.5 (2.5-5.3).** SMS evacuation rations at the box's size mix: cod
  4.6, haddock 2.8, i.e. 3.6 for the gadoid 70 % (ICES WGSAM 2024). Mackinson & Daskalov 2007 adults:
  haddock 4.4, cod 3.5. Greenstreet et al. 1997 rations: 5.3 / 4.8. Christensen 1995 called the gross
  efficiencies behind the raw values (35 %) unrealistic. An energy budget at measured SMR, field
  multiplier ~2 and realistic production needs 4.2 (3.4-5.0) on the mixed diet, i.e. 6.1 (4.9-7.2) on
  infauna alone. J: 3.48 is a low-to-central value.
- **SMR at 7-8 C:** about 2.8 MJ/t/tick (Schurmann & Steffensen 1997; Norin 2019; Fonds et al. 1985).
  Bottom water at 100-300 m is 7-8 C all year (SMHI; A17 100 m mean 7.6). The 50 m layer costs x1.07 on
  the annual mean.
- **Field metabolism:** 8,200 (6,100-10,200) MJ/t/yr including SDA; the engine charges 10,200-10,400.
- **SDA:** 0.10-0.15 of C, scaling with meal size (Jobling & Davies 1980; Secor 2009; Perez-Casanova et
  al. 2009 for haddock: 2-5 % at small meals).
- **Infauna energy density:** diet-weighted 2,620 MJ/t wet weight (realistic 1,900-3,300), from Steimle
  & Terranova 1985 and Ricciardi & Bourget 1998: polychaetes 3.5, whole ophiuroids 1.8-2.1, bivalves 1.2
  kJ/g. 2600 is supported.
- **Assimilation:** energy absorption of carnivorous fish on soft prey 0.8-0.85, about 0.75 after
  excretion (FB4); no measurement for skeleton-rich prey. J: 0.65-0.75; 0.7 is within the evidence,
  giving 1,820 (1,300-2,500) MJ assimilated per t of infauna.
- **Mass balance:** in the North Sea fish take about 4 % of infauna production but 70-80 % of crab and
  Nephrops production (Mackinson & Daskalov 2007; Heath 2005).

## 3. Where starvation arises (spatial probe)

All starvation came from the energy branch (s < u); the extinction sweep removed nothing.

| Year | Starving cells (>0.05 /yr) | Share of B | Share of starvation | Grid starvation /yr |
|---|---:|---:|---:|---:|
| 1 | 0 | 0 | - | 0.000 |
| 4 | 25 | 0.01 | 0.98 | 0.004 |
| 6 | 144 | 0.06 | 0.99 | 0.063 |
| 8 | 422 | 0.10 | 1.00 | 0.156 |

Starving cells grow every year: 2, 9, 25, 57, 144, 284, 422 in years 2-8.

- **Depletion.** Year 8: 207 cells hold 0.5 t/km2 of infauna (0.00 of K); fish there starve at 2.4 /yr,
  gaining 6,284 against costs of 10,573 MJ/t/yr, at fill 0.40. 74 cells at 0.05-0.2 K starve at 0.28
  /yr; every class above 0.2 K starves at less than 0.03 /yr.
- **Origin: aggregation on rich beds.** The year-8 starving cells started richer (infauna 104 vs 89
  t/km2, K 121), shallower (126 vs 173 m; ~0 starvation below 250 m) and less trawled (effort 1.9 vs
  3.2; the lowest trawl quartile starves most, 0.24 /yr). Benthivore density there rose from 1.9 to 3.8
  t/cell by year 6 while the infauna fell from 104 to 60; the fish then starved (1.2 t/cell in year 8).
  These are not the spawn colonies (6-16 % of the cells vs 8 % of the grid).
- **Supply vs demand.** The maximum logistic surplus r K/4 there is ~26.5 t/km2/yr; at Q/B 7.7 it is
  exceeded above 3.45 t/cell of fish. At the budget's infauna ration (1.78) the same bed would carry ~15
  t/cell. 405-483 cells exceed K/4 every year.
- **The policy does not leave.** Move share 0.02-0.03 and eat share 0.97-0.98 on emptied beds; one-year
  training episodes never showed a depleted bed (223.1). If the fish left, recovery to half the trawled
  equilibrium would take a median 5.5 yr (0-9.6): within the horizon a depleted cell is nearly absorbing.
- **Ruled out.** Temperature: starvation spans all months, and in year 6 it peaks in March-May at the
  lowest Q10. Interference: in the starving cells visible infauna is still 12-13x the half-saturation
  (median); only the emptied cells fall below it. Trawling: trawled cells starve least.
- **Why mean fill holds at 0.70:** 90 % of the biomass sits in fed cells at 0.72; the starving 10 % sit
  at 0.51.

## 4. Options (evidence and effect; not a choice)

Effects from `scripts/options.py`. Columns: assimilated need (MJ/t/yr); the infauna Q/B on the engine diet
and its M2; the M2 on the budget diet. Today: 14,521; 7.98 / 0.130; 0.047. Budget M2: 0.029.

| # | Quantity | Engine | Literature range | Effect |
|---|---|---|---|---|
| 1 | `feeding_cost` (SDA form) | 2.0 x RM per eat tick | SDA 0.10-0.15 of C, intake-proportional (Fonds et al. 1985 at satiation; Tirsgaard et al. 2015; FB4 0.14 of C); or 1.3-1.5 x RM (J). Counter-evidence: cod field metabolism 1.9-3.5 x SMR (Chung et al. 2019) | SDA 0.15: 12,080; 6.64 / 0.108; 0.039. SDA 0.10: 6.08 / 0.099. x1.5: 6.60 / 0.108. x1.0: 5.22 / 0.085 |
| 2 | Production charge (structural, J) | ec 4500 whole body + s ME | Whole body 4.5-4.75 kJ/g (lean ~3600 + reserve) | 13,884; 7.63 / 0.125 |
| 3 | Budget Q/B | 3.48 | 3.5 (2.5-5.3) published; 4.2 (3.4-5.0) energy budget | Budget-diet M2 0.035 (0.028-0.042) at 4.2 |
| 4 | No-pool 17.7 % in the budget | Infauna 0.511 of consumption | Redistributed over the pools the fish can eat, infauna ~0.62 of the pooled diet | M2 0.035 at Q/B 3.48, 0.043 at 4.2 |
| 5 | Infauna E x AE | 2600 x 0.7 = 1820 | 1,300-2,500 (E 1,900-3,300; AE 0.65-0.75) | Not the cause. At 2080: 6.98 / 0.114 |
| 6 | Temperature layer | 50 m, mean 8.4 C | Bottom 7-8 C | x0.93 on RM: about -730 MJ, -0.4 t infauna |
| 7 | Diet structure (no parameter) | 99 % infauna | Budget 51 % (noepi still 85-88 %) | x2.4-2.8 on infauna M2 |
| 8 | Behaviour / horizon (mechanics) | Depleted cells absorbing, policy stays | - | Spatial starvation; one-year episodes cannot see it |

**Combined literature-consistent energetics** (feeding 1.0 + SDA 0.15 + lean ec, and ~0.93 temperature):
about 10,900-11,300 MJ/t/yr, i.e. 6.2 infauna Q/B (about 5.8 with temperature) on the engine diet, M2
0.095-0.10; on the budget diet 4.4 (M2 0.037). Both sides meet near Q/B 6 on infauna, or about 4.2 on the
mixed diet. The residual factor of ~3 between the engine's M2 (~0.10) and the budget's 0.029 is the diet
structure (rows 4 and 7), not an energy parameter.

Do not set values from this probe outcome: per project rules, a probe shows where the mechanics fail and
is never the source of a value.
