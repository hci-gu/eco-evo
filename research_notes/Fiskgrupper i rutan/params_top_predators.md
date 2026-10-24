# Seals and porpoises on the three fish groups

Written 2026-10-24 for the split of `gadoids` + `pelagic_fish` into `planktivorous_fish`,
`benthivorous_fish` and `predatory_fish` (`split_brief.md`). Nothing in the code, the library, the
manifest, the tests or `mareld_resume.txt` was edited.

Tags: [F] full text read / table viewed, [A] abstract only, [S] secondary or search snippet,
[D] computed here (`scripts/top_predators.py`, output in `tables/top_predators_stdout.txt`).
"Judgement" marks my reasoning where no source gives the number.

Prey biomass used: planktivores 12 000 t (3.33 t/km2), benthivores 5 500 t (1.53), predators
4 300 t (1.19); grid 3 600 km2. Predators: harbour seals 15 t, harbour porpoises 12 t (the manifest).

## 0. Bottom line

| Pair | Diet share by mass (model pools) | Consumption t/yr | predation_mortality /yr | a | h | Type | Basis |
|---|---|---|---|---|---|---|---|
| seals -> benthivorous_fish | 0.60 (0.45-0.70) | 192 | **0.035** (0.020-0.059) | **0.10** (fit carry-over) | E_benth / 212.5 (25.9 at 5500; 21.2 at 4500) | II | Koster 1977-89 [F], Sorlie 2020 [A], GU 2019 [S]; floor 0.85 kept (vibrissae on resting bottom fish) |
| seals -> predatory_fish | 0.15 (0.08-0.25) | 48 | **0.011** (0.005-0.027) | **0.05** (judgement, 0.03-0.10) | E_pred / 212.5 | II | Koster [F]; haddock/pollack/saithe split is a judgement; floor 0.75 (judgement) |
| seals -> planktivorous_fish | 0.25 (0.10-0.40) | 80 | **0.0067** (0.002-0.015) | **0.002** kept (fit, 199.10) | E_plank / 212.5 (35.3 at 7500) | III kept | Koster [F], Kappa 2026 [A], Harding 2022 [S]; fix72 seals took 14-24 % at this a (206.1) |
| porpoises -> planktivorous_fish | 0.65 (0.55-0.75) | 152 | **0.0127** (0.010-0.018) | **0.0225** kept | 28.6 (mass ceiling) | II | Borjesson 2003 [F via Sveegaard Table 2], Aarefjord 1995 [F via same]; floor 0.95 kept |
| porpoises -> benthivorous_fish | 0.15 (0.10-0.25) | 35 | **0.0064** (0.004-0.013) | **0.12** (fit carry-over) | 28.6 | II | same; cod, hagfish, half of "Gadidae spp."; floor 0.75 kept |
| porpoises -> predatory_fish | 0.20 (0.12-0.25) | 47 | **0.011** (0.006-0.016) | **0.20** (fit carry-over) | 28.6 | II | same; whiting, saithe, half of "Gadidae spp."; floor 0.90 (judgement) |

Totals: seals 320 t/yr (range 250-460), porpoises 234 t/yr (211-280). Seabirds are muted (section 5).

**The library as it stands double-counts.** The modified `fg_library.yaml` copied the old gadoid
pairs onto both demersal pools. That books seals at 0.12 + 0.12 /yr = 660 + 516 t/yr of demersal
fish, 1 194 t/yr in all, 3.7x the 320 t/yr ration. Porpoises are booked at 0.0557 + 0.0557 =
306 + 240 t/yr demersal, 686 t/yr in all, 2.9x their 234 t/yr. These M2 values enter the derived
`growth_rate` of the fish groups, so the parallel fish reviews inherit the error until the pairs
change [D].

| Pair | Current (copied) M2 | Recommended M2 |
|---|---|---|
| seals -> benthivorous_fish | 0.12 | 0.035 |
| seals -> predatory_fish | 0.12 | 0.011 |
| seals -> planktivorous_fish | 0.0015 | 0.0067 |
| porpoises -> benthivorous_fish | 0.0557 | 0.0064 |
| porpoises -> predatory_fish | 0.0557 | 0.011 |
| porpoises -> planktivorous_fish | 0.0117 | 0.0127 |

Main uncertainties, in order:
1. The seal prey item "haddock/pollack/saithe" cannot be split. It is the largest item in both
   recent Skagerrak studies, and it straddles benthivores (haddock) and predators (saithe). It moves
   the seals' benthivore:predator split between about 70:5 and 45:30.
2. All Kattegat/Skagerrak porpoise stomach data are from 1985-1996, when herring were more abundant.
   No recent local porpoise diet exists.
3. Every diet comes from coastal animals (haul-out scats, bycatch). The grid is offshore, 44-530 m on
   the trench slope. Offshore foraging should favour Norway pout, blue whiting, haddock and long rough
   dab, which raises the planktivore and benthivore shares a little (judgement).

## 1. Diets

### 1.1 Harbour seals

Sources, closest to the grid first. The grid is 57.81-58.31 N, 10.07-11.15 E. The nearest
haul-outs (Lysekil: Smogen, Hallo, Maseskar) are 5-15 km off its NE and E edges, but they have no
diet study of their own.

- **Koster archipelago, Swedish Skagerrak, about 60 km N of the grid.** Harkonen 1987 and Harkonen &
  Heide-Jorgensen 1991, % by weight, via Lundstrom et al. 2010 Table 4 [F]. This is the only seal
  source with every species resolved. It is old (1977-79, 1989).

  | Species (pool) | 1977-79 | 1989 |
  |---|---|---|
  | herring (P) | 8 | 21 |
  | sprat (P) | 1 | 3 |
  | blue whiting (P) | 3 | 7 |
  | Norway pout (P) | 6 | 11 |
  | sandeels (not in pool: coastal sand) | 7 | 3 |
  | cod (B) | 20 | 23 |
  | poor cod (B) | 7 | 4 |
  | lemon sole (B) | 15 | 6 |
  | American plaice = long rough dab (B) | 7 | 2 |
  | other flatfish (B) | 5 | 1 |
  | whiting (Pr) | 6 | 8 |
  | other Gadiformes (split B/Pr) | 9 | 3 |
  | viviparous blenny, wrasses, eel, other (none) | 6 | 8 |

  By pool [D]: 1977-79 planktivores 18, benthivores 54, predators 6, unresolved gadiforms 9, not in
  a pool 13. 1989: planktivores 42, benthivores 36, predators 8, gadiforms 3, not in a pool 11.
- **Swedish northern Skagerrak 2020** (Kappa et al. 2026, Mar. Mamm. Sci. 42: e70069; 73 scats from
  three haul-outs, otoliths) [A]. By biomass: haddock/pollack/saithe 18.7 %, ballan wrasse 13.5 %
  (the two largest, so every other item is below 13.5 %). By otolith number: sandeel 39.6 %, Norway
  pout 21.8 %. Blue whiting 7.9 % by number [S, Oceanbites summary]. Cod and herring have fallen
  since 1977-89. The full table sits behind a bot wall (Wiley 403); open access per Unpaywall.
- **Swedish northern Skagerrak 2019** (GU MSc thesis, scats) [S]. Flatfish 46 % of energy, herring
  2.8 % of energy (about 24 % historically).
- **Harding 2022** (GU, popular talk) [S]. The commonest recent Skagerrak prey are blue whiting,
  Norway pout, poor cod, rockling and sandeel. Kattegat seals eat more flatfish (long rough dab,
  dab). Prey mostly 7-20 cm.
- **Norwegian Skagerrak, Telemark/Aust-Agder** (Sorlie et al. 2020, Mar. Biol. Res. 16(4)), about
  100-150 km NW across the trench [A]. By biomass: plaice 24.1 %, gadoids + pleuronectids 87.1 %,
  cod 2.3 %. By index Qi: haddock/pollack/saithe 32.7 %, Trisopterus 12.5 %, plaice 12.4 %,
  herring 10.0 %. Total 315 t/yr for that coast's seals.
- **Kattegat (quantitative)**: none found. **Eastern North Sea**: not searched further; Danish west
  coast haul-outs are far from the grid.
- **Future source**: SLU's Swedish Skagerrak seal diet series, 1999-2024, more than 2 000 scat and
  stomach samples (project 2024-27, M. Mion, K. Lundstrom, M. Lindmark), not yet published [S, SLU
  project page]. It will supersede everything above.

Mapping problems:
- "Haddock/pollack/saithe" is one otolith group. Haddock is a benthivore; saithe is a predator;
  pollack is under 1 % of the predator group's biomass, so in the grid's terms it counts with
  saithe. No source splits the group. Judgement: 50:50.
- "Trisopterus" (Sorlie) mixes Norway pout (planktivore) and poor cod (benthivore).
- Sandeel is mostly a coastal-bank prey; sandeels are under 1 % of the planktivore pool in the grid.
  I count them as "not in a pool".

Working split [D, judgement]:
- Raw, on all prey: benthivores about 50 %, predators about 13 %, planktivores about 20 %, not in a
  pool about 17 % (wrasse, sandeel, blenny/eelpout, gobies).
- Renormalised on the model pools: **benthivores 0.60 (0.45-0.70), predators 0.15 (0.08-0.25),
  planktivores 0.25 (0.10-0.40)**.
- The planktivore target is no longer the old 3-10 % herring figure. Norway pout and blue whiting
  are planktivores now, and the Koster shares on today's pool are 18-42 %. Recent data cut the
  herring part (2.8 % of energy) but keep pout and blue whiting.
- Renormalising is right for the budget. The model's seals take their whole ration from model prey,
  so the M2 they impose is the full ration times the in-pool shares. This is the same convention as
  section 166.3.

### 1.2 Harbour porpoises

Sources, closest first. All are stomach studies of bycaught or stranded animals.

- **Kattegat-Skagerrak, Borjesson et al. 2003** (Mar. Mamm. Sci. 19: 38-58; 112 bycaught porpoises,
  1988-96) [A]. The %W row is from Sveegaard 2011, Paper V, Table 2 [F, table viewed at 250 dpi]:
  cod 5.5, whiting 5.4, Gadidae spp. 6.8, saithe 5.0, Norway pout 4.1, herring 56.2, sprat 9.0,
  sandeels 1.8, gobies 0.8, pearlsides 0.4, hagfish 1.4, fish < 5 % 12.7. **The row sums to 109.1
  as printed.** I renormalise to 100.
- **Kattegat, Aarefjord et al. 1995** (39 porpoises, 1985-90; same table, %W estimated) [F]: cod
  10.6, whiting 8.2, Gadidae 11.1, herring 42.1, sprat 1.1, sandeels 0.1, gobies 13.1, eelpout 2.2,
  fish < 5 % 11.5. Sum 100. The same study gives qualitatively: Kattegat-Skagerrak herring, then
  cod, then poor cod; eastern North Sea whiting, then cod, then long rough dab [F, Sveegaard text].
- **Inner Danish waters** (Andreasen et al. 2017; 339 stomachs, 1980-2011) [S, earlier note]:
  adults cod 36 %, herring 34 %; juveniles cod 26 %, gobies 25 %, herring 18 %. Farther from the
  grid, more cod.
- **Southern North Sea** (Ransijn et al. 2021, Ecol. Evol. 11: 17458; 455 stranded, mostly Dutch)
  [F]: whiting 27.1 %, gobies 20.8 %, sandeels 18.5 %, herring 8.5 %, sprat 6.9 %, cod 6.0 %, other
  12.2 %. Far from the grid; used only for the attack-rate ratios (section 3.2).
- **Norwegian coast 2016-17** (UiT MSc thesis, 134 bycaught) [S]: juvenile saithe dominant, herring
  minor. North Norwegian fjords; not the Skagerrak.
- Kattegat-Skagerrak porpoise density follows herring density, mainly along the trench slope
  (Sveegaard et al. 2012) [A, earlier note]. The grid sits on that slope.

By pool [D] (Gadidae spp. split half benthivore, half predator):

| Study | Planktivores | Benthivores | Predators | Not in a pool |
|---|---|---|---|---|
| Borjesson 2003 (renormalised) | 63.9 | 9.4 | 12.6 | 14.0 |
| Aarefjord 1995 | 43.2 | 16.2 | 13.8 | 26.9 |
| Borjesson, on pools only | 74 | 11 | 15 | - |
| Aarefjord, on pools only | 59 | 22 | 19 | - |

Working split on the model pools: **planktivores 0.65 (0.55-0.75), benthivores 0.15 (0.10-0.25),
predators 0.20 (0.12-0.25)**. The trained fix72 porpoises took 59-67 % herring (206.3), inside
this target.

## 2. Consumption and predation_mortality

**Seals.** The library's ration is 5.9 % of body mass per day, from the model seal's own budget
(spend 283 MJ/t/day, report "Parametervärden för sälar i Skagerrak") [S, report]. That gives
~325 t/yr at 15 t. The trained seals realised ~350 t/yr at ~15 t (199.8). With the new diet the
mass ration depends on the pools' energy [D]:
- E benthivores 5500, predators 5500, planktivores 7500 MJ/t: 5.5 %/day, 300 t/yr.
- E benthivores 4500 (flatfish and long rough dab are lean), predators 5500: 6.1 %/day, 333 t/yr.

I use **320 t/yr**. Range 250-460 t/yr: Olesiuk's 0.33 MJ/kg/day gives ~325; Harkonen &
Heide-Jorgensen's 0.47 gives ~460 as an upper bound.

**Porpoises.** Q/B 19.5 /yr (FMR 0.36 MJ/kg/day on 6.75 MJ/kg prey; porpoises.md) and realised
~19 in fix72 (206.3) give **234 t/yr** at 12 t. Range: Q/B 17.6 (realised, 199.3) to 23.3 (the old
budget, 279 t/yr).

M2 = share x consumption / prey biomass [D]:

| Pair | t/yr | M2 /yr | Range (share x consumption ranges) |
|---|---|---|---|
| seals -> benthivores | 192 | 0.0349 | 0.020-0.059 |
| seals -> predators | 48 | 0.0112 | 0.005-0.027 |
| seals -> planktivores | 80 | 0.0067 | 0.002-0.015 |
| porpoises -> planktivores | 152 | 0.0127 | 0.010-0.018 |
| porpoises -> benthivores | 35 | 0.0064 | 0.004-0.013 |
| porpoises -> predators | 47 | 0.0109 | 0.006-0.016 |

The porpoises' survey-based biomass is ~84 t, 7x the model's 12 t (199.3, porpoises.md). If the
start is ever raised, every porpoise M2 above scales with it.

## 3. Attack rates, handling times and response types

### 3.1 What carries over mechanistically

- **Handling time, seals.** The library uses an energy ceiling: 0.85 MJ/kg/day gross, i.e. 1.5x the
  highest routine intake (Rosen et al. 2021; section 166.3). That is 212.5 MJ/t/tick, so
  **h = E_prey / 212.5** [D]. It reproduces the library (5500 -> 25.9; 7500 -> 35.3).
  At E 4500 h is 21.2 (ceiling 18.9 % BM/day); at 6500 it is 30.6. Use the energy_content the fish
  reviews choose. The planktivore pair uses the same formula: if the pool's energy as eaten is set
  below herring's reference-fill 7500 (Norway pout and blue whiting are ~5 kJ/g), its h falls too. The benthivores will probably be leaner than the copied 5500, so their h falls and
  the mass ceiling rises.
- **Handling time, porpoises.** The library uses a mass ceiling, 0.035 t/t/tick = 14 % BM/day
  (Kastelein 4-9.5 %/day x 1.5; section 149). **h 28.6 on every pair.** Seals and porpoises follow
  different ceiling conventions in the library today. I keep each predator's own one.
- **Visibility floors** describe the predator's sense against the prey's hiding:
  - Seals on benthivores: **0.85**, kept. Vibrissae find resting flatfish and bottom fish
    (Niesterok et al. 2017; section 167). The mechanism fits this pool better than it fitted the old
    gadoids.
  - Seals on predators: **0.75** (judgement). Saithe, whiting and mackerel rest in the water column.
    Seals follow their wakes (Wieskotten et al. 2010), but a resting school off the bottom is harder
    to reach than a fish on it.
  - Seals on planktivores: no pair floor; the prey's own 0.35 applies, as now.
  - Porpoises on planktivores: **0.95**, kept (biosonar; schooling is no defence).
  - Porpoises on benthivores: **0.75**, kept (seabed masking; inferred, section 149).
  - Porpoises on predators: **0.90** (judgement). They rest off the seabed, so no seabed masking.
- **Light.** No dark_ratio on any of these pairs. Seals have no documented light effect (167.2);
  porpoises echolocate.
- **Assimilation** stays per prey type: seals 0.85 on lean fish and 0.88 on herring-like prey;
  porpoises 0.84 on demersal fish and 0.90 on clupeids.

### 3.2 What was fitted, and what that means now

**Saturation first.** At the new demersal densities both demersal pairs of both predators sit near
their ceilings at the grid mean, whatever a is in the plausible range [D; rest share 0.4 at the pair
floor]:

| Pair, a | Visible t/km2 | Intake, % of ceiling | MJ/t/tick | Break-even visible t/km2 |
|---|---|---|---|---|
| seals -> old gadoids, 0.2 (0.69 t/km2) | 0.65 | 77 | 139 | 0.14 |
| seals -> benthivores, 0.2 (carried) | 1.44 | 88 | 159 | 0.14 |
| seals -> benthivores, **0.1** | 1.44 | 79 | 142 | 0.28 |
| seals -> predators, 0.2 (carried) | 1.07 | 85 | 153 | 0.14 |
| seals -> predators, **0.05** | 1.07 | 58 | 105 | 0.55 |
| seals -> planktivores, III 0.002 | 2.47 | 30 | 56 | 3.1 |
| porpoises -> old gadoids, 0.3 (0.69 t/km2) | 0.62 | 84 | 136 | 0.15 |
| porpoises -> benthivores, 0.3 (carried) | 1.38 | 92 | 149 | 0.15 |
| porpoises -> benthivores, **0.12** | 1.38 | 83 | 133 | 0.37 |
| porpoises -> predators, 0.3 (carried) | 1.15 | 91 | 147 | 0.15 |
| porpoises -> predators, **0.20** | 1.15 | 87 | 140 | 0.22 |
| porpoises -> planktivores, 0.0225 (grid mean) | 3.27 | 68 | 160 | 0.96 |
| porpoises -> planktivores, 0.0225 (5 t/km2, where they stand) | 4.90 | 76 | 179 | 0.96 |

Break-even = the visible density where eat-tick gain equals the eat cost (seals 75, porpoises
90 MJ/t/tick): B = I/(a(1 - I h)), I = cost/(assim x E). The seal planktivore figure uses E 7500 and
eat cost only; budget_gate uses its own convention (9.4 t/cell, 199.10).

So the diet in the model is set mainly by the policy's eat-slot allocation and by the floors, not
by a. The eat actions are one softmax over the prey slots, and total intake is at most max_j 1/h_j
(208.1). Between "carry a unchanged" and "rescale a" the energy return per cell changes by under
15 %.

**Two options.**
- (A) Carry a unchanged per predator (seals 0.2, porpoises 0.3 on both demersal pools). Basis: a is a
  clearance rate, a property of predator and prey type, not of prey density. Effect: demersal
  returns rise 8-15 % over the calibrated world, so both predators shift somewhat toward demersal
  prey, and the planktivore pairs (fitted against the old demersal return) would need refitting.
- (B) **Recommended.** Rescale so the demersal return at the new grid mean equals the calibrated old
  return (a x B invariance), then split it between the two pools by the literature's per-tonne
  selectivity. Basis: the old values were fitted in a world whose gadoid biomass was ~4x too low
  (`gadoid_level.md`). Part of the old a only compensated for that. The porpoise fit (151.1) needed
  gadoids to be 4.8x herring per tonne; with the corrected biomass the diet asks for 0.5x
  (benthivores) and 0.86x (predators) [D]. Rescaling undoes the biomass error; it adds no new fit.
  It also leaves the two fitted planktivore pairs valid.

Values under (B) [D]:
- a x B invariance alone gives seals 0.091 (benthivores) and 0.12 (predators), porpoises 0.14 and
  0.16.
- Per-tonne selectivity from the working diets, relative to planktivores: seals benthivores 5.2,
  predators 1.7; porpoises benthivores 0.50, predators 0.86.
- Ransijn et al. 2021 [F] measured relative attack rates per unit prey availability (southern North
  Sea, Type III): sandeel 1, sprat 0.238, whiting 0.120, herring 0.101, other 0.089, cod 0.058. So
  whiting is ~1.2x and cod ~0.6x herring per unit availability. That matches the Skagerrak diets'
  0.86 and 0.50 well. It is the only measured set of relative attack rates found, and it gives no
  support to the old 13x gadoid:herring ratio (151.3 already called that a fit).
- **Porpoises**: benthivores **0.12**, predators **0.20** (ratio 1.7, as the per-tonne selectivity;
  Ransijn whiting/cod 2.1). Planktivores **0.0225** kept.
- **Seals**: benthivores **0.10** (break-even at 0.18 of the pool's grid mean, the same position as
  the old pair at 0.2 / 0.69 t/km2). Predators **0.05** (judgement, 0.03-0.10): selectivity
  0.32 x 0.1 would give 0.03; 0.05 puts break-even at 0.55 t/km2 visible, about half the grid mean,
  so predators pay only in patches. Planktivores **0.002, Type III**, kept: fix72's trained seals
  took 14-24 % pelagic fish at this a (206.1), and the new target is 0.25 (0.10-0.40). This is the
  one place where a measured trained diet already lands in the new target.

**Response types.**
- Literature favours Type III for harbour seals and porpoises: Smout et al. 2014 (grey seals) [A],
  Middlemas et al. 2006 (harbour seals) [A], Ransijn et al. 2021 (porpoises; DIC III 97 202 vs II
  117 216 at the 4-day window) [F], a 2023 St Andrews thesis (all three North Sea mammals) [A].
- In the model, switching already comes from the policy's allocation (199.8). Type III on the main
  prey mainly cuts intake (162, 166.3). Type III also changes the units of a (per (t/cell)^2), so
  none of the Type II values above carry over to it.
- Recommendation: **Type II** on all demersal pairs and on porpoises -> planktivores (as now);
  **Type III** on seals -> planktivores (as now, fitted that way). Judgement.

**Test after retraining**: `tools/probes/pair_intake.py` on the trained policies. Retune a on the
pair that misses: seals predators outside 8-25 % -> 0.03 or 0.1; seals planktivores outside
10-40 % -> 1e-3 or 3e-3; porpoise demersal outside 25-45 % -> scale both demersal a together.
If a pair misses its range while its a already sits on the saturation plateau (section 3.2),
retuning a does little. The library's remaining lever is `predator_share`, the eat-share cap of
section 208. It is a diet fit, so the 208.1 rule applies: use it only knowingly, and say so in the
library comment.

## 4. Seal K and porpoise rates when demersal prey roughly quadruple

- **Seal K (0.0075 t/km2, 27 t).** K stands for off-grid regulation of the coastal metapopulation
  (199.3-199.4). Prey density in the grid does not change that meaning, so keep it. The equilibrium
  is B* = K (g_real - M1 - starvation)/r_max, with g_real the seals' realised gross growth
  (0.17-0.18 /yr in fix72, 206.2) [D]:
  - g 0.18, starvation 0.01: 17 t.
  - g 0.18, no starvation: 20 t.
  - g 0.20, no starvation: 24.5 t.
  More demersal food can remove the seals' late starvation (206.3) and lift g a little, so B* may
  drift up by a few tonnes. Re-read g_real and starvation after the retrain. If the level should
  stay at 15 t, set K = 15 x r_max / (g_real - M1 - starvation).
- **Seal M2 on the fish falls** from 0.12 /yr (on 2 500 t gadoids) to ~0.035 and ~0.011. This
  lowers the derived growth_rate of both demersal groups. The fish reviews should use these values,
  not the copied 0.12 + 0.12.
- **Porpoises.** They have no K, and the 206 x1.5 was food-driven: they starved at a 0.015. Under
  (B) their energy return is unchanged, so the 206 calibration holds. Under (A) they gain ~8-10 %
  energy at the grid mean and may grow, perhaps toward the survey's ~84 t (199.3). That is the open
  food-calibration question, not something to tune here. Watch porpoise biomass and starvation in
  the first long run.

## 5. Seabirds (muted)

Skagerrak seabirds (kittiwake, guillemot, razorbill, gannet, gulls) eat mainly sprat, herring and
sandeel, plus juvenile gadoids, and gannets also take mackerel (general knowledge; not researched
here). If seabirds are unmuted, the old seabirds -> pelagic_fish M2 (0.0236) belongs on
planktivorous_fish, with a small predatory_fish pair at most.

## 6. Sources and access

| Source | Grade | Access |
|---|---|---|
| Lundstrom et al. 2010, NAMMCO Sci. Publ. 8, Table 4 (Harkonen 1987; Harkonen & Heide-Jorgensen 1991) | [F] | `data/top_predators/lundstrom2010_nammco.pdf` |
| Sveegaard 2011 PhD, Paper V, Table 2 (Borjesson 2003; Aarefjord 1995; Lick; Andreasen 2009) | [F] | `data/top_predators/sveegaard2011_phd.pdf`, PDF page 92 |
| Ransijn et al. 2021, Ecol. Evol. 11: 17458 | [F] | `data/top_predators/ransijn2021_PMC8668753.xml` |
| Wallin 2017, SLU thesis (porpoise cod %W, daily intake 1.9-5.5 kg) | [F] | `data/top_predators/wallin2017_slu_thesis.pdf` |
| Borjesson et al. 2003, Mar. Mamm. Sci. 19: 38 | [A] | OpenAlex abstract; %W via Sveegaard [F] |
| Kappa et al. 2026, Mar. Mamm. Sci. 42: e70069 | [A] | OpenAlex abstract; Wiley PDF 403 |
| Sorlie et al. 2020, Mar. Biol. Res. 16(4) | [A] | OpenAlex abstract; Brage copies unreachable |
| GU MSc 2019; Harding 2022; Andreasen et al. 2017; UiT MSc 2019 | [S] | earlier notes (`../Parametervärden för sälar i Skagerrak/diet_and_foraging.md`, `../Vintermetabolism sill och tumlare/porpoise_winter.md`) and search snippets |
| SLU seal diet project 1999-2024 | [S] | slu.se project page; no data published |
| Smout et al. 2014; Middlemas et al. 2006; St Andrews thesis 2023 | [A] | via `diet_and_foraging.md` |
