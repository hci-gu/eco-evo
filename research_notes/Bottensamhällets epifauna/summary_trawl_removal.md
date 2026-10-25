# What the bottom trawl removes from `benthic_community` (summary)

Written by the main agent from the trawl subagent's report (the harness
does not let subagents write summary files). Evidence:
`evidence_C_trawl_catch.md`, `evidence_D_trawl_kill.md`; data and
scripts in `data/`. Mean effort E = 3.014 MW-h/km2/yr (the model's map).

## Bottom line
The catch of edible community members is about 1/33 of the gear's
seabed kill: 0.68 t/km2/yr (0.39-1.43), i.e. 0.0068 /yr at B = 100,
against d x SAR = 0.226 /yr. 92 % of the catch is northern shrimp and
Nephrops; on Nephrops it removes ~6.6 %/yr of its 4.5 t/km2 (ICES
harvest rate 5.4 % by number, ~7 % by weight).

## (a) Catch in the grid at mean effort, t/km2/yr
| Taxon | Low | Central | High | Slope per unit effort |
|---|---|---|---|---|
| Northern shrimp (landings x1.12 boiled-weight, + 8 % dead discards, Norway included) | 0.218 | 0.336 | 0.441 | 0.111 |
| Nephrops (landings + 75 % of discards, discards 17.6 % of landings) | 0.164 | 0.296 | 0.424 | 0.098 |
| Crabs and lobster landed | 0.003 | 0.006 | 0.008 | 0.002 |
| Dead discards of other invertebrates | 0.004 | 0.044 | 0.557 | 0.015 |
| Total | 0.389 | 0.681 | 1.430 | 0.226 |

Low and high are envelopes (every factor at its end together); without
the Nephrops-trawl invertebrate-discard guess the high is 0.89.
Shrimp = (1137 x 0.785 + 171 x 0.559 + 27 x 0.132 + 1.5 x 0.057) x 1.12
+ 7.8 = 1119 t/yr, x 1.08, / 3600 km2. Nephrops = 332 x 0.785 + 1081 x
0.559 + 282 x 0.132 + 678 x 0.057 = 941 t/yr, x (1 + 0.176 x 0.75),
/ 3600. The multipliers are each rectangle's EMODnet effort share inside
the grid (45G0, 44G0, 45G1, 44G1). As a rate: loss = 0.226 E / B per year.

## (b) Kill of fauna that is not caught
- Whole community: 0.226 /yr (0.109-0.381), slope 0.075 /yr per unit
  effort, 5.14e-5 per tick (the library's d 0.10 x SAR 0.75 E).
- Prey-sized infauna alone: 0.136 /yr (0.054-0.381) with d 0.06
  (0.03-0.115). d has no taxon split in the literature (Hiddink pools
  all taxa; Pitcher: "primarily ... epi-fauna"); Kaiser et al. 2006 [F]
  annelids -24 % per multi-pass otter-trawl experiment on mud;
  Sciberras et al. 2018 [S] -3 % per pass; Hinz et al. 2009 [A] chronic
  trawling cuts infauna, not epifauna, biomass on Nephrops mud.

## Combined (user's choice: catch + killed infauna)
0.226 + 0.007 = 0.233 /yr at mean effort; slope 5.29e-5 per tick per
unit effort (0.00529 at table value 100). Shrimp plus Nephrops (0.63)
are additive to d (grab-based d studies miss them); caught crabs and
echinoderms partly count in d already.

## Findings to flag
1. Norway is negligible in the grid (haul-located shrimp ~5 t/yr in the
   grid, 33 t/yr in 45G0), against ~975 t/yr implied by the EIA; the EIA
   figure was set aside (EIA-based high ~1,650 t shrimp in the grid).
2. The effort map level may be too low: EMODnet effort in 44G0 fell 86 %
   from 2018-21 to 2019-22 while EU Nephrops landings held; Danish VMS is
   probably missing from the recent windows, and the map (2021-24 mean)
   and every catchability calibrated to it may be low.
3. Vessels under 12 m (outside VMS) land 0.1 % of the shrimp and ~5 % of
   the Nephrops: (a) and (b) are on nearly the same basis.

## Biggest uncertainties
Invertebrate discards from Nephrops trawls (no Skagerrak data; Clyde
0.5-10 kg per kg Nephrops); where in 44G0/45G0 the Nephrops is caught;
d for deep-mud infauna per pass; today's shrimp catch is 35-60 % below
the 2015-2023 mean (ICES 9.6 kt then, 6.2 kt in 2024, advice 4.0 kt for
2025/26; FDI ends in 2023).

## Key sources
STECF FDI spatial landings EU27 2013-2023; Norwegian ERS logbooks
2011-2024; ICES 2025 advice pra.27.3a4a and nep.fu.3-4; EMODnet bottom
otter effort; SLU observer catch composition 2011-13 (MSC 2015 report);
Bergmann et al. 2002; Kaiser et al. 2006.
