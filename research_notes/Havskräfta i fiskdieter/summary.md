# Nephrops in the benthivores' diet (resume 223.14)

Written by the research subagent, saved by the main agent (2026-10-26). Scripts in `scripts/`
(`ices_nephrops_scan.py`, `dapstom_nephrops_occurrence.py`, `season_factor.py`, `nephrops_group_share.py`),
tables in `tables/` (`species_nephrops_summary.csv`, `group_share_scenarios.csv`, literature rows), data in
`data/`. J = judgement; [F] full text, [A] abstract, [S] secondary.

## Bottom line

- Central: Nephrops is **1.5 % of the benthivore group's consumption** (0.069 of the measured
  benthos-epifauna part 0.2182; 0.048 of the whole epifauna part 0.3143). **M2 on norway_lobster 0.022
  /yr** at Q/B 4.46 (0.017-0.025 at Q/B 3.4-5.0). Range: low 0.49 % (M2 0.007), high 4.3 % (M2 0.064);
  extreme corner 6.5 % (M2 0.10, J). The 221 note's 0.003 (M2 0.001) is ~20x too low; the NNS chain's
  0.118 (M2 0.038) ~1.7x too high (it mixed the NNS benthos composition with the Skagerrak benthos share).
- Split: cod 1.31 %, haddock 0.13 %, long rough dab 0.06 % of group consumption.
- Booking rule (J): Nephrops sits inside cod's benthos share and replaces part of the crab/Pagurus/
  Aphrodita composition; the epifauna total (0.3143) does not rise.
- Whiting eats only traces (<= 0.55 % of its diet in the northern data; one Silver Pit outlier): predators
  -> Nephrops stays unchecked.

## Key data (cod; Nephrops % of stomach mass unless stated)

- Dorman et al. 1992, Skagerrak IYFS, "much of it on the Nephrops grounds", Feb 1991, 294 fed: 3.48 %
  pooled (30-40 cm 6.5, 50-70 cm 4.4, >= 70 cm 3.6, < 30 cm 0); Kattegat 1.97 % [F].
- ICES DB NNS Q1 1981: 4.25 %; NNS Jan-Feb 2023 (77 fed): 8.16 % (100-200 m: 10.1 %, two cod carry most);
  Skagerrak 2023-25: 0.01 % but from only five hauls (the 541 Swedish IBTS 2023 records have no prey rows,
  so they are not "90 % empty") [F].
- North Sea stomach projects 1981 / 1991 by quarter: Q2-Q3 1.5-4.6x Q1; ration-weighted annual / Q1 factor
  median 2.06 (Daan 1989; Hislop et al. 1997) [F].
- On Nephrops grounds elsewhere: occurrence 23-65 % in large cod (Fladen, Farn Deeps, Forth; Howard 1980,
  Kinnear & Livingstone 1979); Irish Sea 12-58 % of intake (Symonds & Elson 1983; Fannon & Hillis 1982;
  Armstrong et al. 1991); S Iceland ~30 % [S].
- Cod eat Nephrops from ~40-50 cm; the Nephrops eaten are mostly 20-30 mm CL (~10 g).
- Haddock 0.3 % (0-1 %); long rough dab 0.5 % (0.2-1.4 %); plaice, witch, dab ~0 [F/J].

## Why the estimates disagreed

The Skagerrak 2023-25 figure rests on five hauls; every Skagerrak/NNS estimate is Q1, when ovigerous
females stay in burrows (season factor 2.06); digestion: stomach share x 0.65 (0.5-0.95) = intake share;
regurgitation does not obviously bias hard Nephrops down (J); station placement factor 1 (0.7-1.3; 70 % of
the grid is UWTV ground); cod size structure is not the explanation (> 50 cm still 48-58 % of grid cod
biomass). The user's observation (hook-caught, summer, large cod on grounds) fits the occurrence data and is
compatible with a few percent of the group's annual consumption.

## Scenarios (nephrops_group_share.py)

| Scenario | Cod intake % | Group p | M2 (Q/B 4.46) | % of Nephrops production |
|---|---|---|---|---|
| 221 note (Skagerrak 2023-25, no levers) | 0.02 | 0.03 % | 0.0005 | 0.1 |
| NNS chain recomputed | 5.69 | 1.66 % | 0.025 | 7.3 |
| LOW (Kattegat 1991, season 1.5, digestion 0.6) | 1.61 | 0.49 % | 0.007 | 2.1 |
| **CENTRAL (Skagerrak Feb 1991, season 2.06, digestion 0.65)** | 4.58 | **1.50 %** | **0.022** | 6.6 |
| NNS 2023 | 7.62 | 2.38 % | 0.035 | 10.4 |
| HIGH (NNS 2023 100-200 m, digestion 0.7) | 12.9 | 4.32 % | 0.064 | 19.0 |

Comparisons: the gap to the Irish Sea (M2 0.04-0.25) is the diet share (4.6 % vs 16-30 % of intake), not
predator density; FU 3-4 natural mortality M 0.2-0.3: central fish M2 0.022 is 7-11 % of M, HIGH 21-32 %.

Not obtained: Pihl 1994, Borje et al. 1987, Brander & Bennett, Mattson 1990/92, Dombaxe 2002, any
hook-vs-trawl diet comparison. Trap: Mackinson & Daskalov 2007's diet matrix has a duplicated page.
