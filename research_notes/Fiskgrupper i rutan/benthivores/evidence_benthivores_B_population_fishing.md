# benthivorous_fish: r_max, natural mortality and fishing mortality per member species

Scope: population parameters for the new group `benthivorous_fish` (1.53 t/km2, grid in 44G0/45G0, 50-300 m).
Shares are the long-run 2001-25 GOV shares from `../tables/long_run_species_by_group.csv`.
Downloads are in `data/` (see `data/README.md`). Scripts are in `scripts/` and were run with `python3 -I`.
Outputs are in `tables/`. Tags: [F] full text read or data downloaded, [A] abstract only, [S] secondary (citing
source named), [D] computed here. "Judgement" marks a value that I chose and that no source gives.

Reused, not redone: cod and haddock r (`../../Torskfiskarnas reglering/r_pool_present.md`), cod and haddock F
(`../../../reports/Fiskeridödlighet i Skagerrak idag.md`, `../../Fiskeridödlighet i Skagerrak idag/catch_by_gear.md`),
and M0 = 0.20 with the non-FG predation split (`../../../reports/Restdödlighet M0 hos torskfiskar.md`).

## 1. Bottom line

All rates are per year. r_prior is the FishBase CMSY prior. r_now is today's low-density productivity, with the
Euler-Lotka adjustment where SAG allows it. M total is the total natural mortality of the species (all predators
plus M0). "Non-model pred." is predation by predators that are not model groups (class c below), which belongs in
M1 next to M0. F is catch (landings + dead discards) / mean total biomass.

| Species (share) | r_prior | r_now (range) | M total | M0 | Non-model pred. (range) | F (range) | Gear map | Main sources |
|---|---|---|---|---|---|---|---|---|
| haddock (38.6 %) | 0.50 | 0.24 (0.15-0.33) | 0.59 (SMS ages 1+) | 0.20 | 0.11 (0.05-0.16) | 0.06 (0.045-0.075) | bottom ~98 %, pelagic 0-1.7 % | earlier notes; SMS 2023 [D]; SAG IBC [F] |
| cod (31.4 %) | 0.51 | 0.35 (0.26-0.40) | 0.70 (ages 1+); 0.59 (ages 2+) | 0.20 | 0.11 (0.05-0.25) | 0.39 (0.35-0.50) | bottom 100 % | earlier notes; SMS 2023 [D] |
| long rough dab (12.6 %) | 0.19 | 0.30 (0.19-0.45), judgement | 0.25 (0.20-0.32) | 0.20 | 0.03 (0.01-0.06) | 0.10 (0.05-0.20), judgement | bottom 100 % | FishBase [F]; Then/Hamel-Cope [D] |
| dab (4.5 %) | 0.68 | 0.55 (0.45-0.68) | 0.35 (ICES) | 0.20 | 0.05 (0.02-0.08) | 0.18 (0.10-0.26) | bottom 100 % | dab stock annex 2025 [F]; SAG 19480 [F] |
| plaice (3.8 %) | 0.50 | 0.40 (0.28-0.50); d = 1 | 0.35 (SMS M1+M2, ages 1+) | 0.20 | 0.05 (0.02-0.08) | 0.04 (0.03-0.08) | bottom 100 % | SAG 22489 [F]; SMS 2023 [D] |
| witch (1.7 %) | 0.32 | 0.28 (0.25-0.42); d = 1.4 | 0.20 (ICES); 0.26 (Then) | 0.17 | 0.02 (0.01-0.04) | 0.12 (0.08-0.18) | bottom 100 % | SAG 22534 [F]; witch annex 2025 [F] |
| starry ray (1.5 %) | 0.09 | 0.15 (0.09-0.25), judgement | 0.23 (Then, tmax 28) | 0.15 | 0.03 (0.01-0.06) | 0.03 (0.02-0.05) | bottom 100 % | ICES advice 2023 [F]; SMS biomass [D] |
| hagfish (1.3 %) | 0.10 | 0.10 (0.07-0.15), judgement | 0.15, judgement | 0.13 | 0.02 (0-0.04) | 0.01 (0-0.03), judgement | bottom 100 % | FishBase [F] |
| grey gurnard (0.9 %) | 0.74 | 0.65 (0.50-0.74) | 0.30 (Then, tmax 21) | 0.20 | 0.05 (0.02-0.08) | 0.03 (0.02-0.05) | bottom 100 % | SAG 22487 [F]; SMS biomass [D] |
| lemon sole (0.9 %) | 0.71 | 0.50 (0.42-0.72) | 0.28 (Then, tmax 23) | 0.20 | 0.03 (0.01-0.05) | 0.09 (0.06-0.13) | bottom 100 % | SAG 22413 [F]; Froese et al. 2016 [F] |
| greater weever (0.8 %) | none | 0.50 (0.30-0.80), judgement | 0.40, judgement | 0.20 | 0.05 (0.02-0.08) | 0.02 (0-0.05), judgement | bottom 100 % | FishBase resilience [F] |
| others (2.5 %) | | 0.40, judgement | 0.30 | 0.20 | 0.05 | 0.10 | bottom | judgement |

**Group values (biomass-weighted, long-run 2001-25 shares; `scripts/group_weights.py`, `tables/group_weights_stdout.txt`) [D]:**

| Quantity | Central | Range | Per tick (6 h) | Note |
|---|---|---|---|---|
| r_prior (FishBase) | 0.46 | | | the old library style |
| **r_now** | **0.31** | 0.22-0.41 | | cod 0.35 and haddock 0.24 carry 70 % of it |
| **M0** | **0.20** | 0.15-0.30 | 1.36e-4 | SMS convention for every bony fish; hagfish and ray slightly lower |
| **Non-model predation** | **0.09** | 0.04-0.16 | 6.0e-5 | squid, halibut, megrim, grey seal and others; within-group and model-group predators removed |
| **M1 = M0 + non-model** | **0.29** | 0.19-0.46 | 2.0e-4 | |
| M total (all causes, for reference) | 0.53 | | | the rest is M2 by model groups (pairs) and within-group predation |
| **F** | **0.17** | 0.14-0.24 | 1.19e-4 | cod alone gives 0.12 of the 0.17 |
| F on the pelagic map | 0.0004 | 0-0.001 | | 0.2 % of the group F (haddock industrial bycatch); the rest is on the bottom-trawl map |

Weights used: haddock 0.386, cod 0.314, long rough dab 0.126, dab 0.045, plaice 0.038, witch 0.017, starry ray 0.015,
hagfish 0.013, grey gurnard 0.009, lemon sole 0.009, greater weever 0.008, others 0.025 (sum 1.005, normalised).

**Weight sensitivity [D].** With the 2021-25 shares (haddock 74 %, cod 8.5 %), F falls to about 0.09 and r_now to
about 0.26. The group F is really a statement about how much cod the group holds. This is the largest single
uncertainty in the group F.

**Reading for K = B*/(1 - F/r) (judgement).** With r_now 0.31 and F 0.17, 1 - F/r = 0.44 and K = 1.53/0.44 =
3.5 t/km2. The range is wide: with F 0.14 and r 0.41, K = 2.3; with F 0.24 and r 0.22, F > r and no K exists. The
pooled number hides that cod alone has F (0.39) above r_now (0.35); the Viking cod substock is indeed declining.
The flatfish have F/r of 0.1-0.35 and should sit near their own carrying capacity (section 6).

## 2. r_max per species

### 2.1 FishBase CMSY priors [F]
Fetched from the species summary pages (`data/fishbase/`, fishbase.se, 2026-10-24). Verbatim "Prior r" lines:

| Species | FishBase text | Resilience text |
|---|---|---|
| Pleuronectes platessa | "Prior r = 0.50, 95% CL = 0.33 - 0.75, Based on 28 full stock assessments." | "Medium ... (K=0.06-0.34; tm=2-6; tmax=30; Fec=50,000)" |
| Limanda limanda | "Prior r = 0.68, 95% CL = 0.45 - 1.02, Based on 3 full stock assessments." | "Low ... (K=0.3-0.6; tm=2-3; tmax=12; Fec=50,000)" |
| Hippoglossoides platessoides | "Prior r = 0.19, 95% CL = 0.12 - 0.28, Based on 2 full stock assessments." | "Low ... (rm=0.43; K=0.06-0.12; tm=2-11; tmax=30; Fec=50,000)" |
| Glyptocephalus cynoglossus | "Prior r = 0.32, 95% CL = 0.21 - 0.48, Based on 8 full stock assessments." | "Medium ... (K=0.15-0.2; tmax=25)" |
| Microstomus kitt | "Prior r = 0.71, 95% CL = 0.47 - 1.07, Based on 1 full stock assessment." | "Low ... (K=0.08-0.42; tm=3-6; tmax=23; Fec=19,680)" |
| Eutrigla gurnardus | "Prior r = 0.74, 95% CL = 0.49 - 1.10, Based on 1 data-limited stock assessment." | "Low ... (K=0.16; tm=3-4; tmax=21; Fec=200,000)" |
| Amblyraja radiata | "Prior r = 0.09, 95% CL = 0.06 - 0.13, Based on 1 full stock assessment." | "Low ... (K=0.17; tm=4; Fec=13-20)" |
| Myxine glutinosa | "Prior r = 0.10, 95% CL = 0.07 - 0.15, Based on 1 full stock assessment." | "Low ... (Fec= 20-30)" |
| Trachinus draco | no prior | "Medium, minimum population doubling time 1.4 - 4.4 years (K=0.08-0.5; Assuming tm=2-4)" |
| Gadus morhua / Melanogrammus aeglefinus | 0.51 (66 assessments) / 0.50 (41) | (reused) |

Notes:
- The priors pool assessments worldwide. For long rough dab, starry ray and hagfish the one or two assessments are
  almost certainly NW Atlantic stocks (American plaice, thorny skate), which are colder and slower. That is a
  judgement from the species ranges, not stated on the pages.
- Dab, lemon sole and grey gurnard carry "Low" resilience text but high assessment-based priors (0.68-0.74). The
  prior rests on 1-3 assessments.

### 2.2 Life-history priors and Schaefer fits, Froese et al. 2016 [F]
Froese R, Garilao C, Winker H, et al. 2016, "Exploitation and status of European stocks", report for Oceana
(`data/froese2016/StockStatusReport_newVersion.pdf`, oceanrep.geomar.de/34476).
- Table 2, life-history r priors (r, 2 SD range): plaice 0.39 (0.20-0.77), dab 0.49 (0.24-0.98), lemon sole 0.42
  (0.20-0.86), witch 0.33 (0.18-0.62), red gurnard (Chelidonichthys cuculus) 0.65 (0.28-1.51). No grey gurnard,
  long rough dab, starry ray, hagfish or weever.
- Table 7, Greater North Sea BSM/CMSY results (Fmsy, F, B/Bmsy) and r = 2 Fmsy [D]:

  | Stock | Last year | Fmsy | F | B/Bmsy | r = 2 Fmsy |
  |---|---|---|---|---|---|
  | dab-nsea | 2014 | 0.29 | 0.08 | 1.68 | 0.58 |
  | lem-nsea | 2014 | 0.36 | 0.24 | 0.81 | 0.72 |
  | ple-nsea | 2015 | 0.14 | 0.05 | 1.74 | 0.28 |
  | wit-nsea | 2014 | 0.24 | 0.19 | 1.07 | 0.48 |
  | ple-2123 (Kattegat) | 2015 | 0.38 | 0.17 | 1.29 | 0.76 |

- The ple-nsea fit is on a stock far above Bmsy. r is then poorly identified (judgement). The dab F 0.08 uses
  landings only (4955 t against 23 kt total catch today), so it understates F.
- Myers et al. 1999 maximum reproductive rates for these flatfish were not retrieved (gap).

### 2.3 Euler-Lotka present-day adjustment from SAG [F data, D arithmetic]
Method as in `r_pool_present.md`: r_now = r_prior (1 - ln(d)/3), d = early / recent median recruits per SSB.
Script `scripts/sag_metrics.py`, output `tables/sag_metrics_stdout.txt`. R is lagged by the recruitment age.

**Plaice, ple.27.420 (North Sea + Skagerrak 3a.20; SAG key 22489, 2026).** The Skagerrak belongs to this stock.

| SSB years | Median R/SSB (thousands per t) | Median R age 1 (millions) | Median SSB (kt) |
|---|---|---|---|
| 1957-89 | 3.52 | 2554 | 677 |
| 1990-99 | 5.17 | 2306 | 427 |
| 2000-09 | 5.51 | 3784 | 542 |
| 2010-19 | 4.06 | 4277 | 1135 |
| 2015-24 | 5.47 | 4925 | 885 |
| 2020-25 | 5.93 | 5538 | 870 |

- Recruits per spawner dipped in 2010-19 (4.06 against 5.2-5.5 in 1990-2009) while SSB doubled to a record
  1.1-1.3 Mt. Absolute recruitment in the same years was the highest in the series and rose further after 2019.
- **Reading: the dip is density dependence, not a productivity loss.** Under a Beverton-Holt curve, doubling SSB
  near the plateau halves R/SSB and leaves R almost unchanged. Here R even rose. Against the long baseline,
  R/SSB is higher today (5.5-5.9 against 3.5). I set **d = 1**: r_now = r_prior. I do not raise r above the prior,
  because the higher R/SSB may partly be an artefact of the 2015 stock-area change (3a.20 added) and of the
  back-calculated early series (judgement).
- r_now plaice = 0.40 (0.28-0.50): mid-point of FishBase 0.50, life history 0.39 and the Schaefer 0.28 (judgement).

**Plaice, Kattegat-Belts-Baltic, ple.27.21-32 (key 22470).** Not the box stock. R/SSB rose from 20 (2000-19) to 105
(2020-25) and SSB rose ninefold. Productivity rose sharply; no adjustment needed.

**Witch, wit.27.3a47d (key 22534).** Median R/SSB 11.4 (1950-89), 10.8 (1990s), 10.7 (2000s), 10.0 (2010s), 8.0
(2015-24 and 2020-25). SSB is similar in both periods (2.8 against 3.3-3.5 kt), so the decline is not compensation.
d = 11.4 / 8.0 = 1.42, r_now = 0.32 × (1 - ln 1.42 / 3) = **0.28**. With the Schaefer 0.48 it is 0.42. Range 0.25-0.42.

**Lemon sole, lem.27.3a47d (key 22413).** Category 3, no recruitment series. The stock-size index (relative)
fell from 1.37 (2016) to 0.79 (2024-25) while the harvest rate fell by about 60 %. A falling index at falling F
suggests lower productivity (judgement). I take the life-history prior 0.42 as the low end and 0.50 as central.

**Dab, dab.27.3a4 (key 19480).** Category 3. The "Recruitment" field holds a small integer index (1-4); no R/SSB.
The SURBAR biomass index rose 2003-2014 (dab stock annex 2025). r_now = 0.55 (0.45-0.68), judgement between the
life-history 0.49, the Schaefer 0.58 and FishBase 0.68.

**Grey gurnard, gug.27.3a47d (key 22487).** Index only. The survey index fell from 27-30 kg/h (2015-17) to
11-13 kg/h (2024-26) while F was low (section 4). The species rose strongly in the 1990s-2000s (not checked here).
r_now 0.65 (0.50-0.74), judgement.

**Starry ray.** No SAG entry. The ICES survey index "has been in continuous decline since 1990, especially since
2002" (ICES Advice 2023, rjr.27.23a4, [F]). r_now 0.15 (0.09-0.25) is a judgement: FishBase 0.09 is a NW Atlantic
thorny skate; the North Sea fish is smaller (Linf 70.9 cm, k 0.1 in the ICES assessment) and should be somewhat faster.

**Long rough dab.** No ICES assessment in SAG 2024-26 (searched by label and species name). r_now 0.30 (0.19-0.45)
is a judgement. FishBase gives 0.19 from two cold-water assessments and "rm=0.43" in the same resilience line.
Stratoudakis et al. 1997 found North Sea long rough dab larger and longer-lived than west of Scotland fish
(MEPS 158:303-306, doi 10.3354/meps158303) [A].

**Hagfish and weever.** Judgement only: hagfish 0.10 (FishBase), weever 0.50 (FishBase "Medium" resilience implies
the 0.2-0.8 default range of Froese et al. 2016 Table 2).

**Cod 0.35 and haddock 0.24** are reused from `r_pool_present.md` (d 2.6 and 4.7).

## 3. Natural mortality

### 3.1 ICES and life-history M [F, D]
- Plaice ple.27.420 stock annex: "Natural mortality is assumed to be 0.1 for all age groups and constant over time.
  These values are probably derived from war-time estimates (Beverton and Holt, 1957; Beverton 1964)." [F]
- Witch stock annex 2025: "The assessment currently uses a constant natural mortality rate of 0.2 for all ages and
  years." [F]
- Dab stock annex 2025: "A natural mortality for dab of M=0.35 was estimated following the method of Then et al.
  (2014)." [F]
- Lemon sole stock annex 2024: "Natural mortality (M) estimates for lemon sole are not available." [F]
- Life-history estimators [D], with FishBase tmax: Then et al. 2015 M = 4.899 tmax^-0.916; Hamel & Cope 2022
  M = 5.4/tmax (formulas as used in `../../Restdödlighet M0 hos torskfiskar/empirical_estimators.md`).

  | tmax (yr) | Species | Then | Hamel & Cope |
  |---|---|---|---|
  | 12 | dab | 0.50 | 0.45 |
  | 20-30 | long rough dab (FishBase 30; North Sea possibly shorter) | 0.32-0.22 | 0.27-0.18 |
  | 21 | grey gurnard | 0.30 | 0.26 |
  | 23 | lemon sole | 0.28 | 0.24 |
  | 25 | witch | 0.26 | 0.22 |
  | 28 | starry ray | 0.23 | 0.19 |

- These are total M (predation + everything else). They are inputs to M total in the table, not to M0.

### 3.2 SMS key run 2023, age-pooled and biomass-weighted [D]
Files from `https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/` (`data/sms2023/`).
Script `scripts/sms_pool.py`: for each species and year, sum over quarters and ages of rate x mean biomass
(N.bar x west), divided by the annual mean biomass; mean of 2018-2022. That is "removal / mean biomass", the same
definition as the model's predation_mortality. Output `tables/sms_pool_2018_2022.txt`.

| Prey (ages) | Mean B (kt) | M1 | M2 | F | M2 by predator |
|---|---|---|---|---|---|
| Cod (1+) | 85 | 0.291 | 0.405 | 0.319 | harbour porpoise 0.189, grey seal 0.146, cod 0.035, gannet 0.030, whiting 0.005 |
| Cod (2+) | 71 | 0.309 | 0.284 | 0.369 | porpoise 0.135, grey seal 0.127, cod 0.021 |
| Haddock (1+) | 251 | 0.200 | 0.386 | 0.221 | saithe 0.184, cod 0.084, whiting 0.069, grey seal 0.049 |
| Haddock (2+) | 177 | 0.200 | 0.183 | 0.304 | saithe 0.072, cod 0.062, grey seal 0.048 |
| Plaice (1+) | 490 | 0.336 | 0.013 | 0.135 | grey seal 0.013 |
| (Age 0 included, cod / haddock) | | | 0.506 / 0.562 | | adds grey gurnard 0.047 / 0.006, whiting, birds |

- Cod M1 0.29 contains the ad hoc 0.353 for ages 3+ since 2011 (emigration to west of Scotland, see the M0 report).
  Its biological part is the 0.20 convention.
- Plaice M1 in SMS is age-varying, 0.49/yr at age 1 down to 0.25 at age 10 (quarterly 0.12 to 0.06, read from
  `summary.out`). The SMS stock annex says plaice M1 already contains seal predation (quoted in the M0 report).
  The explicit grey-seal M2 of 0.013 is small. Plaice M total = 0.35 (ages 1+).
- Biomass in the box: cod is 58 % above 50 cm and only 9 % below 30 cm; haddock 78 % below 40 cm
  (`tables/stage_shares_longrun.csv`, a sibling script `scripts/length_structure.py`) [D]. The box cod is an adult
  stock, so the cod 2+ row is the better guide. The haddock 1+ row fits the haddock.
- SMS is North-Sea-wide. Its grey seals are mostly at the UK coast. Grey seals are rare in the Skagerrak and Kattegat
  (stable at about 25 animals since the 1970s, Härkönen et al. 2007 [S, `../../Bärförmåga för säl och tumlare/seals.md`];
  "remain uncommon", Carroll et al. 2025 [S, same note]). In the box I put grey-seal predation at about 0.005
  (0-0.02), judgement; the raw SMS value (0.13-0.15 on cod, 0.05 on haddock) is the upper end of the cod range.

### 3.3 Classification of predators [D, judgement where marked]

| Class | Predators | Where it goes |
|---|---|---|
| (a) inside benthivorous_fish | cod, grey gurnard, starry ray, long rough dab, haddock | neither M1 nor the budget (within-group, section 160) |
| (b) model groups | predatory_fish (saithe, whiting, hake, anglerfish/monkfish, mackerel, horse mackerel, ling); planktivorous_fish (none of note on this group); seals (harbour seal only); porpoises; seabirds (gannet, guillemot, gulls) | pairs, M2 on the prey |
| (c) not in the model | grey seal, halibut, squid, megrim, other large demersals, sharks other than spurdog | M1 |

SMS M2 on cod and haddock, sorted by class (ages 1+, North Sea scale) [D]:

| | Cod | Haddock | Cod + haddock, weighted 0.45/0.55 |
|---|---|---|---|
| (a) within: cod | 0.035 | 0.084 | 0.062 |
| (b) predatory_fish: saithe + whiting | 0.005 | 0.253 | 0.142 |
| (b) porpoises | 0.189 | 0 | 0.085 |
| (b) seabirds (gannet) | 0.030 | ~0 | 0.013 |
| (c) grey seal, raw SMS / box judgement | 0.146 / ~0.005 | 0.049 / ~0.005 | 0.093 / ~0.005 |

- Seabirds are a model group but muted in `mareld2.yaml`. Muted groups are absent from the budget (AGENTS.md). Their
  part (~0.01 on the group) is small. I suggest leaving it out of M1 while muted and flagging it for the
  seabird -> benthivore pair if the group is switched on (judgement).
- SMS has no squid, halibut or megrim. Their predation hides inside SMS M1 = 0.20. The M0 report therefore used the
  Ecopath North Sea 1991 diet matrix for them. Its re-split of the old gadoid pool's 0.214 of non-FG predation was:
  squid 0.036, halibut 0.030, gurnards 0.029, long rough dab 0.029, megrim 0.019, rays 0.03, monkfish 0.013,
  others ~0.028 (`reports/Restdödlighet M0 hos torskfiskar.md`, the section on the benthos artefact) [S].
  Re-classified for the new group [D]:
  - gurnards + long rough dab + rays = 0.088 move to (a) and leave M1;
  - monkfish 0.013 moves to (b) (predatory_fish pair);
  - squid + halibut + megrim + others = **0.113** stay in (c).
  This was computed on the old gadoid pool (with whiting and saithe juveniles). For cod and haddock I use
  **0.11 (0.05-0.16)**. Grey seal adds ~0.005 in the box. The cod upper end 0.25 adds the raw SMS grey-seal value.
- Flatfish, rays, hagfish, weever (judgement). Their class (c) predators are halibut, grey seal, large crabs on
  small fish, and some squid. Juvenile flatfish predation is mostly by cod, gurnards, starry ray, whiting and
  seabirds, which are class (a) or (b). I put class (c) at 0.02-0.05 per species.
- The (b) M2 on the flatfish (harbour seal, porpoise, predatory_fish on dab, long rough dab and plaice) is the
  remainder M total - M0 - (c) - (a). It is not quantified here; the diet evidence file (C) holds the consumption.

### 3.4 M0 for the age-pooled group
- **M0 = 0.20 (0.15-0.30).** I keep the SMS convention used for cod, haddock, whiting and saithe for every bony fish.
  The reasons are those of the M0 report: it is the ICES default for everything except modelled predators and
  fishing; the adult Ecopath M0 is 0.18; corrected juvenile groups give 0.14. For hagfish (0.13), starry ray
  (0.15) and witch (0.17) I take a little less, because their total M (0.15, 0.23, and the ICES 0.20 for witch)
  cannot hold 0.20 plus their predation (judgement). Biomass-weighted: 0.198.
- **Juveniles in the pool.** The group holds GOV-sampled fish of all sizes: 62 % of the long rough dab biomass is
  below 20 cm, 44 % of the dab below 20 cm, 78 % of the haddock below 40 cm. Juvenile predation is mostly by
  model groups (cod, saithe, whiting, gurnards, porpoise, birds). Juvenile M0 is not higher than adult M0 once the
  Ecopath balancing residuals are removed (M0 report). So the juveniles raise M2 through the pairs, not M0.
- **Non-model predation for the group: 0.09 (0.04-0.16).** It is 0.11 on cod and haddock and 0.02-0.05 on the
  rest. It is mostly juvenile predation by squid and adult predation by halibut and megrim (Ecopath 1991 balance),
  so it carries the low-moderate confidence of that balance.
- **M1 = 0.29 (0.19-0.46) /yr = 2.0e-4 per tick.** The old gadoid M1 was 0.41. The difference is the re-split:
  within-group predators (gurnards, long rough dab, rays) and monkfish leave M1.

## 4. Fishing mortality today

### 4.1 Per species
- **Cod 0.39 (0.35-0.50); haddock 0.06 (0.045-0.075).** Reused from the F report (biomass-weighted F). Gear: in the
  local data, haddock 74 % bottom trawl, 25 % seines, 0 % pelagic; cod 75 % bottom trawl, 9 % seines, 16 % static
  gear, 0 % pelagic (`catch_by_gear.md`) [S]. Haddock industrial bycatch (IBC) in SAG 22417: 1201-2036 t in
  2023-25 against 67-70 kt catch, 1.7-3.0 % [F data, D]. Most of it came from the Danish sprat fishery in the
  North Sea (HAWG 2026, via `catch_by_gear.md`) [S]; in 3a.20 it is about 0. I give haddock 0-1.7 % on the pelagic
  map. The SMS pooled F (cod 0.32, haddock 0.22 for 2018-22, North Sea) is older and higher; the haddock F has
  fallen since (Fbar 0.06-0.08 in 2023-25).
- **Plaice 0.04 (0.03-0.08).** SAG 22489 gives total biomass. Catch / TB [D]: 2015-19 0.082-0.101, 2020-21
  0.065-0.076, 2022-23 0.045-0.046, 2024 0.030, 2025 0.036. Mean 2023-25: 0.037. ICES Fbar(2-6) 2025 = 0.043,
  FMSY 0.123. The catch counts all discards as dead. Discards are 51-68 % of the catch in 2021-25. The stock annex:
  "it is assumed that the majority of discarded animals do not survive"; in beam trawls "less than 10% of the plaice
  and sole discards ... survive"; pulse trawl 15 % [F]. Survival would lower F by up to ~10 %. The upper end 0.08
  allows for the box's 1.5-2x higher bottom-trawl effort (F report) (judgement). Skagerrak plaice is in this stock
  since 2015; the Kattegat stock ple.27.21-32 has catch/TB 0.005-0.017 in 2022-25 (landings + dead discards) [D].
  Gear: seines, otter trawls, gillnets; bottom map 100 %.
- **Witch 0.12 (0.08-0.18).** SAG 22534: Fbar(4-8) 0.18 (2024), 0.15 (2025); FMSY 0.163. No total biomass is
  published. Catch/SSB is 0.36 in 2025, which overstates catch/TB because SSB excludes the many immature fish.
  Froese et al. 2016 gave F 0.19 when ICES Fbar was 0.31 (2014); that ratio (0.61) applied to 0.15 gives 0.09 [D].
  Central 0.12, judgement between 0.09 and Fbar 0.15. Discard rate 6-8 % in 2023-25. The witch annex: Danish
  discards "are reported in mixed Nephrops (OTB_CRU_90-119_0_0_all)", Swedish discards "mainly in Nephrops fishery"
  in 3.a [F]. Bottom map 100 %.
- **Lemon sole 0.09 (0.06-0.13).** SAG 22413 (category 3): harvest rate (catch / index) 1830 in 2024 and 1651 in
  2025 against the HR_msy proxy 4019, so HR/HRmsy = 0.41-0.46 [F data, D]. The chr-rule proxy is set where the
  catch length indicator says F ≈ M. With M 0.24-0.28, F ≈ 0.10-0.13 (judgement on the proxy). Independently,
  Froese et al. 2016 gave F 0.24 for 2014; the harvest rate fell from 4309 (2014) to 1651 (2025), so F ≈ 0.24 x 0.38
  = 0.09 [D]. Catch 1.3 kt in 2025, 18 % discards. Bottom map 100 %.
- **Dab 0.18 (0.10-0.26).** Dab stock annex 2025 and SAG 19480: total catch 23.2 kt in 2024, of which 21.5 kt (93 %)
  discards; M = 0.35; SURBAR total mortality "mean z" 0.53 (0.45-0.61) in 2024; L_F=M / L_mean 0.95 [F]. F = Z - M
  = 0.18, range from the z interval [D]. This is independent of discard survival because z comes from the survey.
  Heath & Cook 2015 (PLoS ONE 10:e0117078, PMC4361349) [F]: "Common dab and gurnards stood out as having high overall
  discard rates (>0.6)". Bottom map 100 % (otter trawls, beam trawls, Nephrops and shrimp trawls).
- **Long rough dab 0.10 (0.05-0.20), judgement.** No ICES assessment, no catch series retrieved. It has no market
  and is discarded; its F is the discard mortality of the Nephrops, Pandalus and mixed otter trawls in the
  Skagerrak. I set it a little below dab (an all-discard flatfish with F 0.18) because the Skagerrak shrimp
  trawls must use a sorting grid with 19 mm bar spacing that sends larger fish out (F report, `local_fishery.md`),
  and because 62 % of the box biomass is below 20 cm, near or below trawl retention. Grid studies in other areas
  show long rough dab escape through grids (search summary only, not read). Discard survival of long rough dab
  was not found. Bottom map 100 %.
- **Grey gurnard 0.03 (0.02-0.05).** SAG 22487: catch 9.6-9.9 kt in 2022-23, 3.3-3.8 kt in 2024-25, 78-92 % discards
  [F]. SMS other-predator input: North Sea grey gurnard biomass 162-241 kt in 2018-22 (N x west from
  `other_pred_N.in` and `west.in`, assuming N in thousands and weight in kg; `scripts/sms_otherpred_biomass.py`,
  `tables/sms_otherpred_biomass.txt`) [D]. Catch / B: 0.02 (2024-25) to 0.05 (2022-23), all discards counted dead.
  Bottom map 100 %.
- **Starry ray 0.03 (0.02-0.05).** ICES Advice 2023 [F]: on the EU prohibited list since 2015; landings 0-3 t;
  discards 393-1115 t in 2018-22 (mean ~780 t); "Discard rate 100%"; fishing pressure indicator above the FMSY
  proxy (Lmean/LF=M 0.95), which "reflects a high total mortality (as it is estimated from discard data). The
  contribution of fishing and other factors to the total mortality is unknown." SMS biomass of starry ray 2018-22:
  12-29 kt, mean 22 kt [D]. Catch / B = 0.035 if all discards die. Skate discard survival is often substantial
  (not checked here), so 0.03 central, judgement. Bottom map 100 %.
- **Hagfish 0.01 (0-0.03), weever 0.02 (0-0.05).** Judgement. No directed fishery in the area; bycatch only.

### 4.2 Group F and gear split [D]
- **F = 0.17 (0.14-0.24)** with the long-run weights. Contributions: cod 0.122, haddock 0.023, long rough dab 0.013,
  dab 0.008, others 0.01 together.
- **Pelagic map: 0.2 % of the group F** (haddock IBC, upper bound 1.7 % of haddock F). Bottom-trawl map: 99.8 %.
  No flatfish, ray, hagfish or weever catch in pelagic gear was found. In this group, seines and gillnets go to the
  bottom-trawl map as in the F report's method.
- Cod, F 0.39, is an ICES Viking-substock value. The F report placed Skagerrak cod there. With 2021-25 weights the
  group F is ~0.09.

## 5. Density dependence in the area's flatfish (for K = B*/(1 - F/r))

Flatfish in the North Sea show compensation in both recruitment and growth. Plaice recruitment is set in the
coastal nurseries. Van der Veer et al. 2000 (ICES J. Mar. Sci. 57:202-215, doi 10.1006/jmsc.1999.0523) [A, via
search summary] state that recruitment "appears to be related to the quantity of juvenile nursery habitats,
suggesting that either larval supply or the carrying capacity of the nurseries is the limiting factor". The SAG data
fit that picture: North Sea plaice SSB doubled after 2008, R/SSB fell in 2010-19 and absolute R still rose to the
highest in the series (section 2.3). Growth is density dependent too. Van Kooten et al. 2023 (Wageningen Marine
Research report C063/23, doi 10.18174/640016) [F] find "clearly" a numbers-related growth rate for plaice of 17-27 cm
and not for other sizes. They cite the early finding of density-dependent growth up to ~25 cm (Rijnsdorp & van
Leeuwen 1992, 1996), the null result of Lorenzen & Enberg 2001, and two analyses of the post-2005 rebuilding that
"find clear density-dependent individual growth" (Zimmerman et al. 2018; Rindorf et al. 2022) [S, all via van Kooten
et al.]. The plaice stock annex adds "a long-term decline in weight-at-age for the older ages" [F]. For the model this
means (judgement): the flatfish members are near their own carrying capacity today. Plaice F/r is ~0.1, so
B*/K ≈ 0.9, and plaice SSB is at a record high at a record-low F. Dab and long rough dab have F/r of ~0.3. The
compensation acts on recruits and small fish, which matches a logistic K on the pooled biomass in kind but not in
the age it acts on. The group's 1 - F/r is set mostly by cod. The flatfish argue for a K close to today's B for
their part of the group. No density-dependence study for dab, long rough dab or witch in the Skagerrak was found.

## 6. Gaps
- Long rough dab: no assessment, no Skagerrak discard tonnage, no North Sea M, growth or discard survival. Its r, M
  and F are judgements and carry 12.6 % of the group.
- Grey seals in the box: no recent counts. The box value 0.005 is a judgement against the SMS North-Sea value.
- The class (c) predation on cod and haddock (0.11) comes from the Ecopath 1991 balance of the old gadoid pool, not
  from data on the box.
- The class (b) M2 on the flatfish (harbour seals, porpoise, predatory_fish) is not quantified here.
- Witch has no total biomass in SAG; its F is between two indirect estimates.
- SMS units for the other-predator biomass (N thousands, weight kg) were inferred, not read in the annex. The
  gurnard and ray F depend on them.
- Starry ray and other skate discard survival was not retrieved.
- Myers et al. 1999 maximum reproductive rates for the flatfish were not retrieved.
- Unpaywall was not used.
