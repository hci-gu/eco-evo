# Benthivores on benthic prey: food-limited or time-limited? Evidence trail

Research subagent, 2026-11-07. Research only (no code, library, manifest, resume or AGENTS.md edit).
Tags: [F] stated in a source read in full (or its abstract where marked [A]), [D] derived here
(script and stdout named), [J] judgement. Units: B t/km2 wet, box-mean convention; a km2 per t
predator per 6 h tick; h ticks; x = a h B_vis; X = sum over the searched menu (shared handling,
section 242).

## 0. What was already done (not redone)

Three earlier notes; the third reconciles the first two and the brief did not name it.
- "Exponering av bottenbyten" (223.14): mechanistic e 0.165 (infauna), 0.43 (epifauna); a 0.036 /
  0.15; pairs saturated (x 19-66) so field feeding levels cannot come from a.
- "Födonivå och mättnad i fält" (223.12): inverted the field ration (Q/B 4.46, F 0.40 of the
  ceiling) into a with tau = 1 -> infauna a ~1e-4 to 2e-4 (`tables/field_implied_a_stdout.txt`),
  e ~0.002-0.01; named Hiddink 2016 as the test.
- "Mättnad hos bottenbyten" (after 223.27): both right about different quantities; no field FR in
  density units for adult gadoids/flatfish on benthos; Persson 2012 lab a on Crangon 0.065-0.13;
  Hvingel & Kingsley 2006 stock-scale cod->Pandalus on the lower limb; Smith 2013 / Berenboim 2000
  diets track prey while total intake is flat; the missing step was a SHARED search and gut over
  the bed prey (now in the engine, section 242). Kattegat and Irish Sea Dryad reanalyses
  (`tables/kattegat_station_analysis_stdout.txt`, `field_density_x_stdout.txt` there).
- "Täthetsberoende och interferens hos torskfiskar": North Sea cod, haddock, Norway pout show no
  significant density-dependent growth in Lorenzen & Enberg 2002 [F there]; Rijnsdorp et al. 1991
  (second-hand): none for plaice, sole, cod.

New here: (1) the tau logic made explicit; (2) the Irish Sea gradient run as a quantitative test of
both candidate a's with a detectability bound; (3) Kattegat re-run on Hiddink's own "fish food"
class; (4) grab-scale and km-scale patchiness; (5) the model's shared-handling elasticities; (6) a
few new sources (Hiddink slides, Casini 2016, abstracts of Hiddink 2011/2016, Rijnsdorp &
Vingerhoed 2001, Rijnsdorp & van Leeuwen 1996, Shephard 2014, Neuenfeldt 2020, Haase 2020).

## 1. The tau point (why the feeding-level inversion assumed its answer)

Intake per tick = tau x [T/h] x X/(1 + X) for a one-prey-class menu [D, the engine's form].
A field ration F (share of Cmax) fixes only the product tau X/(1 + X). With tau = 1, F 0.40 gives
X 0.67, and a follows; with tau 0.5 the same F gives X 4.1, and a is 4-30x higher
(`Födonivå.../tables/field_implied_a_stdout.txt` already lists this). So the field feeding level
carries no information on a unless tau is known [J]. Only a prey-density GRADIENT with intake data
separates the two: food-limited (X ~ 1) predicts an intake elasticity 1/(1 + X) ~ 0.5-0.8 to prey;
time-limited (X >> 1) predicts ~0. A lower bound on X read from a flat response bounds the
foraging-time a from below for any tau (a_day = a_model x tau, and tau <= 1) [D].

## 2. Irish Sea gradient as the test (scripts/pair_K_bounds.py, tables/pair_K_bounds_stdout.txt)

Source facts, Johnson et al. 2015 (Proc R Soc B; [F], text in the Födonivå note's data):
- 15 stations on one mud ground, trawling 2.7-11.9 passes/yr (11 with public station data).
- Biomass of the plaice's common prey fell with trawling (R2 0.51, F1,13 = 13.09, p < 0.001) [F].
- Plaice stomach fullness ~ trawling F1,13 = 0.004, p 0.949; stomach energy per g fish F < 0.001,
  p 0.981; total prey biomass in stomachs p 0.468; dab fullness p 0.759 [F].
- "the ratio of prey biomass to fish biomass in the environment did not change significantly"
  (plaice p 0.648) [F]; fish declined with the prey.
- Plaice mean stomach energy 3.22 +- 0.54 J/g (s.e.), dab 19.18 +- 2.57 [F].
Station prey biomass (grab, large infauna excluded; Hinz et al. data, earlier note) 2.4-18.5,
geometric mean 8.2 t/km2; slope ln B on trawling -0.128 per pass [D].

Prediction per candidate a (benthivores -> infauna, h 112, one-prey menu: plaice is an infauna
specialist [F]; reachable share 0.2 as in the model, or 1) [D]:

| candidate a | share | x at 8.2 t/km2 | I(2.4)/I(18.5) | predicted slope ln I per pass |
|---|---|---|---|---|
| 0.036 (model, Exposure note) | 0.2 | 6.6 | 0.70 | -0.019 |
| 0.016 / 0.092 (Exposure 5-95 %) | 0.2 | 3.0 / 17 | 0.53 / 0.85 | -0.036 / -0.009 |
| 0.0025 (Feeding-level e 0.01) | 0.2 | 0.46 | 0.23 | -0.088 |
| 0.0005 (Feeding-level e 0.002) | 0.2 | 0.09 | 0.15 | -0.117 |
| 0.00015 (Feeding-level tau=1 inversion) | 0.2 | 0.03 | 0.14 | -0.124 |
| 0.036 | 1.0 | 33 | 0.92 | -0.005 |

Detectability [D + J]: SE of the fullness slope = sigma / sqrt(Sxx), sigma = between-station SD of
ln mean fullness. Not reported; J 0.3-0.6 (0.6 from the energy content: CV ~2.4 per fish x
~13 fish per station; fullness is usually less variable than energy content, J).
- 95 % largest intake elasticity compatible with the flat slope: 0.49 (sigma 0.3), 0.74 (0.45),
  0.99 (0.6). Hence X >= 1.0 / 0.35 / ~0 at 8.2 t/km2; K_menu <= 8 / 24 / unbounded t/km2 of total
  grab B (<= 1.6 / 4.7 reachable at share 0.2); a >= 0.0056 / 0.0019 / none (share 0.2), i.e. the
  model a is 6.5-19x above the field lower bound at sigma 0.3-0.45.
- Likelihood ratio of the observed ~0 slope, model a vs the Feeding-level a's: 73-6700 (sigma 0.3),
  7-50 (0.45), 3-9 (0.6). The model-scale a (0.016-0.092) are all compatible (LR 0.8-1.7).
Reading [J]: the Irish Sea gradient disfavours the low a; strength depends on sigma (unknown).
Two caveats [J]: (i) the prey:fish ratio was constant, so a RATIO-dependent food limitation also
predicts flat fullness - the test falsifies only a prey-dependent Holling term with no consumer
effect (the model's form, w = 0); (ii) the fullness/trawl regression was on trawling, not prey;
I translated through the prey-trawl slope, which wastes the residual prey variation.

Within-group conflict [F][A]: Hiddink et al. 2011 (abstract only, same fish) report plaice
condition falling with trawling, "explained by a reduced production of the infaunal invertebrates",
density-dependent competition ruled out; dab unaffected. Johnson 2015 found fullness and energy
flat in the same stomachs. Condition integrates weeks and responds to more than intake (prey
quality: plaice eat low-energy prey, 3.2 vs 19.2 J/g in dab [F]); it is not an intake response
[J]. Hiddink 2011 full text not obtained (Cloudflare).

## 3. Kattegat (scripts/kattegat_fishfood_patchiness.py, tables/kattegat_fishfood_patchiness_stdout.txt)

Hiddink et al. 2016 [A]: plaice prey fell more slowly than plaice with trawling, so food per
plaice rose, then fell above ~5 passes/yr; "This pattern was mirrored in both the condition and
stomach contents of plaice and for long-rough dab"; dab no effect; Nephrops condition rose as its
biomass fell. Full text and stomach data not public. The authors' slides (ICES symposium,
`data/ices_pres_hiddink/pres.txt`, work in progress) define fish food as individuals > 0.01 and
< 0.2 g WW and note "Need to discern between optimum curve and increasing variance" [F].

Re-run on that class (row mean mass WEIGHT/NO., 14-15 fine-sediment stations) [D]:
- fish food 11.6-77.9 t/km2 (7x), mean 54; 0.49 (0.09-0.73) of prey excl. large.
- plaice condition ~ ln food r2 0.00; LRD 0.05. ~ ln(food / own catch per haul): slopes NEGATIVE
  (plaice -0.009, r2 0.15; LRD -0.013, r2 0.19), the opposite sign to the paper's claim; ~ ln(food /
  flatfish catch) r2 0.02 / 0.11. Catch per haul is relative (q and swept area unknown).
- trawled (>4/yr) vs closed (<2/yr): fish food x0.66; plaice catch 0.65 -> 2.50 kg/haul.
- predicted intake I(11.6)/I(77.9): model a 0.92 (share 0.2) / 0.98 (1.0); Feeding-level a 0.18-0.48
  (share 0.2) / 0.29-0.80 (1.0).
Reading [J]: condition does not track food density or food per fish in the public data; I could not
reproduce the paper's ratio signal with Hiddink's own size class (his prey definition per species
and absolute fish densities are not public). Condition sensitivity to intake is not known, so this
is weak support for saturation, not a test.

## 4. Patchiness (same script) [D]

Kattegat fine muds, 5 x 0.1 m2 grabs per station:
- within-station CV between grabs: prey 0.35 (0.15-0.65), fish food 0.50 (0.30-1.45), Amphiura 0.35.
- between-station CV of station means (stations ~2-10 km apart, ~45 km area): prey 0.38, fish food
  0.42, Amphiura 0.48.
With the Jensen table of the Födonivå note (`tables/patchiness_jensen_stdout.txt`, gamma prey in a
cell): CV 0.5 costs 0-6 % of the mean-field intake at any x; CV 1 costs 2-19 % (most at x ~1-3). So the 1-km cell
mean is a fair stand-in for local density for these prey, and a field half-saturation read at
station scale converts to box-mean units within ~10 % [D/J]. Caveat: grab CV misses sub-0.1 m2
structure and the exposed fraction's own patchiness (arm tips, siphons), not measured.

## 5. Model frame (pair_K_bounds.py section 1) [D]

Brief's numbers; non-bed prey approximate (stage_access ~0.3 for planktivores, J).

| pair | x = a h B_vis | single-pair K = 1/(a h) (reachable) | own-B elasticity, shared (bed menu / full menu) | intake at 0.1 B_i |
|---|---|---|---|---|
| infauna | 75.8 | 0.25 t/km2 (1.24 of total B) | 0.58 / 0.62 | 0.16 / 0.15 |
| epifauna | 90.7 | 0.026 (0.042 of B) | 0.49 / 0.54 | 0.18 / 0.17 |
| Nephrops | 11.5 | 0.0079 (0.40 of B) | 0.94 / 0.94 | 0.11 |
| X | 178 (bed) / 197 (all) | | total-menu elasticity 0.006 | total at 0.5 x all prey: 0.994 |

So in the engine each pair's intake is already strongly density-dependent (Nephrops near linear),
while the TOTAL bed intake is saturated. The field evidence speaks to the total (Irish Sea fullness,
Smith 2013 total stomach index, Kattegat condition) and to the shares (diets track prey: Smith 2013,
Berenboim 2000, Johnson 2015 diet composition shifted, DISTLM p 0.007 [F]), and both match.

## 6. Supply vs demand on the exposed infauna [D]

Benthivore infauna consumption at field Q/B 4.46 and diet share 0.54: 1.53 x 4.46 x 0.54 = 3.7
t/km2/yr = 0.20 /yr of the exposed stock 18.8 t/km2, vs refuge exchange 2.5 /yr and infauna
production ~94 t/km2/yr. Box-mean depletion of the exposed pool is not what limits intake at the
field ration [J]; encounter-limited removal if fish searched all the time would be 80 /yr (gut-capped).
A ratio-dependent signal (Hiddink 2016) would need local depletion at the patch/home-range scale,
which the 1-km cells do not resolve (unquantified, gap).

## 7. Epifauna and Nephrops

Epifauna: no field FR in density units. Lab, Persson et al. 2012 [F, via the Mättnad note]:
juvenile cod 30-60 g on sand, a 0.065-0.13 (Crangon), 0.76-1.5 (Palaemon). Scaled to adult
benthivores by ~1/L (2.5-3x, J): a 0.022-0.052 (Crangon) and 0.25-0.60 (Palaemon); K = 1/(a h) at h
258: 0.0065-0.18 t/km2 accessible; x at today's 2.34 accessible: 13-363 [D]. The model a 0.15 and K
0.026 sit inside. Smith et al. 2013 [F]: haddock total stomach index showed no consistent sign over
5-250x epifauna contrasts, diets tracked prey (menu saturated, shares density-dependent).
Hvingel & Kingsley 2006 [F extract]: cod -> Pandalus at stock scale on the lower limb (x ~0.25-0.5,
Type III, P50 rests on a prior). In the shared form a minority prey's own-B elasticity is ~0.9-1
(section 5), so a near-linear per-prey stock response is the shared denominator, not evidence of a
low a [J]; it also includes overlap at 10^4-10^5 km2.
Nephrops: no field FR. Accessible 0.092 t/km2 (burrows, emergence); the pair is a minority prey, own-B
elasticity 0.94 for any a within ~10x of 0.44 [D]; the field bound is only the booked M2 (cod -> Nephrops
0.04-0.25 /yr in the Irish Sea, Mättnad note). No K obtainable.

## 8. Population-scale density dependence (context)

- North Sea plaice growth (Rijnsdorp & van Leeuwen 1996 [A]): growth of < 25 cm plaice rose from the
  1950s and fell in the 1980s, "significantly correlated with indices of plaice density,
  eutrophication, and seabed disturbance by beam trawling", not temperature; beam trawling dominated
  offshore. Small fish, population scale, a production (food supply) signal; no intake data.
- Rijnsdorp & Vingerhoed 2001 [A]: plaice and sole diets did not differ between grounds of different
  trawling intensity in and outside the plaice box.
- Shephard et al. 2014 [A]: Celtic Sea, plaice and lemon sole trophic level unaffected by fishing
  activity; whiting and megrim scavenge more on fished grounds.
- North Sea gadoids: no significant density-dependent growth (Lorenzen & Enberg 2002, earlier note).
- Baltic cod: condition linked to feeding opportunities on PELAGIC prey and to hypoxic area
  (Casini et al. 2016 [F], GAMs, 29.3 % deviance for the pelagic term); benthic feeding level among
  the lowest on record after benthic prey loss to hypoxia (Neuenfeldt et al. 2020 [A]); flounder
  competition for Saduria hypothesised (Haase et al. 2020 [A]). Collapse-scale contrasts, outside
  today's Skagerrak range [J].

## 9. Translation per pair (box-mean, the model's prey class)

| pair | field K (t/km2) | implied a = 1/(K h) | model a | model / field-implied | tau | sources |
|---|---|---|---|---|---|---|
| infauna (menu) | <= 8-24 of total B; <= 1.6-4.7 reachable (sigma 0.3-0.45; none at 0.6) | >= 0.0019-0.0056 (share 0.2); >= 0.0004-0.0011 (share 1) | 0.036 | >= 6.5-19 (0.2); >= 32-95 (1): compatible, not a point value | any (lower bound on foraging-time a) | Johnson 2015 [F]; Hinz data [D] |
| infauna, Kattegat | none (condition flat over 11.6-78 t/km2 fish food) | - | 0.036 | - | - | Hiddink 2016 [A]; Dryad [D] |
| epifauna | 0.0065-0.18 accessible (lab, scaled; J) | 0.022-0.60 | 0.15 | 0.25-7 | 1 within a lab trial | Persson 2012 [F]; Smith 2013 [F] |
| Nephrops | none | - | 0.44 | - | - | gap |

At the Skagerrak box (infauna 94 t/km2), the Irish Sea bound gives X_infauna >= 94 / (8-24) = 4-12
for a fish while it forages [D]: >= 0.8 of the ceiling, i.e. saturated in the field too; the field
feeding level 0.22-0.53 then means a foraging/time share of that order [J].

Caveats on section 9 [J]: (i) the Irish Sea bound is on PLAICE's foraging-time a; the model's 0.036
is a biomass-weighted whole-group value (haddock, cod, flatfish), so the comparison is order of
magnitude, not species-matched. (ii) The Skagerrak extrapolation assumes the Irish Sea grab class and
the model's infauna are comparable prey; the Irish Sea muds are 4-40x poorer and differ in
composition (Skagerrak/Kattegat rich in Amphiura). (iii) sigma is the open parameter: Johnson 2015
gives no station-level fullness means or SDs (checked; only the dab-plaice t 3.512, d.f. 27, without
the means), so the LR row cannot be pinned.

## 10. Model-side consequence [J]

Every fish group has `appetite: reserve_room` since 244.10, so the benthivores have no hunger gate
(h = 1). At X ~180 the engine's benthivore intake per tick is the learned eat share x the ceiling
(capped by reserve room), nothing else. "Time-limited" therefore means the field feeding level
0.22-0.53 of Cmax can only come from the policy's eat share (foraging time); no library parameter
produces it, and lowering a by 1-3 orders to produce it would contradict the Irish Sea gradient.
The check is the long-run probe's realised benthivore feeding level against 0.22-0.53, not a
parameter change.

## 11. Gaps

sigma of station-mean fullness (Johnson 2015 raw stomach data); Hiddink 2011 and 2016 full texts
and the Kattegat stomach data (ratio-dependence test); any lab or field FR of gadoids or flatfish on
infauna (polychaetes, siphons, brittle-star arms); adult-scale epifauna FR; Nephrops FR; local
(home-range) depletion of the exposed infauna fraction; a Skagerrak gradient (e.g. the Kosterfjord or
trawl-closure contrasts) with stomach data.
